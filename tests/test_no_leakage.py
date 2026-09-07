"""
tests/test_no_leakage.py
========================
Data leakage verification tests.
Mandatory Rules:
1. 'label' (ground truth) is NEVER used as an input feature in Model A or Model B.
2. Spatial context features use ONLY 'old_label', never 'label'.
3. Cross-validation splits strictly group by wafer_id with zero wafer overlap.
"""

import numpy as np
import pandas as pd
import pytest

from sandisk_yield.schema import create_eligible_mask, create_new_failure_target
from sandisk_yield.features.spatial import compute_spatial_features
from sandisk_yield.features.pipeline import FeaturePipeline
from sandisk_yield.data.splitter import make_grouped_folds, make_calibration_split


@pytest.fixture
def synthetic_wafer_df():
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "wafer_id": [f"W_F_{i:04d}" if i % 2 == 0 else f"W_N_{i:04d}" for i in range(n)],
        "die_row": rng.integers(0, 25, n),
        "die_col": rng.integers(0, 25, n),
        "feature_1": rng.normal(10, 2, n),
        "feature_2": rng.normal(50, 5, n),
        "old_label": rng.choice([0, 1], p=[0.9, 0.1], size=n),
        "label": rng.choice([0, 1], p=[0.85, 0.15], size=n),
    })


def test_target_mask_definition(synthetic_wafer_df):
    eligible = create_eligible_mask(synthetic_wafer_df)
    # Eligibility must strictly equal old_label == 0
    assert (eligible == (synthetic_wafer_df["old_label"] == 0)).all()


def test_spatial_features_no_target_leakage(synthetic_wafer_df):
    df_altered_target = synthetic_wafer_df.copy()
    # Invert final label
    df_altered_target["label"] = 1 - df_altered_target["label"]
    
    feats_orig = compute_spatial_features(synthetic_wafer_df)
    feats_altered = compute_spatial_features(df_altered_target)
    
    # Spatial features must be IDENTICAL because they must only rely on old_label
    pd.testing.assert_frame_equal(feats_orig, feats_altered)


def test_feature_pipeline_does_not_output_label(synthetic_wafer_df):
    pipeline = FeaturePipeline(scale_features=False)
    y, elig = create_new_failure_target(synthetic_wafer_df)
    pipeline.fit(synthetic_wafer_df.loc[elig], y=y.loc[elig])
    
    X_transformed = pipeline.transform(synthetic_wafer_df)
    assert "label" not in X_transformed.columns
    assert "old_label" not in X_transformed.columns


def test_grouped_folds_zero_wafer_overlap(synthetic_wafer_df):
    folds = make_grouped_folds(synthetic_wafer_df, n_splits=3, group_col="wafer_id")
    for tr_idx, va_idx in folds:
        tr_wafers = set(synthetic_wafer_df["wafer_id"].iloc[tr_idx])
        va_wafers = set(synthetic_wafer_df["wafer_id"].iloc[va_idx])
        assert len(tr_wafers.intersection(va_wafers)) == 0


def test_calibration_split_zero_wafer_overlap(synthetic_wafer_df):
    fit_df, calib_df = make_calibration_split(synthetic_wafer_df, calibration_ratio=0.3, group_col="wafer_id")
    fit_wafers = set(fit_df["wafer_id"])
    calib_wafers = set(calib_df["wafer_id"])
    assert len(fit_wafers.intersection(calib_wafers)) == 0


@pytest.mark.parametrize("ratio", [0, 1, -0.1, np.nan])
def test_calibration_ratio_validation(synthetic_wafer_df, ratio):
    with pytest.raises(ValueError):
        make_calibration_split(synthetic_wafer_df, calibration_ratio=ratio)


def test_blocks_cannot_leak_into_model_a(synthetic_wafer_df):
    from sandisk_yield.training.cross_validation import train_models_cv
    with pytest.raises(ValueError, match="cannot use block"):
        train_models_cv(synthetic_wafer_df, feature_params={"include_blocks": True})


def test_pipeline_transform_ignores_outcomes_and_block_readings(synthetic_wafer_df):
    df = synthetic_wafer_df.copy()
    y, eligible = create_new_failure_target(df)
    pipeline = FeaturePipeline(top_k_parametric=2).fit(df.loc[eligible], y.loc[eligible])
    before = pipeline.transform(df)
    df["label"] = 1 - df.label
    df["block_readings"] = "invalid unused expensive measurements"
    pd.testing.assert_frame_equal(before, pipeline.transform(df))


def test_production_validator_checks_late_blocks_and_ids():
    from sandisk_yield.data.validator import validate_dataframe, ValidationError
    df = pd.DataFrame({"wafer_id": ["w"] * 60, "die_row": range(60), "die_col": [0] * 60,
                       "old_label": [0] * 60, "label": [0] * 60, "feature_1": np.arange(60),
                       "block_readings": ["1 2 3 4"] * 60})
    assert validate_dataframe(df, expected_block_length=4)["status"] == "VALID"
    corrupt = df.copy()
    corrupt.loc[59, "block_readings"] = "1 2"
    with pytest.raises(ValidationError, match="row 59"):
        validate_dataframe(corrupt, expected_block_length=4)
    with pytest.raises(ValidationError, match="Duplicate die"):
        validate_dataframe(pd.concat([df, df.iloc[:1]], ignore_index=True), expected_block_length=4)
    with pytest.raises(ValidationError, match="Missing block"):
        validate_dataframe(df.drop(columns="block_readings"), expected_block_length=4)
    corrupt = df.copy()
    corrupt.loc[0, "die_row"] = -1
    with pytest.raises(ValidationError, match="nonnegative"):
        validate_dataframe(corrupt, expected_block_length=4)


@pytest.mark.parametrize("column", ["die_row", "die_col", "old_label", "label"])
def test_loader_rejects_fractional_integers(tmp_path, column):
    from sandisk_yield.data.loader import load_dataset
    df = pd.DataFrame({"wafer_id": ["w"], "die_row": [0], "die_col": [0], "old_label": [0], "label": [0]})
    df[column] = .7
    path = tmp_path / "invalid.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="integer"):
        load_dataset(path)


def make_synthetic_cluster_fixture(n_wafers=24, seed=2026, prefix="synthetic"):
    """Artificial clustered-block test fixture, NOT fab data or a production benchmark.

    Both classes have spikes; only their adjacency carries the synthetic label.
    This specifically exercises the information lost by five global summaries.
    """
    rng = np.random.default_rng(seed)
    records = []
    for wafer in range(n_wafers):
        failure_position = rng.integers(1, 25)
        for die in range(25):
            old = int(die == 0)
            failure = int(die == failure_position)
            arr = rng.normal(100, 1, 128)
            positions = (np.arange(3) + rng.integers(0, 126) if failure
                         else rng.choice(np.arange(0, 128, 4), size=3, replace=False))
            arr[positions] += 12
            records.append(dict(wafer_id=f"{prefix}_{wafer}", die_row=die // 5,
                                die_col=die % 5, old_label=old, label=max(old, failure),
                                feature_1=rng.normal(), feature_2=rng.normal(),
                                block_readings=" ".join(map(str, arr))))
    return pd.DataFrame(records)


def test_cli_train_reload_and_predict_without_training_labels(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    train = make_synthetic_cluster_fixture()
    inference = make_synthetic_cluster_fixture(3, seed=22, prefix="unseen").drop(columns="label")
    train.to_csv(tmp_path / "train.csv", index=False)
    inference.to_csv(tmp_path / "unlabeled.csv", index=False)
    cfg = yaml.safe_load((root / "configs/base.yaml").read_text())
    for key in cfg["paths"]:
        cfg["paths"][key] = str(tmp_path / key)
    cfg["paths"]["raw_train"] = str(tmp_path / "train.csv")
    cfg["data"]["num_block_readings"] = 128
    cfg["model_b"]["block_length"] = 128
    cfg["model_a"]["params"].update(n_estimators=12, n_jobs=1, min_child_samples=5)
    cfg["cv"]["n_splits"] = 3
    cfg["features"]["top_k_parametric"] = 2
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    def run(*args):
        process = subprocess.run([sys.executable, *map(str, args)], cwd=root, env=env,
                                 capture_output=True, text=True, timeout=120)
        assert process.returncode == 0, process.stdout + process.stderr
    run("scripts/run_all.py", "--config", config_path, "--compare-block-modes")
    model_dir = Path(cfg["paths"]["models_dir"])
    provenance = json.loads((model_dir / "training_provenance.json").read_text())
    assert provenance["n_splits"] == 3
    assert provenance["block_mode"] == "stats"
    assert set(provenance["fitting_wafers"]).isdisjoint(provenance["calibration_wafers"])
    oof_path = Path(cfg["paths"]["predictions_dir"]) / "oof_predictions.csv"
    oof = pd.read_csv(oof_path)
    assert set(oof.wafer_id).isdisjoint(provenance["calibration_wafers"])
    assert not oof.filter(like="probability").isna().any().any()
    # A deployable inference bundle must not require private training labels.
    oof_path.unlink()
    (tmp_path / "train.csv").unlink()
    run("scripts/predict.py", "--config", config_path, "--input", tmp_path / "unlabeled.csv",
        "--output", tmp_path / "submission.csv")
    sub = pd.read_csv(tmp_path / "submission.csv")
    assert len(sub) == len(inference)
    assert (sub.loc[inference.old_label == 1, "predicted_label"] == 1).all()
    prob = pd.read_csv(tmp_path / "prediction_probabilities.csv")
    assert np.isfinite(prob.final_probability).all()
    assert prob.loc[prob.old_label == 0, "routed_to_model_b"].all()  # conservative small-sample gate
    run("scripts/predict.py", "--config", config_path, "--from-probabilities",
        tmp_path / "prediction_probabilities.csv", "--output", tmp_path / "rethresh.csv")
    pd.testing.assert_frame_equal(sub, pd.read_csv(tmp_path / "rethresh.csv"))
