"""Generate deliverables 4 and 5 from frozen models and saved OOF scores.

This command performs feature transformation and explanation only. It never calls
fit(), changes thresholds, writes predictions, or modifies model artifacts.
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sandisk_yield.analysis.imbalance import generate_imbalance_analysis
from sandisk_yield.explainability.block_explain import explain_block_batch
from sandisk_yield.explainability.tree_explain import get_feature_importances


IDS = ["wafer_id", "die_row", "die_col"]
warnings.filterwarnings("ignore", category=PerformanceWarning)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def representative_sample(test, pass_count=1000, seed=42):
    eligible = test[test.old_label == 0]
    failures = eligible[eligible.label == 1]
    passes = eligible[eligible.label == 0]
    if failures.empty:
        raise ValueError("No eligible failures found in test data")
    per_wafer = max(1, pass_count // max(1, passes.wafer_id.nunique()))
    sampled_passes = pd.concat([
        group.sample(n=min(per_wafer, len(group)), random_state=seed)
        for _, group in passes.groupby("wafer_id", sort=False)
    ])
    remaining = pass_count - len(sampled_passes)
    if remaining > 0:
        pool = passes.drop(sampled_passes.index)
        sampled_passes = pd.concat([
            sampled_passes, pool.sample(n=min(remaining, len(pool)), random_state=seed)
        ])
    sample = pd.concat([failures, sampled_passes]).sort_index()
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="input/train.csv")
    parser.add_argument("--test", default="input/test.csv")
    parser.add_argument("--passes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.passes < 1:
        parser.error("--passes must be positive")

    model_a_path = Path("outputs/models/model_a.joblib")
    model_b_path = Path("outputs/models/model_b_cnn.joblib")
    pipeline_path = Path("outputs/models/feature_pipeline.joblib")
    oof_path = Path("outputs/predictions/oof_predictions.csv")
    probability_path = Path("outputs/predictions/prediction_probabilities.csv")
    required = [model_a_path, model_b_path, pipeline_path, oof_path,
                probability_path, Path(args.train), Path(args.test)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing required frozen artifact/data: " + ", ".join(missing))

    model_hashes_before = {path.name: sha256(path) for path in (model_a_path, model_b_path, pipeline_path)}
    model_a = joblib.load(model_a_path)
    model_b = joblib.load(model_b_path)
    pipeline = joblib.load(pipeline_path)
    if getattr(model_b, "mode", None) != "cnn":
        raise ValueError("model_b_cnn.joblib is not a CNN adapter")

    # Exclude the very large block string while building tabular/spatial features.
    test = pd.read_csv(args.test, usecols=lambda column: column != "block_readings")
    sample = representative_sample(test, args.passes, args.seed)
    transformed_test = pipeline.transform(test)
    if list(transformed_test.columns) != list(model_a.feature_names_):
        raise ValueError("Current test feature schema does not match Model A")
    sample_features = transformed_test.loc[sample.index]

    importance = get_feature_importances(model_a, top_n=20)
    top_features = importance.feature.tolist()
    contributions = model_a.model.predict(sample_features, pred_contrib=True)
    if contributions.shape != (len(sample), len(model_a.feature_names_) + 1):
        raise ValueError("Unexpected LightGBM contribution matrix shape")
    contribution_frame = pd.DataFrame(
        contributions[:, :-1], index=sample.index, columns=model_a.feature_names_)
    shap_output = sample[IDS].join(contribution_frame[top_features])
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    shap_path = report_dir / "model_a_per_die_shap.csv"
    shap_output.to_csv(shap_path, index=False)

    spatial_features = [column for column in model_a.feature_names_ if column.startswith("spatial_")]
    if not spatial_features:
        raise ValueError("No spatial features found in current Model A schema")
    spatial_output = sample[IDS + ["label"]].copy()
    for column in spatial_features:
        spatial_output[column] = contribution_frame[column]
    spatial_output["spatial_total_shap"] = contribution_frame[spatial_features].sum(axis=1)
    spatial_csv = report_dir / "model_a_spatial_contributions.csv"
    spatial_output.to_csv(spatial_csv, index=False)

    representative = (sample[sample.label == 1].wafer_id.value_counts().index[0])
    wafer = spatial_output[spatial_output.wafer_id == representative]
    limit = float(np.quantile(np.abs(wafer.spatial_total_shap), 0.98)) or 1.0
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    points = ax.scatter(wafer.die_col, wafer.die_row, c=wafer.spatial_total_shap,
                        cmap="coolwarm", vmin=-limit, vmax=limit, s=38,
                        edgecolors="black", linewidths=0.2)
    ax.invert_yaxis()
    ax.set(title=f"Model A spatial SHAP contribution — {representative}",
           xlabel="Die column", ylabel="Die row")
    fig.colorbar(points, ax=ax, label="Sum of spatial-feature SHAP values (log-odds)")
    fig.tight_layout()
    spatial_plot = report_dir / "model_a_spatial_contribution.png"
    fig.savefig(spatial_plot)
    plt.close(fig)

    block_data = pd.read_csv(args.test, usecols=IDS + ["block_readings"])
    selected_blocks = sample[IDS].merge(block_data, on=IDS, validate="one_to_one", sort=False)
    predictions = pd.read_csv(probability_path, usecols=IDS + ["routed_to_model_b"])
    selected_blocks = selected_blocks.merge(predictions, on=IDS, validate="one_to_one", sort=False)
    region_lists = explain_block_batch(
        model_b, selected_blocks.block_readings.tolist(), top_n_regions=3,
        batch_size=args.batch_size)
    block_rows = []
    for (_, die), regions in zip(selected_blocks.iterrows(), region_lists):
        for region in regions:
            block_rows.append({**{key: die[key] for key in IDS},
                               "routed_to_model_b": bool(die.routed_to_model_b), **region})
    block_path = report_dir / "model_b_block_pattern_analysis_current.csv"
    pd.DataFrame(block_rows).to_csv(block_path, index=False)

    del transformed_test, sample_features, contribution_frame, block_data
    train = pd.read_csv(args.train, usecols=lambda column: column != "block_readings")
    eligible_train = train.old_label == 0
    oof = pd.read_csv(oof_path)
    eligible_keys = train.loc[eligible_train, IDS].reset_index(drop=True)
    if not eligible_keys.equals(oof[IDS].reset_index(drop=True)):
        raise ValueError("Current OOF predictions do not align with current train.csv")
    train_features = pipeline.transform(train).loc[eligible_train].reset_index(drop=True)
    oof = oof.reset_index(drop=True)
    generate_imbalance_analysis(
        oof, train_features[top_features], top_features,
        report_dir / "imbalance_analysis")

    model_hashes_after = {path.name: sha256(path) for path in (model_a_path, model_b_path, pipeline_path)}
    if model_hashes_before != model_hashes_after:
        raise RuntimeError("A frozen model artifact changed during analysis")
    policy = pd.read_csv(probability_path, usecols=["threshold_objective", "fn_fp_cost_ratio", "ensemble_mode"]).drop_duplicates()
    metadata = {
        "models_retrained": False,
        "sample_definition": f"all eligible test failures plus {args.passes} wafer-stratified eligible passes",
        "sample_rows": len(sample),
        "sample_failures": int(sample.label.sum()),
        "sample_passes": int((sample.label == 0).sum()),
        "representative_spatial_wafer": representative,
        "top_features": top_features,
        "spatial_features": spatial_features,
        "model_artifact_sha256": model_hashes_after,
        "submission_policy": policy.iloc[0].to_dict(),
        "interpretation_note": "LightGBM native TreeSHAP values are log-odds contributions; CNN attention is learned attention, not a causal attribution.",
    }
    (report_dir / "interpretability_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"interpretability_sample_rows": len(sample),
                      "failures": int(sample.label.sum()),
                      "passes": int((sample.label == 0).sum()),
                      "model_artifacts_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
