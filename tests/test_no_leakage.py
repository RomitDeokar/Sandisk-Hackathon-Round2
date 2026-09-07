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
