"""
tests/conftest.py
==================
Shared pytest fixtures for the validation test suite.

Fixtures build minimal synthetic DataFrames that mimic the exact output
of generate_data.py — without needing the actual WM-811K data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import (
    BLOCK_READINGS_COL,
    DEFAULT_NUM_BLOCK_READINGS,
    DEFAULT_NUM_FEATURES,
    feature_cols,
    train_cols,
    validation_cols,
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic block_readings string
# ---------------------------------------------------------------------------

def _make_block_str(n: int = DEFAULT_NUM_BLOCK_READINGS, label: int = 0) -> str:
    rng = np.random.default_rng(42)
    vals = rng.normal(100.0, 15.0, size=n)
    if label == 1:
        # inject anomaly in 5% of blocks (matches generator)
        anom_idx = rng.integers(0, n, size=max(1, int(n * 0.05)))
        vals[anom_idx] += 4.5
    return " ".join(f"{v:.2f}" for v in vals)


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def num_features() -> int:
    return 10  # Use small count for speed in tests


@pytest.fixture(scope="session")
def num_block_readings() -> int:
    return 20  # Use small count for speed in tests


@pytest.fixture()
def valid_train_df(num_features, num_block_readings) -> pd.DataFrame:
    """A perfectly valid train DataFrame with small feature/block counts."""
    rng = np.random.default_rng(0)
    n_rows = 500

    data = {
        "wafer_id": [f"W_F_{i:04d}" if i % 4 == 0 else f"W_N_{i:04d}" for i in range(n_rows)],
        "die_row": rng.integers(0, 30, size=n_rows),
        "die_col": rng.integers(0, 30, size=n_rows),
    }

    for fname in feature_cols(num_features):
        data[fname] = rng.normal(100.0, 15.0, size=n_rows)

    labels = (rng.random(n_rows) < 0.03).astype(int)
    old_labels = (labels & (rng.random(n_rows) < 0.5)).astype(int)
    data["old_label"] = old_labels
    data[BLOCK_READINGS_COL] = [
        _make_block_str(num_block_readings, lbl) for lbl in labels
    ]
    data["label"] = labels

    return pd.DataFrame(data)


@pytest.fixture()
def valid_validation_df(valid_train_df) -> pd.DataFrame:
    """Validation split = train without 'label'."""
    return valid_train_df.drop(columns=["label"])


@pytest.fixture()
def df_missing_feature_col(valid_train_df, num_features) -> pd.DataFrame:
    """Train df with one feature column removed."""
    return valid_train_df.drop(columns=[f"feature_{num_features}"])


@pytest.fixture()
def df_missing_label(valid_train_df) -> pd.DataFrame:
    """Train df with 'label' column removed."""
    return valid_train_df.drop(columns=["label"])


@pytest.fixture()
def df_bad_label_values(valid_train_df) -> pd.DataFrame:
    """Train df with a rogue label value (e.g. 2)."""
    df = valid_train_df.copy()
    df.loc[0, "label"] = 2
    return df


@pytest.fixture()
def df_bad_block_length(valid_train_df, num_block_readings) -> pd.DataFrame:
    """Train df where one row has wrong number of block readings."""
    df = valid_train_df.copy()
    # Give row 0 only 5 readings instead of num_block_readings
    df.loc[0, BLOCK_READINGS_COL] = "1.00 2.00 3.00 4.00 5.00"
    return df


@pytest.fixture()
def df_bad_block_not_float(valid_train_df, num_block_readings) -> pd.DataFrame:
    """Train df where one row has a non-float token in block_readings."""
    df = valid_train_df.copy()
    # Replace valid reading with text string (same length of tokens)
    bad_str = " ".join(["abc"] * num_block_readings)
    df.loc[0, BLOCK_READINGS_COL] = bad_str
    return df


@pytest.fixture()
def df_nulls_in_features(valid_train_df, num_features) -> pd.DataFrame:
    """Train df with NaN injected into a feature column."""
    df = valid_train_df.copy()
    df.loc[5, "feature_1"] = np.nan
    df.loc[10, "feature_2"] = np.nan
    return df


@pytest.fixture()
def df_null_in_wafer_id(valid_train_df) -> pd.DataFrame:
    """Train df with a null wafer_id."""
    df = valid_train_df.copy()
    df.loc[0, "wafer_id"] = None
    return df


@pytest.fixture()
def df_bad_wafer_id_format(valid_train_df) -> pd.DataFrame:
    """Train df where some wafer_ids don't follow W_F/W_N_NNNN pattern."""
    df = valid_train_df.copy()
    df.loc[0, "wafer_id"] = "wafer_1"
    df.loc[1, "wafer_id"] = "WF0001"
    return df


@pytest.fixture()
def df_float_label(valid_train_df) -> pd.DataFrame:
    """Train df where label is float (common CSV read artifact)."""
    df = valid_train_df.copy()
    df["label"] = df["label"].astype(float)
    return df


@pytest.fixture()
def df_empty() -> pd.DataFrame:
    """Empty DataFrame."""
    return pd.DataFrame()
