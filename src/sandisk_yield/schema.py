"""
src/sandisk_yield/schema.py
===========================
Canonical schema definitions, constants, and target/eligibility masking.
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np

# Column constants
ID_COLS: List[str] = ["wafer_id", "die_row", "die_col"]
FEATURE_PREFIX: str = "feature_"
OLD_LABEL_COL: str = "old_label"
TARGET_COL: str = "label"
BLOCK_READINGS_COL: str = "block_readings"

# Label semantics
LABEL_PASS: int = 0
LABEL_FAIL: int = 1


def detect_feature_cols(df: pd.DataFrame, prefix: str = FEATURE_PREFIX) -> List[str]:
    """Dynamically detect all parametric feature column names."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    # Sort naturally by numerical suffix if present
    def sort_key(c: str):
        suffix = c[len(prefix):]
        return int(suffix) if suffix.isdigit() else suffix
    try:
        return sorted(cols, key=sort_key)
    except Exception:
        return sorted(cols)


def create_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """
    Returns boolean Series of eligible dies (old_label == 0).
    Only eligible dies are evaluated for new failure prediction.
    """
    if OLD_LABEL_COL not in df.columns:
        raise KeyError(f"Expected column '{OLD_LABEL_COL}' to determine eligibility mask")
    return df[OLD_LABEL_COL] == LABEL_PASS


def create_new_failure_target(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Extracts the new-failure prediction target and eligibility mask.
    Returns (target, eligible_mask).
    target is only meaningful where eligible_mask is True.
    """
    eligible_mask = create_eligible_mask(df)
    if TARGET_COL in df.columns:
        target = df[TARGET_COL].copy()
    else:
        target = pd.Series(np.nan, index=df.index, dtype=float)
    return target, eligible_mask
