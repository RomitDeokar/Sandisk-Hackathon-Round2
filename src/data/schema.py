"""
src/data/schema.py
==================
Canonical schema definitions for the Sandisk Die Yield dataset.

All validation, loading, and feature engineering code should import
from this module rather than hardcoding column names or dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Column-group constants
# ---------------------------------------------------------------------------

ID_COLS: List[str] = ["wafer_id", "die_row", "die_col"]

# Parametric test features — default 500, configurable in config.yaml
FEATURE_PREFIX: str = "feature_"
DEFAULT_NUM_FEATURES: int = 500

LABEL_COLS: List[str] = ["old_label", "label"]

# block_readings: space-separated string of NUM_BLOCK_READINGS floats
BLOCK_READINGS_COL: str = "block_readings"
DEFAULT_NUM_BLOCK_READINGS: int = 2000

# Validation set drops 'label' but keeps 'old_label'
VALIDATION_DROPS: List[str] = ["label"]


def feature_cols(n: int = DEFAULT_NUM_FEATURES) -> List[str]:
    """Return the canonical list of feature column names."""
    return [f"{FEATURE_PREFIX}{i}" for i in range(1, n + 1)]


def train_cols(n: int = DEFAULT_NUM_FEATURES) -> List[str]:
    """Full ordered column list for train / test files."""
    return ID_COLS + feature_cols(n) + ["old_label", BLOCK_READINGS_COL, "label"]


def validation_cols(n: int = DEFAULT_NUM_FEATURES) -> List[str]:
    """Full ordered column list for validation file (no label)."""
    return ID_COLS + feature_cols(n) + ["old_label", BLOCK_READINGS_COL]


# ---------------------------------------------------------------------------
# Expected dtypes (after loading — before block_readings parsing)
# ---------------------------------------------------------------------------

@dataclass
class ColumnSpec:
    name: str
    dtype_kind: str          # 'O' = object/str, 'i' = int, 'f' = float
    nullable: bool = False
    description: str = ""


SCHEMA_SPECS: List[ColumnSpec] = [
    ColumnSpec("wafer_id",       "O", nullable=False, description="Wafer identifier string"),
    ColumnSpec("die_row",        "i", nullable=False, description="Row index on wafer grid"),
    ColumnSpec("die_col",        "i", nullable=False, description="Column index on wafer grid"),
    # feature_1 ... feature_N  are added dynamically (all float64)
    ColumnSpec("old_label",      "i", nullable=False, description="Pre-test label 0/1"),
    ColumnSpec("block_readings", "O", nullable=False, description="Space-separated block float string"),
    ColumnSpec("label",          "i", nullable=False, description="Post-test label 0/1 (TARGET)"),
]

# Wafer-ID prefix conventions
WAFER_ID_FAILURE_PREFIX: str = "W_F_"
WAFER_ID_NONE_PREFIX: str = "W_N_"

# Label semantics
LABEL_PASS: int = 0
LABEL_FAIL: int = 1

# Eligible dies for evaluation: only those with old_label == LABEL_PASS
ELIGIBLE_MASK_COL: str = "old_label"
ELIGIBLE_MASK_VALUE: int = LABEL_PASS
