"""
src/sandisk_yield/data/validator.py
====================================
Comprehensive schema, type, and distribution validation.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from sandisk_yield.schema import (
    ID_COLS,
    OLD_LABEL_COL,
    TARGET_COL,
    BLOCK_READINGS_COL,
    detect_feature_cols,
    LABEL_PASS,
    LABEL_FAIL,
)


class ValidationError(Exception):
    pass


def validate_dataframe(
    df: pd.DataFrame,
    is_training: bool = True,
    expected_block_length: Optional[int] = None,
    sample_block_rows: int = 50,
) -> Dict[str, any]:
    """
    Validates DataFrame against required schema, types, and constraints.
    Returns summary metrics or raises ValidationError.
    """
    errors: List[str] = []
    if df.empty:
        errors.append("Dataset must not be empty")
    if not df.columns.is_unique:
        raise ValidationError("Duplicate column names")
    if not df.index.is_unique:
        errors.append("Duplicate row index")
    warnings: List[str] = []

    # 1. Structural ID columns
    for col in ID_COLS:
        if col not in df.columns:
            errors.append(f"Missing required ID column: '{col}'")

    # 2. old_label column
    if OLD_LABEL_COL not in df.columns:
        errors.append(f"Missing required '{OLD_LABEL_COL}' column")
    else:
        unique_old = set(df[OLD_LABEL_COL].dropna().unique())
        if not unique_old.issubset({LABEL_PASS, LABEL_FAIL}):
            errors.append(f"Invalid values in '{OLD_LABEL_COL}': {unique_old}. Expected {{0, 1}}")

    # 3. Target column (if training)
    if is_training:
        if TARGET_COL not in df.columns:
            errors.append(f"Missing required target column '{TARGET_COL}' for training data")
        else:
            unique_target = set(df[TARGET_COL].dropna().unique())
            if not unique_target.issubset({LABEL_PASS, LABEL_FAIL}):
                errors.append(f"Invalid values in '{TARGET_COL}': {unique_target}. Expected {{0, 1}}")

    # 4. Parametric feature columns
    feat_cols = detect_feature_cols(df)
    if len(feat_cols) == 0:
        errors.append("No parametric feature columns (feature_*) detected")

    # 5. Null checks in critical columns
    crit_cols = [c for c in ID_COLS + [OLD_LABEL_COL] if c in df.columns]
    if is_training and TARGET_COL in df.columns:
        crit_cols.append(TARGET_COL)
    for c in crit_cols:
        n_null = df[c].isna().sum()
        if n_null > 0:
            errors.append(f"Found {n_null} null values in critical column '{c}'")

    # Grid indexing requires unique, nonnegative integer coordinates.
    if all(c in df for c in ID_COLS):
        if df.duplicated(ID_COLS).any():
            errors.append("Duplicate die IDs (wafer_id, die_row, die_col)")
        for col in ("die_row", "die_col"):
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if (not np.isfinite(vals).all() or (vals < 0).any()
                    or (vals != np.floor(vals)).any()
                    or not pd.api.types.is_integer_dtype(df[col])):
                errors.append(f"'{col}' must contain nonnegative integer coordinates (integer dtype)")
    if is_training and OLD_LABEL_COL in df and TARGET_COL in df:
        if ((df[OLD_LABEL_COL] == 1) & (df[TARGET_COL] == 0)).any():
            errors.append("An old failure cannot become a passing final label")

    # Validate EVERY row. Sampling can miss corrupt late rows and silently
    # fabricated padding would change model predictions. Keep sample_block_rows
    # in the signature for backward compatibility, but no longer sample.
    if BLOCK_READINGS_COL not in df:
        if expected_block_length is not None:
            errors.append("Missing block_readings required by Model B")
    else:
        from sandisk_yield.features.blocks import parse_block_readings_row
        for idx, item in df[BLOCK_READINGS_COL].items():
            try:
                parse_block_readings_row(item, expected_block_length or 0)
            except (ValueError, TypeError) as exc:
                errors.append(f"Invalid block_readings at row {idx}: {exc}")
                break

    if errors:
        raise ValidationError("Dataset validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return {
        "status": "VALID",
        "n_rows": len(df),
        "n_wafers": df["wafer_id"].nunique() if "wafer_id" in df.columns else 0,
        "n_features": len(feat_cols),
        "warnings": warnings,
    }
