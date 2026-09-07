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

    # 6. block_readings column validation
    if BLOCK_READINGS_COL in df.columns:
        if not (pd.api.types.is_string_dtype(df[BLOCK_READINGS_COL]) or pd.api.types.is_object_dtype(df[BLOCK_READINGS_COL])):
            errors.append(f"'{BLOCK_READINGS_COL}' must be string/object dtype, got {df[BLOCK_READINGS_COL].dtype}")
        elif expected_block_length is not None and len(df) > 0:
            sample_n = min(sample_block_rows, len(df))
            sampled = df[BLOCK_READINGS_COL].iloc[:sample_n]
            for idx, item in enumerate(sampled):
                if isinstance(item, str):
                    toks = item.split()
                    if len(toks) != expected_block_length:
                        errors.append(f"Sample row {idx} in block_readings has {len(toks)} tokens, expected {expected_block_length}")
                        break
                elif isinstance(item, (list, np.ndarray)):
                    if len(item) != expected_block_length:
                        errors.append(f"Sample row {idx} in block_readings has length {len(item)}, expected {expected_block_length}")
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
