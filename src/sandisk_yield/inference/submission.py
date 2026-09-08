"""
src/sandisk_yield/inference/submission.py
========================================
Submission generation and rigorous schema validation.
"""

from pathlib import Path
from typing import List, Union
import pandas as pd
import numpy as np


class SubmissionValidationError(Exception):
    pass


def validate_submission_df(
    sub_df: pd.DataFrame,
    ref_df: pd.DataFrame
) -> None:
    """
    Rigorously validates submission against reference raw input:
    - Exact required columns: wafer_id, die_row, die_col, predicted_label
    - Row count matches exactly
    - No duplicate die identifiers
    - Binary predicted_label (0 or 1)
    - Wafer IDs and coordinate alignments match
    - No missing values
    """
    errors: List[str] = []
    
    # Required columns
    expected_cols = ["wafer_id", "die_row", "die_col", "predicted_label"]
    if list(sub_df.columns) != expected_cols:
        errors.append(f"Columns mismatch: expected {expected_cols}, got {list(sub_df.columns)}")
        
    # Row count
    if len(sub_df) != len(ref_df):
        errors.append(f"Row count mismatch: submission has {len(sub_df):,} rows, expected {len(ref_df):,}")
        
    if errors:
        raise SubmissionValidationError("Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    # Null values
    if sub_df.isna().sum().sum() > 0:
        errors.append(f"Found {sub_df.isna().sum().sum()} null values in submission")
        
    # Binary predictions
    if "predicted_label" in sub_df.columns:
        unique_preds = set(sub_df["predicted_label"].unique())
        if not unique_preds.issubset({0, 1}):
            errors.append(f"Predicted labels must be 0 or 1, found: {unique_preds}")
        
    # Duplicate dies check
    dups = sub_df.duplicated(subset=["wafer_id", "die_row", "die_col"]).sum()
    if dups > 0:
        errors.append(f"Found {dups} duplicate die coordinate records")
        
    # Coordinate alignment with reference
    if len(sub_df) == len(ref_df):
        if "wafer_id" in sub_df.columns and "wafer_id" in ref_df.columns:
            if not (sub_df["wafer_id"].to_numpy() == ref_df["wafer_id"].to_numpy()).all():
                errors.append("Wafer ID ordering does not match reference dataframe")
        if "die_row" in sub_df.columns and "die_row" in ref_df.columns:
            if not (sub_df["die_row"].to_numpy() == ref_df["die_row"].to_numpy()).all():
                errors.append("die_row ordering does not match reference dataframe")
        # Bug: rows in the same wafer/row could be permuted undetected.
        if not np.array_equal(sub_df["die_col"].to_numpy(), ref_df["die_col"].to_numpy()):
            errors.append("die_col ordering does not match reference dataframe")
            
    if errors:
        raise SubmissionValidationError("Submission validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def export_submission(
    results_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    submission_path: Union[str, Path] = "outputs/predictions/submission.csv",
    probabilities_path: Union[str, Path] = "outputs/predictions/prediction_probabilities.csv"
) -> None:
    """
    Validates and exports final submission.csv and prediction_probabilities.csv.
    """
    sub_cols = ["wafer_id", "die_row", "die_col", "predicted_label"]
    sub_df = results_df[sub_cols].copy()
    
    # Cast types
    sub_df["die_row"] = sub_df["die_row"].astype(int)
    sub_df["die_col"] = sub_df["die_col"].astype(int)
    sub_df["predicted_label"] = sub_df["predicted_label"].astype(int)
    
    # Run strict validation
    validate_submission_df(sub_df, ref_df)
    
    # Write submission.csv
    p_sub = Path(submission_path)
    p_sub.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(p_sub, index=False)
    
    # Write prediction_probabilities.csv
    p_prob = Path(probabilities_path)
    p_prob.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(p_prob, index=False)
