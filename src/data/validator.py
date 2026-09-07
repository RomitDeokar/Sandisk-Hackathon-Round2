"""
src/data/validator.py
======================
Data validation layer for the Sandisk Die Yield dataset.

Validates that a loaded DataFrame matches the expected schema from
generate_data.py / config.yaml. Raises DataValidationError with a
structured report of every discovered problem.

Usage
-----
    from src.data.validator import validate_dataset, DataValidationError

    try:
        report = validate_dataset(df, split="train", num_features=500)
    except DataValidationError as e:
        print(e.report)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.schema import (
    BLOCK_READINGS_COL,
    DEFAULT_NUM_BLOCK_READINGS,
    DEFAULT_NUM_FEATURES,
    FEATURE_PREFIX,
    ID_COLS,
    LABEL_FAIL,
    LABEL_PASS,
    feature_cols,
    train_cols,
    validation_cols,
)


# ---------------------------------------------------------------------------
# Error / Report types
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: str          # "ERROR" | "WARNING" | "INFO"
    check: str             # short check name
    message: str


@dataclass
class ValidationReport:
    split: str
    n_rows: int
    n_cols: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        lines = [
            f"=== Validation Report: {self.split} ===",
            f"  Rows : {self.n_rows:,}",
            f"  Cols : {self.n_cols}",
            f"  Status: {'PASS' if self.passed else 'FAIL'}",
            "",
        ]
        for issue in self.issues:
            lines.append(f"  [{issue.severity}] {issue.check}: {issue.message}")
        return "\n".join(lines)


class DataValidationError(Exception):
    """Raised when a DataFrame fails critical schema checks."""

    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(str(report))


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
    split: str = "train",
    num_features: int = DEFAULT_NUM_FEATURES,
    num_block_readings: int = DEFAULT_NUM_BLOCK_READINGS,
    sample_block_rows: int = 200,
    raise_on_error: bool = True,
) -> ValidationReport:
    """
    Validate a loaded dataset DataFrame against the generator schema.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset (train, test, or validation).
    split : str
        One of 'train', 'test', 'validation'. Determines which columns
        are expected (validation has no 'label').
    num_features : int
        Number of parametric feature columns expected.
    num_block_readings : int
        Expected number of floats in each block_readings string.
    sample_block_rows : int
        Number of rows to sample when checking block_readings content.
    raise_on_error : bool
        If True, raise DataValidationError when any ERROR-level issue found.

    Returns
    -------
    ValidationReport
        Structured report with issues by severity.
    """
    report = ValidationReport(split=split, n_rows=len(df), n_cols=len(df.columns))
    is_validation = split == "validation"

    # --- 1. Required columns present ---
    _check_columns(df, report, split, num_features, is_validation)

    # --- 2. No unexpected extra columns ---
    _check_no_extra_columns(df, report, num_features, is_validation)

    # --- 3. ID column dtypes ---
    _check_id_dtypes(df, report)

    # --- 4. Feature column dtypes (float) ---
    _check_feature_dtypes(df, report, num_features)

    # --- 5. Label column checks (only for train/test) ---
    if not is_validation:
        _check_label_columns(df, report)

    # --- 6. old_label column (present in all splits) ---
    _check_old_label(df, report)

    # --- 7. block_readings column ---
    _check_block_readings(df, report, num_block_readings, sample_block_rows)

    # --- 8. No NaN in critical columns ---
    _check_no_nulls(df, report, num_features, is_validation)

    # --- 9. Wafer-ID format ---
    _check_wafer_id_format(df, report)

    # --- 10. Class balance info ---
    if not is_validation:
        _report_class_balance(df, report)

    # --- 11. Row count sanity ---
    _check_row_count(df, report)

    if raise_on_error and not report.passed:
        raise DataValidationError(report)

    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _add(report: ValidationReport, severity: str, check: str, msg: str) -> None:
    report.issues.append(ValidationIssue(severity=severity, check=check, message=msg))


def _check_columns(
    df: pd.DataFrame,
    report: ValidationReport,
    split: str,
    num_features: int,
    is_validation: bool,
) -> None:
    expected = set(validation_cols(num_features) if is_validation else train_cols(num_features))
    actual = set(df.columns)
    missing = expected - actual
    if missing:
        # Distinguish feature-column mismatches from structural mismatches
        missing_feats = sorted(c for c in missing if c.startswith(FEATURE_PREFIX))
        missing_struct = sorted(c for c in missing if not c.startswith(FEATURE_PREFIX))
        if missing_struct:
            _add(report, "ERROR", "missing_structural_cols",
                 f"Missing structural columns: {missing_struct}")
        if missing_feats:
            _add(report, "ERROR", "missing_feature_cols",
                 f"Missing {len(missing_feats)} feature columns "
                 f"(first: {missing_feats[0]}, last: {missing_feats[-1]})")
    else:
        _add(report, "INFO", "required_columns", f"All {len(expected)} expected columns present")


def _check_no_extra_columns(
    df: pd.DataFrame,
    report: ValidationReport,
    num_features: int,
    is_validation: bool,
) -> None:
    expected = set(validation_cols(num_features) if is_validation else train_cols(num_features))
    extra = sorted(set(df.columns) - expected)
    if extra:
        _add(report, "WARNING", "extra_columns",
             f"Unexpected extra columns (will be ignored): {extra}")


def _check_id_dtypes(df: pd.DataFrame, report: ValidationReport) -> None:
    for col in ID_COLS:
        if col not in df.columns:
            continue
        if col == "wafer_id":
            if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                _add(report, "ERROR", "id_dtype",
                     f"'{col}' should be string/object, got {df[col].dtype}")
        else:
            if not pd.api.types.is_integer_dtype(df[col]):
                _add(report, "ERROR", "id_dtype",
                     f"'{col}' should be integer, got {df[col].dtype}")


def _check_feature_dtypes(
    df: pd.DataFrame, report: ValidationReport, num_features: int
) -> None:
    feat_cols = [c for c in feature_cols(num_features) if c in df.columns]
    bad = [c for c in feat_cols if not pd.api.types.is_float_dtype(df[c])]
    if bad:
        _add(report, "ERROR", "feature_dtypes",
             f"{len(bad)} feature cols not float64 (e.g. {bad[:3]})")
    else:
        _add(report, "INFO", "feature_dtypes",
             f"All {len(feat_cols)} feature columns are float dtype")


def _check_label_columns(df: pd.DataFrame, report: ValidationReport) -> None:
    for col in ["label", "old_label"]:
        if col not in df.columns:
            continue
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({LABEL_PASS, LABEL_FAIL}):
            bad_vals = unique_vals - {LABEL_PASS, LABEL_FAIL}
            _add(report, "ERROR", f"{col}_values",
                 f"'{col}' contains unexpected values: {bad_vals}. Expected {{0, 1}}")
        else:
            _add(report, "INFO", f"{col}_values",
                 f"'{col}' contains only valid values {{0, 1}}")


def _check_old_label(df: pd.DataFrame, report: ValidationReport) -> None:
    if "old_label" not in df.columns:
        _add(report, "ERROR", "old_label_present", "'old_label' column is missing")
        return
    unique_vals = set(df["old_label"].dropna().unique())
    if not unique_vals.issubset({LABEL_PASS, LABEL_FAIL}):
        _add(report, "ERROR", "old_label_values",
             f"'old_label' has unexpected values: {unique_vals - {0, 1}}")


def _check_block_readings(
    df: pd.DataFrame,
    report: ValidationReport,
    num_block_readings: int,
    sample_rows: int,
) -> None:
    col = BLOCK_READINGS_COL
    if col not in df.columns:
        _add(report, "ERROR", "block_readings_present", f"'{col}' column is missing")
        return

    if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
        _add(report, "ERROR", "block_readings_dtype",
             f"'{col}' should be object/string dtype, got {df[col].dtype}")
        return

    # Sample some rows to verify content
    n_sample = min(sample_rows, len(df))
    sample_idx = np.random.default_rng(0).choice(len(df), size=n_sample, replace=False)
    sample_df = df.iloc[sample_idx]

    length_errors: List[str] = []
    parse_errors: List[str] = []

    for idx, val in zip(sample_idx, sample_df[col]):
        if not isinstance(val, str):
            parse_errors.append(f"row {idx}: not a string ({type(val).__name__})")
            continue
        tokens = val.split()
        if len(tokens) != num_block_readings:
            length_errors.append(
                f"row {idx}: expected {num_block_readings} values, got {len(tokens)}"
            )
        else:
            # Try parsing first and last few tokens
            try:
                _ = [float(t) for t in tokens[:5] + tokens[-5:]]
            except ValueError as exc:
                parse_errors.append(f"row {idx}: parse error — {exc}")

    if parse_errors:
        _add(report, "ERROR", "block_readings_parse",
             f"{len(parse_errors)} rows have non-parseable block_readings "
             f"(sample: {parse_errors[:2]})")
    if length_errors:
        _add(report, "ERROR", "block_readings_length",
             f"{len(length_errors)} rows have wrong block count "
             f"(sample: {length_errors[:2]})")
    if not parse_errors and not length_errors:
        _add(report, "INFO", "block_readings",
             f"block_readings format OK — sampled {n_sample} rows, "
             f"each has {num_block_readings} float values")


def _check_no_nulls(
    df: pd.DataFrame,
    report: ValidationReport,
    num_features: int,
    is_validation: bool,
) -> None:
    # Critical columns that must never be null
    critical = ID_COLS + [BLOCK_READINGS_COL, "old_label"]
    if not is_validation:
        critical.append("label")

    for col in critical:
        if col not in df.columns:
            continue
        n_null = df[col].isna().sum()
        if n_null > 0:
            _add(report, "ERROR", "nulls",
                 f"'{col}' has {n_null:,} null values ({n_null/len(df)*100:.2f}%)")

    # Feature nulls (just warn, don't fail)
    feat_cols_present = [c for c in feature_cols(num_features) if c in df.columns]
    if feat_cols_present:
        total_feat_nulls = df[feat_cols_present].isna().sum().sum()
        if total_feat_nulls > 0:
            _add(report, "WARNING", "feature_nulls",
                 f"Feature columns contain {total_feat_nulls:,} null values total")
        else:
            _add(report, "INFO", "feature_nulls", "No null values in feature columns")


def _check_wafer_id_format(df: pd.DataFrame, report: ValidationReport) -> None:
    if "wafer_id" not in df.columns:
        return
    pattern = re.compile(r"^W_[FN]_\d{4}$")
    bad = df["wafer_id"][~df["wafer_id"].str.match(pattern, na=False)]
    if len(bad) > 0:
        unique_bad = bad.unique()[:5]
        _add(report, "WARNING", "wafer_id_format",
             f"{len(bad):,} rows have non-standard wafer_id format "
             f"(examples: {unique_bad.tolist()}). Expected W_F_NNNN or W_N_NNNN")
    else:
        _add(report, "INFO", "wafer_id_format",
             "All wafer_ids match expected W_F_NNNN / W_N_NNNN pattern")


def _report_class_balance(df: pd.DataFrame, report: ValidationReport) -> None:
    if "label" not in df.columns:
        return
    n_total = len(df)
    n_fail = int(df["label"].sum())
    n_pass = n_total - n_fail
    fail_pct = n_fail / n_total * 100 if n_total > 0 else 0.0

    _add(report, "INFO", "class_balance",
         f"label distribution — pass: {n_pass:,} ({100-fail_pct:.2f}%), "
         f"fail: {n_fail:,} ({fail_pct:.2f}%)")

    # Eligibility: only old_label=0 dies count for evaluation
    if "old_label" in df.columns:
        eligible = df[df["old_label"] == LABEL_PASS]
        n_elig = len(eligible)
        n_elig_fail = int(eligible["label"].sum())
        elig_fail_pct = n_elig_fail / n_elig * 100 if n_elig > 0 else 0.0
        _add(report, "INFO", "eligible_class_balance",
             f"eligible (old_label=0) — {n_elig:,} dies, "
             f"fail: {n_elig_fail:,} ({elig_fail_pct:.2f}%)")

    if fail_pct > 20:
        _add(report, "WARNING", "class_balance",
             f"Fail rate {fail_pct:.2f}% is unexpectedly high (expected ~3%)")


def _check_row_count(df: pd.DataFrame, report: ValidationReport) -> None:
    if len(df) == 0:
        _add(report, "ERROR", "row_count", "DataFrame is empty (0 rows)")
    elif len(df) < 1000:
        _add(report, "WARNING", "row_count",
             f"Only {len(df):,} rows — unusually small for a wafer dataset")
    else:
        _add(report, "INFO", "row_count", f"{len(df):,} rows — looks reasonable")
