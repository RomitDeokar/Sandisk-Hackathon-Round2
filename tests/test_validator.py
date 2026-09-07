"""
tests/test_validator.py
========================
Tests for src/data/validator.py.

Every check function in the validator is exercised through a combination
of the "happy path" (valid data) and targeted "corrupt" fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.schema import BLOCK_READINGS_COL, feature_cols
from src.data.validator import (
    DataValidationError,
    ValidationReport,
    validate_dataset,
)


# ============================================================
# Happy path: valid DataFrames should pass with no ERRORs
# ============================================================

class TestValidTrainDf:
    def test_returns_report(self, valid_train_df, num_features, num_block_readings):
        report = validate_dataset(
            valid_train_df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert isinstance(report, ValidationReport)

    def test_no_errors(self, valid_train_df, num_features, num_block_readings):
        report = validate_dataset(
            valid_train_df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert report.passed, f"Expected no errors, got:\n{report}"

    def test_has_info_messages(self, valid_train_df, num_features, num_block_readings):
        report = validate_dataset(
            valid_train_df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        info = [i for i in report.issues if i.severity == "INFO"]
        assert len(info) > 0, "Expected at least one INFO message from the validator"

    def test_correct_row_count(self, valid_train_df, num_features, num_block_readings):
        report = validate_dataset(
            valid_train_df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert report.n_rows == len(valid_train_df)

    def test_correct_col_count(self, valid_train_df, num_features, num_block_readings):
        report = validate_dataset(
            valid_train_df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert report.n_cols == len(valid_train_df.columns)


class TestValidValidationDf:
    def test_no_errors(self, valid_validation_df, num_features, num_block_readings):
        report = validate_dataset(
            valid_validation_df, split="validation",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert report.passed, f"Expected no errors, got:\n{report}"

    def test_label_not_required_for_validation(self, valid_validation_df, num_features, num_block_readings):
        """Validation split should NOT require a 'label' column."""
        assert "label" not in valid_validation_df.columns
        report = validate_dataset(
            valid_validation_df, split="validation",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert report.passed


# ============================================================
# Column presence checks
# ============================================================

class TestMissingColumns:
    def test_missing_feature_col_raises(self, df_missing_feature_col, num_features, num_block_readings):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                df_missing_feature_col, split="train",
                num_features=num_features, num_block_readings=num_block_readings,
                raise_on_error=True,
            )
        report = exc_info.value.report
        error_checks = [i.check for i in report.errors]
        assert "missing_feature_cols" in error_checks or "missing_structural_cols" in error_checks

    def test_missing_label_raises_for_train(self, df_missing_label, num_features, num_block_readings):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                df_missing_label, split="train",
                num_features=num_features, num_block_readings=num_block_readings,
                raise_on_error=True,
            )
        report = exc_info.value.report
        assert not report.passed

    def test_missing_feature_no_raise(self, df_missing_feature_col, num_features, num_block_readings):
        report = validate_dataset(
            df_missing_feature_col, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert not report.passed

    def test_extra_columns_are_warning(self, valid_train_df, num_features, num_block_readings):
        df_extra = valid_train_df.copy()
        df_extra["extra_col"] = 0
        report = validate_dataset(
            df_extra, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        # Extra column should be WARNING, not ERROR
        assert report.passed
        warn_checks = [i.check for i in report.warnings]
        assert "extra_columns" in warn_checks


# ============================================================
# Label value checks
# ============================================================

class TestLabelValues:
    def test_bad_label_value_raises(self, df_bad_label_values, num_features, num_block_readings):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                df_bad_label_values, split="train",
                num_features=num_features, num_block_readings=num_block_readings,
                raise_on_error=True,
            )
        report = exc_info.value.report
        assert any("label_values" in i.check for i in report.errors)

    def test_float_label_passes_after_cast(self, num_features, num_block_readings, valid_train_df):
        """Float label (CSV artifact) should pass after int cast by loader."""
        df = valid_train_df.copy()
        df["label"] = df["label"].astype(float)  # simulate CSV read
        df["label"] = df["label"].astype(int)    # simulate loader cast
        report = validate_dataset(
            df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert report.passed


# ============================================================
# Block readings checks
# ============================================================

class TestBlockReadings:
    def test_wrong_block_length_raises(self, df_bad_block_length, num_features, num_block_readings):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                df_bad_block_length, split="train",
                num_features=num_features, num_block_readings=num_block_readings,
                raise_on_error=True,
                sample_block_rows=500,  # ensure the bad row is sampled
            )
        report = exc_info.value.report
        checks = [i.check for i in report.errors]
        assert "block_readings_length" in checks

    def test_non_float_block_raises(self, df_bad_block_not_float, num_features, num_block_readings):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                df_bad_block_not_float, split="train",
                num_features=num_features, num_block_readings=num_block_readings,
                raise_on_error=True,
                sample_block_rows=500,
            )
        report = exc_info.value.report
        checks = [i.check for i in report.errors]
        assert "block_readings_parse" in checks

    def test_block_readings_missing_col(self, valid_train_df, num_features, num_block_readings):
        df = valid_train_df.drop(columns=[BLOCK_READINGS_COL])
        report = validate_dataset(
            df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert not report.passed
        checks = [i.check for i in report.errors]
        assert "block_readings_present" in checks or "missing_structural_cols" in checks

    def test_block_readings_parse_correct_format(self, num_features, num_block_readings):
        """Verify that a correctly formatted block_readings string is accepted."""
        vals = " ".join(f"{v:.2f}" for v in np.random.default_rng(1).normal(100, 15, num_block_readings))
        n_rows = 50
        rng = np.random.default_rng(1)
        data = {"wafer_id": [f"W_N_{i:04d}" for i in range(n_rows)],
                "die_row": rng.integers(0, 10, n_rows),
                "die_col": rng.integers(0, 10, n_rows)}
        for f in feature_cols(num_features):
            data[f] = rng.normal(0, 1, n_rows)
        data["old_label"] = np.zeros(n_rows, dtype=int)
        data["label"] = np.zeros(n_rows, dtype=int)
        data[BLOCK_READINGS_COL] = [vals] * n_rows
        df = pd.DataFrame(data)
        report = validate_dataset(
            df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
            sample_block_rows=n_rows,
        )
        assert report.passed, str(report)


# ============================================================
# Null checks
# ============================================================

class TestNullChecks:
    def test_null_in_wafer_id_raises(self, df_null_in_wafer_id, num_features, num_block_readings):
        with pytest.raises(DataValidationError):
            validate_dataset(
                df_null_in_wafer_id, split="train",
                num_features=num_features, num_block_readings=num_block_readings,
                raise_on_error=True,
            )

    def test_null_in_feature_is_warning(self, df_nulls_in_features, num_features, num_block_readings):
        report = validate_dataset(
            df_nulls_in_features, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        # Feature nulls → WARNING only, report still passes
        warn_checks = [i.check for i in report.warnings]
        assert "feature_nulls" in warn_checks
        assert report.passed


# ============================================================
# Wafer ID format checks
# ============================================================

class TestWaferIdFormat:
    def test_bad_format_is_warning(self, df_bad_wafer_id_format, num_features, num_block_readings):
        report = validate_dataset(
            df_bad_wafer_id_format, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        warn_checks = [i.check for i in report.warnings]
        assert "wafer_id_format" in warn_checks
        # Should still pass (not an error)
        assert report.passed


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_empty_dataframe_raises(self, df_empty):
        with pytest.raises(DataValidationError):
            validate_dataset(
                df_empty, split="train",
                raise_on_error=True,
            )

    def test_empty_dataframe_no_raise(self, df_empty):
        report = validate_dataset(
            df_empty, split="train",
            raise_on_error=False,
        )
        assert not report.passed
        checks = [i.check for i in report.errors]
        assert "row_count" in checks or "missing_structural_cols" in checks

    def test_single_wafer_single_die(self, num_features, num_block_readings):
        """Minimal 1-row DataFrame should still validate correctly."""
        rng = np.random.default_rng(99)
        vals_str = " ".join(f"{v:.2f}" for v in rng.normal(100, 15, num_block_readings))
        data = {"wafer_id": ["W_F_0000"], "die_row": [5], "die_col": [5]}
        for f in feature_cols(num_features):
            data[f] = [float(rng.normal(100, 15))]
        data["old_label"] = [0]
        data[BLOCK_READINGS_COL] = [vals_str]
        data["label"] = [1]
        df = pd.DataFrame(data)
        report = validate_dataset(
            df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
            sample_block_rows=1,
        )
        # Should have warning about small size but no errors
        assert report.passed, str(report)

    def test_report_str_contains_status(self, valid_train_df, num_features, num_block_readings):
        report = validate_dataset(
            valid_train_df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        s = str(report)
        assert "PASS" in s or "FAIL" in s
        assert "train" in s

    def test_no_raise_mode_returns_report_on_error(self, df_missing_label, num_features, num_block_readings):
        """raise_on_error=False must return report instead of raising."""
        report = validate_dataset(
            df_missing_label, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        assert isinstance(report, ValidationReport)
        assert not report.passed


# ============================================================
# Class balance reporting
# ============================================================

class TestClassBalance:
    def test_class_balance_info_present(self, valid_train_df, num_features, num_block_readings):
        report = validate_dataset(
            valid_train_df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        info_checks = [i.check for i in report.issues if i.severity == "INFO"]
        assert "class_balance" in info_checks

    def test_high_fail_rate_triggers_warning(self, num_features, num_block_readings):
        """If >20% of dies fail, warn."""
        rng = np.random.default_rng(7)
        n = 300
        vals_str = " ".join(f"{v:.2f}" for v in rng.normal(100, 15, num_block_readings))
        data = {"wafer_id": [f"W_F_{i:04d}" for i in range(n)],
                "die_row": rng.integers(0, 20, n),
                "die_col": rng.integers(0, 20, n)}
        for f in feature_cols(num_features):
            data[f] = rng.normal(0, 1, n).tolist()
        data["old_label"] = [0] * n
        data["label"] = [1] * n          # 100% fail — deliberately extreme
        data[BLOCK_READINGS_COL] = [vals_str] * n
        df = pd.DataFrame(data)
        report = validate_dataset(
            df, split="train",
            num_features=num_features, num_block_readings=num_block_readings,
            raise_on_error=False,
        )
        warn_checks = [i.check for i in report.warnings]
        assert "class_balance" in warn_checks
