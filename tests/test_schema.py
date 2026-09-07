"""
tests/test_schema.py
====================
Tests for src/data/schema.py — verifies that column-name generators
and constant values are correct and internally consistent.
"""

from __future__ import annotations

import pytest

from src.data.schema import (
    BLOCK_READINGS_COL,
    DEFAULT_NUM_BLOCK_READINGS,
    DEFAULT_NUM_FEATURES,
    ELIGIBLE_MASK_COL,
    ELIGIBLE_MASK_VALUE,
    FEATURE_PREFIX,
    ID_COLS,
    LABEL_COLS,
    LABEL_FAIL,
    LABEL_PASS,
    VALIDATION_DROPS,
    WAFER_ID_FAILURE_PREFIX,
    WAFER_ID_NONE_PREFIX,
    feature_cols,
    train_cols,
    validation_cols,
)


class TestConstants:
    def test_id_cols_order(self):
        """ID cols must be wafer_id, die_row, die_col in that order."""
        assert ID_COLS == ["wafer_id", "die_row", "die_col"]

    def test_label_values_distinct(self):
        """Pass and fail labels must be different integers."""
        assert LABEL_PASS != LABEL_FAIL

    def test_label_pass_is_zero(self):
        """Convention: 0 = pass."""
        assert LABEL_PASS == 0

    def test_label_fail_is_one(self):
        """Convention: 1 = fail."""
        assert LABEL_FAIL == 1

    def test_block_readings_col_name(self):
        assert BLOCK_READINGS_COL == "block_readings"

    def test_default_features(self):
        assert DEFAULT_NUM_FEATURES == 500

    def test_default_block_readings(self):
        assert DEFAULT_NUM_BLOCK_READINGS == 2000

    def test_eligible_mask(self):
        """Eligible dies for evaluation are those with old_label == 0."""
        assert ELIGIBLE_MASK_COL == "old_label"
        assert ELIGIBLE_MASK_VALUE == LABEL_PASS

    def test_wafer_id_prefixes(self):
        assert WAFER_ID_FAILURE_PREFIX.startswith("W_F")
        assert WAFER_ID_NONE_PREFIX.startswith("W_N")

    def test_validation_drops_label(self):
        assert "label" in VALIDATION_DROPS


class TestFeatureCols:
    def test_default_count(self):
        cols = feature_cols()
        assert len(cols) == DEFAULT_NUM_FEATURES

    def test_naming(self):
        cols = feature_cols(5)
        assert cols == ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

    def test_prefix(self):
        cols = feature_cols(3)
        for c in cols:
            assert c.startswith(FEATURE_PREFIX)

    def test_custom_count(self):
        cols = feature_cols(10)
        assert len(cols) == 10
        assert cols[0] == "feature_1"
        assert cols[-1] == "feature_10"

    def test_zero_features(self):
        cols = feature_cols(0)
        assert cols == []


class TestTrainCols:
    def test_starts_with_ids(self):
        cols = train_cols(5)
        assert cols[:3] == ID_COLS

    def test_ends_with_label(self):
        cols = train_cols(5)
        assert cols[-1] == "label"

    def test_contains_old_label(self):
        cols = train_cols(5)
        assert "old_label" in cols

    def test_contains_block_readings(self):
        cols = train_cols(5)
        assert BLOCK_READINGS_COL in cols

    def test_feature_count(self):
        cols = train_cols(10)
        feat = [c for c in cols if c.startswith(FEATURE_PREFIX)]
        assert len(feat) == 10

    def test_total_col_count_default(self):
        # 3 id + 500 features + old_label + block_readings + label = 506
        cols = train_cols()
        assert len(cols) == 3 + DEFAULT_NUM_FEATURES + 3

    def test_old_label_before_block_readings(self):
        """old_label should come before block_readings."""
        cols = train_cols(5)
        assert cols.index("old_label") < cols.index(BLOCK_READINGS_COL)

    def test_block_readings_before_label(self):
        """block_readings should come before label."""
        cols = train_cols(5)
        assert cols.index(BLOCK_READINGS_COL) < cols.index("label")


class TestValidationCols:
    def test_no_label(self):
        cols = validation_cols(5)
        assert "label" not in cols

    def test_has_old_label(self):
        cols = validation_cols(5)
        assert "old_label" in cols

    def test_has_block_readings(self):
        cols = validation_cols(5)
        assert BLOCK_READINGS_COL in cols

    def test_one_fewer_than_train(self):
        tcols = train_cols(5)
        vcols = validation_cols(5)
        assert len(vcols) == len(tcols) - 1

    def test_starts_with_ids(self):
        cols = validation_cols(5)
        assert cols[:3] == ID_COLS
