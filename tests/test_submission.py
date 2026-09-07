"""
tests/test_submission.py
========================
Submission export and validation tests.
"""

import numpy as np
import pandas as pd
import pytest

from sandisk_yield.inference.submission import validate_submission_df, SubmissionValidationError


@pytest.fixture
def valid_ref_and_sub():
    ref_df = pd.DataFrame({
        "wafer_id": ["W1", "W1", "W2"],
        "die_row": [10, 11, 5],
        "die_col": [20, 21, 8],
        "old_label": [0, 1, 0]
    })
    sub_df = pd.DataFrame({
        "wafer_id": ["W1", "W1", "W2"],
        "die_row": [10, 11, 5],
        "die_col": [20, 21, 8],
        "predicted_label": [0, 1, 1]
    })
    return ref_df, sub_df


def test_valid_submission(valid_ref_and_sub):
    ref_df, sub_df = valid_ref_and_sub
    # Must not raise
    validate_submission_df(sub_df, ref_df)


def test_missing_column_raises(valid_ref_and_sub):
    ref_df, sub_df = valid_ref_and_sub
    bad_sub = sub_df.drop(columns=["predicted_label"])
    with pytest.raises(SubmissionValidationError):
        validate_submission_df(bad_sub, ref_df)


def test_row_count_mismatch_raises(valid_ref_and_sub):
    ref_df, sub_df = valid_ref_and_sub
    bad_sub = sub_df.iloc[:2]
    with pytest.raises(SubmissionValidationError):
        validate_submission_df(bad_sub, ref_df)


def test_non_binary_label_raises(valid_ref_and_sub):
    ref_df, sub_df = valid_ref_and_sub
    bad_sub = sub_df.copy()
    bad_sub.loc[0, "predicted_label"] = 2
    with pytest.raises(SubmissionValidationError):
        validate_submission_df(bad_sub, ref_df)
