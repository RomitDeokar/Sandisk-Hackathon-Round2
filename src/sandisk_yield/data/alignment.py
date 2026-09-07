"""Explicit schema and row contracts shared by training and inference."""
import logging


def assert_feature_alignment(train_cols, val_cols):
    train_cols, val_cols = list(train_cols), list(val_cols)
    assert len(train_cols) == len(set(train_cols)), "Duplicate training features"
    assert train_cols == val_cols, "Training/prediction feature names or order differ"
    assert not {"label", "predicted_label", "final_probability"}.intersection(train_cols), "Outcome leakage"


def assert_row_alignment(*objects):
    reference = objects[0].index
    assert reference.is_unique, "Use unique row indices before masking/grouping"
    for obj in objects[1:]:
        assert reference.equals(obj.index), "Features, labels and raw rows differ in order"


def log_feature_uniqueness(features, context):
    logger = logging.getLogger("sandisk_yield.features")
    for name, count in features.nunique(dropna=False).items():
        logger.info("%s: %s nunique=%s%s", context, name, count,
                    " (CONSTANT: inspect wafer context)" if count <= 1 else "")
