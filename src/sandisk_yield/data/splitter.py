"""
src/sandisk_yield/data/splitter.py
==================================
Grouped train/validation and calibration splitting by wafer_id.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_grouped_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "wafer_id",
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates wafer_id-grouped cross-validation train/val index splits.
    Guarantees no wafer overlap between train and validation.
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' missing from DataFrame")
        
    unique_groups = df[group_col].unique()
    if df[group_col].isna().any() or not 2 <= n_splits <= len(unique_groups):
        raise ValueError("Grouped CV requires nonmissing groups and 2 <= folds <= wafer count")
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)
    
    # Map back to dataframe order
    group_map = {g: i for i, g in enumerate(shuffled_groups)}
    group_indices = df[group_col].map(group_map).values
    
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    dummy_x = np.zeros(len(df))
    for train_idx, val_idx in gkf.split(dummy_x, groups=group_indices):
        folds.append((train_idx, val_idx))
        
    return folds


def make_calibration_split(
    df: pd.DataFrame,
    calibration_ratio: float = 0.2,
    group_col: str = "wafer_id",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame into fit_df and calib_df by wafer_id.
    """
    unique_wafers = df[group_col].unique()
    rng = np.random.default_rng(seed)
    shuffled_wafers = rng.permutation(unique_wafers)
    
    n_calib = max(1, int(len(unique_wafers) * calibration_ratio))
    calib_wafers = set(shuffled_wafers[:n_calib])
    fit_wafers = set(shuffled_wafers[n_calib:])
    
    fit_df = df[df[group_col].isin(fit_wafers)].copy()
    calib_df = df[df[group_col].isin(calib_wafers)].copy()
    
    return fit_df, calib_df
