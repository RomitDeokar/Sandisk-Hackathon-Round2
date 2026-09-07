"""
src/features/block_features.py
================================
Utilities for parsing and featurizing the block_readings column.

block_readings is a space-separated string of k float values per die.
The signal is sparse: only ~5% of blocks in a failing die are anomalous,
and those anomalous blocks are spatially clustered within the 2000-element array.

Recommended usage
-----------------
    from src.features.block_features import extract_block_features_df

    feat_df = extract_block_features_df(df)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import BLOCK_READINGS_COL, DEFAULT_NUM_BLOCK_READINGS


def parse_block_readings(s: str, k: int = DEFAULT_NUM_BLOCK_READINGS) -> np.ndarray:
    """
    Parse a single block_readings string into a float32 numpy array.

    Parameters
    ----------
    s : str
        Space-separated float string, e.g. "98.34 101.22 99.87 ..."
    k : int
        Expected number of values. Raises ValueError if count mismatches.

    Returns
    -------
    np.ndarray of shape (k,) dtype float32
    """
    arr = np.fromstring(s, dtype=np.float32, sep=" ")
    if arr.shape[0] != k:
        raise ValueError(
            f"Expected {k} block readings, got {arr.shape[0]}"
        )
    return arr


def extract_block_features(arr: np.ndarray) -> dict:
    """
    Extract statistical and anomaly-detection features from a block array.

    The generator injects anomalies as a local cluster of shifted values,
    so local extreme statistics are more informative than global ones.

    Parameters
    ----------
    arr : np.ndarray, shape (k,)

    Returns
    -------
    dict of scalar features
    """
    k = len(arr)

    # Global statistics
    mean_ = float(arr.mean())
    std_ = float(arr.std())
    min_ = float(arr.min())
    max_ = float(arr.max())
    q5 = float(np.percentile(arr, 5))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    q95 = float(np.percentile(arr, 95))
    iqr = q75 - q25

    # Z-score based: fraction of blocks that are outliers
    if std_ > 0:
        z = (arr - mean_) / std_
        frac_gt2z = float((np.abs(z) > 2).mean())    # blocks > 2 sigma
        frac_gt3z = float((np.abs(z) > 3).mean())    # blocks > 3 sigma
        max_z = float(np.abs(z).max())
    else:
        frac_gt2z = 0.0
        frac_gt3z = 0.0
        max_z = 0.0

    # Local anomaly: max deviation in a sliding window of size 50
    window = 50
    if k >= window:
        rolling_mean = np.convolve(arr, np.ones(window) / window, mode="valid")
        rolling_deviations = np.abs(arr[:len(rolling_mean)] - rolling_mean)
        max_local_dev = float(rolling_deviations.max())
        mean_local_dev = float(rolling_deviations.mean())
    else:
        max_local_dev = float(np.abs(arr - mean_).max())
        mean_local_dev = float(np.abs(arr - mean_).mean())

    # Skewness (3rd moment)
    if std_ > 0:
        skewness = float(((arr - mean_) ** 3).mean() / (std_ ** 3))
    else:
        skewness = 0.0

    # Kurtosis (excess)
    if std_ > 0:
        kurtosis = float(((arr - mean_) ** 4).mean() / (std_ ** 4)) - 3.0
    else:
        kurtosis = 0.0

    return {
        "br_mean": mean_,
        "br_std": std_,
        "br_min": min_,
        "br_max": max_,
        "br_q05": q5,
        "br_q25": q25,
        "br_q75": q75,
        "br_q95": q95,
        "br_iqr": iqr,
        "br_range": max_ - min_,
        "br_frac_gt2z": frac_gt2z,
        "br_frac_gt3z": frac_gt3z,
        "br_max_abs_z": max_z,
        "br_max_local_dev": max_local_dev,
        "br_mean_local_dev": mean_local_dev,
        "br_skewness": skewness,
        "br_kurtosis": kurtosis,
    }


def extract_block_features_df(
    df: pd.DataFrame,
    num_block_readings: int = DEFAULT_NUM_BLOCK_READINGS,
) -> pd.DataFrame:
    """
    Apply block feature extraction to every row in a DataFrame.

    Returns a new DataFrame (same index) with block-derived feature columns.
    Does NOT modify the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'block_readings' string column.
    num_block_readings : int
        Expected number of floats per block_readings string.

    Returns
    -------
    pd.DataFrame with columns: br_mean, br_std, ..., br_kurtosis
    """
    if BLOCK_READINGS_COL not in df.columns:
        raise KeyError(f"Column '{BLOCK_READINGS_COL}' not found in DataFrame")

    records = []
    for val in df[BLOCK_READINGS_COL]:
        arr = parse_block_readings(val, k=num_block_readings)
        records.append(extract_block_features(arr))

    return pd.DataFrame(records, index=df.index)
