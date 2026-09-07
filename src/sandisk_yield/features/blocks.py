"""
src/sandisk_yield/features/blocks.py
===================================
Sub-die block readings parser and explicit anomaly feature engineering.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def parse_block_readings_row(val: Union[str, list, np.ndarray], expected_len: int = 2000) -> np.ndarray:
    """
    Parses string, list, or array of block readings into float32 ndarray.
    """
    if isinstance(val, np.ndarray):
        arr = val.astype(np.float32)
    elif isinstance(val, list):
        arr = np.array(val, dtype=np.float32)
    elif isinstance(val, str):
        arr = np.fromstring(val, dtype=np.float32, sep=" ")
    else:
        arr = np.zeros(expected_len, dtype=np.float32)
        
    if len(arr) != expected_len and expected_len > 0:
        if len(arr) < expected_len:
            pad = np.zeros(expected_len - len(arr), dtype=np.float32)
            arr = np.concatenate([arr, pad])
        else:
            arr = arr[:expected_len]
    return arr


def compress_block_readings_global(df, expected_len=2000):
    """Simple per-die summaries; no fitted statistics or outcome columns."""
    records = []
    for value in df["block_readings"]:
        arr = parse_block_readings_row(value, expected_len)
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite block readings")
        records.append({"blk_mean": arr.mean(), "blk_std": arr.std(),
                        "blk_min": arr.min(), "blk_max": arr.max(),
                        "blk_median": np.median(arr)})
    return pd.DataFrame(records, index=df.index, dtype=np.float32)


def compress_block_readings_zonal(df, expected_len=2000, n_zones=8):
    """Contiguous sequence zones; assumes block ordering is meaningful and stable."""
    if not 1 <= n_zones <= expected_len:
        raise ValueError("n_zones must be between 1 and block length")
    records = []
    for value in df["block_readings"]:
        arr = parse_block_readings_row(value, expected_len)
        if not np.isfinite(arr).all():
            raise ValueError("Nonfinite block readings")
        row = {}
        for zone, values in enumerate(np.array_split(arr, n_zones)):
            for stat, number in (("mean", values.mean()), ("std", values.std()),
                                 ("min", values.min()), ("max", values.max())):
                row[f"blk_zone_{zone}_{stat}"] = number
        records.append(row)
    return pd.DataFrame(records, index=df.index, dtype=np.float32)


def extract_block_anomaly_features(arr: np.ndarray) -> dict:
    """
    Extracts statistical and sparse localized anomaly metrics from a single block array.
    """
    k = len(arr)
    if k == 0:
        return {}
        
    mean_ = float(arr.mean())
    std_ = float(arr.std())
    min_ = float(arr.min())
    max_ = float(arr.max())
    rng_val = max_ - min_
    
    # Quantiles
    q01, q05, q25, q50, q75, q95, q99 = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    iqr = q75 - q25
    
    # Z-scores & anomaly counts
    if std_ > 1e-6:
        z = np.abs((arr - mean_) / std_)
        cnt_z2 = int((z > 2.0).sum())
        cnt_z3 = int((z > 3.0).sum())
        frac_z2 = float(cnt_z2 / k)
        frac_z3 = float(cnt_z3 / k)
        max_abs_z = float(z.max())
    else:
        cnt_z2 = cnt_z3 = 0
        frac_z2 = frac_z3 = max_abs_z = 0.0
        
    # Local deviations (sliding window smoothed)
    smooth = uniform_filter1d(arr, size=max(3, int(k * 0.025)), mode="nearest")
    local_devs = np.abs(arr - smooth)
    max_local_dev = float(local_devs.max())
    mean_local_dev = float(local_devs.mean())
    
    # Anomaly cluster identification & largest contiguous run
    anom_mask = (arr > q95 + 1.5 * iqr) | (arr < q01 - 1.5 * iqr) if iqr > 1e-6 else (z > 2.5 if std_ > 1e-6 else np.zeros(k, dtype=bool))
    
    # Run-length of True values
    max_run = 0
    cur_run = 0
    n_clusters = 0
    in_cluster = False
    anom_indices = []
    
    for i, is_anom in enumerate(anom_mask):
        if is_anom:
            cur_run += 1
            max_run = max(max_run, cur_run)
            anom_indices.append(i)
            if not in_cluster:
                n_clusters += 1
                in_cluster = True
        else:
            cur_run = 0
            in_cluster = False
            
    # Center of mass of anomalies (normalized to [0, 1])
    if anom_indices:
        center_of_mass = float(np.mean(anom_indices) / k)
    else:
        center_of_mass = 0.5
        
    return {
        "blk_mean": np.float32(mean_),
        "blk_std": np.float32(std_),
        "blk_min": np.float32(min_),
        "blk_max": np.float32(max_),
        "blk_range": np.float32(rng_val),
        "blk_q01": np.float32(q01),
        "blk_q05": np.float32(q05),
        "blk_q25": np.float32(q25),
        "blk_q50": np.float32(q50),
        "blk_q75": np.float32(q75),
        "blk_q95": np.float32(q95),
        "blk_q99": np.float32(q99),
        "blk_iqr": np.float32(iqr),
        "blk_count_z_gt_2": np.float32(cnt_z2),
        "blk_count_z_gt_3": np.float32(cnt_z3),
        "blk_frac_z_gt_2": np.float32(frac_z2),
        "blk_frac_z_gt_3": np.float32(frac_z3),
        "blk_max_abs_z": np.float32(max_abs_z),
        "blk_max_local_dev": np.float32(max_local_dev),
        "blk_mean_local_dev": np.float32(mean_local_dev),
        "blk_max_contiguous_run": np.float32(max_run),
        "blk_cluster_count": np.float32(n_clusters),
        "blk_center_of_mass": np.float32(center_of_mass),
    }


def compute_block_features_df(
    df: pd.DataFrame,
    block_col: str = "block_readings",
    expected_len: int = 2000
) -> pd.DataFrame:
    """
    Computes dataframe of explicit block anomaly features for all rows.
    """
    if block_col not in df.columns:
        raise KeyError(f"Block column '{block_col}' not found in DataFrame")
        
    records = []
    for item in df[block_col]:
        arr = parse_block_readings_row(item, expected_len=expected_len)
        records.append(extract_block_anomaly_features(arr))
        
    return pd.DataFrame(records, index=df.index)
