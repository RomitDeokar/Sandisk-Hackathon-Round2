"""
src/sandisk_yield/features/wafer.py
===================================
Wafer-level context and localized parametric process variations.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


def compute_wafer_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes wafer-level contextual features using only pre-test data (old_label).
    """
    # These features have at most one value per wafer (only ten independent
    # observations here); identical wafer geometry may make them all constant.
    out_dfs = []
    for wid, w_df in df.groupby("wafer_id", sort=False):
        n_dies = len(w_df)
        old_fails = (w_df["old_label"] == 1).sum()
        old_fail_rate = float(old_fails) / max(1, n_dies)
        
        row_span = float(w_df["die_row"].max() - w_df["die_row"].min() + 1)
        col_span = float(w_df["die_col"].max() - w_df["die_col"].min() + 1)
        aspect_ratio = row_span / max(1.0, col_span)
        die_density = n_dies / (row_span * col_span)
        
        wafer_dict = {
            "wafer_die_count": np.float32(n_dies),
            "wafer_old_fail_count": np.float32(old_fails),
            "wafer_old_fail_rate": np.float32(old_fail_rate),
            "wafer_row_span": np.float32(row_span),
            "wafer_col_span": np.float32(col_span),
            "wafer_aspect_ratio": np.float32(aspect_ratio),
            "wafer_die_density": np.float32(die_density),
        }
        out_dfs.append(pd.DataFrame(wafer_dict, index=w_df.index))
        
    return pd.concat(out_dfs).loc[df.index]


class LocalProcessFeatureExtractor:
    """
    Selects top-K parametric features strictly on training data (e.g. Cohen's d / variance),
    and computes localized neighborhood z-scores:
        local_z = (die_val - local_mean) / (local_std + eps)
    """

    def __init__(self, top_k: int = 30, window_size: int = 5, epsilon: float = 1e-6):
        self.top_k = top_k
        self.window_size = window_size
        self.epsilon = epsilon
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, feature_cols: Optional[List[str]] = None):
        if feature_cols is None:
            from sandisk_yield.schema import detect_feature_cols
            feature_cols = detect_feature_cols(X)
            
        # If labels are provided for training, select by separability (Cohen's d) or variance
        if y is not None and not y.isna().all() and y.nunique() > 1:
            scores = []
            for col in feature_cols:
                p_vals = X.loc[y == 0, col].dropna()
                f_vals = X.loc[y == 1, col].dropna()
                if len(p_vals) > 0 and len(f_vals) > 0:
                    diff = abs(f_vals.mean() - p_vals.mean())
                    pooled_std = np.sqrt((p_vals.var() + f_vals.var()) / 2.0) + self.epsilon
                    d = diff / pooled_std
                    scores.append((d, col))
                else:
                    scores.append((0.0, col))
            scores.sort(key=lambda x: x[0], reverse=True)
            self.selected_features_ = [col for _, col in scores[:self.top_k]]
        else:
            # Fallback to feature variance
            variances = [(X[col].var(), col) for col in feature_cols]
            variances.sort(key=lambda x: x[0], reverse=True)
            self.selected_features_ = [col for _, col in variances[:self.top_k]]
            
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            return pd.DataFrame(index=df.index)
            
        out_dfs = []
        for wid, w_df in df.groupby("wafer_id", sort=False):
            rows = w_df["die_row"].values
            cols = w_df["die_col"].values
            assert (rows >= 0).all() and (cols >= 0).all(), "Negative grid coordinates"
            max_r = rows.max() + 1
            max_c = cols.max() + 1
            
            valid_mask = np.zeros((max_r, max_c), dtype=np.float32)
            valid_mask[rows, cols] = 1.0
            
            local_dict = {}
            for feat in self.selected_features_:
                if feat not in w_df.columns:
                    continue
                feat_vals = w_df[feat].values.astype(np.float32)
                grid = np.zeros((max_r, max_c), dtype=np.float32)
                grid[rows, cols] = feat_vals
                
                # Neighborhood mean & std via uniform_filter
                w_sz = self.window_size
                n_count = uniform_filter(valid_mask, size=w_sz, mode="constant", cval=0.0) * (w_sz * w_sz)
                n_sum = uniform_filter(grid, size=w_sz, mode="constant", cval=0.0) * (w_sz * w_sz)
                n_mean = n_sum / (n_count + self.epsilon)
                
                # Local variance
                grid_sq = np.zeros((max_r, max_c), dtype=np.float32)
                grid_sq[rows, cols] = feat_vals ** 2
                n_sq_sum = uniform_filter(grid_sq, size=w_sz, mode="constant", cval=0.0) * (w_sz * w_sz)
                n_var = np.maximum(0.0, (n_sq_sum / (n_count + self.epsilon)) - (n_mean ** 2))
                n_std = np.sqrt(n_var)
                
                die_local_mean = n_mean[rows, cols]
                die_local_std = n_std[rows, cols]
                die_local_z = (feat_vals - die_local_mean) / (die_local_std + self.epsilon)
                
                local_dict[f"local_z_{feat}"] = die_local_z.astype(np.float32)
                local_dict[f"local_dev_{feat}"] = (feat_vals - die_local_mean).astype(np.float32)
                
            out_dfs.append(pd.DataFrame(local_dict, index=w_df.index))
            
        return pd.concat(out_dfs).loc[df.index]
