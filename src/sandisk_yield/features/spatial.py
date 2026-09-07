"""
src/sandisk_yield/features/spatial.py
=====================================
Spatial geometry and neighborhood feature engineering.
STRICT LEAKAGE RULE: Uses ONLY old_label (pre-test status), never final label.
"""

from typing import List
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


def compute_spatial_features(
    df: pd.DataFrame,
    windows: List[int] = [3, 5, 7],
    epsilon: float = 1e-6
) -> pd.DataFrame:
    """
    Computes spatial features per wafer without target leakage.
    - Normalized row, col
    - Radial distance & radial distance squared
    - Edge distances & minimum edge distance
    - Multi-scale neighborhood old failure counts & densities (3x3, 5x5, 7x7)
    - Nearest old failure Euclidean distance
    """
    out_dfs = []
    
    # Process group by group to maintain clean wafer boundaries
    for wid, w_df in df.groupby("wafer_id", sort=False):
        w_df = w_df.copy()
        
        rows = w_df["die_row"].values
        cols = w_df["die_col"].values
        old_labels = w_df["old_label"].values
        # Grid indexing assumes nonnegative integer die coordinates; fail loudly
        # rather than silently wrapping a negative index into the opposite edge.
        assert (rows >= 0).all() and (cols >= 0).all(), "Negative grid coordinates"
        
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        
        row_span = max(1, max_r - min_r)
        col_span = max(1, max_c - min_c)
        
        # 1. Coordinate normalization
        r_norm = (rows - min_r) / row_span
        c_norm = (cols - min_c) / col_span
        
        # 2. Radial geometry
        center_r = (min_r + max_r) / 2.0
        center_c = (min_c + max_c) / 2.0
        
        dy = (rows - center_r) / (row_span / 2.0)
        dx = (cols - center_c) / (col_span / 2.0)
        radial_dist = np.sqrt(dy ** 2 + dx ** 2).astype(np.float32)
        radial_dist_sq = (radial_dist ** 2).astype(np.float32)
        
        # 3. Edge distances
        top_edge_dist = (rows - min_r).astype(np.float32)
        bottom_edge_dist = (max_r - rows).astype(np.float32)
        left_edge_dist = (cols - min_c).astype(np.float32)
        right_edge_dist = (max_c - cols).astype(np.float32)
        min_edge_dist = np.minimum(
            np.minimum(top_edge_dist, bottom_edge_dist),
            np.minimum(left_edge_dist, right_edge_dist)
        )
        
        # Build 2D grid representations for fast spatial neighborhood convolution
        grid_rows = max_r + 1
        grid_cols = max_c + 1
        
        valid_mask_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        old_fail_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        
        valid_mask_grid[rows, cols] = 1.0
        old_fail_grid[rows, cols] = (old_labels == 1).astype(np.float32)
        
        spatial_dict = {
            "spatial_row_norm": r_norm.astype(np.float32),
            "spatial_col_norm": c_norm.astype(np.float32),
            "spatial_radial_dist": radial_dist,
            "spatial_radial_dist_sq": radial_dist_sq,
            "spatial_top_edge_dist": top_edge_dist,
            "spatial_bottom_edge_dist": bottom_edge_dist,
            "spatial_left_edge_dist": left_edge_dist,
            "spatial_right_edge_dist": right_edge_dist,
            "spatial_min_edge_dist": min_edge_dist,
        }
        
        # 4. Multi-scale neighborhood old_fail filtering
        for w in windows:
            # Local window sum
            valid_count_grid = uniform_filter(valid_mask_grid, size=w, mode="constant", cval=0.0) * (w * w)
            fail_count_grid = uniform_filter(old_fail_grid, size=w, mode="constant", cval=0.0) * (w * w)
            
            # Extract at die coordinates
            val_cnt = valid_count_grid[rows, cols]
            fail_cnt = fail_count_grid[rows, cols]
            density = fail_cnt / (val_cnt + epsilon)
            
            spatial_dict[f"spatial_neigh_valid_count_{w}x{w}"] = val_cnt.astype(np.float32)
            spatial_dict[f"spatial_neigh_old_fail_count_{w}x{w}"] = fail_cnt.astype(np.float32)
            spatial_dict[f"spatial_neigh_old_fail_density_{w}x{w}"] = density.astype(np.float32)
            
        # 5. Nearest old failure Euclidean distance
        old_fail_indices = np.argwhere(old_fail_grid == 1.0)
        if len(old_fail_indices) > 0:
            die_coords = np.column_stack([rows, cols])
            # Pairwise distance calculation
            diffs = die_coords[:, np.newaxis, :] - old_fail_indices[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs ** 2, axis=2))
            min_dist = dists.min(axis=1)
        else:
            # Structurally constant on wafers with no old failures. In particular,
            # masking out old failures BEFORE spatial construction destroys this
            # signal; callers must transform complete wafers then mask targets.
            min_dist = np.full(len(rows), np.sqrt(grid_rows**2 + grid_cols**2), dtype=np.float32)
            
        spatial_dict["spatial_nearest_old_fail_dist"] = min_dist.astype(np.float32)
        
        sp_df = pd.DataFrame(spatial_dict, index=w_df.index)
        out_dfs.append(sp_df)
        
    return pd.concat(out_dfs).loc[df.index]
