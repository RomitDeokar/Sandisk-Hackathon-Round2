"""
src/sandisk_yield/cascade/cascade.py
====================================
End-to-end WaferFusion-Cascade orchestration.
Supports TWO explicit evaluation modes:
  1. Full Model B ('full_model_b'): All eligible dies processed by Model B.
  2. Cascade ('cascade'): Model A screens first, only routed dies inspected by Model B.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sandisk_yield.models.model_a import ModelA
from sandisk_yield.models.model_b import ModelB
from sandisk_yield.models.calibration import Calibrator
from sandisk_yield.cascade.conformal_gate import ConformalGate
from sandisk_yield.cascade.router import CascadeRouter
from sandisk_yield.features.blocks import compute_block_features_df
from sandisk_yield.schema import create_eligible_mask
from sandisk_yield.data.alignment import assert_feature_alignment, assert_row_alignment


class WaferFusionCascade:
    """
    Orchestrator combining Model A, Calibrator, Conformal Gate, Router, and Model B.
    """

    def __init__(
        self,
        model_a: ModelA,
        calibrator_a: Calibrator,
        conformal_gate: ConformalGate,
        router: CascadeRouter,
        model_b: Optional[ModelB] = None,
        calibrator_b: Optional[Calibrator] = None,
        threshold: float = 0.5,
        threshold_b: Optional[float] = None
    ):
        self.model_a = model_a
        self.calibrator_a = calibrator_a
        self.conformal_gate = conformal_gate
        self.router = router
        self.model_b = model_b
        self.calibrator_b = calibrator_b
        self.threshold = threshold
        self.threshold_b = threshold if threshold_b is None else threshold_b

    def predict_detailed(
        self,
        df_raw: pd.DataFrame,
        X_tab: pd.DataFrame,
        mode: str = "cascade",
        batch_size: int = 64
    ) -> pd.DataFrame:
        """
        Executes pipeline in specified mode ('cascade' or 'full_model_b').
        Returns rich DataFrame containing:
          - p_A_raw, p_A_calibrated
          - uncertainty_state
          - routed_to_model_b
          - p_B_raw, p_B_calibrated
          - final_probability
          - predicted_label
        """
        assert_row_alignment(df_raw, X_tab)
        assert_feature_alignment(self.model_a.feature_names_, X_tab.columns)
        n = len(df_raw)
        results = pd.DataFrame(index=df_raw.index)
        
        # 1. Identify eligible dies (old_label == 0)
        eligible_mask = create_eligible_mask(df_raw).values
        
        # Initialize output arrays
        p_a_calib = np.zeros(n, dtype=np.float32)
        p_b_calib = np.full(n, np.nan, dtype=np.float32)
        final_probs = np.zeros(n, dtype=np.float32)
        uncertainty_states = np.full(n, "OLD_FAILURE", dtype=object)
        routed_to_b = np.zeros(n, dtype=bool)
        
        # For old_label == 1, final_prob is 1.0
        final_probs[~eligible_mask] = 1.0
        
        # If there are eligible dies, run screening
        if eligible_mask.sum() > 0:
            elig_indices = np.where(eligible_mask)[0]
            X_tab_elig = X_tab.iloc[elig_indices]
            
            # Step 1: Model A fast screening
            raw_p_a = self.model_a.predict_proba(X_tab_elig)[:, 1]
            cal_p_a = self.calibrator_a.predict(raw_p_a)
            p_a_calib[elig_indices] = cal_p_a
            
            # Step 2: Conformal prediction & uncertainty states
            conf_states = self.conformal_gate.derive_routing_states(cal_p_a)
            uncertainty_states[elig_indices] = conf_states
            
            # Step 3: Determine routing
            if mode == "full_model_b":
                to_route = np.ones(len(elig_indices), dtype=bool)
            else:
                to_route = self.router.route(
                    cal_p_a,
                    conf_states,
                    spatial_features=X_tab_elig,
                    threshold=self.threshold
                )
            routed_to_b[elig_indices] = to_route
            
            # Dies resolved purely by Model A
            resolved_by_a_mask = ~to_route
            if resolved_by_a_mask.sum() > 0:
                resolved_orig_idx = elig_indices[resolved_by_a_mask]
                final_probs[resolved_orig_idx] = cal_p_a[resolved_by_a_mask]
                
            # Dies routed to Model B
            if to_route.sum() > 0 and self.model_b is not None:
                routed_orig_idx = elig_indices[to_route]
                X_tab_routed = X_tab_elig.iloc[to_route].values
                block_strings = df_raw["block_readings"].iloc[routed_orig_idx].tolist()
                
                # Compute block features for routed batch
                df_routed_raw = df_raw.iloc[routed_orig_idx]
                X_blk_feats = compute_block_features_df(df_routed_raw, expected_len=self.model_b.block_length).values
                
                # Model B inference
                raw_p_b = self.model_b.predict_proba(
                    X_tab_routed,
                    block_strings,
                    X_blk_feats,
                    batch_size=batch_size
                )[:, 1]
                
                if self.calibrator_b is not None:
                    cal_p_b = self.calibrator_b.predict(raw_p_b)
                else:
                    cal_p_b = raw_p_b
                    
                p_b_calib[routed_orig_idx] = cal_p_b
                final_probs[routed_orig_idx] = cal_p_b
            elif to_route.sum() > 0:
                # If Model B is not loaded, fallback to Model A
                routed_orig_idx = elig_indices[to_route]
                final_probs[routed_orig_idx] = cal_p_a[to_route]
                
        # Final binary decision based on optimal threshold
        # A and B have different OOF probability scales and decision thresholds.
        thresholds = np.where(np.isfinite(p_b_calib), getattr(self, "threshold_b", self.threshold), self.threshold)
        predicted_labels = (final_probs >= thresholds).astype(int)
        # Ensure hard rule: old_label == 1 always produces predicted_label = 1
        predicted_labels[~eligible_mask] = 1
        
        results["wafer_id"] = df_raw["wafer_id"].values
        results["die_row"] = df_raw["die_row"].values
        results["die_col"] = df_raw["die_col"].values
        results["old_label"] = df_raw["old_label"].values
        results["model_a_probability"] = p_a_calib
        results["model_b_probability"] = p_b_calib
        results["final_probability"] = final_probs
        results["uncertainty_state"] = uncertainty_states
        results["routed_to_model_b"] = routed_to_b
        results["predicted_label"] = predicted_labels
        results["decision_threshold"] = thresholds
        
        return results
