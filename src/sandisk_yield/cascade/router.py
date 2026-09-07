"""
src/sandisk_yield/cascade/router.py
===================================
Safety Routing Policy combining Conformal State, Boundary Margin, and Spatial Anomalies.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


class CascadeRouter:
    """
    Intelligent safety router deciding whether a die can be resolved by Model A
    or requires deep Model B block inspection.
    
    Routes to Model B when:
    1. Conformal state is AMBIGUOUS
    2. Model A probability is near the decision threshold (|p_A - threshold| < prob_margin)
    3. Spatial anomaly trigger: die is located near a cluster of old failures or wafer edge
    4. Safety override trigger: non-negligible risk (p_A > min_model_b_prob and p_A < max_model_b_prob)
    """

    def __init__(
        self,
        prob_margin: float = 0.20,
        min_model_b_prob: float = 0.10,
        max_model_b_prob: float = 0.90,
        spatial_anomaly_threshold: float = 0.75,
        force_model_b_on_edge: bool = False
    ):
        self.prob_margin = prob_margin
        self.min_model_b_prob = min_model_b_prob
        self.max_model_b_prob = max_model_b_prob
        self.spatial_anomaly_threshold = spatial_anomaly_threshold
        self.force_model_b_on_edge = force_model_b_on_edge

    def route(
        self,
        model_a_probs: np.ndarray,
        conformal_states: np.ndarray,
        spatial_features: Optional[pd.DataFrame] = None,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Returns boolean array: True = Route to Model B, False = Resolve with Model A.
        """
        n = len(model_a_probs)
        p_a = np.asarray(model_a_probs, dtype=np.float32)
        route_to_b = np.zeros(n, dtype=bool)
        
        # Rule 1: Conformal ambiguity
        is_ambiguous = (conformal_states == "AMBIGUOUS")
        route_to_b |= is_ambiguous
        
        # Rule 2: Decision boundary margin
        near_boundary = np.abs(p_a - threshold) < self.prob_margin
        route_to_b |= near_boundary
        
        # Rule 3: Intermediate risk zone
        in_suspicious_range = (p_a >= self.min_model_b_prob) & (p_a <= self.max_model_b_prob)
        route_to_b |= in_suspicious_range
        
        # Rule 4: Spatial context triggers (if spatial dataframe provided)
        if spatial_features is not None:
            # Check neighborhood old fail density if available
            density_cols = [c for c in spatial_features.columns if "old_fail_density" in c]
            if density_cols:
                max_density = spatial_features[density_cols].max(axis=1).values
                high_spatial_risk = max_density > self.spatial_anomaly_threshold
                # If high spatial risk and Model A says pass, escalate to Model B
                route_to_b |= (high_spatial_risk & (p_a < threshold))
                
            if self.force_model_b_on_edge and "spatial_radial_dist" in spatial_features.columns:
                is_edge = (spatial_features["spatial_radial_dist"].values > 0.85)
                route_to_b |= (is_edge & (p_a > 0.05))
                
        return route_to_b
