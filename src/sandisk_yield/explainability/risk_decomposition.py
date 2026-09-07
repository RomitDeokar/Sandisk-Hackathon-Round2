"""
src/sandisk_yield/explainability/risk_decomposition.py
======================================================
Die Risk Card generation decomposing risk into parametric, spatial, and block evidence.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def generate_die_risk_card(
    wafer_id: str,
    die_row: int,
    die_col: int,
    final_risk: float,
    threshold: float,
    modal_weights: Optional[np.ndarray] = None,
    top_parametric_drivers: Optional[List[str]] = None,
    spatial_summary: Optional[Dict[str, float]] = None,
    block_summary: Optional[Dict[str, any]] = None
) -> Dict[str, any]:
    """
    Generates structured Die Risk Card combining multi-scale evidence.
    """
    # Map risk to category
    # Bug: a tuned threshold below 0.20 previously marked above-threshold dies PASS.
    if final_risk < min(0.20, threshold):
        risk_category = "LOW_RISK"
        action = "PASS"
    elif final_risk < threshold:
        risk_category = "MEDIUM_RISK"
        action = "REVIEW"
    else:
        risk_category = "HIGH_RISK"
        action = "HOLD / FAIL"
        
    weights = modal_weights if modal_weights is not None else np.array([0.45, 0.35, 0.20])
    w_sum = weights.sum() if weights.sum() > 0 else 1.0
    
    evidence_breakdown = {
        "parametric_pct": float((weights[0] / w_sum) * 100),
        "spatial_pct": float((weights[1] / w_sum) * 100),
        "block_pct": float((weights[2] / w_sum) * 100),
    }
    
    return {
        "die_id": f"{wafer_id}_r{die_row}_c{die_col}",
        "wafer_id": wafer_id,
        "die_row": int(die_row),
        "die_col": int(die_col),
        "final_risk": float(final_risk),
        "risk_category": risk_category,
        "action": action,
        "evidence_breakdown": evidence_breakdown,
        "top_parametric_drivers": top_parametric_drivers or ["feature_1", "feature_12", "feature_48"],
        "spatial_context": spatial_summary or {},
        "block_anomaly_summary": block_summary or {},
    }
