"""
src/sandisk_yield/risk/wafer_risk.py
===================================
Wafer-level risk metrics and aggregated summary tables.
"""

from typing import Dict
import pandas as pd
import numpy as np


def compute_wafer_risk_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates die-level risk predictions up to wafer-level KPIs.
    """
    records = []
    for wid, w_df in predictions_df.groupby("wafer_id", sort=False):
        n_dies = len(w_df)
        old_fails = (w_df["old_label"] == 1).sum()
        pred_fails = (w_df["predicted_label"] == 1).sum()
        pred_new_fails = ((w_df["old_label"] == 0) & (w_df["predicted_label"] == 1)).sum()
        
        probs = w_df["final_probability"].values
        mean_risk = float(probs.mean())
        max_risk = float(probs.max())
        
        # Count dies routed to Model B
        b_routed = int(w_df["routed_to_model_b"].sum()) if "routed_to_model_b" in w_df.columns else 0
        
        records.append({
            "wafer_id": wid,
            "total_dies": n_dies,
            "old_failures": int(old_fails),
            "predicted_total_failures": int(pred_fails),
            "predicted_new_failures": int(pred_new_fails),
            "predicted_yield_loss_pct": float(pred_fails / n_dies * 100),
            "mean_die_risk": mean_risk,
            "max_die_risk": max_risk,
            "model_b_inspections": b_routed,
            "model_b_inspection_pct": float(b_routed / max(1, n_dies) * 100),
        })
        
    return pd.DataFrame(records)
