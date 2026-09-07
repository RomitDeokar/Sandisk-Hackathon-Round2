"""
src/sandisk_yield/training/trainer_cascade.py
============================================
Cascade end-to-end training, calibration, and threshold optimization coordinator.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from sandisk_yield.models.model_a import ModelA
from sandisk_yield.models.model_b import ModelB, WaferBlockDataset
from sandisk_yield.models.calibration import Calibrator
from sandisk_yield.cascade.conformal_gate import ConformalGate
from sandisk_yield.cascade.router import CascadeRouter
from sandisk_yield.cascade.cascade import WaferFusionCascade
from sandisk_yield.data.splitter import make_calibration_split
from sandisk_yield.features.blocks import compute_block_features_df
from sandisk_yield.training.evaluation import compute_die_yield_metrics, optimize_threshold
from sandisk_yield.schema import create_eligible_mask
from sandisk_yield.logging_utils import setup_logger
from sandisk_yield.data.alignment import assert_feature_alignment, assert_row_alignment

logger = setup_logger("sandisk_yield.trainer_cascade")


def build_and_calibrate_cascade(
    df_raw: pd.DataFrame,
    X_tab: pd.DataFrame,
    y: pd.Series,
    model_a: ModelA,
    model_b: Optional[ModelB] = None,
    calibration_ratio: float = 0.2,
    calibration_method: str = "isotonic",
    conformal_coverage: float = 0.95,
    routing_config: Optional[dict] = None,
    seed: int = 42
) -> Tuple[WaferFusionCascade, Dict[str, float]]:
    """
    Splits into model-fit and calibration wafers.
    Calibrates Model A, fits Conformal Gate, determines optimal threshold,
    and initializes the full WaferFusionCascade.
    """
    routing_cfg = routing_config or {}
    assert_row_alignment(df_raw, X_tab, y)
    
    # 1. Split into fit and calibration wafers (Strict grouping by wafer_id)
    fit_df, calib_df = make_calibration_split(df_raw, calibration_ratio=calibration_ratio, group_col="wafer_id", seed=seed)
    
    # Extract eligible masks for calibration
    calib_elig = create_eligible_mask(calib_df).values
    calib_indices = np.where(calib_elig)[0]
    
    X_tab_calib = X_tab.loc[calib_df.index].iloc[calib_indices]
    y_calib = y.loc[calib_df.index].iloc[calib_indices].values
    assert_feature_alignment(model_a.feature_names_, X_tab_calib.columns)
    # Bug: splitting after a full-data fit does not create held-out predictions.
    # Index overlap is a conservative guard; callers must preserve original row
    # IDs across fit/calibration datasets (do not reset each partition separately).
    if hasattr(model_a, "training_index_") and model_a.training_index_.intersection(X_tab_calib.index).size:
        raise ValueError("Calibration rows overlap model fitting rows; use independent wafers or the OOF run_all workflow")
    logger.warning("Caller must also ensure Model B and feature selection never saw calibration wafers; probability calibration and gate reuse is not independent split conformal.")
    
    logger.info(f"Calibrating Cascade on {calib_df['wafer_id'].nunique()} holdout calibration wafers ({len(calib_indices)} eligible dies)...")
    
    # 2. Get Model A raw probabilities on calibration set
    raw_p_a_calib = model_a.predict_proba(X_tab_calib)[:, 1]
    
    # 3. Fit probability calibrator
    calibrator_a = Calibrator(method=calibration_method)
    calibrator_a.fit(raw_p_a_calib, y_calib)
    calibrated_p_a = calibrator_a.predict(raw_p_a_calib)
    
    # 4. Calibrate Conformal Uncertainty Gate
    conformal_gate = ConformalGate(default_coverage=conformal_coverage)
    conformal_gate.calibrate(calibrated_p_a, y_calib)
    
    # 5. Fit Calibrator for Model B if present
    calibrator_b = None
    if model_b is not None:
        block_strings_calib = calib_df["block_readings"].iloc[calib_indices].tolist()
        df_calib_raw = calib_df.iloc[calib_indices]
        X_blk_feats_calib = compute_block_features_df(df_calib_raw, expected_len=model_b.block_length).values
        
        raw_p_b_calib = model_b.predict_proba(
            X_tab_calib.values,
            block_strings_calib,
            X_blk_feats_calib
        )[:, 1]
        
        calibrator_b = Calibrator(method=calibration_method)
        calibrator_b.fit(raw_p_b_calib, y_calib)
        
    # 6. Initialize Router
    router = CascadeRouter(
        prob_margin=routing_cfg.get("prob_margin", 0.20),
        min_model_b_prob=routing_cfg.get("min_model_b_prob", 0.10),
        max_model_b_prob=routing_cfg.get("max_model_b_prob", 0.90),
        spatial_anomaly_threshold=routing_cfg.get("spatial_anomaly_threshold", 0.75),
        force_model_b_on_edge=routing_cfg.get("force_model_b_on_edge", False)
    )
    
    # 7. Find optimal decision threshold on calibrated probabilities
    opt_thresh, best_metrics = optimize_threshold(y_calib, calibrated_p_a, metric="fail_f1")
    logger.info(f"Optimal decision threshold selected on calibration set: {opt_thresh:.2f} (Fail F1: {best_metrics['fail_f1']:.4f})")
    
    cascade = WaferFusionCascade(
        model_a=model_a,
        calibrator_a=calibrator_a,
        conformal_gate=conformal_gate,
        router=router,
        model_b=model_b,
        calibrator_b=calibrator_b,
        threshold=opt_thresh
    )
    
    return cascade, best_metrics
