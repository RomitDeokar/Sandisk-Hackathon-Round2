"""
src/sandisk_yield/training/trainer_a.py
======================================
Model A grouped cross-validation trainer.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sandisk_yield.models.model_a import ModelA
from sandisk_yield.data.splitter import make_grouped_folds
from sandisk_yield.training.evaluation import compute_die_yield_metrics, optimize_threshold
from sandisk_yield.logging_utils import setup_logger
from sandisk_yield.data.alignment import assert_feature_alignment, assert_row_alignment, log_feature_uniqueness
from sandisk_yield.training.evaluation import summarize_oof
from sklearn.metrics import average_precision_score

logger = setup_logger("sandisk_yield.trainer_a")


def train_model_a_cv(
    df_raw: pd.DataFrame,
    X_tab: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lightgbm",
    model_params: Optional[dict] = None,
    n_splits: int = 5,
    seed: int = 42,
    feature_pipeline_factory=None,
    raw_context: Optional[pd.DataFrame] = None
) -> Tuple[ModelA, np.ndarray, Dict[str, float]]:
    """
    Trains Model A using Wafer-Grouped K-Fold Cross Validation.
    Computes out-of-fold probability predictions and evaluates metrics.
    """
    assert_row_alignment(df_raw, X_tab, y)
    assert (df_raw.old_label == 0).all(), "CV target rows must be eligible"
    folds = make_grouped_folds(df_raw, n_splits=n_splits, group_col="wafer_id", seed=seed)
    oof_probs = np.full(len(df_raw), np.nan, dtype=np.float64)
    seen = np.zeros(len(df_raw), dtype=int)
    if feature_pipeline_factory is None:
        logger.warning("Precomputed X_tab must be target-independent; use a feature_pipeline_factory for learned preprocessing")
    
    logger.info(f"Starting Model A {n_splits}-fold grouped CV across {df_raw['wafer_id'].nunique()} wafers...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_tab.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X_tab.iloc[val_idx], y.iloc[val_idx]
        if feature_pipeline_factory is not None:
            context = df_raw if raw_context is None else raw_context
            pipeline = feature_pipeline_factory().fit(df_raw.iloc[train_idx], y_tr)
            full_tr = context.loc[context.wafer_id.isin(df_raw.iloc[train_idx].wafer_id)]
            full_va = context.loc[context.wafer_id.isin(df_raw.iloc[val_idx].wafer_id)]
            X_tr = pipeline.transform(full_tr).loc[y_tr.index]
            X_va = pipeline.transform(full_va).loc[y_va.index]
        assert_feature_alignment(X_tr.columns, X_va.columns)
        assert_row_alignment(X_tr, y_tr)
        assert_row_alignment(X_va, y_va)
        log_feature_uniqueness(X_tr, f"Model A fold {fold_idx + 1}")
        if y_tr.nunique() != 2:
            raise ValueError("Training fold lacks one class")
        
        clf = ModelA(model_type=model_type, params=model_params)
        clf.fit(X_tr, y_tr)
        
        val_probs = clf.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = val_probs
        seen[val_idx] += 1
        logger.info("Fold %s train PR-AUC=%.6f validation PR-AUC=%.6f", fold_idx + 1,
                    average_precision_score(y_tr, clf.predict_proba(X_tr)[:, 1]),
                    average_precision_score(y_va, val_probs))
        

    # Overall OOF metrics
    assert (seen == 1).all() and np.isfinite(oof_probs).all(), "Incomplete OOF predictions"
    oof_metrics = summarize_oof(y.values, oof_probs, folds)
    opt_thresh = oof_metrics["threshold"]
    logger.info("Fold metrics at the pooled OOF-tuned threshold: %s", oof_metrics["fold_metrics"])
    logger.warning("OOF threshold-tuned decision metrics are descriptive, not nested-CV estimates")
    logger.info(f"OOF Evaluation - Optimal Threshold: {opt_thresh:.2f}, Fail PR-AUC: {oof_metrics['pr_auc']:.4f}, Fail F1: {oof_metrics['fail_f1']:.4f}")
    
    # Train final Model A on all data
    final_model_a = ModelA(model_type=model_type, params=model_params)
    if feature_pipeline_factory is not None:
        pipeline = feature_pipeline_factory().fit(df_raw, y)
        X_tab = pipeline.transform(df_raw if raw_context is None else raw_context).loc[y.index]
        final_model_a.feature_pipeline_ = pipeline
    final_model_a.fit(X_tab, y)
    
    return final_model_a, oof_probs, oof_metrics
