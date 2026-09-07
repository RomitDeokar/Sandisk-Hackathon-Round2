"""
src/sandisk_yield/training/evaluation.py
========================================
Comprehensive evaluation metrics and threshold optimization for Die Yield Prediction.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix
)


def compute_die_yield_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes all standard and imbalanced classification metrics for die yield prediction.
    Primary metric: Fail PR-AUC (average precision score).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    
    # 1. Curve AUCs
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0
        
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    # 2. Precision, Recall, F1 for pass (0) and fail (1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    
    pass_prec, fail_prec = float(prec[0]), float(prec[1])
    pass_rec, fail_rec = float(rec[0]), float(rec[1])
    pass_f1, fail_f1 = float(f1[0]), float(f1[1])
    
    # 3. Balanced and Overall Accuracy
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "fail_precision": fail_prec,
        "fail_recall": fail_rec,
        "fail_f1": fail_f1,
        "pass_precision": pass_prec,
        "pass_recall": pass_rec,
        "pass_f1": pass_f1,
        "balanced_accuracy": bacc,
        "accuracy": acc,
        "threshold": threshold,
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_thresh: float = 0.05,
    max_thresh: float = 0.95,
    step: float = 0.01,
    metric: str = "fail_f1"
) -> Tuple[float, Dict[str, float]]:
    """
    Searches candidate thresholds on validation predictions to maximize chosen metric.
    """
    thresholds = np.arange(min_thresh, max_thresh + 1e-5, step)
    best_thresh = 0.5
    best_score = -1.0
    best_metrics = {}
    
    for th in thresholds:
        m = compute_die_yield_metrics(y_true, y_prob, threshold=th)
        score = m.get(metric, m["fail_f1"])
        if score > best_score:
            best_score = score
            best_thresh = float(th)
            best_metrics = m
            
    return best_thresh, best_metrics


def find_best_threshold(y_true, proba, metric="f1", fn_fp_cost_ratio=8.0, fp_cost=1.0):
    """Scan distinct probability cuts; class 1 is failure. Ties favor recall."""
    y, p = np.asarray(y_true), np.asarray(proba, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty aligned one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Expected binary targets and finite probabilities in [0, 1]")
    metric = {"fail_f1": "f1", "fail_precision": "precision", "fail_recall": "recall"}.get(metric, metric)
    if not np.isfinite(fn_fp_cost_ratio) or fn_fp_cost_ratio <= 0 or not np.isfinite(fp_cost) or fp_cost <= 0:
        raise ValueError("False-negative/false-positive cost ratio and FP cost must be positive and finite")
    if metric not in {"f1", "precision", "recall", "accuracy", "balanced_accuracy", "cost_weighted"}:
        raise ValueError("Threshold tuning requires a decision metric, not an AUC")
    # Tied probabilities enter together. Cumulative counts avoid a quadratic scan.
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(p) - 1]
    tp = np.r_[0, np.cumsum(sorted_y)[ends]].astype(float)
    fp = np.r_[0, ends + 1] - tp
    fn, tn = y.sum() - tp, len(y) - y.sum() - fp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    recall = tp / max(1, y.sum())
    scores = {
        "f1": np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=2 * tp + fp + fn > 0),
        "precision": precision, "recall": recall,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall + tn / max(1, len(y) - y.sum())) / 2,
        # Minimize empirical OOF cost; probabilities need not be calibrated.
        "cost_weighted": -(fn * fn_fp_cost_ratio + fp) * fp_cost,
    }[metric]
    thresholds = np.r_[np.nextafter(p.max(), np.inf), sorted_p[ends]]
    return float(thresholds[np.flatnonzero(scores == scores.max())[-1]])


def summarize_oof(y, probabilities, folds, threshold_objective="f1", fn_fp_cost_ratio=8.0):
    """Descriptive fold metrics at a pooled OOF-tuned threshold, not nested CV."""
    threshold = find_best_threshold(y, probabilities, threshold_objective, fn_fp_cost_ratio)
    summary = compute_die_yield_metrics(y, probabilities, threshold)
    records = [compute_die_yield_metrics(np.asarray(y)[va], probabilities[va], threshold)
               for _, va in folds]
    for metric in records[0]:
        if metric != "threshold":
            values = np.array([record[metric] for record in records])
            summary[metric + "_mean"] = float(values.mean())
            summary[metric + "_std"] = float(values.std(ddof=0))
    summary["fold_metrics"] = records
    return summary


def compare_models(results):
    """Consolidate named model/mode CV summaries into one mean ± std table."""
    rows = []
    for name, summary in results.items():
        row = {"model": name, "threshold": summary["threshold"]}
        for metric in ("pr_auc", "fail_f1", "fail_recall", "fail_precision", "accuracy"):
            row[metric] = f"{summary[metric + '_mean']:.4f} ± {summary[metric + '_std']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)
