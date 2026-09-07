"""
src/sandisk_yield/visualization/curves.py
=========================================
Precision-Recall, ROC, Calibration, and Threshold-vs-F1 curves generation.
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
from sklearn.calibration import calibration_curve


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Union[str, Path], model_name: str = "Model"):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    score = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC = {score:.4f})")
    ax.set_xlabel("Recall (Fail)")
    ax.set_ylabel("Precision (Fail)")
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(p)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Union[str, Path], model_name: str = "Model"):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (ROC-AUC = {score:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(p)
    plt.close(fig)


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Union[str, Path]):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    ax.plot(prob_pred, prob_true, "s-", label="Calibrated Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram / Calibration Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(p)
    plt.close(fig)
