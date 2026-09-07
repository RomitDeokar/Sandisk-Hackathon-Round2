"""
src/sandisk_yield/cascade/uncertainty.py
========================================
Uncertainty metrics estimation (Entropy, Margin, Variation Ratio).
"""

import numpy as np


def compute_uncertainty_metrics(probs: np.ndarray, epsilon: float = 1e-7) -> dict:
    """
    Computes die-level prediction uncertainty metrics from predicted probabilities.
    probs: 1D array of failure probabilities in [0, 1].
    """
    p1 = np.clip(np.asarray(probs, dtype=np.float64), epsilon, 1.0 - epsilon)
    p0 = 1.0 - p1
    
    # 1. Shannon Entropy: - sum(p * log(p))
    entropy = - (p0 * np.log2(p0) + p1 * np.log2(p1))
    
    # 2. Decision Margin: distance to decision boundary |p1 - 0.5|
    margin = np.abs(p1 - 0.5)
    
    # 3. Least confidence: 1 - max(p0, p1)
    least_confidence = 1.0 - np.maximum(p0, p1)
    
    return {
        "entropy": entropy.astype(np.float32),
        "margin": margin.astype(np.float32),
        "least_confidence": least_confidence.astype(np.float32),
    }
