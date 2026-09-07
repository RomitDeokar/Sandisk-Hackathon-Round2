"""
src/sandisk_yield/models/calibration.py
======================================
Post-hoc probability calibration (Isotonic Regression & Platt Scaling).
"""

from pathlib import Path
from typing import Optional, Union
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    """
    Fits probability calibrator on validation/calibration probabilities to produce
    reliable, true risk estimates.
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method.lower()
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
        elif self.method in ("sigmoid", "platt"):
            self.model = LogisticRegression(C=1.0, solver="lbfgs")
        else:
            raise ValueError(f"Unknown calibration method: '{method}'. Use 'isotonic' or 'sigmoid'.")
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        raw_probs = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        y_true = np.asarray(y_true, dtype=int)
        
        if self.method == "isotonic":
            self.model.fit(raw_probs, y_true)
        else:
            # Logistic regression expects 2D features
            self.model.fit(raw_probs.reshape(-1, 1), y_true)
            
        self.is_fitted = True
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.asarray(raw_probs)
            
        raw_probs = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        if self.method == "isotonic":
            calibrated = self.model.predict(raw_probs)
        else:
            calibrated = self.model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            
        return np.clip(calibrated, 0.0, 1.0)

    def save(self, path: Union[str, Path]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Calibrator":
        return joblib.load(path)
