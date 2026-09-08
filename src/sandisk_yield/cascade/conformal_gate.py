"""
src/sandisk_yield/cascade/conformal_gate.py
==========================================
Conformal Prediction / Uncertainty Coverage Gate.
Calibrates prediction set non-conformity scores on a holdout wafer partition.
Derives routing states: CONFIDENT_PASS, CONFIDENT_FAIL, AMBIGUOUS.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np


class ConformalGate:
    """
    Split-conformal prediction classifier for coverage-controlled routing.
    Non-conformity score: s_i = 1 - p(y_true | x_i)
    """

    def __init__(self, coverage_levels: List[float] = [0.90, 0.95, 0.975, 0.99], default_coverage: float = 0.95):
        self.coverage_levels = coverage_levels
        self.default_coverage = default_coverage
        self.quantiles_: Dict[float, float] = {}
        self.is_calibrated = False

    def calibrate(self, calib_probs: np.ndarray, y_calib: np.ndarray):
        """
        Calibrate quantile thresholds on an independent holdout set of calibration wafers.
        """
        p1 = np.asarray(calib_probs, dtype=np.float64)
        y = np.asarray(y_calib, dtype=int)
        n = len(y)
        if n == 0:
            raise ValueError("Empty calibration set passed to ConformalGate")
            
        # Non-conformity scores for true labels
        # If y=1, s = 1 - p1. If y=0, s = 1 - (1 - p1) = p1.
        scores = np.where(y == 1, 1.0 - p1, p1)
        
        self.quantiles_ = {}
        for alpha_cov in sorted(set(self.coverage_levels + [self.default_coverage])):
            # Finite-sample conformal quantile level
            q_level = np.ceil((n + 1) * alpha_cov) / n
            q_level = min(1.0, max(0.0, q_level))
            q_val = float(np.quantile(scores, q_level, method="higher"))
            self.quantiles_[alpha_cov] = q_val
            
        self.is_calibrated = True
        return self

    def predict_prediction_sets(self, probs: np.ndarray, coverage: Optional[float] = None) -> List[List[int]]:
        """
        Generates conformal prediction sets {0}, {1}, or {0, 1} for each die.
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalGate must be calibrated before predicting sets")
            
        cov = coverage or self.default_coverage
        q_val = self.quantiles_.get(cov, 0.95)
        
        p1 = np.asarray(probs, dtype=np.float64)
        p0 = 1.0 - p1
        
        # Class included if 1 - p_k <= q_val  =>  p_k >= 1 - q_val
        thresh = 1.0 - q_val
        
        pred_sets = []
        for prob0, prob1 in zip(p0, p1):
            s = []
            if prob0 >= thresh:
                s.append(0)
            if prob1 >= thresh:
                s.append(1)
            # Guarantee non-empty set
            if not s:
                s = [0] if prob0 >= prob1 else [1]
            pred_sets.append(s)
            
        return pred_sets

    def derive_routing_states(self, probs: np.ndarray, coverage: Optional[float] = None) -> np.ndarray:
        """
        Maps prediction sets to routing states:
        - CONFIDENT_PASS: set is {0}
        - CONFIDENT_FAIL: set is {1}
        - AMBIGUOUS: set is {0, 1}
        """
        sets = self.predict_prediction_sets(probs, coverage=coverage)
        states = []
        for s in sets:
            if len(s) == 1:
                states.append("CONFIDENT_PASS" if s[0] == 0 else "CONFIDENT_FAIL")
            else:
                states.append("AMBIGUOUS")
        return np.array(states, dtype=object)

    def save(self, path: Union[str, Path]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConformalGate":
        return joblib.load(path)
