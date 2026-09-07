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
import pandas as pd
from mapie.conformity_scores import LACConformityScore


class ConformalGate:
    """
    Split-conformal prediction classifier for coverage-controlled routing.
    Non-conformity score: s_i = 1 - p(y_true | x_i)
    """

    def __init__(self, coverage_levels: Optional[List[float]] = None, default_coverage: float = 0.99):
        self.coverage_levels = list(coverage_levels) if coverage_levels is not None else [0.90, 0.95, 0.975, 0.99]
        if any(not 0 < c < 1 for c in self.coverage_levels + [default_coverage]):
            raise ValueError("Coverage levels must be strictly between zero and one")
        self.default_coverage = default_coverage
        self.quantiles_: Dict[float, float] = {}
        self.is_calibrated = False

    @staticmethod
    def _probabilities(values):
        p = np.asarray(values, dtype=np.float64)
        if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Expected one-dimensional finite probabilities in [0, 1]")
        return p

    def calibrate(self, calib_probs: np.ndarray, y_calib: np.ndarray, groups=None):
        """Use independent holdout scores; optional wafer maxima handle within-wafer dependence.

        Group mode gives simultaneous set coverage on an exchangeable NEW wafer,
        not a guarantee on final cascade decisions or under process distribution shift.
        Calibration wafers must not be used for fitting, tuning, or feature selection.
        """
        p1 = self._probabilities(calib_probs)
        y = np.asarray(y_calib)
        if not len(y) or y.shape != p1.shape or not np.isin(y, [0, 1]).all():
            raise ValueError("Expected nonempty aligned binary calibration labels")
        scores = LACConformityScore().get_conformity_scores(
            y.astype(int), np.column_stack([1 - p1, p1]), y_enc=y.astype(int)
        ).ravel()
        self.calibration_unit_ = "die"
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape != y.shape or pd.isna(groups).any():
                raise ValueError("Calibration groups must be aligned and nonmissing")
            scores = pd.DataFrame({"group": groups, "score": scores}).groupby(
                "group", sort=False)["score"].max().to_numpy()
            self.calibration_unit_ = "wafer_maximum"
        n = len(scores)
        self.n_calibration_units_ = n
        self.quantiles_ = {}
        for coverage in sorted(set(self.coverage_levels + [self.default_coverage])):
            rank = int(np.ceil((n + 1) * coverage))
            # Exact finite-sample order statistic. When rank > n the missing
            # calibration score is +infinity, NOT the largest observed score.
            self.quantiles_[coverage] = (float(np.partition(scores, rank - 1)[rank - 1])
                                         if rank <= n else float("inf"))
        self.is_calibrated = True
        return self

    def predict_prediction_sets(self, probs: np.ndarray, coverage: Optional[float] = None) -> List[List[int]]:
        """
        Generates conformal prediction sets {0}, {1}, or {0, 1} for each die.
        """
        if not self.is_calibrated:
            raise RuntimeError("ConformalGate must be calibrated before predicting sets")
            
        cov = self.default_coverage if coverage is None else coverage
        if cov not in self.quantiles_:
            raise ValueError(f"Coverage {cov} was not calibrated")
        q_val = self.quantiles_[cov]
        
        p1 = self._probabilities(probs)
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
            # Empty sets indicate insufficient evidence and MUST be escalated.
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
