"""
tests/test_cascade.py
======================
Unit tests for ConformalGate, CascadeRouter, and WaferFusionCascade orchestration.
"""

import numpy as np
import pandas as pd
import pytest

from sandisk_yield.models.model_a import ModelA
from sandisk_yield.models.calibration import Calibrator
from sandisk_yield.cascade.conformal_gate import ConformalGate
from sandisk_yield.cascade.router import CascadeRouter
from sandisk_yield.cascade.cascade import WaferFusionCascade


@pytest.fixture
def dummy_cascade():
    # Model A
    clf = ModelA(model_type="hist_gbm", params={"max_iter": 10})
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "f2": [0.1, 0.2, 0.3, 0.4, 0.5]})
    y = pd.Series([0, 0, 0, 1, 1])
    clf.fit(X, y)
    
    # Calibrator
    calib = Calibrator(method="isotonic")
    calib.fit(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), np.array([0, 0, 0, 1, 1]))
    
    # Conformal Gate
    gate = ConformalGate(default_coverage=0.95)
    gate.calibrate(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), np.array([0, 0, 0, 1, 1]))
    
    # Router
    router = CascadeRouter(prob_margin=0.20, min_model_b_prob=0.20, max_model_b_prob=0.80)
    
    cascade = WaferFusionCascade(
        model_a=clf,
        calibrator_a=calib,
        conformal_gate=gate,
        router=router,
        model_b=None,  # pure Model A fallback
        threshold=0.5
    )
    return cascade, X


def test_conformal_gate_prediction_sets():
    gate = ConformalGate(default_coverage=0.95)
    calib_probs = np.array([0.05, 0.10, 0.20, 0.85, 0.95])
    y_true = np.array([0, 0, 0, 1, 1])
    gate.calibrate(calib_probs, y_true)
    
    states = gate.derive_routing_states(np.array([0.01, 0.50, 0.99]))
    assert states[0] == "CONFIDENT_PASS"
    assert states[2] == "CONFIDENT_FAIL"


def test_router_margin_escalation():
    router = CascadeRouter(prob_margin=0.20)
    probs = np.array([0.05, 0.45, 0.95])
    conformal_states = np.array(["CONFIDENT_PASS", "AMBIGUOUS", "CONFIDENT_FAIL"])
    
    routes = router.route(probs, conformal_states, threshold=0.5)
    assert routes[0] == False  # Far from boundary, confident pass
    assert routes[1] == True   # Within margin |0.45 - 0.50| < 0.20
    assert routes[2] == False  # Far from boundary, confident fail


def test_cascade_prediction_detailed(dummy_cascade):
    cascade, X = dummy_cascade
    df_raw = pd.DataFrame({
        "wafer_id": ["W1", "W1", "W1", "W1", "W1"],
        "die_row": [0, 1, 2, 3, 4],
        "die_col": [0, 0, 0, 0, 0],
        "old_label": [0, 0, 1, 0, 0],  # row index 2 is an old failure
    })
    
    res = cascade.predict_detailed(df_raw, X, mode="cascade")
    
    # Verify mandatory hard rule: old_label == 1 must strictly produce predicted_label == 1
    assert res.loc[2, "predicted_label"] == 1
    assert res.loc[2, "final_probability"] == 1.0
    assert "model_a_probability" in res.columns
    assert "uncertainty_state" in res.columns
