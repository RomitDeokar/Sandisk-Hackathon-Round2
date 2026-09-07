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
    gate.calibrate(np.tile(calib_probs, 30), np.tile(y_true, 30))
    
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


def test_tiny_calibration_does_not_fabricate_confidence():
    gate = ConformalGate(default_coverage=.99).calibrate([.1, .2, .9], [0, 0, 1])
    assert np.isinf(gate.quantiles_[.99])
    assert gate.predict_prediction_sets([0, .5, 1]) == [[0, 1]] * 3


def test_empty_prediction_set_is_ambiguous():
    gate = ConformalGate(default_coverage=.9).calibrate([.1] * 100, [0] * 100)
    assert gate.predict_prediction_sets([.5]) == [[]]
    assert gate.derive_routing_states([.5]).tolist() == ["AMBIGUOUS"]


def test_exact_finite_sample_order_statistic():
    scores = np.arange(1, 11) / 100
    gate = ConformalGate(default_coverage=.8).calibrate(scores, np.zeros(10))
    assert gate.quantiles_[.8] == pytest.approx(.09)


def test_wafer_maximum_calibration_counts_wafers_not_dies():
    gate = ConformalGate().calibrate([.1] * 200, [0] * 200, groups=["W1"] * 100 + ["W2"] * 100)
    assert gate.n_calibration_units_ == 2
    assert gate.calibration_unit_ == "wafer_maximum"
    assert gate.derive_routing_states([.01]).tolist() == ["AMBIGUOUS"]


@pytest.mark.parametrize("p,y", [([np.nan], [0]), ([1.1], [1]), ([[.1]], [0]),
                                  ([.1], [2]), ([.1], [.5]), ([.1, .2], [0]), ([], [])])
def test_invalid_calibration_rejected(p, y):
    with pytest.raises(ValueError):
        ConformalGate().calibrate(p, y)


@pytest.mark.parametrize("coverage", [0, 1, -1, np.nan])
def test_invalid_coverage_rejected(coverage):
    with pytest.raises(ValueError):
        ConformalGate(default_coverage=coverage)


def test_uncalibrated_coverage_rejected():
    gate = ConformalGate().calibrate([.1], [0])
    with pytest.raises(ValueError, match="not calibrated"):
        gate.predict_prediction_sets([.1], coverage=.88)


def test_missing_model_b_does_not_count_inspections(dummy_cascade):
    cascade, X = dummy_cascade
    raw = pd.DataFrame({"wafer_id": ["w"] * 5, "die_row": range(5),
                        "die_col": [0] * 5, "old_label": [0] * 5})
    out = cascade.predict_detailed(raw, X)
    assert not out.routed_to_model_b.any()
    with pytest.raises(ValueError, match="requires"):
        cascade.predict_detailed(raw, X, mode="full_model_b")
    with pytest.raises(ValueError, match="Unknown"):
        cascade.predict_detailed(raw, X, mode="typo")
