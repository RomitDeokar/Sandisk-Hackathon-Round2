"""
src/sandisk_yield/models/model_a.py
===================================
Model A: Fast die-level, spatial, and wafer screening classifier.
Supports LightGBM with robust fallbacks to XGBoost or HistGradientBoostingClassifier.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
import pandas as pd

from sandisk_yield.logging_utils import setup_logger
from sandisk_yield.data.alignment import assert_feature_alignment, assert_row_alignment

logger = setup_logger("sandisk_yield.model_a")


class ModelA:
    """
    Fast screening model combining parametric features, spatial geometry,
    wafer context, and localized process variations.
    """

    def __init__(self, model_type: str = "lightgbm", params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type.lower()
        # Defaults precede user overrides; never overwrite requested weighting.
        self.params = {"n_estimators": 100, **(params or {})}
        if self.model_type == "lightgbm" and not any(
            key in self.params for key in ("class_weight", "is_unbalance", "scale_pos_weight")
        ):
            self.params["is_unbalance"] = True
        self.model = None
        self.feature_names_: list = []
        self._init_model()

    def _init_model(self):
        if self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(**self.params)
                logger.info("Effective LightGBM parameters: %s", self.model.get_params())
                logger.info("Initialized Model A with LightGBM")
                return
            except ImportError:
                logger.warning("LightGBM not found, falling back to XGBoost")
                self.model_type = "xgboost"
                
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                # Translate LightGBM params if needed
                xgb_params = self.params.copy()
                if "num_leaves" in xgb_params:
                    xgb_params.pop("num_leaves")
                self._xgb_balance = bool(xgb_params.pop("is_unbalance", False))
                if "class_weight" in xgb_params and xgb_params["class_weight"] == "balanced":
                    xgb_params.pop("class_weight")
                    self._xgb_balance = True
                # Derive class weight in fit from the actual training fold.
                self.model = xgb.XGBClassifier(**xgb_params)
                logger.info("Initialized Model A with XGBoost")
                return
            except ImportError:
                logger.warning("XGBoost not found, falling back to sklearn HistGradientBoostingClassifier")
                self.model_type = "hist_gbm"
                
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            max_iter=self.params.get("n_estimators", 150),
            learning_rate=self.params.get("learning_rate", 0.05),
            max_depth=self.params.get("max_depth", 6),
            class_weight="balanced",
            random_state=self.params.get("random_state", 42)
        )
        logger.info("Initialized Model A with HistGradientBoostingClassifier")

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: Optional[list] = None):
        assert_row_alignment(X, y)
        assert_feature_alignment(X.columns, X.columns)
        self.feature_names_ = list(X.columns)
        self.training_index_ = X.index.copy()
        if self.model_type == "xgboost" and getattr(self, "_xgb_balance", False):
            if "scale_pos_weight" not in self.params:
                self.model.set_params(scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())))
        if self.model_type == "lightgbm":
            self.model.fit(X, y)
        elif self.model_type == "xgboost":
            self.model.fit(X, y)
        else:
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert_feature_alignment(self.feature_names_, X.columns)
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return np.zeros(len(self.feature_names_))

    def save(self, path: Union[str, Path]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelA":
        return joblib.load(path)
