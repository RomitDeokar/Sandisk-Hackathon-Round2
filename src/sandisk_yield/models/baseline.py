"""
src/sandisk_yield/models/baseline.py
===================================
Baseline simple classification models (LogisticRegression / Dummy).
"""

from pathlib import Path
from typing import Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class BaselineModel:
    """Fast linear / baseline model for benchmark comparison."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = LogisticRegression(class_weight="balanced", max_iter=200, random_state=random_state)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, path: Union[str, Path]):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaselineModel":
        return joblib.load(path)
