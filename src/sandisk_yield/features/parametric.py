"""
src/sandisk_yield/features/parametric.py
========================================
Robust parametric feature cleaning, filtering, and optional scaling.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sandisk_yield.schema import detect_feature_cols


class ParametricPreprocessor(BaseEstimator, TransformerMixin):
    """
    Fits parametric preprocessing statistics ONLY on training data:
    - Identifies non-constant numeric features
    - Imputes non-finite values using training medians
    - Computes robust scaling parameters (median and IQR) if requested
    """

    def __init__(
        self,
        scale_features: bool = False,
        feature_prefix: str = "feature_",
        variance_threshold: float = 1e-8
    ):
        self.scale_features = scale_features
        self.feature_prefix = feature_prefix
        self.variance_threshold = variance_threshold
        
        self.valid_features_: List[str] = []
        self.medians_: dict = {}
        self.iqrs_: dict = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        all_feats = detect_feature_cols(X, prefix=self.feature_prefix)
        self.valid_features_ = []
        self.medians_ = {}
        self.iqrs_ = {}

        for col in all_feats:
            vals = X[col].replace([np.inf, -np.inf], np.nan)
            med = float(vals.median())
            if np.isnan(med):
                med = 0.0
            
            # Check variance on non-nan
            cleaned = vals.fillna(med)
            var = float(cleaned.var())
            if var > self.variance_threshold:
                self.valid_features_.append(col)
                self.medians_[col] = med
                q75 = float(cleaned.quantile(0.75))
                q25 = float(cleaned.quantile(0.25))
                iqr = q75 - q25
                self.iqrs_[col] = iqr if iqr > 1e-6 else 1.0

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.valid_features_:
            if col in X.columns:
                vals = X[col].replace([np.inf, -np.inf], np.nan).fillna(self.medians_[col]).astype(np.float32)
                if self.scale_features:
                    out[col] = (vals - self.medians_[col]) / self.iqrs_[col]
                else:
                    out[col] = vals
            else:
                # Bug: a missing column used zero even for unscaled features,
                # unlike missing cells which used the fitted training median.
                out[col] = np.float32(0.0 if self.scale_features else self.medians_[col])
        return out
