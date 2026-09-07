"""
src/sandisk_yield/features/pipeline.py
======================================
Unified feature engineering pipeline combining parametric, spatial, wafer, and local process features.
"""

from typing import List, Optional, Tuple
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sandisk_yield.features.parametric import ParametricPreprocessor
from sandisk_yield.features.spatial import compute_spatial_features
from sandisk_yield.features.wafer import compute_wafer_context, LocalProcessFeatureExtractor
from sandisk_yield.features.blocks import compute_block_features_df
from sandisk_yield.data.alignment import assert_feature_alignment, assert_row_alignment


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Fits training-only preprocessors (Parametric Imputer/Scaler, Local Process Extractor)
    and transforms raw inputs into feature-rich tables for Model A and Model B.
    """

    def __init__(
        self,
        scale_features: bool = False,
        top_k_parametric: int = 30,
        spatial_windows: List[int] = [3, 5, 7],
        include_blocks: bool = False,
        num_block_readings: int = 2000,
        epsilon: float = 1e-6
    ):
        self.scale_features = scale_features
        self.top_k_parametric = top_k_parametric
        self.spatial_windows = spatial_windows
        self.include_blocks = include_blocks
        self.num_block_readings = num_block_readings
        self.epsilon = epsilon
        
        self.parametric_preprocessor = ParametricPreprocessor(scale_features=scale_features)
        self.local_process_extractor = LocalProcessFeatureExtractor(
            top_k=top_k_parametric,
            window_size=5,
            epsilon=epsilon
        )
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.feature_names_ = []
        if y is not None:
            assert_row_alignment(X, y)
        # 1. Fit parametric preprocessor on raw feature columns
        self.parametric_preprocessor.fit(X)
        
        # 2. Fit local process extractor
        cleaned_param = self.parametric_preprocessor.transform(X)
        self.local_process_extractor.fit(cleaned_param, y=y)
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Parametric features
        feat_param = self.parametric_preprocessor.transform(df)
        
        # 2. Spatial geometry & neighborhood features (ONLY old_label)
        feat_spatial = compute_spatial_features(df, windows=self.spatial_windows, epsilon=self.epsilon)
        
        # 3. Wafer context features
        feat_wafer = compute_wafer_context(df)
        
        # 4. Localized process variation features
        # Bug: raw NaNs/infinities entered neighborhood filters, despite fitting
        # selection on cleaned values. Reuse training-fitted cleaning here.
        local_input = df[["wafer_id", "die_row", "die_col"]].join(feat_param)
        feat_local = self.local_process_extractor.transform(local_input)
        
        parts = [feat_param, feat_spatial, feat_wafer, feat_local]
        
        # 5. Optional explicit block features
        if self.include_blocks and "block_readings" in df.columns:
            feat_blocks = compute_block_features_df(df, expected_len=self.num_block_readings)
            parts.append(feat_blocks)
            
        combined = pd.concat(parts, axis=1)
        assert_row_alignment(df, combined)
        if self.feature_names_:
            assert_feature_alignment(self.feature_names_, combined.columns)
        else:
            assert_feature_alignment(combined.columns, combined.columns)
            self.feature_names_ = list(combined.columns)
        return combined
