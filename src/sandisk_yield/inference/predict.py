"""
src/sandisk_yield/inference/predict.py
======================================
High-level prediction inference pipeline.
"""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from sandisk_yield.cascade.cascade import WaferFusionCascade
from sandisk_yield.features.pipeline import FeaturePipeline
from sandisk_yield.data.loader import load_dataset
from sandisk_yield.inference.submission import export_submission


def run_inference(
    cascade: WaferFusionCascade,
    feature_pipeline: FeaturePipeline,
    data_path: Union[str, Path],
    mode: str = "cascade",
    batch_size: int = 64
) -> pd.DataFrame:
    """
    Runs full inference on raw dataset and returns prediction dataframe.
    """
    df_raw = load_dataset(data_path, split_name="inference")
    X_tab = feature_pipeline.transform(df_raw)
    
    results = cascade.predict_detailed(
        df_raw=df_raw,
        X_tab=X_tab,
        mode=mode,
        batch_size=batch_size
    )
    return results
