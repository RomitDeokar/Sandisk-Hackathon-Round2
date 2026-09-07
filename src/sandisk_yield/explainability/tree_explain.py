"""
src/sandisk_yield/explainability/tree_explain.py
===============================================
Tree explainability (Feature Importance / Permutation Importance / SHAP fallback).
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from sandisk_yield.models.model_a import ModelA


def get_feature_importances(model_a: ModelA, top_n: int = 20) -> pd.DataFrame:
    """
    Extracts top feature importance scores from Model A.
    """
    importances = model_a.feature_importances_
    names = model_a.feature_names_
    
    if len(importances) != len(names):
        importances = np.zeros(len(names))
        
    df = pd.DataFrame({
        "feature": names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)
    
    return df.head(top_n)
