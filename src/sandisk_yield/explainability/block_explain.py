"""
src/sandisk_yield/explainability/block_explain.py
================================================
Sub-die block sequence saliency and localized anomaly localization.
"""

from typing import Dict, Iterable, List, Tuple, Union
import numpy as np
import torch

from sandisk_yield.models.model_b import ModelB
from sandisk_yield.features.blocks import parse_block_readings_row


def explain_die_block(
    model_b: Union[ModelB, object],
    block_raw_str: str,
    top_n_regions: int = 3
) -> Dict[str, any]:
    """
    Computes attention saliency over 2000 block readings to pinpoint suspicious sub-die defects.
    """
    # Persisted CNN artifacts are BlockFeatureModel adapters. Unwrap the inner
    # neural network while retaining support for callers that pass ModelB.
    neural_model = getattr(model_b, "model", model_b)
    if not isinstance(neural_model, ModelB):
        raise TypeError("Expected ModelB or a saved CNN BlockFeatureModel adapter")
    device = next(neural_model.parameters()).device
    arr = parse_block_readings_row(block_raw_str, expected_len=neural_model.block_length)
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite block readings")
    x_tensor = torch.as_tensor(arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    neural_model.eval()
    with torch.no_grad():
        _, attn_weights = neural_model.encoder(x_tensor)
        attn = attn_weights.squeeze().cpu().numpy()
        
    # Map attention back to sequence bins
    seq_len = len(arr)
    n_attn = len(attn)
    bin_size = seq_len / n_attn
    
    top_attn_bins = np.argsort(attn)[-top_n_regions:][::-1]
    suspicious_ranges = []
    for b in top_attn_bins:
        start_idx = int(b * bin_size)
        end_idx = min(seq_len, int((b + 1) * bin_size))
        sub_arr = arr[start_idx:end_idx]
        suspicious_ranges.append({
            "block_range": f"{start_idx}-{end_idx}",
            "attention_weight": float(attn[b]),
            "mean_signal": float(sub_arr.mean()) if len(sub_arr) > 0 else 0.0,
            "max_signal": float(sub_arr.max()) if len(sub_arr) > 0 else 0.0,
        })
        
    return {
        "raw_signal": arr,
        "attention_profile": attn,
        "suspicious_regions": suspicious_ranges
    }


def explain_block_batch(
    model_b: Union[ModelB, object],
    block_values: Iterable,
    top_n_regions: int = 3,
    batch_size: int = 128,
) -> List[List[Dict[str, float]]]:
    """Return the strongest learned-attention regions for many dies efficiently."""
    neural_model = getattr(model_b, "model", model_b)
    if not isinstance(neural_model, ModelB):
        raise TypeError("Expected ModelB or a saved CNN BlockFeatureModel adapter")
    if batch_size < 1 or top_n_regions < 1:
        raise ValueError("batch_size and top_n_regions must be positive")
    device = next(neural_model.parameters()).device
    values = list(block_values)
    results = []
    neural_model.eval()
    for start in range(0, len(values), batch_size):
        arrays = np.stack([
            parse_block_readings_row(v, expected_len=neural_model.block_length)
            for v in values[start:start + batch_size]
        ])
        if not np.isfinite(arrays).all():
            raise ValueError("Nonfinite block readings")
        tensor = torch.as_tensor(arrays, dtype=torch.float32, device=device).unsqueeze(1)
        with torch.no_grad():
            _, weights = neural_model.encoder(tensor)
        weights = weights.detach().cpu().numpy()
        for arr, attn in zip(arrays, weights):
            bin_size = len(arr) / len(attn)
            regions = []
            for rank, index in enumerate(np.argsort(attn)[-top_n_regions:][::-1], 1):
                lo = int(index * bin_size)
                hi = min(len(arr), int((index + 1) * bin_size))
                signal = arr[lo:hi]
                regions.append({
                    "attention_rank": rank,
                    "block_start": lo,
                    "block_end": hi,
                    "attention_weight": float(attn[index]),
                    "mean_signal": float(signal.mean()),
                    "max_signal": float(signal.max()),
                })
            results.append(regions)
    return results
