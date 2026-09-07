"""
src/sandisk_yield/seed.py
=========================
Global seed and reproducibility utilities for Python, NumPy, PyTorch.
"""

import os
import random
import sys
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def seed_everything(seed: int = 42) -> None:
    """Set seeds across Python random, NumPy, OS, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(requested: str = "auto"):
    """Resolve PyTorch device safely based on availability and config."""
    if not HAS_TORCH:
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")
