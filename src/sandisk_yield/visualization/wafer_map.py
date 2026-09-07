"""
src/sandisk_yield/visualization/wafer_map.py
============================================
Wafer risk maps using actual 2D die coordinates on the wafer disk.
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_wafer_risk_map(
    w_df: pd.DataFrame,
    wafer_id: str,
    output_path: Optional[Union[str, Path]] = None,
    value_col: str = "final_probability"
):
    """
    Renders 2D spatial scatter / wafer map of die risk probabilities.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    
    rows = w_df["die_row"].values
    cols = w_df["die_col"].values
    vals = w_df[value_col].values
    
    sc = ax.scatter(
        cols, rows, c=vals, cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
        s=40, edgecolors="black", linewidths=0.3
    )
    
    # Invert y-axis to match wafer matrix convention
    ax.invert_yaxis()
    ax.set_title(f"Wafer Risk Map: {wafer_id}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Die Column")
    ax.set_ylabel("Die Row")
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Failure Probability Risk")
    
    plt.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p)
        plt.close(fig)
    else:
        return fig
