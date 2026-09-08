"""
src/sandisk_yield/data/loader.py
================================
Dataset loading with format autodetection and startup statistics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from sandisk_yield.logging_utils import setup_logger
from sandisk_yield.data.delimited import read_delimited
from sandisk_yield.schema import (
    ID_COLS,
    OLD_LABEL_COL,
    TARGET_COL,
    BLOCK_READINGS_COL,
    detect_feature_cols,
    create_eligible_mask,
)

logger = setup_logger("sandisk_yield.loader")


def load_dataset(
    path: Union[str, Path],
    split_name: str = "dataset",
    separator: Optional[str] = None
) -> pd.DataFrame:
    """
    Load CSV/TSV (delimiter detected unless separator is supplied), Parquet or Pickle.
    Prints informative startup statistics.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = p.suffix.lower()
    if ext in (".csv", ".tsv"):
        df = read_delimited(p, separator=separator)
    elif ext == ".parquet":
        df = pd.read_parquet(p)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(p)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Expected .csv, .tsv, .parquet, or .pkl")

    # Cast integer identifiers and labels if present
    for col in ["die_row", "die_col", OLD_LABEL_COL]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Compute diagnostics
    feature_cols = detect_feature_cols(df)
    n_rows = len(df)
    n_wafers = df["wafer_id"].nunique() if "wafer_id" in df.columns else 0
    has_block = BLOCK_READINGS_COL in df.columns
    old_label_dist = df[OLD_LABEL_COL].value_counts().to_dict() if OLD_LABEL_COL in df.columns else {}
    target_dist = df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else None

    logger.info(
        f"Loaded {split_name} from: {p.resolve()} | "
        f"Rows: {n_rows:,} | Wafers: {n_wafers} | Features: {len(feature_cols)} | "
        f"Block column present: {has_block}"
    )
    if old_label_dist:
        logger.info(f"  Old label dist (0=pass, 1=fail): {old_label_dist}")
    if target_dist is not None:
        eligible = create_eligible_mask(df)
        elig_fail = df.loc[eligible, TARGET_COL].sum()
        elig_total = eligible.sum()
        fail_rate = (elig_fail / elig_total * 100) if elig_total > 0 else 0.0
        logger.info(f"  Target dist: {target_dist} | Eligible New Fails: {elig_fail:,}/{elig_total:,} ({fail_rate:.2f}%)")

    return df
