"""
src/data/loader.py
==================
Dataset loader for the Sandisk Die Yield dataset.

Handles CSV, Parquet, and Pickle formats, runs schema validation,
and provides a clean interface to get train/test/validation DataFrames.

Usage
-----
    from src.data.loader import load_split, load_all_splits

    train_df = load_split("input/train.csv", split="train")
    splits   = load_all_splits("input/")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import pandas as pd

from src.data.validator import ValidationReport, validate_dataset
from src.sandisk_yield.data.delimited import read_delimited

SplitName = Literal["train", "test", "validation"]


def load_split(
    path: Union[str, Path],
    split: SplitName = "train",
    num_features: int = 500,
    num_block_readings: int = 2000,
    validate: bool = True,
    raise_on_error: bool = True,
    separator: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[ValidationReport]]:
    """
    Load a single split file and optionally validate it.

    Parameters
    ----------
    path : str or Path
        Path to the data file (.csv, .parquet, .pkl).
    split : 'train' | 'test' | 'validation'
        Which split this file represents.
    num_features : int
        Expected number of parametric feature columns.
    num_block_readings : int
        Expected number of float values in each block_readings string.
    validate : bool
        Whether to run schema validation after loading.
    raise_on_error : bool
        If True and validation finds ERRORs, raises DataValidationError.

    Returns
    -------
    (df, report) : DataFrame and ValidationReport (None if validate=False)
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".csv", ".tsv"):
        df = read_delimited(path, separator=separator)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file extension: '{ext}'. Use .csv, .tsv, .parquet, or .pkl")

    # Cast integer columns that CSV may have read as float
    for col in ["die_row", "die_col", "old_label"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    report = None
    if validate:
        from src.data.validator import validate_dataset
        report = validate_dataset(
            df,
            split=split,
            num_features=num_features,
            num_block_readings=num_block_readings,
            raise_on_error=raise_on_error,
        )

    return df, report


def load_all_splits(
    input_dir: Union[str, Path] = "input",
    fmt: str = "csv",
    num_features: int = 500,
    num_block_readings: int = 2000,
    validate: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load train, test, and validation splits from a directory.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing train.{fmt}, test.{fmt}, validation.{fmt}.
    fmt : str
        File format: 'csv', 'parquet', or 'pkl'.
    num_features : int
        Expected number of parametric feature columns.
    num_block_readings : int
        Expected number of float values per block_readings string.
    validate : bool
        Whether to run schema validation on each split.

    Returns
    -------
    dict with keys 'train', 'test', 'validation'
    """
    input_dir = Path(input_dir)
    splits: Dict[str, pd.DataFrame] = {}

    for split_name in ("train", "test", "validation"):
        fpath = input_dir / f"{split_name}.{fmt}"
        if not fpath.exists():
            print(f"[loader] WARNING: {fpath} not found — skipping '{split_name}' split")
            continue

        print(f"[loader] Loading {split_name} from {fpath} ...")
        df, report = load_split(
            fpath,
            split=split_name,
            num_features=num_features,
            num_block_readings=num_block_readings,
            validate=validate,
        )
        splits[split_name] = df

        if report is not None:
            status = "PASS" if report.passed else "FAIL"
            print(f"[loader] Validation {status} for '{split_name}' "
                  f"({len(report.errors)} errors, {len(report.warnings)} warnings)")

    return splits
