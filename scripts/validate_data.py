"""
scripts/validate_data.py
========================
Standalone script to validate input datasets against generator schema.
"""

import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sandisk_yield.config import load_config, add_data_arguments, resolve_data_paths, require_input_files
from sandisk_yield.data.loader import load_dataset
from sandisk_yield.data.validator import validate_dataframe
from sandisk_yield.logging_utils import setup_logger

logger = setup_logger("validate_data")


def main():
    parser = argparse.ArgumentParser(description="Validate SanDisk dataset schema and constraints")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument("--file", type=str, default=None, help="Explicit dataset path")
    add_data_arguments(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolve_data_paths(cfg, args, parser, required=() if args.file else ("train", "validation", "test"))
    paths_to_check = [args.file] if args.file else [
        cfg["paths"]["raw_train"],
        cfg["paths"]["raw_test"],
        cfg["paths"]["raw_validation"]
    ]

    require_input_files(parser, paths_to_check)
    for p in paths_to_check:
        if p and Path(p).exists():
            is_training = "validation" not in str(p).lower()
            df = load_dataset(p, split_name=Path(p).name)
            report = validate_dataframe(
                df,
                is_training=is_training,
                expected_block_length=cfg["data"].get("num_block_readings", 2000)
            )
            logger.info(f"Validation Result for {p}: {report['status']} ({report['n_rows']:,} rows, {report['n_features']} features)")


if __name__ == "__main__":
    main()
