"""
src/sandisk_yield/config.py
===========================
Configuration loading and dataclass representation.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


def add_data_arguments(parser):
    """Shared optional overrides; relative CLI paths follow the working directory."""
    parser.add_argument("--train", default=None, help="Override input/train.csv")
    parser.add_argument("--val", "--validation", dest="val", default=None,
                        help="Override input/validation.csv")
    parser.add_argument("--test", default=None, help="Override input/test.csv")


def require_input_files(parser, paths):
    """Check existence only, before loading any dataset or fitting a model."""
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        parser.error("Missing input file(s): " + ", ".join(missing)
                     + f". Place train.csv, validation.csv, and test.csv in {REPO_ROOT / 'input'} "
                     + "with those exact names, or supply explicit CLI file paths.")


def resolve_data_paths(cfg, args, parser, required=("train", "validation", "test")):
    """CLI override > configured input path > repo/input/<split>.csv.

    Relative configured/default paths are repo-relative; explicit CLI paths are
    relative to the caller's working directory. Never change the working directory.
    """
    paths = cfg.setdefault("paths", {})
    for split, argument in (("train", "train"), ("validation", "val"), ("test", "test")):
        override = getattr(args, argument, None)
        path = Path(override or paths.get("raw_" + split, f"input/{split}.csv")).expanduser()
        if not path.is_absolute() and override is None:
            path = REPO_ROOT / path
        paths["raw_" + split] = str(path.resolve())
    require_input_files(parser, [paths["raw_" + split] for split in required])
    return cfg


def load_config(config_path: str = "configs/base.yaml") -> Dict[str, Any]:
    """Load configuration dictionary with optional inheritance (_base_)."""
    p = Path(config_path)
    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    if "_base_" in cfg:
        base_path = p.parent.parent / cfg["_base_"] if not Path(cfg["_base_"]).is_absolute() else Path(cfg["_base_"])
        if not base_path.exists():
            base_path = Path(cfg["_base_"])
        base_cfg = load_config(str(base_path))
        
        # Deep merge
        merged = deep_merge(base_cfg, cfg)
        merged.pop("_base_", None)
        return merged
        
    return cfg


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result
