"""
scripts/build_features.py
=========================
Builds and persists transformed feature matrices for fast iterative model training.
"""

import argparse
import sys
from pathlib import Path
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sandisk_yield.config import load_config, add_data_arguments, resolve_data_paths
from sandisk_yield.seed import seed_everything
from sandisk_yield.data.loader import load_dataset
from sandisk_yield.schema import create_eligible_mask, create_new_failure_target
from sandisk_yield.features.pipeline import FeaturePipeline
from sandisk_yield.logging_utils import setup_logger

logger = setup_logger("build_features")


def main():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config")
    add_data_arguments(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolve_data_paths(cfg, args, parser, required=("train",))
    seed_everything(cfg.get("seed", 42))

    raw_train_path = Path(cfg["paths"]["raw_train"])
    if not raw_train_path.exists():
        logger.error(f"Train data not found at {raw_train_path}")
        return

    train_df = load_dataset(raw_train_path, split_name="train")
    y_train, eligible_mask = create_new_failure_target(train_df)

    # Initialize feature pipeline
    feat_cfg = cfg.get("features", {})
    pipeline = FeaturePipeline(
        scale_features=feat_cfg.get("scale_features", False),
        top_k_parametric=feat_cfg.get("top_k_parametric", 30),
        spatial_windows=feat_cfg.get("spatial_windows", [3, 5, 7]),
        include_blocks=False,  # Tabular pipeline for Model A
        epsilon=float(feat_cfg.get("epsilon", 1e-6))
    )

    logger.info("Fitting FeaturePipeline on training data...")
    # Fit only on eligible dies to prevent old failures from biasing local feature selection
    pipeline.fit(train_df.loc[eligible_mask], y=y_train.loc[eligible_mask])

    logger.info("Transforming full training dataset...")
    X_tab_train = pipeline.transform(train_df)

    # Persist artifacts
    feat_dir = Path(cfg["paths"]["features_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    
    art_dir = Path(cfg["paths"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, art_dir / "feature_pipeline.joblib")
    X_tab_train.to_parquet(feat_dir / "train_tabular_features.parquet")
    logger.info(f"Features built successfully! ({X_tab_train.shape[1]} tabular columns saved to {feat_dir})")


if __name__ == "__main__":
    main()
