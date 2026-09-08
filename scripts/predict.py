"""
scripts/predict.py
==================
Inference CLI script for predicting on any custom raw dataset.
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sandisk_yield.config import load_config, resolve_data_paths, require_input_files
from sandisk_yield.inference.predict import run_inference
from sandisk_yield.inference.submission import export_submission
from sandisk_yield.data.loader import load_dataset
from sandisk_yield.logging_utils import setup_logger
from sandisk_yield.training.operating_points import (
    add_operating_arguments, tune_policy, apply_saved_policy)

logger = setup_logger("predict")


def main():
    parser = argparse.ArgumentParser(description="Run WaferFusion-Cascade inference on raw data")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config")
    parser.add_argument("--input", type=str, default=None, help="Input path (default: input/validation.csv)")
    parser.add_argument("--output", type=str, default="outputs/predictions/submission.csv", help="Submission output path")
    parser.add_argument("--mode", type=str, default="cascade", choices=["cascade", "full_model_b"])
    add_operating_arguments(parser)
    parser.add_argument("--from-probabilities", default=None,
                        help="Re-threshold saved predictions without model inference")
    parser.add_argument("--oof", default=None, help="OOF CSV for threshold tuning; never supply test predictions here")
    args = parser.parse_args()
    output_path = Path(args.output)
    probability_output = output_path.with_name(
        "prediction_probabilities.csv" if output_path.name == "submission.csv"
        else output_path.stem + "_probabilities.csv")

    cfg = load_config(args.config)
    resolve_data_paths(cfg, args, parser, required=())
    args.input = args.input or cfg["paths"]["raw_validation"]
    if not np.isfinite(args.fn_fp_cost_ratio) or args.fn_fp_cost_ratio <= 0:
        parser.error("--fn-fp-cost-ratio must be positive and finite")
    require_input_files(parser, [args.from_probabilities or args.input])
    models_dir = Path(cfg["paths"]["models_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])

    cascade = joblib.load(models_dir / "cascade_orchestrator.joblib")
    # Saved OOF scores use raw probability scales. Do not apply those thresholds
    # to older artifacts whose calibrators transform the probabilities.
    if any(getattr(c, "is_fitted", False) for c in (cascade.calibrator_a, cascade.calibrator_b)):
        parser.error("OOF threshold tuning requires artifacts with uncalibrated probability scales")
    oof_path = args.oof or str(Path(cfg["paths"]["predictions_dir"]) / "oof_predictions.csv")
    require_input_files(parser, [oof_path])
    policy = tune_policy(pd.read_csv(oof_path), args.threshold_objective, args.fn_fp_cost_ratio)
    b_name = f"Model B ({getattr(cascade.model_b, 'mode', 'cnn')})"
    cascade.threshold = policy["thresholds"]["Model A"]
    cascade.threshold_b = policy["thresholds"][b_name]
    if args.from_probabilities:
        ref_df = pd.read_csv(args.from_probabilities)
        try:
            results = apply_saved_policy(ref_df, policy, args.ensemble_mode, b_name)
        except ValueError as exc:
            parser.error(str(exc))
        export_submission(results, ref_df, submission_path=args.output,
                          probabilities_path=str(probability_output))
        logger.info("Exported selected OOF policy without inference or retraining: %s", args.output)
        return
    # Training now saves preprocessing beside the models; retain legacy lookup.
    pipeline_path = models_dir / "feature_pipeline.joblib"
    if not pipeline_path.exists():
        pipeline_path = artifacts_dir / "feature_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)

    logger.info(f"Running inference on {args.input} in mode '{args.mode}'...")
    results = run_inference(cascade, pipeline, args.input,
                            mode="full_model_b" if args.ensemble_mode != "none" else args.mode)
    results = apply_saved_policy(results, policy, args.ensemble_mode, b_name)
    ref_df = load_dataset(args.input, split_name="reference")

    export_submission(results, ref_df, submission_path=args.output,
                      probabilities_path=str(probability_output))
    logger.info(f"Inference complete! Submission saved to {args.output}")


if __name__ == "__main__":
    main()
