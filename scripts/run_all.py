"""Five-fold wafer CV, OOF threshold selection, final refits and prediction export.

Code-review assumptions: pre-test parametric/block readings are available for
complete inference wafers; feature_* columns must not encode post-test outcomes.
"""
import argparse
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sandisk_yield.config import load_config, add_data_arguments, resolve_data_paths
from sandisk_yield.seed import seed_everything, get_device
from sandisk_yield.data.loader import load_dataset
from sandisk_yield.data.validator import validate_dataframe
from sandisk_yield.data.alignment import assert_feature_alignment, assert_row_alignment
from sandisk_yield.schema import create_new_failure_target
from sandisk_yield.training.cross_validation import train_models_cv
from sandisk_yield.training.evaluation import compare_models
from sandisk_yield.training.operating_points import (
    add_operating_arguments, save_operating_reports, comparison_table, apply_saved_policy)
from sandisk_yield.models.calibration import Calibrator
from sandisk_yield.cascade.conformal_gate import ConformalGate
from sandisk_yield.cascade.router import CascadeRouter
from sandisk_yield.cascade.cascade import WaferFusionCascade
from sandisk_yield.inference.submission import export_submission
from sandisk_yield.explainability.tree_explain import get_feature_importances
from sandisk_yield.risk.wafer_risk import compute_wafer_risk_summary
from sandisk_yield.logging_utils import setup_logger

logger = setup_logger("run_all")


def main():
    parser = argparse.ArgumentParser(description="Wafer-grouped CV and block-mode comparison")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--fast", action="store_true", help="Reduce CNN epochs; retain all five folds")
    parser.add_argument("--block-mode", choices=["cnn", "zonal", "global"], default="cnn")
    parser.add_argument("--compare-block-modes", action="store_true",
                        help="Evaluate all three modes on identical folds; deploy --block-mode")
    parser.add_argument("--n-estimators", type=int, default=100)
    add_operating_arguments(parser)
    parser.add_argument("--postprocess-only", action="store_true",
                        help="Only analyze saved OOF probabilities; no training or submission changes")
    add_data_arguments(parser)
    args = parser.parse_args()
    if args.n_estimators < 1:
        parser.error("--n-estimators must be positive")
    if not np.isfinite(args.fn_fp_cost_ratio) or args.fn_fp_cost_ratio <= 0:
        parser.error("--fn-fp-cost-ratio must be positive and finite")
    if args.ensemble_mode != "none" and args.block_mode != "cnn":
        parser.error("Ensembling requires --block-mode cnn")
    cfg = load_config(args.config)
    if args.postprocess_only:
        oof_path = Path(cfg["paths"]["predictions_dir"]) / "oof_predictions.csv"
        if not oof_path.is_file():
            parser.error(f"Saved OOF probabilities not found: {oof_path}")
        records, _, _ = save_operating_reports(pd.read_csv(oof_path), cfg["paths"]["metrics_dir"], args.fn_fp_cost_ratio)
        logger.info("OOF operating-point comparison (no retraining):\n%s", comparison_table(records).to_string(index=False))
        logger.info("Submission and model artifacts unchanged. Objective selection uses OOF labels, not test labels.")
        return
    resolve_data_paths(cfg, args, parser)
    seed = cfg.get("seed", 42)
    seed_everything(seed)
    train_df = load_dataset(Path(cfg["paths"]["raw_train"]), split_name="train")
    block_length = cfg.get("model_b", {}).get("block_length", cfg["data"]["num_block_readings"])
    validate_dataframe(train_df, is_training=True, expected_block_length=block_length)
    y_all, eligible = create_new_failure_target(train_df)
    assert_row_alignment(train_df, y_all, eligible)
    logger.info("Training wafers=%s; eligible dies=%s; eligible failures=%s",
                train_df.wafer_id.nunique(), eligible.sum(), y_all.loc[eligible].sum())
    feat_cfg = cfg.get("features", {})
    feature_params = dict(
        scale_features=feat_cfg.get("scale_features", False),
        top_k_parametric=feat_cfg.get("top_k_parametric", 30),
        spatial_windows=feat_cfg.get("spatial_windows", [3, 5, 7]),
        include_blocks=False, epsilon=float(feat_cfg.get("epsilon", 1e-6)))
    model_params = dict(cfg.get("model_a", {}).get("params", {}))
    # CLI override is last: YAML and defaults cannot silently replace it.
    model_params["n_estimators"] = args.n_estimators
    model_params.setdefault("random_state", seed)
    b_cfg = cfg.get("model_b", {})
    training_params = dict(
        epochs=3 if args.fast else b_cfg.get("epochs", 15),
        batch_size=b_cfg.get("batch_size", 64),
        embedding_dim=b_cfg.get("embedding_dim", 8),
        hidden_dim=b_cfg.get("hidden_dim", 32),
        dropout=b_cfg.get("dropout", 0.4),
        use_attention=b_cfg.get("use_attention", True),
        lr=float(b_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(b_cfg.get("weight_decay", 1e-4)),
        device=get_device(cfg.get("device", "auto")))
    modes = ("cnn", "zonal", "global") if args.compare_block_modes else (args.block_mode,)
    pipeline, models, oof, summaries, folds = train_models_cv(
        train_df, modes=modes, feature_params=feature_params, model_params=model_params,
        training_params=training_params, block_length=block_length, n_splits=5, seed=seed,
        model_type=cfg.get("model_a", {}).get("type", "lightgbm"))
    oof_frame = train_df.loc[eligible, ["wafer_id", "die_row", "die_col", "old_label", "label"]].copy()
    fold_ids = np.zeros(len(oof_frame), dtype=int)
    for number, (_, val) in enumerate(folds, 1):
        fold_ids[val] = number
    oof_frame["fold"] = fold_ids
    for name, probabilities in oof.items():
        oof_frame[name + " probability"] = probabilities
    records, _, policies = save_operating_reports(oof_frame, cfg["paths"]["metrics_dir"], args.fn_fp_cost_ratio)
    policy = policies[args.threshold_objective]
    comparison = comparison_table(records)
    logger.info("\nFive-fold CV (mean ± population std):\n%s", comparison.to_string(index=False))
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    predictions_dir = Path(cfg["paths"]["predictions_dir"])
    for directory in (metrics_dir, models_dir, reports_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    with (metrics_dir / "cv_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    thresholds = policy["thresholds"]
    with (models_dir / "thresholds.json").open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
    for name, probabilities in oof.items():
        oof_frame[name + " probability"] = probabilities
        oof_frame[name + " prediction"] = (probabilities >= thresholds[name]).astype(int)
    oof_frame.to_csv(predictions_dir / "oof_predictions.csv", index=False)

    selected = f"Model B ({args.block_mode})"
    # Bug: previous calibration split was taken AFTER both models had seen every
    # training wafer. OOF scores avoid in-sample calibration inputs. An unfitted
    # Calibrator is intentionally identity, preserving the OOF threshold scale.
    # Pooled OOF gate scores are heuristic, not independent split-conformal scores.
    gate = ConformalGate(default_coverage=cfg.get("conformal", {}).get("default_coverage", 0.95))
    gate.calibrate(oof["Model A"], y_all.loc[eligible].to_numpy())
    route_cfg = cfg.get("routing", {})
    router = CascadeRouter(**{key: route_cfg[key] for key in (
        "prob_margin", "min_model_b_prob", "max_model_b_prob",
        "spatial_anomaly_threshold", "force_model_b_on_edge") if key in route_cfg})
    cascade = WaferFusionCascade(
        models["Model A"], Calibrator(), gate, router, models[selected], None,
        thresholds["Model A"], threshold_b=thresholds[selected])
    models["Model A"].save(models_dir / "model_a.joblib")
    for model in models.values():
        if getattr(model, "mode", None) == "cnn":
            # CPU tensors make the persisted adapter loadable without a GPU.
            model.model.cpu()
    # The adapter persists neural input scaling and mode-specific feature schema.
    joblib.dump(models[selected], models_dir / "model_b.joblib")
    joblib.dump(pipeline, models_dir / "feature_pipeline.joblib")
    joblib.dump(cascade, models_dir / "cascade_orchestrator.joblib")
    for name, model in models.items():
        if name.startswith("Model B"):
            joblib.dump(model, models_dir / f"model_b_{model.mode}.joblib")
    get_feature_importances(models["Model A"], top_n=30).to_csv(
        reports_dir / "model_a_feature_importance.csv", index=False)

    # No training-set fallback masquerading as an evaluation benchmark.
    prediction_path = Path(cfg["paths"]["raw_validation"])
    if not prediction_path.exists():
        prediction_path = Path(cfg["paths"]["raw_test"])
    if prediction_path.exists():
        target = load_dataset(prediction_path, split_name="prediction")
        validate_dataframe(target, is_training=False, expected_block_length=block_length)
        X_target = pipeline.transform(target)
        assert_feature_alignment(models["Model A"].feature_names_, X_target.columns)
        assert_feature_alignment(models[selected].feature_names_, X_target.columns)
        assert_row_alignment(target, X_target)
        results = cascade.predict_detailed(target, X_target,
            mode="full_model_b" if args.ensemble_mode != "none" else "cascade",
            batch_size=training_params["batch_size"])
        results = apply_saved_policy(results, policy, args.ensemble_mode, selected)
        export_submission(results, target, predictions_dir / "submission.csv",
                          predictions_dir / "prediction_probabilities.csv")
        compute_wafer_risk_summary(results).to_csv(reports_dir / "wafer_risk_summary.csv", index=False)
    else:
        logger.info("No prediction dataset present; saved CV results and deployable artifacts.")

    report = [
        "# Wafer-grouped cross-validation",
        comparison.to_string(index=False),
        "",
        "All five folds hold out whole wafers. Every eligible die has one OOF probability.",
        "Feature selection, imputation and neural scaling are fitted only on training folds.",
        "Fold metrics use the pooled OOF-tuned threshold. They are descriptive tuning results,",
        "not nested-CV estimates or evidence that predictive performance improved.",
        "Standard deviations use ddof=0; the folds are not independent confidence intervals.",
        "CNN uses neural fusion; zonal/global use LightGBM plus summaries, so the learner also changes.",
        f"Configured tree learner: {cfg.get('model_a', {}).get('type', 'lightgbm')} (see logs for effective backend).",
        "Zonal compression assumes stable, meaningful ordering of the block sequence.",
        "The deployment cascade uses model-specific OOF thresholds and an OOF routing heuristic.",
        "No independent split-conformal coverage guarantee or probability calibration is claimed.",
        f"Selected deployment mode: {args.block_mode}.",
        f"Threshold objective: {args.threshold_objective}; FN/FP cost ratio: {args.fn_fp_cost_ratio}; ensemble: {args.ensemble_mode}.",
        "Union runs both models on every eligible die; it has no single probability threshold or PR-AUC.",
        "New neural architecture requires retraining; old neural checkpoints are incompatible.",
    ]
    (reports_dir / "final_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
