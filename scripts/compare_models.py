"""
scripts/compare_models.py

Rigorous benchmark of:
  1. Baseline Logistic Regression
  2. Model A - die-level + spatial features
  3. Model B - Model A + 2000 block readings
  4. WaferFusion Cascade

Evaluation protocol:
  TRAIN -> FIT + CALIBRATION
  TEST  -> untouched final evaluation

No test wafer is used for:
  - training
  - calibration
  - threshold selection
  - cascade calibration
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve

# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sandisk_yield.config import load_config, add_data_arguments, resolve_data_paths
from sandisk_yield.seed import seed_everything, get_device
from sandisk_yield.data.loader import load_dataset
from sandisk_yield.data.validator import validate_dataframe
from sandisk_yield.schema import (
    create_eligible_mask,
    create_new_failure_target,
)
from sandisk_yield.data.splitter import make_calibration_split
from sandisk_yield.features.pipeline import FeaturePipeline
from sandisk_yield.features.blocks import compute_block_features_df
from sandisk_yield.models.baseline import BaselineModel
from sandisk_yield.models.model_a import ModelA
from sandisk_yield.models.model_b import (
    ModelB,
    WaferBlockDataset,
)
from sandisk_yield.training.losses import FocalLoss

from sandisk_yield.models.calibration import Calibrator
from sandisk_yield.training.trainer_cascade import (
    build_and_calibrate_cascade,
)
from sandisk_yield.training.evaluation import (
    compute_die_yield_metrics,
    optimize_threshold,
)
from sandisk_yield.logging_utils import setup_logger


logger = setup_logger("compare_models")

def train_model_b(
    train_dataset,
    val_dataset,
    tabular_dim,
    block_length,
    embedding_dim,
    hidden_dim,
    batch_size,
    epochs,
    device,
):
    """Train Model B using the existing ModelB implementation."""

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = ModelB(
        tabular_dim=tabular_dim,
        block_length=block_length,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    # Keep the benchmark aligned with the project's imbalance-aware trainer.
    criterion = FocalLoss(alpha=0.35, gamma=2.0)

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for batch in train_loader:

            optimizer.zero_grad()

            if isinstance(batch, dict):
                    x_tab = batch["tab"].to(device)
                    x_block = batch["raw_block"].to(device)
                    x_block_features = batch["blk_feats"].to(device)
                    y = batch["y"].float().to(device)

                    output = model(
                    x_tab,
                    x_block,
                    x_block_features,
                )

            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            logits = logits.squeeze(-1)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += loss.item()

        logger.info(
            f"Model B epoch "
            f"{epoch + 1}/{epochs} "
            f"loss={total_loss / max(1, len(train_loader)):.4f}"
        )

    return model, None
def main():

    # ==============================================================
    # 0. ARGUMENTS
    # ==============================================================

    parser = argparse.ArgumentParser(
        description="Rigorous Model A vs Model B benchmark"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast development run",
    )

    add_data_arguments(parser)
    args = parser.parse_args()

    start_time = time.time()

    # ==============================================================
    # 1. CONFIG
    # ==============================================================

    cfg = load_config(args.config)
    resolve_data_paths(cfg, args, parser, required=("train", "test"))

    seed = cfg.get("seed", 42)

    seed_everything(seed)

    device = get_device(
        cfg.get("device", "auto")
    )

    logger.info(
        "=== Starting rigorous benchmark ==="
    )

    # ==============================================================
    # 2. LOAD TRAIN + FINAL TEST
    # ==============================================================

    train_path = Path(
        cfg["paths"]["raw_train"]
    )

    test_path = Path(
        cfg["paths"]["raw_test"]
    )

    if not train_path.exists():
        logger.error(
            f"Training data not found: {train_path}"
        )
        sys.exit(1)

    if not test_path.exists():
        logger.error(
            f"Test data not found: {test_path}"
        )
        sys.exit(1)

    train_df = load_dataset(
        train_path,
        split_name="train",
    )

    test_df = load_dataset(
        test_path,
        split_name="test",
    )

    validate_dataframe(
        train_df,
        is_training=True,
        expected_block_length=cfg["data"]["num_block_readings"],
    )

    validate_dataframe(
        test_df,
        is_training=True,
        expected_block_length=cfg["data"]["num_block_readings"],
    )

    logger.info(
        f"Train: {len(train_df):,} dies / "
        f"{train_df['wafer_id'].nunique()} wafers"
    )

    logger.info(
        f"Final test: {len(test_df):,} dies / "
        f"{test_df['wafer_id'].nunique()} wafers"
    )

    # ==============================================================
    # 3. TRAIN -> FIT + CALIBRATION
    #    TEST -> NEVER TOUCHED UNTIL FINAL EVALUATION
    # ==============================================================

    calib_ratio = (
        0.30
        if args.fast
        else cfg.get(
            "cv",
            {}
        ).get(
            "calibration_size",
            0.25,
        )
    )

    fit_df, calib_df = make_calibration_split(
        train_df,
        calibration_ratio=calib_ratio,
        group_col="wafer_id",
        seed=seed,
    )

    fit_wafers = set(
        fit_df["wafer_id"].unique()
    )

    calib_wafers = set(
        calib_df["wafer_id"].unique()
    )

    test_wafers = set(
        test_df["wafer_id"].unique()
    )

    assert fit_wafers.isdisjoint(
        calib_wafers
    ), "Fit/calibration wafer overlap!"

    assert fit_wafers.isdisjoint(
        test_wafers
    ), "Fit/test wafer overlap!"

    assert calib_wafers.isdisjoint(
        test_wafers
    ), "Calibration/test wafer overlap!"

    logger.info(
        "Wafer-group isolation: PASSED"
    )

    logger.info(
        f"Fit:         {fit_df['wafer_id'].nunique()} wafers"
    )

    logger.info(
        f"Calibration: {calib_df['wafer_id'].nunique()} wafers"
    )

    logger.info(
        f"FINAL TEST:  {test_df['wafer_id'].nunique()} wafers"
    )

    # ==============================================================
    # 4. TARGET / ELIGIBILITY
    # ==============================================================

    y_fit_all, fit_eligible = (
        create_new_failure_target(fit_df)
    )

    y_calib_all, calib_eligible = (
        create_new_failure_target(calib_df)
    )

    y_test_all, test_eligible = (
        create_new_failure_target(test_df)
    )

    fit_mask = fit_eligible.values
    calib_mask = calib_eligible.values
    test_mask = test_eligible.values

    # ==============================================================
    # 5. FEATURE PIPELINE
    # ==============================================================

    feat_cfg = cfg.get(
        "features",
        {}
    )

    pipeline = FeaturePipeline(
        scale_features=feat_cfg.get(
            "scale_features",
            False,
        ),
        top_k_parametric=feat_cfg.get(
            "top_k_parametric",
            30,
        ),
        spatial_windows=feat_cfg.get(
            "spatial_windows",
            [3, 5, 7],
        ),
        include_blocks=False,
        epsilon=float(
            feat_cfg.get(
                "epsilon",
                1e-6,
            )
        ),
    )

    # IMPORTANT:
    # Feature pipeline learns ONLY from FIT data.
    pipeline.fit(
        fit_df.loc[fit_mask],
        y=y_fit_all.loc[fit_mask],
    )

    X_fit = pipeline.transform(
        fit_df
    )

    X_calib = pipeline.transform(
        calib_df
    )

    X_test = pipeline.transform(
        test_df
    )

    # Eligible subsets
    X_fit_elig = X_fit.loc[fit_mask]
    X_calib_elig = X_calib.loc[calib_mask]
    X_test_elig = X_test.loc[test_mask]

    y_fit = y_fit_all.loc[fit_mask]
    y_calib = y_calib_all.loc[calib_mask]
    y_test = y_test_all.loc[test_mask]

    df_fit_elig = fit_df.loc[fit_mask]
    df_calib_elig = calib_df.loc[calib_mask]
    df_test_elig = test_df.loc[test_mask]

    y_calib_true = y_calib.values
    y_test_true = y_test.values

    n_calib = len(y_calib_true)
    n_calib_fails = int(
        y_calib_true.sum()
    )

    n_test = len(y_test_true)
    n_test_fails = int(
        y_test_true.sum()
    )

    logger.info(
        f"Calibration: {n_calib:,} eligible dies / "
        f"{n_calib_fails:,} failures"
    )

    logger.info(
        f"FINAL TEST: {n_test:,} eligible dies / "
        f"{n_test_fails:,} failures"
    )

    # ==============================================================
    # 6. BASELINE
    # ==============================================================

    logger.info(
        "Training Baseline Logistic Regression..."
    )

    t0 = time.time()

    baseline = BaselineModel(
        random_state=seed
    )

    baseline.fit(
        X_fit_elig.fillna(0.0),
        y_fit,
    )

    time_base_train = (
        time.time() - t0
    )

    # Threshold is learned ONLY on calibration.
    p_base_calib = baseline.predict_proba(
        X_calib_elig.fillna(0.0)
    )[:, 1]

    thresh_base, _ = optimize_threshold(
        y_calib_true,
        p_base_calib,
        metric="fail_f1",
    )

    # ==============================================================
    # 7. MODEL A
    # ==============================================================

    logger.info(
        "Training Model A..."
    )

    t0 = time.time()

    model_a_params = (
        cfg.get(
            "model_a",
            {}
        )
        .get(
            "params",
            {}
        )
        .copy()
    )

    if args.fast:
        model_a_params[
            "n_estimators"
        ] = 40

    model_a = ModelA(
        model_type=cfg.get(
            "model_a",
            {}
        ).get(
            "type",
            "lightgbm",
        ),
        params=model_a_params,
    )

    model_a.fit(
        X_fit_elig,
        y_fit,
    )

    time_a_train = (
        time.time() - t0
    )

    # --------------------------------------------------------------
    # Model A calibration
    # --------------------------------------------------------------

    raw_a_calib = model_a.predict_proba(
        X_calib_elig
    )[:, 1]

    calibrator_a = Calibrator(
        method=cfg.get(
            "calibration",
            {}
        ).get(
            "method",
            "isotonic",
        )
    )

    calibrator_a.fit(
        raw_a_calib,
        y_calib_true,
    )

    p_a_calib = calibrator_a.predict(
        raw_a_calib
    )

    thresh_a, _ = optimize_threshold(
        y_calib_true,
        p_a_calib,
        metric="fail_f1",
    )

    # ==============================================================
    # 8. MODEL B
    # ==============================================================

    logger.info(
        "Training Model B "
        "(CNN + Evidence Fusion)..."
    )

    b_cfg = cfg.get(
        "model_b",
        {}
    )

    block_length = b_cfg.get(
        "block_length",
        2000,
    )

    batch_size = b_cfg.get(
        "batch_size",
        64,
    )

    epochs = (
        3
        if args.fast
        else b_cfg.get(
            "epochs",
            15,
        )
    )

    # --------------------------------------------------------------
    # Block features for FIT
    # --------------------------------------------------------------

    t0 = time.time()

    block_fit = compute_block_features_df(
        df_fit_elig,
        expected_len=block_length,
    ).values

    train_b_dataset = WaferBlockDataset(
        X_tab=X_fit_elig.values,
        block_strings=df_fit_elig[
            "block_readings"
        ].tolist(),
        X_blk_feats=block_fit,
        y=y_fit.values,
        block_length=block_length,
    )

    model_b, _ = train_model_b(
        train_dataset=train_b_dataset,
        val_dataset=None,
        tabular_dim=X_fit_elig.shape[1],
        block_length=block_length,
        embedding_dim=b_cfg.get(
            "embedding_dim",
            64,
        ),
        hidden_dim=b_cfg.get(
            "hidden_dim",
            128,
        ),
        batch_size=batch_size,
        epochs=epochs,
        device=device,
    )

    time_b_train = (
        time.time() - t0
    )

    # --------------------------------------------------------------
    # Model B calibration
    # --------------------------------------------------------------

    block_calib = compute_block_features_df(
        df_calib_elig,
        expected_len=block_length,
    ).values

    raw_b_calib = model_b.predict_proba(
        X_calib_elig.values,
        df_calib_elig[
            "block_readings"
        ].tolist(),
        block_calib,
        batch_size=batch_size,
        device=device,
    )[:, 1]

    calibrator_b = Calibrator(
        method=cfg.get(
            "calibration",
            {}
        ).get(
            "method",
            "isotonic",
        )
    )

    calibrator_b.fit(
        raw_b_calib,
        y_calib_true,
    )

    p_b_calib = calibrator_b.predict(
        raw_b_calib
    )

    thresh_b, _ = optimize_threshold(
        y_calib_true,
        p_b_calib,
        metric="fail_f1",
    )

    # ==============================================================
    # 9. CASCADE
    # ==============================================================

    logger.info(
        "Building WaferFusion Cascade..."
    )

    cascade, _ = build_and_calibrate_cascade(
        df_raw=df_fit_elig,
        X_tab=X_fit_elig,
        y=y_fit,
        model_a=model_a,
        model_b=model_b,
        calibration_ratio=0.20,
        calibration_method=cfg.get(
            "calibration",
            {}
        ).get(
            "method",
            "isotonic",
        ),
        conformal_coverage=cfg.get(
            "conformal",
            {}
        ).get(
            "default_coverage",
            0.95,
        ),
        routing_config=cfg.get(
            "routing",
            {}
        ),
        seed=seed,
    )

    thresh_casc = cascade.threshold

    # ==============================================================
    # 10. FINAL TEST
    # ==============================================================

    logger.info(
        "================================================"
    )

    logger.info(
        "FINAL EVALUATION ON UNTOUCHED TEST WAFERS"
    )

    logger.info(
        "================================================"
    )

    # --------------------------------------------------------------
    # Baseline
    # --------------------------------------------------------------

    t0 = time.time()

    p_base = baseline.predict_proba(
        X_test_elig.fillna(0.0)
    )[:, 1]

    time_base_inf = (
        time.time() - t0
    )

    m_base = compute_die_yield_metrics(
        y_test_true,
        p_base,
        thresh_base,
    )

    # --------------------------------------------------------------
    # Model A
    # --------------------------------------------------------------

    t0 = time.time()

    raw_a_test = model_a.predict_proba(
        X_test_elig
    )[:, 1]

    p_a_test = calibrator_a.predict(
        raw_a_test
    )

    time_a_inf = (
        time.time() - t0
    )

    m_a = compute_die_yield_metrics(
        y_test_true,
        p_a_test,
        thresh_a,
    )

    # --------------------------------------------------------------
    # Model B
    # --------------------------------------------------------------

    t0 = time.time()

    block_test = compute_block_features_df(
        df_test_elig,
        expected_len=block_length,
    ).values

    raw_b_test = model_b.predict_proba(
        X_test_elig.values,
        df_test_elig[
            "block_readings"
        ].tolist(),
        block_test,
        batch_size=batch_size,
        device=device,
    )[:, 1]

    p_b_test = calibrator_b.predict(
        raw_b_test
    )

    time_b_inf = (
        time.time() - t0
    )

    m_b = compute_die_yield_metrics(
        y_test_true,
        p_b_test,
        thresh_b,
    )

    # --------------------------------------------------------------
    # Cascade
    # --------------------------------------------------------------

    t0 = time.time()

    cascade_result = cascade.predict_detailed(
        test_df,
        X_test,
        mode="cascade",
        batch_size=batch_size,
    )

    time_casc_inf = (
        time.time() - t0
    )

    p_casc_test = cascade_result.loc[
        test_mask,
        "final_probability",
    ].values

    routed_to_b = cascade_result.loc[
        test_mask,
        "routed_to_model_b",
    ].values

    b_usage_pct = float(
        routed_to_b.mean() * 100
    )

    m_casc = compute_die_yield_metrics(
        y_test_true,
        p_casc_test,
        thresh_casc,
    )

    # ==============================================================
    # 11. BRIER SCORES
    # ==============================================================

    brier_a = brier_score_loss(
        y_test_true,
        p_a_test,
    )

    brier_b = brier_score_loss(
        y_test_true,
        p_b_test,
    )

    brier_casc = brier_score_loss(
        y_test_true,
        p_casc_test,
    )

    # ==============================================================
    # 12. COMPARISON TABLE
    # ==============================================================

    comp_rows = []

    def add_model_row(
        name,
        metrics,
        threshold,
        inference_time,
        usage,
    ):
        comp_rows.append(
            {
                "model": name,
                "accuracy": round(
                    metrics["accuracy"],
                    4,
                ),
                "balanced_accuracy": round(
                    metrics["balanced_accuracy"],
                    4,
                ),
                "roc_auc": round(
                    metrics["roc_auc"],
                    4,
                ),
                "pr_auc": round(
                    metrics["pr_auc"],
                    4,
                ),
                "fail_precision": round(
                    metrics["fail_precision"],
                    4,
                ),
                "fail_recall": round(
                    metrics["fail_recall"],
                    4,
                ),
                "fail_f1": round(
                    metrics["fail_f1"],
                    4,
                ),
                "pass_precision": round(
                    metrics["pass_precision"],
                    4,
                ),
                "pass_recall": round(
                    metrics["pass_recall"],
                    4,
                ),
                "pass_f1": round(
                    metrics["pass_f1"],
                    4,
                ),
                "threshold": round(
                    threshold,
                    4,
                ),
                "inference_time": (
                    f"{inference_time:.3f}s"
                ),
                "model_b_usage": usage,
            }
        )

    add_model_row(
        "Baseline",
        m_base,
        thresh_base,
        time_base_inf,
        "0.0%",
    )

    add_model_row(
        "Model_A",
        m_a,
        thresh_a,
        time_a_inf,
        "0.0%",
    )

    add_model_row(
        "Model_B",
        m_b,
        thresh_b,
        time_b_inf,
        "100.0%",
    )

    add_model_row(
        "Cascade",
        m_casc,
        thresh_casc,
        time_casc_inf,
        f"{b_usage_pct:.1f}%",
    )

    comp_df = pd.DataFrame(
        comp_rows
    )

    metrics_dir = Path(
        cfg["paths"]["metrics_dir"]
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comp_df.to_csv(
        metrics_dir
        / "model_comparison.csv",
        index=False,
    )

    # ==============================================================
    # 13. MODEL B DELTAS
    # ==============================================================

    d_pr_auc = (
        m_b["pr_auc"]
        - m_a["pr_auc"]
    )

    d_f1 = (
        m_b["fail_f1"]
        - m_a["fail_f1"]
    )

    d_rec = (
        m_b["fail_recall"]
        - m_a["fail_recall"]
    )

    # ==============================================================
    # 14. PREDICTION ANALYSIS
    # ==============================================================

    pred_a = (
        p_a_test >= thresh_a
    ).astype(int)

    pred_b = (
        p_b_test >= thresh_b
    ).astype(int)

    pred_casc = (
        p_casc_test >= thresh_casc
    ).astype(int)

    # --------------------------------------------------------------
    # Block-rescued failures
    # --------------------------------------------------------------

    rescued = (
        (y_test_true == 1)
        & (pred_a == 0)
        & (pred_b == 1)
    )

    n_rescued = int(
        rescued.sum()
    )

    pct_failures_rescued = (
        n_rescued
        / max(1, n_test_fails)
        * 100
    )

    b_tp = int(
        (
            (y_test_true == 1)
            & (pred_b == 1)
        ).sum()
    )

    pct_b_tp_rescued = (
        n_rescued
        / max(1, b_tp)
        * 100
    )

    # --------------------------------------------------------------
    # Transition matrix
    # --------------------------------------------------------------

    pp_mask = (
        (pred_a == 0)
        & (pred_b == 0)
    )

    pf_mask = (
        (pred_a == 0)
        & (pred_b == 1)
    )

    fp_mask = (
        (pred_a == 1)
        & (pred_b == 0)
    )

    ff_mask = (
        (pred_a == 1)
        & (pred_b == 1)
    )

    t_pass_pass = int(pp_mask.sum())
    t_pass_fail = int(pf_mask.sum())
    t_fail_pass = int(fp_mask.sum())
    t_fail_fail = int(ff_mask.sum())

    def failure_rate(mask):
        if mask.sum() == 0:
            return 0.0
        return float(
            y_test_true[mask].mean()
            * 100
        )

    rate_pp = failure_rate(pp_mask)
    rate_pf = failure_rate(pf_mask)
    rate_fp = failure_rate(fp_mask)
    rate_ff = failure_rate(ff_mask)

    # --------------------------------------------------------------
    # False negatives
    # --------------------------------------------------------------

    fn_a = int(
        (
            (y_test_true == 1)
            & (pred_a == 0)
        ).sum()
    )

    fn_b = int(
        (
            (y_test_true == 1)
            & (pred_b == 0)
        ).sum()
    )

    fn_casc = int(
        (
            (y_test_true == 1)
            & (pred_casc == 0)
        ).sum()
    )

    cascade_missed = (
        (y_test_true == 1)
        & (pred_a == 0)
        & (~routed_to_b)
    )

    n_cascade_missed = int(
        cascade_missed.sum()
    )

    # ==============================================================
    # 15. FIGURES
    # ==============================================================

    fig_dir = Path(
        cfg["paths"]["figures_dir"]
    )

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Metric comparison
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 4.5),
        dpi=120,
    )

    names = [
        "Model A",
        "Model B",
        "Cascade",
    ]

    pr = [
        m_a["pr_auc"],
        m_b["pr_auc"],
        m_casc["pr_auc"],
    ]

    f1 = [
        m_a["fail_f1"],
        m_b["fail_f1"],
        m_casc["fail_f1"],
    ]

    x = np.arange(
        len(names)
    )

    width = 0.35

    ax.bar(
        x - width / 2,
        pr,
        width,
        label="PR-AUC",
    )

    ax.bar(
        x + width / 2,
        f1,
        width,
        label="Fail F1",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    ax.set_title(
        "Final Test Model Comparison"
    )

    ax.legend()
    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "model_metric_comparison_bar.png"
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # PR curve
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5),
        dpi=120,
    )

    for probs, label in [
        (
            p_a_test,
            f"Model A "
            f"(AUC={m_a['pr_auc']:.3f})",
        ),
        (
            p_b_test,
            f"Model B "
            f"(AUC={m_b['pr_auc']:.3f})",
        ),
        (
            p_casc_test,
            f"Cascade "
            f"(AUC={m_casc['pr_auc']:.3f})",
        ),
    ]:

        precision, recall, _ = (
            precision_recall_curve(
                y_test_true,
                probs,
            )
        )

        ax.plot(
            recall,
            precision,
            lw=2,
            label=label,
        )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Final Test Precision-Recall Curves"
    )

    ax.legend(
        loc="lower left"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "pr_curves_comparison.png"
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # ROC curve
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5),
        dpi=120,
    )

    for probs, label in [
        (
            p_a_test,
            f"Model A "
            f"(AUC={m_a['roc_auc']:.3f})",
        ),
        (
            p_b_test,
            f"Model B "
            f"(AUC={m_b['roc_auc']:.3f})",
        ),
        (
            p_casc_test,
            f"Cascade "
            f"(AUC={m_casc['roc_auc']:.3f})",
        ),
    ]:

        fpr, tpr, _ = roc_curve(
            y_test_true,
            probs,
        )

        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=label,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Chance",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Final Test ROC Curves"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "roc_curves_comparison.png"
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Threshold curve
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 4.5),
        dpi=120,
    )

    thresholds = np.linspace(
        0.05,
        0.95,
        50,
    )

    f1_a_curve = [
        compute_die_yield_metrics(
            y_test_true,
            p_a_test,
            t,
        )["fail_f1"]
        for t in thresholds
    ]

    f1_b_curve = [
        compute_die_yield_metrics(
            y_test_true,
            p_b_test,
            t,
        )["fail_f1"]
        for t in thresholds
    ]

    ax.plot(
        thresholds,
        f1_a_curve,
        lw=2,
        label=(
            f"Model A "
            f"(calibration threshold={thresh_a:.2f})"
        ),
    )

    ax.plot(
        thresholds,
        f1_b_curve,
        lw=2,
        label=(
            f"Model B "
            f"(calibration threshold={thresh_b:.2f})"
        ),
    )

    ax.axvline(
        thresh_a,
        linestyle="--",
        alpha=0.6,
    )

    ax.axvline(
        thresh_b,
        linestyle="--",
        alpha=0.6,
    )

    ax.set_xlabel(
        "Decision Threshold"
    )

    ax.set_ylabel(
        "Fail F1"
    )

    ax.set_title(
        "Test F1 vs Threshold"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "threshold_vs_f1_comparison.png"
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5),
        dpi=120,
    )

    true_a, pred_a_cal = calibration_curve(
        y_test_true,
        p_a_test,
        n_bins=8,
    )

    true_b, pred_b_cal = calibration_curve(
        y_test_true,
        p_b_test,
        n_bins=8,
    )

    ax.plot(
        pred_a_cal,
        true_a,
        "s-",
        label=(
            f"Model A "
            f"(Brier={brier_a:.4f})"
        ),
    )

    ax.plot(
        pred_b_cal,
        true_b,
        "o-",
        label=(
            f"Model B "
            f"(Brier={brier_b:.4f})"
        ),
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Perfect",
    )

    ax.set_xlabel(
        "Mean Predicted Probability"
    )

    ax.set_ylabel(
        "Observed Failure Fraction"
    )

    ax.set_title(
        "Final Test Probability Calibration"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "calibration_comparison.png"
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        dpi=120,
    )

    cm_a = confusion_matrix(
        y_test_true,
        pred_a,
    )

    cm_b = confusion_matrix(
        y_test_true,
        pred_b,
    )

    axes[0].imshow(
        cm_a,
        cmap="Blues",
    )

    axes[0].set_title(
        f"Model A "
        f"(threshold={thresh_a:.2f})"
    )

    axes[1].imshow(
        cm_b,
        cmap="Greens",
    )

    axes[1].set_title(
        f"Model B "
        f"(threshold={thresh_b:.2f})"
    )

    for ax, cm in [
        (axes[0], cm_a),
        (axes[1], cm_b),
    ]:

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        ax.set_xticks(
            [0, 1],
            ["Pass", "Fail"],
        )

        ax.set_yticks(
            [0, 1],
            ["Pass", "Fail"],
        )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "confusion_matrices_comparison.png"
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Transition matrix
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 5),
        dpi=120,
    )

    transition = np.array(
        [
            [
                t_pass_pass,
                t_pass_fail,
            ],
            [
                t_fail_pass,
                t_fail_fail,
            ],
        ]
    )

    ax.imshow(
        transition,
        cmap="YlOrRd",
    )

    rates = [
        rate_pp,
        rate_pf,
        rate_fp,
        rate_ff,
    ]

    for i in range(2):
        for j in range(2):

            idx = i * 2 + j

            ax.text(
                j,
                i,
                (
                    f"Count: "
                    f"{transition[i, j]:,}\n"
                    f"Failure: "
                    f"{rates[idx]:.1f}%"
                ),
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax.set_title(
        "Model A → Model B Decision Matrix"
    )

    ax.set_xlabel(
        "Model B"
    )

    ax.set_ylabel(
        "Model A"
    )

    ax.set_xticks(
        [0, 1],
        ["Pass", "Fail"],
    )

    ax.set_yticks(
        [0, 1],
        ["Pass", "Fail"],
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "transition_disagreement_matrix.png"
    )

    plt.close(fig)

    # --------------------------------------------------------------
    # Cascade routing
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(6, 4),
        dpi=120,
    )

    ax.bar(
        [
            "Model A\nOnly",
            "Model B\nInspection",
        ],
        [
            100.0 - b_usage_pct,
            b_usage_pct,
        ],
    )

    ax.set_ylabel(
        "Percentage of eligible test dies"
    )

    ax.set_title(
        "Cascade Routing Distribution"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        fig_dir
        / "cascade_routing_distribution.png"
    )

    plt.close(fig)

    # ==============================================================
    # 16. FINAL REPORT
    # ==============================================================

    reports_dir = Path(
        cfg["paths"]["reports_dir"]
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Best standalone model by PR-AUC.
    if m_b["pr_auc"] >= m_a["pr_auc"]:
        best_predictive = "Model B"
    else:
        best_predictive = "Model A"

    if d_pr_auc > 0:
        block_prauc_answer = "Yes"
    else:
        block_prauc_answer = "No"

    if d_f1 > 0:
        block_f1_answer = "Yes"
    else:
        block_f1_answer = "No"

    if d_rec > 0:
        block_recall_answer = "Yes"
    else:
        block_recall_answer = "No"

    block_answer = (
        "Yes"
        if (
            d_pr_auc > 0
            and d_f1 > 0
        )
        else "No clear improvement"
    )

    cascade_retains = (
        m_casc["pr_auc"]
        >= m_b["pr_auc"] - 0.01
        and
        m_casc["fail_f1"]
        >= m_b["fail_f1"] - 0.01
    )

    cascade_answer = (
        "Yes"
        if cascade_retains
        else "Not demonstrated"
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    # Build report using a list rather than one huge f-string.
    # This avoids accidental unterminated-string errors.
    # --------------------------------------------------------------

    report_lines = [

        "# Model Comparison & Architecture Benchmark Report",
        "",
        "**Evaluation Protocol:** Wafer-Grouped Train / Calibration / Final Test",
        "",
        f"**Training Scope:** "
        f"{fit_df['wafer_id'].nunique()} fit wafers",
        "",
        f"**Calibration Scope:** "
        f"{calib_df['wafer_id'].nunique()} calibration wafers",
        "",
        f"**Final Test Scope:** "
        f"{test_df['wafer_id'].nunique()} completely unseen wafers",
        "",
        f"**Final Test Holdout:** "
        f"{n_test:,} eligible dies, "
        f"{n_test_fails:,} actual new failures",
        "",
        "**Strict Leakage Isolation:** "
        "Models are trained only on fit wafers. "
        "Calibration and threshold selection use only calibration wafers. "
        "Final metrics are computed only on untouched test wafers. "
        "Evaluation is restricted to `old_label == 0`.",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Model | PR-AUC | Fail F1 | Fail Recall | Model B Usage |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Baseline | "
            f"{m_base['pr_auc']:.4f} | "
            f"{m_base['fail_f1']:.4f} | "
            f"{m_base['fail_recall']:.4f} | "
            f"0.0% |"
        ),
        (
            f"| Model A | "
            f"{m_a['pr_auc']:.4f} | "
            f"{m_a['fail_f1']:.4f} | "
            f"{m_a['fail_recall']:.4f} | "
            f"0.0% |"
        ),
        (
            f"| Model B | "
            f"{m_b['pr_auc']:.4f} | "
            f"{m_b['fail_f1']:.4f} | "
            f"{m_b['fail_recall']:.4f} | "
            f"100.0% |"
        ),
        (
            f"| Cascade | "
            f"{m_casc['pr_auc']:.4f} | "
            f"{m_casc['fail_f1']:.4f} | "
            f"{m_casc['fail_recall']:.4f} | "
            f"{b_usage_pct:.1f}% |"
        ),
        "",
        f"- **Best Standalone Predictive Model:** {best_predictive}",
        f"- **Model A → Model B PR-AUC Change:** {d_pr_auc:+.4f}",
        f"- **Model A → Model B Fail-F1 Change:** {d_f1:+.4f}",
        f"- **Model A → Model B Fail-Recall Change:** {d_rec:+.4f}",
        f"- **Block-Level Improvement Demonstrated:** {block_answer}",
        f"- **Cascade Retains Model B Advantage:** {cascade_answer}",
        "",
        "---",
        "",
        "## 2. Comprehensive Comparison",
        "",
        "```text",
        comp_df.to_string(index=False),
        "```",
        "",
        "---",
        "",
        "## 3. Block-Level Evidence",
        "",
        f"- Model B PR-AUC improvement: **{d_pr_auc:+.4f}**",
        f"- Model B Fail-F1 improvement: **{d_f1:+.4f}**",
        f"- Model B Fail-Recall improvement: **{d_rec:+.4f}**",
        f"- PR-AUC improved: **{block_prauc_answer}**",
        f"- Fail-F1 improved: **{block_f1_answer}**",
        f"- Fail-Recall improved: **{block_recall_answer}**",
        "",
        f"- Actual failures rescued by Model B over Model A: "
        f"**{n_rescued:,}**",
        f"- Percentage of actual failures rescued: "
        f"**{pct_failures_rescued:.2f}%**",
        f"- Percentage of Model B true positives represented by rescues: "
        f"**{pct_b_tp_rescued:.2f}%**",
        "",
        "---",
        "",
        "## 4. False-Negative Safety Analysis",
        "",
        f"| Model | False Negatives |",
        f"|---|---:|",
        f"| Model A | {fn_a:,} |",
        f"| Model B | {fn_b:,} |",
        f"| Cascade | {fn_casc:,} |",
        "",
        f"**Cascade failures missed without routing:** "
        f"{n_cascade_missed:,}",
        "",
        "---",
        "",
        "## 5. Cascade Routing",
        "",
        f"- Model B routing rate: **{b_usage_pct:.2f}%**",
        f"- Model A-only rate: "
        f"**{100.0 - b_usage_pct:.2f}%**",
        "",
        "---",
        "",
        "## 6. Brier Scores",
        "",
        f"- Model A: **{brier_a:.4f}**",
        f"- Model B: **{brier_b:.4f}**",
        f"- Cascade: **{brier_casc:.4f}**",
        "",
        "---",
        "",
        "## 7. Training / Inference Time",
        "",
        f"- Baseline training: **{time_base_train:.3f}s**",
        f"- Model A training: **{time_a_train:.3f}s**",
        f"- Model B training: **{time_b_train:.3f}s**",
        f"- Baseline inference: **{time_base_inf:.3f}s**",
        f"- Model A inference: **{time_a_inf:.3f}s**",
        f"- Model B inference: **{time_b_inf:.3f}s**",
        f"- Cascade inference: **{time_casc_inf:.3f}s**",
        "",
        "---",
        "",
        "## 8. Decision",
        "",
        (
            "The benchmark uses a completely unseen wafer-level "
            "test set. Thresholds and calibration parameters were "
            "determined before final testing."
        ),
        "",
        (
            "Block-level improvement is considered demonstrated "
            "only when Model B improves both PR-AUC and Fail-F1 "
            "over Model A."
        ),
        "",
        f"**Final block-reading conclusion:** {block_answer}",
        "",
        f"**Total benchmark runtime:** "
        f"{time.time() - start_time:.2f}s",
    ]

    report_md = "\n".join(
        report_lines
    )

    report_path = (
        reports_dir
        / "model_comparison_report.md"
    )

    report_path.write_text(
        report_md,
        encoding="utf-8",
    )

    # ==============================================================
    # 17. SAVE FINAL PREDICTIONS
    # ==============================================================

    predictions_dir = Path(
        cfg["paths"]["predictions_dir"]
    )

    predictions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_df = df_test_elig[
        [
            "wafer_id",
            "die_row",
            "die_col",
            "old_label",
            "label",
        ]
    ].copy()

    prediction_df["baseline_probability"] = p_base
    prediction_df["model_a_probability"] = p_a_test
    prediction_df["model_b_probability"] = p_b_test
    prediction_df["cascade_probability"] = p_casc_test

    prediction_df["baseline_prediction"] = (
        pred_a * 0
        + (
            p_base >= thresh_base
        ).astype(int)
    )

    prediction_df["model_a_prediction"] = pred_a
    prediction_df["model_b_prediction"] = pred_b
    prediction_df["cascade_prediction"] = pred_casc

    prediction_df["routed_to_model_b"] = (
        routed_to_b
    )

    prediction_df.to_csv(
        predictions_dir
        / "final_test_predictions.csv",
        index=False,
    )

    # ==============================================================
    # 18. FINAL CONSOLE SUMMARY
    # ==============================================================

    logger.info(
        "================================================"
    )

    logger.info(
        "BENCHMARK COMPLETE"
    )

    logger.info(
        "================================================"
    )

    logger.info(
        f"Final test dies: {n_test:,}"
    )

    logger.info(
        f"Final test failures: {n_test_fails:,}"
    )

    logger.info(
        f"Model A PR-AUC: {m_a['pr_auc']:.4f}"
    )

    logger.info(
        f"Model B PR-AUC: {m_b['pr_auc']:.4f}"
    )

    logger.info(
        f"Model A → B PR-AUC: {d_pr_auc:+.4f}"
    )

    logger.info(
        f"Model A Fail-F1: {m_a['fail_f1']:.4f}"
    )

    logger.info(
        f"Model B Fail-F1: {m_b['fail_f1']:.4f}"
    )

    logger.info(
        f"Model A → B Fail-F1: {d_f1:+.4f}"
    )

    logger.info(
        f"Block improvement: {block_answer}"
    )

    logger.info(
        f"Cascade Model B usage: "
        f"{b_usage_pct:.1f}%"
    )

    logger.info(
        f"Report saved to: {report_path}"
    )

    logger.info(
        f"Predictions saved to: "
        f"{predictions_dir / 'final_test_predictions.csv'}"
    )

    logger.info(
        f"Total runtime: "
        f"{time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()
