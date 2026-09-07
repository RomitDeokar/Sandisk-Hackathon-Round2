"""Imbalance and class-distribution overlap reporting from frozen artifacts."""
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve


MODEL_COLUMNS = {
    "Model A": "Model A probability",
    "Model B (cnn)": "Model B (cnn) probability",
}


def histogram_overlap(pass_values, fail_values, bins=30):
    """Histogram intersection in [0, 1], using common quantile-robust edges."""
    pass_values = np.asarray(pass_values, dtype=float)
    fail_values = np.asarray(fail_values, dtype=float)
    pass_values = pass_values[np.isfinite(pass_values)]
    fail_values = fail_values[np.isfinite(fail_values)]
    if not len(pass_values) or not len(fail_values):
        return np.nan
    combined = np.concatenate([pass_values, fail_values])
    lo, hi = np.quantile(combined, [0.001, 0.999])
    if not np.isfinite(lo + hi) or hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(np.clip(pass_values, lo, hi), edges)
    f, _ = np.histogram(np.clip(fail_values, lo, hi), edges)
    p = p / max(1, p.sum())
    f = f / max(1, f.sum())
    return float(np.minimum(p, f).sum())


def generate_imbalance_analysis(oof, features, top_features: Sequence[str], output_dir):
    """Generate plots, tables and narrative strictly from saved OOF scores/features."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(oof) != len(features) or not oof.index.equals(features.index):
        raise ValueError("OOF rows and feature rows are not aligned")
    y = oof["label"].to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("OOF labels must contain both classes")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    model_stats = []
    calibration_rows = []
    for name, column in MODEL_COLUMNS.items():
        probabilities = oof[column].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Nonfinite OOF scores for {name}")
        precision, recall, _ = precision_recall_curve(y, probabilities)
        ap = average_precision_score(y, probabilities)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
        observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
        quantile_bins = pd.Series(pd.qcut(probabilities, q=10, duplicates="drop"))
        counts = quantile_bins.value_counts(sort=False).to_numpy()
        for bin_number, (pred, obs, count) in enumerate(zip(predicted, observed, counts), 1):
            calibration_rows.append({"model": name, "bin": bin_number,
                                     "mean_predicted_probability": pred,
                                     "observed_failure_rate": obs, "count": int(count)})
        ece = float(sum(r["count"] * abs(r["observed_failure_rate"] - r["mean_predicted_probability"])
                        for r in calibration_rows if r["model"] == name) / len(y))
        model_stats.append({"model": name, "average_precision": ap,
                            "brier_score": brier_score_loss(y, probabilities),
                            "expected_calibration_error": ece})
    ax.axhline(y.mean(), color="grey", linestyle="--", label=f"Failure prevalence ({y.mean():.1%})")
    ax.set(xlabel="Recall", ylabel="Precision", title="OOF precision–recall: imbalanced eligible dies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png")
    plt.close(fig)

    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, group in calibration.groupby("model", sort=False):
        ax.plot(group.mean_predicted_probability, group.observed_failure_rate, marker="o", label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability", labelpad=8)
    ax.set_ylabel("Observed failure frequency")
    ax.set_title("OOF reliability curves (10 equal-frequency bins)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_reliability_curves.png")
    plt.close(fig)

    overlap_rows = []
    for feature in top_features:
        values = features[feature].to_numpy(dtype=float)
        overlap_rows.append({
            "feature": feature,
            "histogram_overlap": histogram_overlap(values[y == 0], values[y == 1]),
            "pass_mean": float(np.nanmean(values[y == 0])),
            "fail_mean": float(np.nanmean(values[y == 1])),
            "pass_median": float(np.nanmedian(values[y == 0])),
            "fail_median": float(np.nanmedian(values[y == 1])),
        })
    overlap = pd.DataFrame(overlap_rows).sort_values("histogram_overlap")
    overlap.to_csv(output_dir / "feature_distribution_overlap.csv", index=False)
    stats = pd.DataFrame(model_stats)
    stats.to_csv(output_dir / "model_imbalance_metrics.csv", index=False)

    strongest = overlap.iloc[0]
    weakest = overlap.iloc[-1]
    a = stats.set_index("model").loc["Model A"]
    b = stats.set_index("model").loc["Model B (cnn)"]
    report = f"""# How the current models handle imbalanced, overlapping distributions

This report uses the current five-fold wafer-grouped OOF predictions for
{len(y):,} eligible training dies: {int(y.sum()):,} failures and
{int((y == 0).sum()):,} passes ({y.mean():.2%} failure prevalence). It does not
use in-sample predictions, retrain either model, or change the Cost 8:1 policy.

## Precision–recall behavior

Model A achieves pooled OOF average precision {a.average_precision:.4f}; Model B (cnn)
achieves {b.average_precision:.4f}. PR curves are the appropriate ranking view
under this class imbalance because they expose the precision cost of recovering
more of the rare failures. The CNN's higher average precision shows that adding
block-level evidence improves ranking of failures, although the classes remain
substantially overlapping.

These pooled values differ from the reported mean of five fold-level PR-AUCs:
pooling ranks scores across folds, whereas the CV table calculates each fold's
PR-AUC first and then averages the five results.

## Probability reliability

Model A has Brier score {a.brier_score:.4f} and 10-bin calibration error
{a.expected_calibration_error:.4f}. Model B (cnn) has Brier score
{b.brier_score:.4f} and calibration error {b.expected_calibration_error:.4f}.
The saved scores are uncalibrated; deviations from the diagonal in the reliability
plot mean score magnitudes should not be interpreted as literal failure rates.
Cost-weighted thresholding changes decisions, not calibration.

## Feature-distribution overlap

Histogram intersection ranges from 0 (separated) to 1 (indistinguishable).
Among the current Model A top-20 features, `{strongest.feature}` separates the
classes most strongly (overlap {strongest.histogram_overlap:.3f}), while
`{weakest.feature}` overlaps most (overlap {weakest.histogram_overlap:.3f}).
No single feature resolves the task; Model A combines weak and spatial signals.

## Direct answer to the brief

Model A addresses imbalance through balanced LightGBM class weights and combines
many overlapping parametric/spatial features. Model B uses focal loss to emphasize
hard examples and adds raw-block CNN attention plus block summaries. Its higher
OOF average precision indicates better rare-failure ranking, while the reliability
curves show neither model should be treated as calibrated. The final Cost 8:1
operating point explicitly trades additional false alarms for fewer false negatives;
it does not remove the underlying distribution overlap.
"""
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    return stats, calibration, overlap
