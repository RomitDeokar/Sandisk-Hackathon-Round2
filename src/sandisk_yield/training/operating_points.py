"""Tune decisions on saved OOF scores only; no estimator fitting or test tuning."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sandisk_yield.training.evaluation import find_best_threshold

A = "Model A"
B = "Model B (cnn)"
AVERAGE = "Ensemble (average)"
UNION = "Ensemble (union)"
KEYS = ["wafer_id", "die_row", "die_col"]


def add_operating_arguments(parser):
    parser.add_argument("--threshold-objective", choices=["f1", "cost_weighted"], default="f1")
    parser.add_argument("--fn-fp-cost-ratio", type=float, default=8.0)
    parser.add_argument("--ensemble-mode", choices=["none", "union", "average"], default="none")


def decision_metrics(y, predicted, ratio=8.0):
    y, predicted = np.asarray(y), np.asarray(predicted)
    if y.ndim != 1 or y.shape != predicted.shape or not len(y):
        raise ValueError("Expected aligned nonempty labels and decisions")
    if not np.isin(y, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels and decisions must be binary")
    tp = int(((y == 1) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, recall=tp / max(1, tp + fn),
                precision=tp / max(1, tp + fp), f1=2 * tp / max(1, 2 * tp + fp + fn),
                accuracy=(tp + tn) / len(y), weighted_cost=fn * ratio + fp)


def validate_oof(oof):
    required = KEYS + ["old_label", "label", "fold"]
    if not set(required).issubset(oof.columns):
        raise ValueError("OOF file must contain die IDs, old_label, label and fold")
    if oof.empty or oof[required].isna().any().any() or oof.duplicated(KEYS).any():
        raise ValueError("OOF file is empty, incomplete or has duplicate die IDs")
    if not (oof.old_label == 0).all() or not oof.label.isin([0, 1]).all():
        raise ValueError("OOF tuning requires only eligible dies with binary targets")
    if oof.groupby("wafer_id").fold.nunique().max() != 1:
        raise ValueError("A wafer appears in multiple validation folds")


def tune_policy(oof, objective="f1", ratio=8.0):
    validate_oof(oof)
    thresholds = {}
    for column in oof.columns:
        if column.endswith(" probability"):
            thresholds[column[:-12]] = find_best_threshold(oof.label, oof[column], objective, ratio)
    if A not in thresholds:
        raise ValueError("Model A OOF probabilities are missing")
    if B in thresholds:
        p = (oof[A + " probability"].to_numpy() + oof[B + " probability"].to_numpy()) / 2
        thresholds[AVERAGE] = find_best_threshold(oof.label, p, objective, ratio)
    return dict(objective=objective, fn_fp_cost_ratio=float(ratio), thresholds=thresholds)


def policy_decisions(probabilities, policy):
    """Return decisions and ranking scores. Union has no probability/AP score."""
    thresholds = policy["thresholds"]
    decisions = {name: np.asarray(p) >= thresholds[name] for name, p in probabilities.items()}
    scores = dict(probabilities)
    if A in probabilities and B in probabilities:
        scores[AVERAGE] = (np.asarray(probabilities[A]) + np.asarray(probabilities[B])) / 2
        decisions[AVERAGE] = scores[AVERAGE] >= thresholds[AVERAGE]
        decisions[UNION] = decisions[A] | decisions[B]
    return decisions, scores


def analyze_oof(oof, ratio=8.0):
    """Both objectives, full confusion counts, fold variability and fail overlap."""
    records, overlaps, policies = [], [], {}
    probabilities = {c[:-12]: oof[c].to_numpy() for c in oof if c.endswith(" probability")}
    for objective in ("f1", "cost_weighted"):
        policy = tune_policy(oof, objective, ratio)
        policies[objective] = policy
        decisions, scores = policy_decisions(probabilities, policy)
        for name, predicted in decisions.items():
            record = dict(model=name, objective=objective, fn_fp_cost_ratio=ratio,
                          threshold=policy["thresholds"].get(name, np.nan),
                          threshold_a=policy["thresholds"][A],
                          threshold_b=policy["thresholds"].get(B, np.nan),
                          **decision_metrics(oof.label, predicted, ratio))
            fold_records = []
            for fold in oof.fold.unique():
                mask = (oof.fold == fold).to_numpy()
                metrics = decision_metrics(oof.label.to_numpy()[mask], predicted[mask], ratio)
                if name in scores:
                    metrics["pr_auc"] = average_precision_score(oof.label.to_numpy()[mask], scores[name][mask])
                fold_records.append(metrics)
            for metric in ("recall", "precision", "f1", "accuracy", "pr_auc"):
                values = [m.get(metric, np.nan) for m in fold_records]
                record[metric + "_mean"] = float(np.mean(values))
                record[metric + "_std"] = float(np.std(values))
            records.append(record)
        if B in decisions:
            fails = oof.label.to_numpy() == 1
            a, b = decisions[A], decisions[B]
            overlaps.append(dict(objective=objective, fn_fp_cost_ratio=ratio,
                fails_caught_by_both=int((fails & a & b).sum()),
                fails_only_a=int((fails & a & ~b).sum()),
                fails_only_b=int((fails & ~a & b).sum()),
                fails_caught_by_neither=int((fails & ~a & ~b).sum())))
    return pd.DataFrame(records), pd.DataFrame(overlaps), policies


def comparison_table(records):
    rows = []
    for _, record in records.iterrows():
        row = {k: record[k] for k in ("model", "objective", "fn_fp_cost_ratio", "threshold", "TP", "FN", "FP", "TN")}
        for metric in ("pr_auc", "recall", "precision", "f1", "accuracy"):
            mean, std = record[metric + "_mean"], record[metric + "_std"]
            row[metric] = "N/A (decision union)" if pd.isna(mean) else f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def save_operating_reports(oof, directory, ratio=8.0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records, overlaps, policies = analyze_oof(oof, ratio)
    records.to_csv(directory / "threshold_tradeoffs.csv", index=False)
    # Wide form places the F1 and cost objectives literally side by side.
    records.pivot(index="model", columns="objective",
                  values=["threshold", "TP", "FN", "FP", "TN", "recall", "precision", "f1", "weighted_cost"]).to_csv(
                      directory / "threshold_tradeoffs_side_by_side.csv")
    overlaps.to_csv(directory / "model_overlap_analysis.csv", index=False)
    comparison_table(records).to_csv(directory / "operating_point_comparison.csv", index=False)
    return records, overlaps, policies


def apply_saved_policy(predictions, policy, ensemble_mode="none", model_b_name=B):
    """Re-threshold saved raw-scale scores; preserve routing for mode none.

    Complete B scores are mandatory for ensembles. Never substitute A for a
    missing B prediction or treat the routed B subset as a full-model evaluation.
    """
    if ensemble_mode not in ("none", "union", "average"):
        raise ValueError("Unknown ensemble mode")
    if predictions.duplicated(KEYS).any():
        raise ValueError("Duplicate prediction die IDs")
    if predictions[KEYS + ["old_label"]].isna().any().any() or not predictions.old_label.isin([0, 1]).all():
        raise ValueError("Incomplete prediction IDs or invalid old_label")
    out = predictions.copy()
    mask = (out.old_label == 0).to_numpy()
    a = out.loc[mask, "model_a_probability"].to_numpy(dtype=float)
    b = out.loc[mask, "model_b_probability"].to_numpy(dtype=float)
    if np.isinf(b).any():
        raise ValueError("Infinite Model B probabilities")
    thresholds = policy["thresholds"]
    for p in (a, b[np.isfinite(b)]):
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid probability values")
    if model_b_name not in thresholds:
        raise ValueError("OOF policy does not contain the selected Model B")
    if ensemble_mode != "none":
        if model_b_name != B or not np.isfinite(b).all():
            raise ValueError("CNN ensembles require CNN probabilities for EVERY eligible die; generate full_model_b predictions first")
    if ensemble_mode == "union":
        labels = (a >= thresholds[A]) | (b >= thresholds[B])
        score = np.maximum(a, b)  # Display only; not a calibrated union probability.
        decision_threshold = np.full(len(a), np.nan)
    elif ensemble_mode == "average":
        score = (a + b) / 2
        decision_threshold = np.full(len(a), thresholds[AVERAGE])
        labels = score >= decision_threshold
    else:
        has_b = np.isfinite(b)
        score = np.where(has_b, b, a)
        decision_threshold = np.where(has_b, thresholds[model_b_name], thresholds[A])
        labels = score >= decision_threshold
    out.loc[mask, "predicted_label"] = labels.astype(int)
    out.loc[~mask, "predicted_label"] = 1
    out.loc[mask, "final_probability"] = score
    out.loc[mask, "decision_threshold"] = decision_threshold
    out["threshold_objective"] = policy["objective"]
    out["fn_fp_cost_ratio"] = policy["fn_fp_cost_ratio"]
    out["ensemble_mode"] = ensemble_mode
    out["threshold_a"] = thresholds[A]
    out["threshold_b"] = thresholds[model_b_name]
    return out


def analyze_saved_test(oof, predictions, labels, ratios=(4.0, 8.0, 12.0)):
    """Evaluate OOF-chosen policies without fitting or tuning on test labels.

    Existing cascade routing is frozen. Standalone/ensemble comparisons explicitly
    use only rows with both model scores; coverage columns expose any exclusions.
    Inputs must refer to the same inference measurements and raw score scale.
    """
    if set(oof.wafer_id).intersection(labels.wafer_id):
        raise ValueError("Test wafers overlap OOF training wafers")
    joined = labels[KEYS + ["old_label", "label"]].merge(
        predictions.drop(columns="label", errors="ignore"), on=KEYS, how="outer", validate="one_to_one",
        suffixes=("_truth", ""), indicator=True)
    if not (joined["_merge"] == "both").all():
        raise ValueError("Predictions do not cover exactly the labeled test dies")
    if not (joined.old_label_truth == joined.old_label).all():
        raise ValueError("Test and prediction eligibility differ")
    eligible = joined.loc[joined.old_label == 0].copy()
    complete = eligible.model_b_probability.notna()
    both = eligible.loc[complete]
    rows = []
    for ratio in ratios:
        for objective in ("f1", "cost_weighted"):
            policy = tune_policy(oof, objective, ratio)
            result = apply_saved_policy(eligible, policy)
            common = dict(objective=objective, fn_fp_cost_ratio=ratio,
                          threshold_a=policy["thresholds"][A], threshold_b=policy["thresholds"][B])
            rows.append(dict(model="Cascade (frozen saved routing)", scope="all eligible test dies",
                evaluated_rows=len(eligible), excluded_rows=0, **common,
                **decision_metrics(eligible.label, result.predicted_label, ratio)))
            probabilities = {A: both.model_a_probability.to_numpy(), B: both.model_b_probability.to_numpy()}
            decisions, _ = policy_decisions(probabilities, policy)
            for name, predicted in decisions.items():
                rows.append(dict(model=name, scope="eligible test dies with both saved scores",
                    evaluated_rows=len(both), excluded_rows=int((~complete).sum()),
                    threshold=policy["thresholds"].get(name, np.nan), **common,
                    **decision_metrics(both.label, predicted, ratio)))
    return pd.DataFrame(rows)
