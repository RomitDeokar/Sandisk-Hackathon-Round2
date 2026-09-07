"""Generate candidate operating-point reports from saved scores; never train/export."""
import argparse
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sandisk_yield.training.operating_points import analyze_oof, analyze_saved_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default="outputs/predictions/oof_predictions.csv")
    parser.add_argument("--probabilities", default="outputs/predictions/prediction_probabilities.csv")
    parser.add_argument("--test-labels", default=None, help="Optional labeled test file matching the saved predictions")
    parser.add_argument("--ratios", nargs="+", type=float, default=[4, 8, 12])
    parser.add_argument("--output-dir", default="outputs/metrics")
    args = parser.parse_args()
    oof = pd.read_csv(args.oof)
    records, overlaps = [], []
    for ratio in args.ratios:
        report, overlap, _ = analyze_oof(oof, ratio)
        records.append(report)
        overlaps.append(overlap)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(records, ignore_index=True).to_csv(directory / "threshold_cost_candidates_oof.csv", index=False)
    pd.concat(overlaps, ignore_index=True).to_csv(directory / "model_overlap_candidates.csv", index=False)
    if args.test_labels:
        # Read only IDs/labels, not the large block-reading columns. Delimiter
        # detection matches the loader; no inference or model fitting occurs.
        import csv
        with open(args.test_labels, encoding="utf-8-sig") as handle:
            separator = csv.Sniffer().sniff(handle.readline(), delimiters=",\t;").delimiter
        labels = pd.read_csv(args.test_labels, sep=separator,
                             usecols=["wafer_id", "die_row", "die_col", "old_label", "label"])
        report = analyze_saved_test(oof, pd.read_csv(args.probabilities), labels, args.ratios)
        report.to_csv(directory / "threshold_cost_candidates_test.csv", index=False)
        print(report[report.model == "Cascade (frozen saved routing)"].to_string(index=False))
    print("Only metrics reports written; models and submission files unchanged.")


if __name__ == "__main__":
    main()
