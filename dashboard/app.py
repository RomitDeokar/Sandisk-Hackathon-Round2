"""
dashboard/app.py
================
Streamlit dashboard for interactive WaferFusion-Cascade analytics and explainability.
Loads persisted artifacts and predictions — NO retraining.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sandisk_yield.training.operating_points import analyze_oof

st.set_page_config(page_title="WaferFusion-Cascade Dashboard", layout="wide")

st.title("🛡️ WaferFusion-Cascade: SanDisk Die Yield Analytics")

# This panel works from OOF artifacts alone, even before a submission exists.
oof_path = Path("outputs/predictions/oof_predictions.csv")
if oof_path.exists():
    @st.cache_data
    def load_oof(path, modified_time):
        return pd.read_csv(path)

    st.subheader("Failure cost and recall trade-off — OOF preview")
    ratio = st.slider("Cost of one missed failure / one false alarm", 1.0, 50.0, 8.0, 0.5)
    oof_preview = load_oof(str(oof_path), oof_path.stat().st_mtime_ns)
    operating, overlap, _ = analyze_oof(oof_preview, ratio)
    st.dataframe(operating[["model", "objective", "threshold", "TP", "FN", "FP", "TN",
                            "recall", "precision", "f1", "weighted_cost"]], use_container_width=True)
    preview_models = operating.model.unique().tolist()
    default_model = preview_models.index("Model B (cnn)") if "Model B (cnn)" in preview_models else 0
    selected_preview = st.selectbox("Confusion matrix model", preview_models, index=default_model)
    selected_objective = st.radio(
        "Threshold objective", ["f1", "cost_weighted"], index=1, horizontal=True)
    point = operating[(operating.model == selected_preview) & (operating.objective == selected_objective)].iloc[0]
    st.dataframe(pd.DataFrame([[int(point.TN), int(point.FP)], [int(point.FN), int(point.TP)]],
        index=["Actual pass", "Actual fail"], columns=["Predicted pass", "Predicted fail"]))
    st.caption("OOF tuning preview only: no retraining, test-label tuning, or submission changes. "
               "Cost units assume FP=1. The 8:1 ratio is a user-selected operating assumption, not a calibrated package-risk model. "
               "Union/average require both models on every eligible die and do not retain cascade compute savings.")

# Load predictions
pred_path = Path("outputs/predictions/prediction_probabilities.csv")
comp_path = Path("outputs/metrics/operating_point_comparison.csv")

if not pred_path.exists():
    st.warning("No predictions found. Please run the ML pipeline first: `python scripts/run_all.py`")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv(pred_path)

df = load_data()
# The exported probability artifact is the source of truth for the submission's
# operating policy. This prevents a training-time thresholds file from drifting
# away from a later, safely re-thresholded submission.
policy_columns = {"threshold_objective", "fn_fp_cost_ratio", "ensemble_mode",
                  "threshold_a", "threshold_b"}
if policy_columns.issubset(df.columns):
    policies = df[list(sorted(policy_columns))].drop_duplicates()
    if len(policies) != 1:
        st.error("Saved predictions contain multiple operating policies.")
        st.stop()
    selected_policy = policies.iloc[0]
    st.caption("Submission operating policy selected from wafer-grouped OOF predictions")
    st.json({
        "threshold_objective": selected_policy["threshold_objective"],
        "fn_fp_cost_ratio": float(selected_policy["fn_fp_cost_ratio"]),
        "ensemble_mode": selected_policy["ensemble_mode"],
        "Model A": float(selected_policy["threshold_a"]),
        "Model B (cnn)": float(selected_policy["threshold_b"]),
    })
else:
    st.warning("Saved predictions do not contain operating-policy metadata.")
# Predictions already contain the appropriate A/B OOF-tuned decision. Never
# retune on dashboard labels or replace persisted decisions with a fixed cutoff.
st.caption("Risk scores are uncalibrated; OOF cascade routing has no split-conformal coverage guarantee.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "🗺️ Wafer Explorer",
    "🔍 Die Risk Card",
    "⚡ Cascade & Compute Savings",
    "📈 Benchmark Comparison",
    "🧭 Interpretability",
    "⚖️ Imbalance Analysis",
])

with tab1:
    st.header("Pipeline Health & Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    total_dies = len(df)
    eligible_dies = int((df["old_label"] == 0).sum())
    old_fails = int((df["old_label"] == 1).sum())
    pred_new_fails = int(((df["old_label"] == 0) & (df["predicted_label"] == 1)).sum())
    b_inspections = int(df["routed_to_model_b"].sum()) if "routed_to_model_b" in df.columns else 0
    
    c1.metric("Total Dies", f"{total_dies:,}")
    c2.metric("Eligible Dies (old_label=0)", f"{eligible_dies:,}")
    c3.metric("Pre-Test Fails (old_label=1)", f"{old_fails:,}")
    c4.metric("Predicted New Fails", f"{pred_new_fails:,}")
    c5.metric("Model B Inspections", f"{b_inspections:,} ({b_inspections/max(1, eligible_dies)*100:.1f}%)")

with tab2:
    st.header("Wafer Spatial Risk Explorer")
    wafers = df["wafer_id"].unique().tolist()
    sel_wafer = st.selectbox("Select Wafer", wafers)
    
    w_df = df[df["wafer_id"] == sel_wafer]
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sc = ax.scatter(
        w_df["die_col"], w_df["die_row"],
        c=w_df["final_probability"], cmap="RdYlGn_r", vmin=0, vmax=1,
        s=35, edgecolors="black", linewidths=0.2
    )
    ax.invert_yaxis()
    ax.set_title(f"Wafer Risk: {sel_wafer}")
    plt.colorbar(sc, ax=ax, label="Failure Score")
    st.pyplot(fig)

with tab3:
    st.header("Die Risk Card & Multi-Resolution Evidence")
    sel_row = st.selectbox("Select Die Index", range(min(50, len(df))))
    die = df.iloc[sel_row]
    
    st.json({
        "Wafer": die["wafer_id"],
        "Position": f"Row {die['die_row']}, Col {die['die_col']}",
        "Pre-test Status": "Pass" if die["old_label"] == 0 else "Fail",
        "Failure Score": f"{die['final_probability']:.4f}",
        "Decision Threshold": die.get("decision_threshold", "See saved decisions"),
        "Uncertainty State": die.get("uncertainty_state", "N/A"),
        "Inspected by Model B": bool(die.get("routed_to_model_b", False)),
        "Final Decision": "FAIL" if die["predicted_label"] == 1 else "PASS"
    })

with tab4:
    st.header("Cascade Routing & Compute Reduction")
    st.write(
        "By using Conformal Gate uncertainty and decision boundary safety margin, "
        "only ambiguous dies undergo 2000-block deep inspection."
    )
    if "routed_to_model_b" in df.columns:
        counts = df[df["old_label"] == 0]["routed_to_model_b"].value_counts()
        # Bug: boolean counts aligned to string index labels produced empty bars.
        st.bar_chart(pd.DataFrame({"Die Count": [counts.get(False, 0), counts.get(True, 0)]},
                                 index=["Model A Only (Fast)", "Model B (Deep)"]))

with tab5:
    st.header("Model Benchmark Comparison")
    st.caption("Cost-weighted 8:1 operating points from five-fold OOF predictions; descriptive tuning metrics, not nested CV.")
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        comp_df = comp_df[(comp_df["objective"] == "cost_weighted") &
                          (comp_df["fn_fp_cost_ratio"] == 8.0)].copy()
        st.dataframe(comp_df, use_container_width=True)
    else:
        st.info("Run evaluation script to populate comparison table.")

with tab6:
    st.header("Current-model Interpretability")
    shap_path = Path("outputs/reports/model_a_per_die_shap.csv")
    spatial_path = Path("outputs/reports/model_a_spatial_contributions.csv")
    block_path = Path("outputs/reports/model_b_block_pattern_analysis_current.csv")
    if all(path.exists() for path in (shap_path, spatial_path, block_path)):
        shap_df = pd.read_csv(shap_path)
        spatial_df = pd.read_csv(spatial_path)
        block_df = pd.read_csv(block_path)
        explanation_ids = shap_df.apply(
            lambda row: f"{row.wafer_id} / r{int(row.die_row)} / c{int(row.die_col)}", axis=1)
        selected_die = st.selectbox("Select explained die", explanation_ids, key="interpret_die")
        selected_index = explanation_ids[explanation_ids == selected_die].index[0]
        shap_row = shap_df.loc[selected_index]
        key = (shap_row.wafer_id, int(shap_row.die_row), int(shap_row.die_col))
        feature_values = shap_row.drop(labels=["wafer_id", "die_row", "die_col"]).astype(float)
        top_values = feature_values.loc[feature_values.abs().sort_values(ascending=False).head(10).index]
        st.subheader("Model A: largest per-die TreeSHAP contributions")
        st.caption("Positive log-odds contributions push toward failure; negative contributions push toward pass.")
        st.bar_chart(top_values.rename("TreeSHAP contribution"))

        spatial_die = spatial_df[(spatial_df.wafer_id == key[0]) &
                                 (spatial_df.die_row == key[1]) &
                                 (spatial_df.die_col == key[2])].iloc[0]
        st.subheader("Model A: spatial contribution")
        st.metric("Sum of spatial-feature SHAP values", f"{spatial_die.spatial_total_shap:.4f}")
        spatial_columns = [column for column in spatial_df.columns if column.startswith("spatial_")
                           and column != "spatial_total_shap"]
        st.dataframe(pd.DataFrame({"feature": spatial_columns,
                                  "SHAP contribution": spatial_die[spatial_columns].astype(float).values})
                     .sort_values("SHAP contribution", key=abs, ascending=False),
                     use_container_width=True)
        wafer_spatial = spatial_df[spatial_df.wafer_id == key[0]]
        color_limit = float(np.quantile(np.abs(wafer_spatial.spatial_total_shap), 0.98)) or 1.0
        spatial_fig, spatial_ax = plt.subplots(figsize=(6, 5), dpi=120)
        spatial_points = spatial_ax.scatter(
            wafer_spatial.die_col, wafer_spatial.die_row,
            c=wafer_spatial.spatial_total_shap, cmap="coolwarm",
            vmin=-color_limit, vmax=color_limit, s=36,
            edgecolors="black", linewidths=0.2)
        spatial_ax.scatter([key[2]], [key[1]], marker="*", s=180,
                           facecolors="none", edgecolors="black", linewidths=1.5)
        spatial_ax.invert_yaxis()
        spatial_ax.set(title=f"Spatial SHAP contribution — {key[0]}",
                       xlabel="Die column", ylabel="Die row")
        spatial_fig.colorbar(spatial_points, ax=spatial_ax,
                             label="Sum of spatial-feature SHAP values (log-odds)")
        spatial_fig.tight_layout()
        st.pyplot(spatial_fig)

        prediction = df[(df.wafer_id == key[0]) & (df.die_row == key[1]) & (df.die_col == key[2])].iloc[0]
        st.subheader("Model B (cnn): learned block attention")
        if bool(prediction.routed_to_model_b):
            regions = block_df[(block_df.wafer_id == key[0]) &
                               (block_df.die_row == key[1]) &
                               (block_df.die_col == key[2])].copy()
            regions["block_region"] = regions.apply(
                lambda row: f"{int(row.block_start)}–{int(row.block_end)}", axis=1)
            st.bar_chart(regions.set_index("block_region")["attention_weight"])
            st.dataframe(regions[["attention_rank", "block_start", "block_end",
                                  "attention_weight", "mean_signal", "max_signal"]],
                         use_container_width=True)
            st.caption("Attention identifies regions emphasized by the encoder; it is not a causal attribution.")
        else:
            st.info("This die was resolved by Model A and was not routed to Model B in the saved cascade.")
    else:
        st.info("Run `python scripts/generate_analysis_deliverables.py` to create current-model explanations.")

with tab7:
    st.header("Imbalanced and Overlapping Distributions")
    imbalance_dir = Path("outputs/reports/imbalance_analysis")
    summary_path = imbalance_dir / "summary.md"
    pr_path = imbalance_dir / "precision_recall_curves.png"
    calibration_path = imbalance_dir / "calibration_reliability_curves.png"
    overlap_path = imbalance_dir / "feature_distribution_overlap.csv"
    if all(path.exists() for path in (summary_path, pr_path, calibration_path, overlap_path)):
        st.markdown(summary_path.read_text(encoding="utf-8"))
        left, right = st.columns(2)
        left.image(str(pr_path), caption="OOF precision–recall curves")
        right.image(str(calibration_path), caption="OOF reliability curves")
        st.subheader("Top-20 Model A feature distribution overlap")
        st.caption("0 means separated class histograms; 1 means complete overlap.")
        st.dataframe(pd.read_csv(overlap_path), use_container_width=True)
    else:
        st.info("Run `python scripts/generate_analysis_deliverables.py` to create the analysis.")
