# Smart-ATE: Adaptive Yield Gating & Fab Economics Simulator
### (with Package Risk Propagation) — SanDisk Hackathon Project Specification

> **Purpose of this document:** This is the single source of truth for the project. It is written to be uploaded as context to an LLM (Claude/GPT/Gemini) so that the assistant immediately understands the idea, constraints, architecture, and division of labor, and can help write code, debug, or generate content consistent with this plan. Do not deviate from the architecture described here without updating this doc.

---

## 1. THE IDEA (One-Paragraph Pitch)

Smart-ATE is a two-tier predictive pipeline for semiconductor wafer testing. **Model A** is a fast, cheap classifier that looks at die-level parametric tests + spatial wafer context and instantly clears or scraps the "easy" dies (the ones it's confident about). Only the **statistically ambiguous, borderline dies** get escalated to **Model B**, a heavier model that also uses expensive high-dimensional block-level readings. This cascade is wrapped in **conformal prediction** (99% confidence) so the "ambiguous" cutoff is statistically principled, not a guessed threshold. On top of this, we add **Package Risk Propagation**: since SanDisk sells multi-chip packages (stacked NAND), one bad die can force scrapping an entire assembled package — so we simulate package groupings and show that catching a risky die *before* packaging saves far more money than catching it after. The result is presented as a live Streamlit dashboard showing ATE tester-time saved (%), packaging scrap cost prevented ($), and package-level cost avoided ($), plus full SHAP-based explainability per die.

**Why this wins:** It solves a real fab economics problem (not just "detect defects"), it's technically rigorous (GroupKFold, conformal prediction, TreeSHAP), it's laptop-buildable in 72 hours with 100% free/open-source tools, and it has a strong visual 60-second expo hook (a slider that moves and dollar figures that update live).

---

## 2. HACKATHON CONSTRAINTS (DO NOT VIOLATE)

- **72-hour build window.** No architecture requiring heavy compute, long training, or extensive hyperparameter search.
- **Strictly tabular data.** Die-level parametric tests, m×m spatial coordinates, dense k block-level arrays. NOT images. NO computer vision models (no CNNs on images).
- **Must build TWO models:**
  - **Model A:** die-level + spatial context only.
  - **Model B:** Model A's inputs PLUS high-dimensional block-level readings.
- **Must directly compare Model A vs Model B** to quantify how much the block-level signal improves detection.
- **Class imbalance:** ~96%+ healthy dies, rare defect class. Must keep high overall accuracy AND catch the minority class AND not blow up false positives. (Use PR-AUC / F1 / recall-on-minority, not raw accuracy, as the real metric.)
- **Explicit XAI required:** per-die feature importance, spatial contribution visualization, block-reading pattern analysis.
- **Must be SanDisk-fab-relevant:** frame everything around real business value (scrap cost, tester time, multi-chip package failure prevention) — not abstract ML metrics only.
- **Low frontend, high visual hook:** minimize UI engineering time; maximize the "stop and look" factor for a 60-second expo walk-by. One clean dashboard page is enough.
- **Must be free to build.** Laptop-only, no paid compute, no cloud bill, no API keys.

---

## 3. ARCHITECTURE OVERVIEW

```
Raw tabular data (die-level tests, m×m spatial coords, k block-level readings)
        │
        ▼
GroupKFold split on wafer_id (prevents spatial leakage)
        │
        ▼
┌─────────────────────────┐
│   MODEL A (the Gate)    │  LightGBM on die-level features + engineered
│                          │  spatial features (local failure density,
│                          │  radial distance from wafer center, neighbor
│                          │  defect rate)
└─────────────────────────┘
        │
        ▼
Wrap Model A in CONFORMAL PREDICTION (MAPIE, 99% confidence)
        │
        ├── Single-label prediction (confidently Pass or confidently Fail)
        │        → SHIP or SCRAP immediately (no Model B needed → saves tester time)
        │
        └── Ambiguous / multi-label prediction (uncertain)
                 → escalate to Model B
                        │
                        ▼
        ┌─────────────────────────────────┐
        │   MODEL B (the Resolver)        │  LightGBM on [Model A features +
        │                                  │  compressed block-level stats:
        │                                  │  variance, IQR, peak-to-peak,
        │                                  │  skew of the k block readings]
        └─────────────────────────────────┘
                        │
                        ▼
                Final Pass/Fail decision on borderline dies
        │
        ▼
PACKAGE RISK PROPAGATION LAYER
  - Simulate/assume package groupings of dies (e.g., N dies per stack)
  - Package survival probability = f(individual die risk scores in that group)
  - If any die in a package exceeds risk threshold → package flagged at-risk
  - Cost model: die-scrap-cost vs. package-scrap-cost (package >> die)
        │
        ▼
XAI LAYER
  - TreeSHAP on Model A and Model B
  - Waterfall plots (per-die feature importance)
  - Wafer spatial heatmap (risk by die position)
  - Block-reading pattern plots for flagged dies
  - Probability-distribution overlap plot: Model A vs Model B
        │
        ▼
STREAMLIT DASHBOARD (the expo hook)
  - Interactive wafer map (color-coded: healthy / review / high-risk)
  - Model A vs Model B comparison table (PR-AUC, F1, recall, precision)
  - Dynamic threshold slider → live-updates:
       • "ATE Tester Time Saved (%)"
       • "Packaging Scrap Cost Prevented ($)"
       • "Package-Level Cost Avoided ($)"  ← the differentiator
  - Click-a-die → SHAP waterfall + block pattern for that die
```

---

## 4. STEP-BY-STEP EXECUTION PLAN

### Day 1 — Data & Model A
1. Load and explore the dataset. Confirm columns: die-level parametric features, x/y spatial coordinates, wafer_id, block-level readings (k-dimensional), and the true label (pass/fail).
2. Check class imbalance ratio. Confirm ~96%+ healthy.
3. **GroupKFold split on `wafer_id`** — this is mandatory, not optional. Adjacent dies on the same wafer are spatially correlated; random row-splitting leaks information and will produce fake high accuracy.
4. Engineer spatial features from the m×m context:
   - Radial distance from wafer center.
   - Local neighborhood failure density (e.g., failure rate of k-nearest neighboring dies).
   - Edge-vs-center indicator (edge dies fail more often in real fabs — worth checking/using).
5. Train **Model A**: LightGBM classifier on die-level parametric features + engineered spatial features.
6. Evaluate with PR-AUC, F1, recall/precision on the minority (defect) class. Do NOT report plain accuracy as your headline metric.

### Day 2 — Model B, Conformal Gate, Cascade Logic
7. Compress the k block-level readings into a small set of high-signal statistics per die: variance, IQR, peak-to-peak amplitude, skewness/kurtosis (optional). This avoids feeding raw high-dimensional noise into the model.
8. Train **Model B**: LightGBM on [Model A's features + compressed block-level stats].
9. Wrap **Model A** with **MAPIE** conformal prediction, calibrated to 99% confidence, producing prediction sets (not just a single probability).
   - Dies with a single confident label (Pass-only or Fail-only in the prediction set) → resolved immediately, no Model B call.
   - Dies with an ambiguous set (both labels plausible) → escalate to Model B.
10. Wire the full cascade: Model A → conformal filter → (bypass or escalate to Model B) → final decision.
11. Compute the **% of dies resolved by Model A alone** — this is your "ATE tester time saved" number.

### Day 3 — XAI, Package Layer, Dashboard, Rehearsal
12. Run TreeSHAP on both models. Generate:
    - Waterfall plot for individual die feature importance.
    - Wafer-level spatial heatmap of risk.
    - Block-reading pattern plot for flagged dies (e.g., show which block indices are abnormal).
    - Probability-distribution overlap plot comparing Model A vs Model B outputs on the escalated subset.
13. Build the **Package Risk Propagation** layer:
    - Group dies into simulated packages (define a reasonable N dies/package, e.g., 4–8, document your assumption).
    - Package survival probability = combination (e.g., product of survival probs, or "weakest link" logic) of individual die risk scores in that group.
    - Cost model: assign a realistic relative cost ratio (die scrap cost vs. assembled package scrap cost — package should be materially higher, e.g., 5–10x, and you should state this assumption clearly in the dashboard/doc).
14. Build the **Streamlit dashboard**: wafer map, comparison table, threshold slider with live-updating cost/time metrics, click-a-die detail view with SHAP.
15. Prepare the **60-second pitch**: one sentence on the problem, one on the cascade architecture, one on the package-cost insight, then let the slider do the talking.
16. Rehearse explaining conformal prediction and TreeSHAP in one plain-English sentence each — judges will ask.

---

## 5. REQUIREMENTS / TOOLING LIST (ALL FREE, LAPTOP-RUNNABLE)

| Purpose | Tool | Notes |
|---|---|---|
| Core modeling | `lightgbm` | CPU-only, fast on tabular data |
| Conformal prediction | `mapie` | Pin the version early — classification API varies across versions |
| Explainability | `shap` (TreeSHAP) | Fast for tree models |
| Data handling | `pandas`, `numpy` | Standard |
| Validation | `scikit-learn` (GroupKFold, metrics) | PR-AUC, F1, precision/recall |
| Dashboard | `streamlit` | Single-page app, run locally, no deployment needed |
| Plotting | `matplotlib` / `plotly` | Plotly preferred for interactive wafer map in Streamlit |
| Environment | `requirements.txt` with pinned versions | Set up on Hour 1 to avoid "works on my machine" issues |

**Team setup recommendation:** designate one "demo laptop" that the final integrated app is pulled onto by Day 3 morning. Don't debug environment mismatches during expo setup.

---

## 6. WHAT CAN BE AI-ASSISTED vs. WHAT MUST BE DONE BY THE TEAM

### Can be generated/accelerated with an LLM (Claude/GPT/Gemini):
- Boilerplate code: data loading scripts, LightGBM training loops, MAPIE wrapper setup, SHAP plotting code, Streamlit UI layout code.
- Feature engineering function scaffolding (radial distance, neighbor density, block-stat compression).
- Docstrings, README, requirements.txt, presentation slide text/talking points.
- Debugging error messages and library version conflicts.
- Cost-model formula suggestions and sanity-checking your assumptions (e.g., "is a 5-10x package-vs-die cost ratio realistic?").
- Generating synthetic/mock data to test the pipeline BEFORE the real dataset is available, so you're not blocked on Day 1.

### Must be done by the team (not just AI-generated blindly):
- **Understanding and validating the actual dataset schema** — an LLM can't know your real column names, units, or label conventions; you must inspect and confirm this yourselves.
- **Choosing and justifying the package-grouping assumption** (N dies/package, cost ratio) — this needs to be defensible to a real SanDisk engineer, not just plausible-sounding.
- **Interpreting whether Model B's improvement over Model A is actually meaningful** given your specific data — don't let an LLM "explain away" a bad result; report the real number.
- **Sanity-checking for data leakage** — GroupKFold correctness must be manually verified (check that no wafer_id appears in both train and test folds).
- **The live pitch/demo delivery** — explaining conformal prediction and SHAP in your own words, in front of judges, confidently.
- **Final threshold/business-tradeoff decisions** — how aggressive to set the conformal confidence level and risk thresholds is a judgment call the team should own and be able to defend.

---

## 7. RESULTS / METRICS TO LOOK OUT FOR (What "success" looks like)

You need these numbers ready before expo day:

1. **Model A standalone performance:** PR-AUC, F1, recall & precision on the defect class.
2. **Model B standalone performance:** same metrics — must show measurable improvement over Model A, especially on recall of the minority class. This is your core "does block-level data help" proof.
3. **% of dies resolved by Model A alone (bypassing Model B):** this is your "ATE tester time saved" headline number. Higher = more efficient gating, but don't let it come at the cost of missed defects — show the tradeoff.
4. **False positive rate at your chosen operating point** — must not be inflated; show you're not just scrapping everything to catch defects.
5. **Package-level cost avoided figure** — a concrete dollar estimate (even if built on stated assumptions) showing the value of catching a die pre-package vs. post-package.
6. **SHAP-driven explanation for at least 2-3 example flagged dies** — be ready to walk a judge through "why did the model flag this die" live.
7. **Model A vs Model B probability distribution overlap plot** — visually shows how much extra separation the block-level data buys you on ambiguous cases specifically (not on the easy cases Model A already nailed).

**Red flags to catch before demo day:**
- Accuracy looks great (>99%) but recall on the minority class is near zero → you're just predicting "healthy" for everything. This is the single most common failure mode with imbalanced data — watch for it explicitly.
- GroupKFold not actually preventing leakage (check no wafer_id overlap between train/test).
- Package cost ratio assumption is unexplainable/unrealistic if a judge pushes back on it.
- MAPIE version mismatch across team laptops causing last-minute breakage.

---

## 8. THE 60-SECOND EXPO HOOK (Script Skeleton)

1. *(5 sec)* "SanDisk doesn't sell dies — they sell packages. One bad die scraps a whole stack."
2. *(15 sec)* "We built a two-tier system: a fast model clears the easy dies instantly, and only sends the ambiguous ones to a heavier model that reads block-level data."
3. *(20 sec)* [Move the slider live] "Watch — as I tighten the threshold, tester time saved goes up, and here's the packaging cost we prevent by catching this die *before* it gets packaged, not after."
4. *(15 sec)* [Click a flagged die] "And here's exactly why the model flagged it — SHAP shows these three block readings were abnormal."
5. *(5 sec)* "All of this, on a laptop, in 72 hours, on free tools."

---

*End of specification. Any assistant reading this document should treat Section 3 (Architecture) and Section 2 (Constraints) as fixed requirements, and Section 4 (Step-by-Step) as the execution order to follow or help accelerate.*
