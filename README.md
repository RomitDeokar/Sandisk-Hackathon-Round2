# WaferFusion-Cascade: SanDisk Die Yield Prediction System

WaferFusion-Cascade is a multi-resolution, uncertainty-aware ML system designed for the SanDisk Die Yield Hackathon. It implements a two-stage screening cascade that evaluates die-level parametric signatures, wafer-level spatial geometry, and deep sub-die 2,000-block electrical waveforms while ensuring strict data-leakage boundaries and compute efficiency.

---

## 1. System Architecture

```
Raw Data
   ↓
Data Validation (schema, nulls, types, block length)
   ↓
Feature Engineering (500+ parametric, multi-scale spatial 3x3/5x5/7x7, local z-scores)
   ↓
Model A: Fast Screening (LightGBM with XGBoost/HistGBM fallbacks)
   ↓
Probability Calibration (Isotonic Regression on holdout wafers)
   ↓
Uncertainty / Conformal Gate (Coverage-controlled prediction sets)
   ↓
Cascade Safety Router
   ├── Confident → Pure Model A resolution (Fast)
   └── Ambiguous / Suspicious / Boundary Margin → Model B
                                                  ↓
                                       Deep 1D CNN Block Encoder
                                                  ↓
                                       Evidence Fusion (Modal Gating)
                                                  ↓
                                        Final Calibrated Risk
                                                  ↓
                                       Submission & Visualizations
```

---

## 2. Directory Structure

```text
sandisk_die_yield/
├── configs/
│   ├── base.yaml              # Master configuration
│   ├── cpu.yaml               # CPU development profile
│   └── gpu.yaml               # CUDA GPU acceleration profile
├── src/
│   └── sandisk_yield/
│       ├── schema.py          # Column constants, target & eligibility masking
│       ├── seed.py            # Reproducibility seeds (Python, NumPy, PyTorch)
│       ├── config.py          # Config loader supporting deep inheritance
│       ├── logging_utils.py   # Formatted logging handlers
│       ├── data/
│       │   ├── loader.py      # Autodetect CSV, Parquet, Pickle loaders
│       │   ├── validator.py   # Schema & constraint validation
│       │   └── splitter.py    # Grouped wafer-level CV & calibration splitters
│       ├── features/
│       │   ├── parametric.py  # Robust imputation & scaling
│       │   ├── spatial.py     # Geometry & old-failure neighborhood densities
│       │   ├── wafer.py       # Wafer context & local process z-scores
│       │   ├── blocks.py      # Block parser & explicit anomaly features
│       │   └── pipeline.py    # Unified feature transformer
│       ├── models/
│       │   ├── baseline.py    # LogisticRegression baseline
│       │   ├── model_a.py     # LightGBM screening classifier
│       │   ├── block_encoder.py # PyTorch 1D CNN with attention
│       │   ├── fusion.py      # Evidence fusion network with learned gating
│       │   ├── model_b.py     # End-to-end deep inspection model
│       │   └── calibration.py # Isotonic probability calibrator
│       ├── cascade/
│       │   ├── uncertainty.py # Entropy & boundary margin metrics
│       │   ├── conformal_gate.py # Split-conformal uncertainty gate
│       │   ├── router.py      # Safety routing policy
│       │   └── cascade.py     # End-to-end cascade orchestrator
│       ├── training/
│       │   ├── losses.py      # Imbalanced Focal Loss
│       │   ├── evaluation.py  # Die yield PR-AUC & threshold search
│       │   ├── trainer_a.py   # Grouped CV Model A trainer
│       │   ├── trainer_b.py   # PyTorch Model B trainer
│       │   └── trainer_cascade.py # Cascade calibration coordinator
│       ├── explainability/
│       │   ├── tree_explain.py # Feature importance extraction
│       │   ├── block_explain.py # Block saliency localization
│       │   └── risk_decomposition.py # Die Risk Card generator
│       ├── risk/
│       │   └── wafer_risk.py  # Wafer-level summary metrics
│       ├── visualization/
│       │   ├── wafer_map.py   # 2D spatial wafer risk maps
│       │   └── curves.py      # PR, ROC, Calibration curves
│       └── inference/
│           ├── predict.py     # High-level inference orchestrator
│           └── submission.py  # Strict submission validator & exporter
├── scripts/
│   ├── validate_data.py       # Dataset validation CLI
│   ├── build_features.py      # Feature engineering CLI
│   ├── run_all.py             # Master full execution pipeline
│   └── predict.py             # Submission prediction CLI
├── dashboard/
│   └── app.py                 # Interactive Streamlit dashboard
└── tests/
    ├── test_no_leakage.py     # Strict data-leakage boundary tests
    ├── test_cascade.py        # Conformal gate & routing tests
    ├── test_model_b.py        # PyTorch Model B forward pass tests
    ├── test_submission.py     # Submission validation tests
    ├── test_schema.py         # Schema tests
    └── test_validator.py      # Multi-level data validator tests
```

---

## 3. How to Run

### Step 1: Run Full Pytest Test Suite
```bash
pytest tests -v
```

### Step 2: Run Full End-to-End Pipeline
```bash
# Fast mode for rapid validation:
python scripts/run_all.py --fast

# Full production run:
python scripts/run_all.py --config configs/base.yaml
```

### Step 3: Run Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### Step 4: Run Inference on New Unlabeled Datasets
```bash
python scripts/predict.py --input input/validation.csv --output outputs/predictions/submission.csv
```
