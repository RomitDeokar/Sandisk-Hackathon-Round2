# Code-review changes for ten wafers

## Final selected submission policy (2026-09-07)

Use Model B (cnn) single-model cascade with Cost 8:1, not an ensemble.
CNN threshold is 0.451177567243576; the existing Model A bypass uses
0.5690903332956938. Saved routing is preserved, without inference or retraining.

**Test-selection caveat:** Cost 8:1 was selected after inspecting test.csv
performance across several cost ratios. test.csv is therefore no longer a fully
untouched holdout for this policy choice. The following final results are a
confirmatory check, not an independent final evaluation: TP=518, FN=862, FP=43,
TN=31,175; fail recall=37.54%, precision=92.34%, Fail F1=0.5337.
Primary justification is OOF cost minimization: CNN's `8*FN + FP` is 32,437
versus 33,573 at its F1-tuned threshold (descriptive tuning results).

`outputs/predictions/submission.csv` contains all 39,351 dies, including 6,753
old failures forced to predicted failure. The above metrics score only the
32,598 eligible dies. See `outputs/reports/final_report.md` for the exact-file
verification, submission hash and intentional CNN-bypass explanation.

```sh
python scripts/predict.py --from-probabilities outputs/predictions/prediction_probabilities.csv --threshold-objective cost_weighted --fn-fp-cost-ratio 8 --ensemble-mode none --output outputs/predictions/submission.csv
```

Keep these explicit flags: the general CLI default remains F1, not Cost 8:1.

## Cost-sensitive decisions and ensembles without retraining

For the existing saved run, generate F1-versus-cost comparison reports only:

```sh
python scripts/run_all.py --postprocess-only --threshold-objective cost_weighted --fn-fp-cost-ratio 8
python scripts/analyze_thresholds.py --test-labels input/test.csv --ratios 4 8 12
```

These commands never fit a model or overwrite the submission/model artifacts.
The second command scores existing predictions against the specified test labels;
those labels must match the inference dataset. In the current saved run, every
non-label field of validation.csv was verified equal to test.csv. Thresholds
are selected exclusively from OOF labels, never test labels. Choosing a ratio
after inspecting test results makes that test part of operating-policy selection;
it is no longer an untouched final evaluation of the chosen policy.

`--threshold-objective {f1,cost_weighted}` defaults to `f1`. The cost objective
minimizes `FN * ratio + FP` over distinct OOF score cuts (FP cost = 1), including
all-pass/all-fail decisions. `--fn-fp-cost-ratio` defaults to 8 and must be positive
and finite. The 8:1 ratio is a policy assumption supplied for this task; a package
scrap multiplier alone does not establish a per-die FN/FP cost ratio. Ties favor
the lower threshold/higher recall. This works on raw, uncalibrated probabilities;
it does not substitute a theoretical calibrated probability cutoff.

New reports under `outputs/metrics/`:

- `threshold_tradeoffs.csv`: both objectives, thresholds, TP/FN/FP/TN,
  pooled recall/precision/F1, weighted cost, and fold mean/std.
- `threshold_tradeoffs_side_by_side.csv`: the two objectives in adjacent columns.
- `operating_point_comparison.csv`: single models, union and average, both objectives.
- `model_overlap_analysis.csv`: failing dies caught by both, only A, only CNN,
  and neither at each objective's independently tuned A/B thresholds.
- `threshold_cost_candidates_oof.csv`, `model_overlap_candidates.csv`, and optional
  `threshold_cost_candidates_test.csv`: the candidate ratio sweep.

`--ensemble-mode {none,union,average}` is available in run_all.py and predict.py.
Union ORs the A/CNN decisions at their own tuned thresholds; it cannot reduce
recall relative to either constituent on the same rows. Average uses the mean of
the two probabilities and tunes its own single threshold on OOF predictions.
Union has no single probability threshold or PR-AUC: the max score exported for
display is not used as a surrogate decision threshold. Both ensemble modes need
CNN scores for every eligible die, so their runtime compute savings differ from
the cascade. In the current saved test artifact one eligible die lacks a CNN
score. Candidate standalone/ensemble test reports explicitly exclude this row
and report coverage; candidate cascade scores use ALL eligible rows and freeze
the saved routing. Ensemble submission export rejects missing CNN scores.

After choosing a policy, export to a NEW candidate filename without inference:

```sh
python scripts/predict.py --from-probabilities outputs/predictions/prediction_probabilities.csv --threshold-objective cost_weighted --fn-fp-cost-ratio 8 --output outputs/predictions/submission_cost8.csv
```

For a complete union/average submission, predict.py can run saved models in
`--mode full_model_b` with `--ensemble-mode union` or `average`; this is inference,
not retraining. When invoking run_all.py without `--postprocess-only`, its existing
training workflow still runs, then uses the requested operating policy to export.
The saved `cv_metrics.json` retains the original F1 summary; the new operating
reports contain both objectives. Current submission/model files are not changed
by the demo slider or the postprocessing command.

The dashboard slider recalculates both objectives on saved OOF probabilities,
showing confusion counts and pooled recall/precision/F1 without training. These
OOF-tuned metrics are descriptive, not nested-CV performance estimates. Existing
retrained-model artifacts with fitted probability calibrators are rejected by
the raw-OOF submission tuning path to avoid mixing probability scales.

## Using real input files

Place `train.csv`, `validation.csv`, and `test.csv` in the `input/` folder,
then run `python scripts/run_all.py` with no arguments to use them automatically.
The project root is the folder containing `scripts/`, `src/`, and `configs/`
(in this download, it is the nested `Sandisk_Wizards-main` directory).

Default inputs are resolved relative to that project root, regardless of the
shell's working directory. The precedence is an explicit CLI override, then the
configured `paths.raw_*` value, then `input/<split>.csv`. The shipped configuration
already uses those three default names. Relative configured paths are project-
relative; relative CLI paths are relative to the shell's working directory.

```sh
python scripts/run_all.py --train /path/train.csv --val /path/validation.csv --test /path/test.csv
```

`--validation` is also accepted as an alias for `--val`. All three files must
exist before `run_all.py` starts loading or training; otherwise it prints a clear
missing-file error. Training/CV uses `train.csv`; the existing prediction export
uses `validation.csv`. `test.csv` is required by this input contract but is not
automatically scored by the CV entry point. The separate comparison entry point
uses train/test. This wiring does not change those dataset roles.

Feature-building requires only train; comparison scripts require train/test;
the validation script checks all three unless `--file` selects one. Those scripts
also accept the train/val/test overrides. `scripts/predict.py` defaults to
`input/validation.csv` and keeps its `--input` override (saved models are still
required). CSV/TSV loaders detect comma, tab or semicolon from the header and log
the selected delimiter; the `.csv` extension alone does not dictate it. Spaces
inside `block_readings` remain part of that cell. No real data was read to add
this wiring; Python sources were checked for syntax only.

The input directory is ignored except for its `.gitkeep` placeholder, with an
explicit `input/*.csv` ignore rule for confidential data. Git ignore rules do not
remove files that were already tracked; do not force-add data files.

The active entry point is `scripts/run_all.py`. During the original CV changes, no pipeline, import check,
compilation, test, data generation or data validation was executed while making
those edits. The later input-wiring changes received a source-only syntax check
as described above. Runtime assertions/logging are provided for the eventual data run.

1. `data/alignment.py` asserts row identity and feature name/order consistency.
   The feature pipeline freezes its output schema and uses cleaned local inputs.
   CV transforms complete wafers before selecting `old_label == 0` rows, so old
   failures remain visible to neighborhood features. Each feature's distinct
   count is logged per fold. Wafer-level features can have only ten independent
   values; no-old-failure wafers have constant nearest-failure distance.
2. `training/cross_validation.py::train_models_cv` shares five wafer folds across
   A and B, fits preprocessing within each fold, and returns positional OOF
   probabilities aligned with eligible source rows. Each row is checked to occur
   once. `training/trainer_a.py::train_model_a_cv` retains its three-value return
   and adds fold summaries plus optional fold-local preprocessing parameters.
3. `training/evaluation.py::find_best_threshold` considers distinct score cuts,
   including all-positive/all-negative decisions, with failure (1) as the target
   class. OOF thresholds are saved per model and used for exported decisions.
   Dashboard displays saved decisions and thresholds; it does not retune.
4. `models/block_encoder.py` has one eight-channel convolution, an eight-value
   embedding by default, dropout 0.4 and optional attention. Model B defaults to
   a 32-unit fusion head. AdamW already had weight decay; the entry point now
   passes configuration through. Checkpoint weights use deep copies. LayerNorm
   handles singleton batches. New neural checkpoints require retraining.
5. `--block-mode cnn|zonal|global` chooses the deployed Model B. Add
   `--compare-block-modes` to evaluate all three with identical folds and save a
   consolidated table. CNN retains neural fusion and 23 anomaly summaries;
   global/zonal use the configured tree learner on tabular plus block summaries.
   This comparison therefore changes the learner as well as the representation.
   Zonal statistics assume contiguous sequence positions have physical meaning.
6. `--n-estimators` defaults to 100 and overrides YAML last. Effective LightGBM
   parameters and train/validation PR-AUC are logged per fold. CPU/GPU/base
   configuration defaults all use the smaller neural dimensions.
7. `training/evaluation.py::compare_models` writes PR-AUC (average precision),
   failure F1, recall, precision and accuracy as mean ± population standard
   deviation across folds. Raw fold metrics and thresholds are saved in JSON.

Example command for a human to run later from the project directory:

```sh
python scripts/run_all.py --block-mode cnn --compare-block-modes --n-estimators 100
```

The pooled OOF threshold is tuned on the same OOF labels used for reporting:
decision metrics are descriptive tuning results, not unbiased nested-CV results.
No performance improvement is claimed from source inspection. Fold variability
is not a confidence interval. A fold with one validation class logs a warning;
a training fold with one class raises an error rather than creating fake scores.

The previous cascade calibrated on rows already used for training. The new
entry point preserves raw probability scales and uses OOF scores for heuristic
routing, with separate A/B decision thresholds. It does not claim calibrated
probabilities or independent split-conformal coverage. The legacy cascade helper
now rejects detectable fit/calibration row overlap. Callers of that helper must
preserve source indices and keep all learned preprocessing/models off calibration
wafers. The separate legacy `scripts/compare_models.py` remains a holdout
benchmark; the new consolidated CV workflow is `scripts/run_all.py`.

Models, mode adapters, thresholds and preprocessing are saved together under
the configured models directory. `scripts/predict.py` prefers this preprocessing
artifact with a legacy-directory fallback. OOF predictions include die IDs and
fold IDs. Prediction exports include the decision threshold used for each die.
The source feature allowlist is `feature_*`; confirming that those measurements
and block readings are genuinely pre-outcome requires dataset provenance.
