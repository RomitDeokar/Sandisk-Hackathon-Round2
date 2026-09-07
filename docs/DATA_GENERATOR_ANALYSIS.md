# Data Generator Analysis

**Source:** `generate_data.py` · `config.yaml` · `README.md`  
**WM-811K Base:** LSWMD.pkl (must be placed at `data/LSWMD.pkl`)

---

## 1. Exact Generated Schema

Every output file (`train`, `test`, `validation`) shares the same column layout:

| # | Column | dtype | Description |
|---|--------|-------|-------------|
| 1 | `wafer_id` | str | e.g. `W_F_0001` (failure-pattern wafer) or `W_N_0001` (none/all-pass wafer) |
| 2 | `die_row` | int | Row index within the wafer grid (0-based) |
| 3 | `die_col` | int | Column index within the wafer grid (0-based) |
| 4-503 | `feature_1` ... `feature_500` | float64 | Synthetic parametric test measurements (500 by default, configurable) |
| 504 | `old_label` | int (0/1) | **Pre-test** die status from WM-811K map: 0=pass, 1=fail |
| 505 | `block_readings` | str | Space-separated string of 2000 float values (sub-die block readings) |
| 506 | `label` | int (0/1) | **Post-test** die status: 0=pass, 1=fail (**TARGET COLUMN**) |

> **Note:** `validation.csv` has the `label` column **dropped** — it is an unlabeled inference set.
> Column order: `wafer_id, die_row, die_col, feature_1...feature_N, old_label, block_readings, label`

---

## 2. Exact Block-Reading Representation

`block_readings` is a **single string column** where:

- Each value is a **space-separated sequence** of exactly `k = 2000` floats
- Each float is **rounded to 2 decimal places** (e.g. `"98.34 101.22 99.87 ..."`)
- Parsing: `np.array(row["block_readings"].split(), dtype=np.float32)` → shape `(2000,)`

**Generation mechanics:**
```
Base signal:    N(100.0, 15.0) per block, for all k=2000 blocks
Smoothing:      0.6 * raw + 0.4 * uniform_filter1d(raw, size=5)  [spatial correlation]
Fail injection: Only failing dies (label=1) get anomalous blocks
  - Number of anomalous blocks: max(1, int(2000 * 0.05)) = 100 blocks
  - Seed position: uniform random in [0, 2000)
  - Spread: Gaussian offsets with sigma = k*0.05 = 100 positions (clustered)
  - Anomalous shift: N(fail_shift, fail_shift*0.3) added, where fail_shift = 0.3 * 15.0 = 4.5
```

**Key insight:** Only ~5% of blocks in a failing die show anomalous readings —
the signal is **sparse and spatially clustered** within the 2000-element sequence.

---

## 3. Train / Test / Validation Generation

```
WM-811K LSWMD.pkl
  |-- labeled_failure wafers  (failureType != "none" and != "")
  `-- none_wafers             (failureType == "none")
             |
             v  select_wafers(total=200, target_fail_rate=0.03)
             |
   Shuffle -> [0:160] train wafers  |  [160:200] test wafers
                     |                          |
             generate_dataset()         generate_dataset()
                     |                          |
             train_df (labeled)         test_df (labeled)
                                               |
                                        val_df = test_df.drop("label")
```

**Split ratios (default config):**

| Split | Wafers | Has label |
|-------|--------|-----------|
| Train | 160 | Yes |
| Test | 40 | Yes |
| Validation | 40 | No (label column dropped) |

**Wafer selection logic:**
- `target_fail_rate = 0.03` (3%)
- `avg_internal_fail_rate = 0.12` (12% within failure-pattern wafers assumed)
- `fraction_failure_wafers = min(0.03 / 0.12, 0.8) = 0.25`, clamped to min 0.05
- So ~25% of selected wafers have failure patterns, ~75% are "none" (all-pass) wafers

**Wafer ID naming:**
- Failure-pattern wafers: `W_F_0000`, `W_F_0001`, ...
- None (all-pass) wafers: `W_N_0000`, `W_N_0001`, ...
- After shuffling the combined list, train takes first 160, test takes last 40

**RNG determinism:**
- Single `np.random.default_rng(seed=42)` is shared for all selection + feature generation
- Feature definitions use a separate `np.random.default_rng(seed+1000)` — always identical

---

## 4. Spatial Generation Behavior

Each wafer generates dies only for **valid positions** (`wafer_map > 0`).
Grid cells with value `0` are outside the circular wafer boundary and are skipped.

**WM-811K map values:**
- `0` = no die (outside wafer boundary) — skipped entirely
- `1` = passing die
- `2` = failing die (old_label = 1)

### Spatial maps computed per wafer:

| Map | Formula | Range | Purpose |
|-----|---------|-------|---------|
| **Radial** | `sqrt((y-cy)^2 + (x-cx)^2) / max_r` | [0, 1] | Center=0, Edge=1 |
| **Linear gradient** | `cos(theta)*x_norm + sin(theta)*y_norm`, random theta | [-1, 1] | Simulates process chamber asymmetry |
| **Neighborhood fail density** | `uniform_filter(fail_mask, 5x5) / uniform_filter(valid_mask, 5x5)` | [0, 1] | Local clustering of failures |

### Feature generation per die per feature:
```
base = N(base_mean, base_std)                              # process distribution
base += radial_map  * U(-1,1) * 0.3 * base_std            # radial effect  (strength=0.3)
base += linear_map  * U(-1,1) * 0.15 * base_std           # linear drift   (strength=0.15)
base += fail_density * 0.2 * fail_shift                    # neighbor contamination (influence=0.2)

if die is non-marginal fail:  base += fail_shift            # full separation signal
if die is marginal fail:      base += fail_shift * U(0.05, 0.25)  # near-pass signal (5-25% shift)
```

### New failure injection (post-test fails):
```
wafer_base_rate ~ Exponential(mean=0.02), capped at 0.25
P(new_fail | die) = wafer_base_rate
                  + fail_density * wafer_base_rate * 3.0   (proximity boost x3)
                  + radial_map   * wafer_base_rate * 1.5   (edge boost x1.5)
                  clipped to [0, 0.40]
```
Only **passing** dies (`old_label=0`) can become new failures.

---

## 5. Label Construction

Two-stage label scheme:

```
WM-811K map value:
  0 -> no die (skipped)
  1 -> pass die  -> old_label = 0
  2 -> fail die  -> old_label = 1

Post-test fail injection (only on old_label=0 dies):
  Probabilistic draw -> new_fail_mask

Final label:
  label = old_label | new_fail_mask
        = 1  if old fail  OR  new fail
        = 0  if old pass AND no new fail
```

**Marginal fails** (65% of all label=1 dies):
- Selected randomly from ALL label=1 dies (both old and newly added)
- Receive only 5-25% of the normal `fail_shift` in each feature
- Nearly indistinguishable from pass dies — intentionally hard
- 65% marginal fraction makes the problem deliberately challenging

**Evaluation rule (from README):**
> Only consider **eligible dies** where `old_label = 0`.
> Dies with `old_label = 1` are already known failures before the test —
> the model should only predict on `old_label=0` dies.

---

## 6. Available Configuration Parameters

All parameters live in `config.yaml`:

### Core
| Parameter | Default | Description |
|-----------|---------|-------------|
| `seed` | `42` | Master RNG seed (data RNG = seed, feature def RNG = seed+1000) |
| `num_wafers_train` | `160` | Wafers in training set |
| `num_wafers_test` | `40` | Wafers in test/validation set |
| `wm811k_path` | `"data/LSWMD.pkl"` | Path to WM-811K pickle file |
| `output_dir` | `"input"` | Directory where output files are saved |
| `output_formats` | `["csv"]` | List of formats: `parquet`, `pkl`, `csv` |

### Feature Generation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_features` | `500` | Number of parametric features per die |
| `feature_generation.base_mean_range` | `[0.1, 5000.0]` | Log-uniform range for feature means |
| `feature_generation.cv_range` | `[0.02, 0.15]` | Coefficient of variation (std = CV * abs(mean)) |
| `feature_generation.fail_shift_fraction_range` | `[0.1, 0.5]` | fail_shift as fraction of base_std |
| `feature_generation.log_scale_mean` | `true` | Use log-uniform (vs uniform) sampling for base_mean |

### Spatial
| Parameter | Default | Description |
|-----------|---------|-------------|
| `neighborhood_window` | `5` | Convolution window size for local fail density |
| `zone_rows` | `4` | Wafer divided into zone_rows x zone_cols blocks (computed but not in output) |
| `zone_cols` | `4` | Same |
| `radial_gradient_strength` | `0.3` | Max radial effect as fraction of feature std |
| `linear_gradient_strength` | `0.15` | Max linear drift as fraction of feature std |
| `neighborhood_influence` | `0.2` | Neighbor fail density influence coefficient |

### Label / Class Imbalance
| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_fail_rate` | `0.03` | Target ~3% overall die fail rate |
| `new_fail_rate` | `0.02` | Mean of exponential dist for per-wafer new-fail rate |
| `marginal_fail_fraction` | `0.65` | Fraction of failing dies with near-boundary signal |

### Block Readings
| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_block_readings` | `2000` | Number of sub-die block readings per die (k) |
| `block_readings.base_mean` | `100.0` | Normal signal mean for all blocks |
| `block_readings.base_std` | `15.0` | Normal signal std for all blocks |
| `block_readings.fail_shift_fraction` | `0.3` | Anomalous shift = 0.3 * 15 = 4.5 |
| `block_readings.anomalous_block_fraction` | `0.05` | 5% of blocks in failing die are anomalous (=100 blocks) |
| `block_readings.block_correlation_kernel` | `5` | Smoothing kernel size for intra-die block correlation |

### CLI Overrides
```bash
python generate_data.py --num_wafers 500   # overrides train+test total (80/20 split)
python generate_data.py --config alt.yaml  # use alternative config file
python generate_data.py --csv              # force CSV export even if not in formats list
```

---

## Key Design Insights for ML

1. **Massive imbalance:** ~3% fail rate -> 97% pass. Must use SMOTE, weighted loss, or threshold tuning.
2. **65% hard cases:** Most fails are marginal — standard threshold classifiers will struggle without calibration.
3. **Block readings are sparse signals:** Only 5% of 2000 blocks show anomaly, clustered spatially. Aggregate statistics (mean, std, max, percentiles) will lose signal; need local extrema or learned aggregation.
4. **Multi-scale structure:** Die-level features (500) + wafer-level spatial gradients + sub-die block readings (2000).
5. **Evaluation is on `old_label=0` only:** Pre-known failures are excluded from evaluation metric.
6. **`block_readings` is a string column:** Must be parsed with `.split()` + float cast before use.
7. **500 features, correlated by spatial gradients:** Per-feature radial/linear effects create within-wafer correlation. PCA or feature selection will help.
8. **Wafer-level random effects:** Exponential wafer_base_rate creates high between-wafer variance in fail rate. Some wafers have near-zero new fails, others have 20%+.
9. **Die position is informative:** Edge dies (high radial distance) have elevated fail probability. `die_row`, `die_col`, and derived radial distance should be engineered as features.
