# Final submission: Model B (cnn) cascade, Cost 8:1

Final policy: `cost_weighted`, FN/FP cost ratio 8, `ensemble_mode=none`.
CNN threshold: 0.451177567243576; Model A bypass threshold: 0.5690903332956938.
Existing cascade routing is preserved; no models were retrained or rerun.

**Evaluation caveat:** Cost 8:1 was selected after inspecting candidate performance
on test.csv across cost ratios 4, 8 and 12. Consequently test.csv is no longer a
fully untouched holdout for this threshold-policy choice. The numbers immediately
below are a confirmatory check, not an independent final evaluation. OOF cost
minimization is the primary justification: for CNN, `8*FN + FP` decreases from
33,573 at the F1 threshold to 32,437 at the selected threshold. This is an OOF
tuning result, not an unbiased estimate of the selected policy's generalization.

Exact exported `outputs/predictions/submission.csv`, scored against test.csv on
32,598 eligible dies (`old_label == 0`):

| TP | FN | FP | TN | Fail recall | Fail precision | Fail F1 | Overall accuracy |
| --: | --: | --: | --: | --: | --: | --: | --: |
| 518 | 862 | 43 | 31,175 | 37.54% | 92.34% | 0.5337454920 | 97.22% |

The submission contains all 39,351 dies in test input order, including all 6,753
`old_label == 1` dies forced to `predicted_label=1`. IDs were checked one-to-one.
Submission SHA-256: `45a83eb47e3622f10f858e933a02adebd525507b64ba20827b9454d5a6ac17ac`.

The single absent eligible CNN score is intentional: `(W_N_0015, 22, 24)` has
Model A probability 0.906466 and state `CONFIDENT_FAIL`; its saved
`routed_to_model_b` is false. It exceeds the router's upper intermediate-risk
limit 0.90, lies outside the 0.20 boundary margin around the original 0.607530
threshold, and does not meet the pass-only spatial escalation condition. Edge
escalation is disabled in the saved router. `cascade.py` initializes CNN scores
to NaN and fills only routed rows. This die is correctly retained as predicted
fail using Model A, not dropped. No parsing, missing-feature or batch-loss fix
is warranted, and no retraining is needed.

Reproduce this policy explicitly (CLI defaults still select F1 if flags are omitted):

```sh
python scripts/predict.py --from-probabilities outputs/predictions/prediction_probabilities.csv --threshold-objective cost_weighted --fn-fp-cost-ratio 8 --ensemble-mode none --output outputs/predictions/submission.csv
```

The companion probability CSV records the selected objective, ratio, ensemble
mode and model-specific thresholds. Trained model artifacts remain unchanged.

## Historical F1-threshold wafer-grouped cross-validation
           model  threshold          pr_auc         fail_f1     fail_recall  fail_precision        accuracy
         Model A   0.607530 0.3581 ± 0.0629 0.3691 ± 0.0671 0.2871 ± 0.0434 0.6217 ± 0.2220 0.9577 ± 0.0135
   Model B (cnn)   0.599387 0.5072 ± 0.0189 0.5165 ± 0.0075 0.3613 ± 0.0104 0.9148 ± 0.0812 0.9714 ± 0.0075
 Model B (zonal)   0.588701 0.3609 ± 0.0637 0.3606 ± 0.0560 0.3210 ± 0.0468 0.4690 ± 0.1657 0.9519 ± 0.0123
Model B (global)   0.632314 0.3329 ± 0.0522 0.3338 ± 0.0466 0.3139 ± 0.0424 0.3958 ± 0.1311 0.9477 ± 0.0104

All five folds hold out whole wafers. Every eligible die has one OOF probability.
Feature selection, imputation and neural scaling are fitted only on training folds.
Fold metrics use the pooled OOF-tuned threshold. They are descriptive tuning results,
not nested-CV estimates or evidence that predictive performance improved.
Standard deviations use ddof=0; the folds are not independent confidence intervals.
CNN uses neural fusion; zonal/global use LightGBM plus summaries, so the learner also changes.
Configured tree learner: lightgbm (see logs for effective backend).
Zonal compression assumes stable, meaningful ordering of the block sequence.
The deployment cascade uses model-specific OOF thresholds and an OOF routing heuristic.
No independent split-conformal coverage guarantee or probability calibration is claimed.
Selected deployment mode: cnn.
New neural architecture requires retraining; old neural checkpoints are incompatible.
