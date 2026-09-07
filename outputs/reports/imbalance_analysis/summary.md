# How the current models handle imbalanced, overlapping distributions

This report uses the current five-fold wafer-grouped OOF predictions for
154,037 eligible training dies: 6,519 failures and
147,518 passes (4.23% failure prevalence). It does not
use in-sample predictions, retrain either model, or change the Cost 8:1 policy.

## Precision–recall behavior

Model A achieves pooled OOF average precision 0.3180; Model B (cnn)
achieves 0.4876. PR curves are the appropriate ranking view
under this class imbalance because they expose the precision cost of recovering
more of the rare failures. The CNN's higher average precision shows that adding
block-level evidence improves ranking of failures, although the classes remain
substantially overlapping.

These pooled values differ from the reported mean of five fold-level PR-AUCs:
pooling ranks scores across folds, whereas the CV table calculates each fold's
PR-AUC first and then averages the five results.

## Probability reliability

Model A has Brier score 0.1394 and 10-bin calibration error
0.3141. Model B (cnn) has Brier score
0.0487 and calibration error 0.1383.
The saved scores are uncalibrated; deviations from the diagonal in the reliability
plot mean score magnitudes should not be interpreted as literal failure rates.
Cost-weighted thresholding changes decisions, not calibration.

## Feature-distribution overlap

Histogram intersection ranges from 0 (separated) to 1 (indistinguishable).
Among the current Model A top-20 features, `wafer_die_density` separates the
classes most strongly (overlap 0.837), while
`spatial_radial_dist` overlaps most (overlap 0.928).
No single feature resolves the task; Model A combines weak and spatial signals.

## Direct answer to the brief

Model A addresses imbalance through balanced LightGBM class weights and combines
many overlapping parametric/spatial features. Model B uses focal loss to emphasize
hard examples and adds raw-block CNN attention plus block summaries. Its higher
OOF average precision indicates better rare-failure ranking, while the reliability
curves show neither model should be treated as calibrated. The final Cost 8:1
operating point explicitly trades additional false alarms for fewer false negatives;
it does not remove the underlying distribution overlap.
