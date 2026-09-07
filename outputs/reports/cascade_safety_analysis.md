# Cascade Safety & Dangerous False Negative Analysis

## Summary
* **Total Eligible Validation Dies:** 118
* **Actual New Failures:** 5
* **Cascade Dangerous False Negatives:** 2
* **Cascade Missed Failures (Unrouted False Negatives):** 0

## Routing Recommendation
The conformal coverage level of 0.95 and boundary margin of 0.2 successfully catches marginal candidates. If lower risk tolerance is required in production, decrease `min_model_b_prob` from 0.1 to 0.05.
