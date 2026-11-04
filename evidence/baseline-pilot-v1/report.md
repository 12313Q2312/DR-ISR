# Measured results

Descriptive pilot evidence; no statistical or novelty claim.

Selected expression minimizes training MSE + node penalty. Holdouts are never used for search.

Noisy task: training labels contain N(0,0.1²) noise; both holdouts use the clean target.

| Task | Method | Seeds | Train MSE median | Test MSE median [Q1,Q3] | Extrap MSE median | Seconds median | Fitness / probes / context median |
|---|---|---:|---:|---:|---:|---:|---|
| polynomial | gp | 3 | 0.00013333 | 0.00012806 [6.403e-05,0.0003677] | 0.0008995 | 0.119 | 1200 / 0 / 0 |
| additive | gp | 3 | 4.2231e-33 | 3.8342e-33 [1.917e-33,0.0462] | 6.8617e-32 | 0.114 | 1200 / 0 / 0 |
| interaction | gp | 3 | 3.4377e-33 | 3.7213e-33 [1.861e-33,0.05297] | 5.4535e-32 | 0.091 | 1200 / 0 / 0 |
| composition | gp | 3 | 0 | 0 [0,7.223e-05] | 0 | 0.078 | 1200 / 0 / 0 |
| noisy_additive | gp | 3 | 0.0088515 | 0 [0,0.0497] | 0 | 0.110 | 1200 / 0 / 0 |

## Paired comparisons against GP

Negative median difference favors the compared method. A difference of medians is not a median paired difference.

| Task | Method | Paired seeds | Median test MSE difference | Wins / ties / losses |
|---|---|---:|---:|---|

## Budget and limitations

Every training vector pass is charged, including donor probes and context passes. These units differ in tree size and derivative overhead; inspect node-sample work and wall time too. This is a conservative accounting protocol, not a claim of identical FLOPs or equal runtime.

Guidance uses a local linearization. Multiplication/composition can invalidate its prediction. The shuffled-residual control measures whether per-sample residual alignment adds value at similar overhead. All budgets are exhausted even after a solution is found. Duplicate probes and unchanged offspring are charged.

Timing is machine-dependent. Small synthetic tasks and a small seed sample cannot establish general superiority. No hyperparameter tuning or selection is performed using holdout metrics.
