# Measured results

Descriptive pilot evidence; no statistical or novelty claim.

Selected expression minimizes training MSE + node penalty. Holdouts are never used for search.

Noisy task: training labels contain N(0,0.1²) noise; both holdouts use the clean target.

| Task | Method | Seeds | Train MSE median | Test MSE median [Q1,Q3] | Extrap MSE median | Seconds median | Fitness / probes / context median |
|---|---|---:|---:|---:|---:|---:|---|
| polynomial | gp | 10 | 0.00013289 | 0.00012997 [6.238e-33,0.001576] | 0.0016199 | 0.369 | 3000 / 0 / 0 |
| polynomial | residual_gp | 10 | 0.00029302 | 0.0002282 [7.152e-33,0.008363] | 1.6149 | 0.234 | 680 / 1856 / 464 |
| polynomial | shuffled_gp | 10 | 0.030983 | 0.032335 [0.008695,0.04145] | 9.8132 | 0.200 | 688 / 1849.5 / 462.5 |
| additive | gp | 10 | 0.010068 | 0.011343 [2.564e-33,0.08535] | 0.45364 | 0.285 | 3000 / 0 / 0 |
| additive | residual_gp | 10 | 0.079124 | 0.095099 [0.02384,0.185] | 4.8851 | 0.112 | 685 / 1852 / 463 |
| additive | shuffled_gp | 10 | 0.080588 | 0.091114 [0.01493,0.1011] | 4.6412 | 0.151 | 682.5 / 1854 / 463.5 |
| interaction | gp | 10 | 1.7188e-33 | 1.8607e-33 [0,0.02376] | 2.7268e-32 | 0.234 | 3000 / 0 / 0 |
| interaction | residual_gp | 10 | 0.10632 | 0.10594 [0.1059,0.1063] | 5.3848 | 0.098 | 679 / 1856.5 / 464.5 |
| interaction | shuffled_gp | 10 | 0.054136 | 0.054664 [0.0003384,0.1059] | 3.121 | 0.127 | 688 / 1849 / 463 |
| composition | gp | 10 | 0 | 0 [0,0] | 0 | 0.191 | 3000 / 0 / 0 |
| composition | residual_gp | 10 | 0 | 0 [0,0.0001445] | 0 | 0.102 | 680 / 1856 / 464 |
| composition | shuffled_gp | 10 | 6.2142e-05 | 7.2227e-05 [0,0.0001445] | 0.75914 | 0.103 | 688 / 1849.5 / 462.5 |
| noisy_additive | gp | 10 | 0.0097847 | 0.00066968 [0,0.07] | 0.025109 | 0.254 | 3000 / 0 / 0 |
| noisy_additive | residual_gp | 10 | 0.0588 | 0.050297 [0.0009228,0.09978] | 2.6725 | 0.143 | 681.5 / 1854.5 / 464 |
| noisy_additive | shuffled_gp | 10 | 0.10255 | 0.097416 [0.003691,0.1037] | 5.5342 | 0.166 | 685.5 / 1851 / 463.5 |

## Paired comparisons against GP

Negative median difference favors the compared method. A difference of medians is not a median paired difference.

| Task | Method | Paired seeds | Median test MSE difference | Wins / ties / losses |
|---|---|---:|---:|---|
| polynomial | residual_gp | 10 | -6.40313e-05 | 5 / 2 / 3 |
| polynomial | shuffled_gp | 10 | 0.0296961 | 1 / 0 / 9 |
| additive | residual_gp | 10 | 0.00186334 | 2 / 3 / 5 |
| additive | shuffled_gp | 10 | 0.00573823 | 2 / 2 / 6 |
| interaction | residual_gp | 10 | 0.0901096 | 1 / 1 / 8 |
| interaction | shuffled_gp | 10 | 0.00235466 | 2 / 2 / 6 |
| composition | residual_gp | 10 | 0 | 0 / 6 / 4 |
| composition | shuffled_gp | 10 | 7.22268e-05 | 0 / 5 / 5 |
| noisy_additive | residual_gp | 10 | 0.00180907 | 3 / 2 / 5 |
| noisy_additive | shuffled_gp | 10 | 0.0514024 | 2 / 2 / 6 |

## Budget and limitations

Every training vector pass is charged, including donor probes and context passes. These units differ in tree size and derivative overhead; inspect node-sample work and wall time too. This is a conservative accounting protocol, not a claim of identical FLOPs or equal runtime.

Guidance uses a local linearization. Multiplication/composition can invalidate its prediction. The shuffled-residual control measures whether per-sample residual alignment adds value at similar overhead. All budgets are exhausted even after a solution is found. Duplicate probes and unchanged offspring are charged.

Timing is machine-dependent. Small synthetic tasks and a small seed sample cannot establish general superiority. No hyperparameter tuning or selection is performed using holdout metrics.
