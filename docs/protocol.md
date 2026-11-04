# Protocol v1 — fixed before measured experiments

Question: does residual-informed donor selection improve search outcomes enough to pay for its own overhead?

This is an initial, independently implemented expression-tree GP study. It is not a replication of VSR, neural training, or a test of every mechanism proposed for DR-ISR. The historical project name is retained; this first implementation is steady-state GP, not a complete incremental/backfitting system.

## Arms

- `gp`: tournament selection, random subtree crossover (0.75) or random subtree mutation (0.25), steady-state worst-member replacement, incumbent retained.
- `residual_gp`: identical engine and operators; on crossover, score up to four donor subtree proposals using training residuals and local context sensitivity. Replace the same uniformly sampled recipient path with the highest-scoring donor. One donor parent is selected per crossover in every arm. Proposals are sampled with replacement.
- `shuffled_gp`: same guidance calculation, but permute residual values across samples once per guided event. This is an alignment/overhead control; it is not expected to share exactly the same random proposal stream after divergence.

All arms share the initial population, operators `{+, -, *, protected_div, sin, cos}`, terminals (variables and uniform constants in [-2,2]), parsimony penalty 1e-4 per expanded tree node, depth/node limits, data, seeds and charged budget. No constant fitting, extra operators, external pretrained model or knowledge of the target expression enters the search. Root replacement is allowed. Mutation size violations become charged no-ops. Full parent and offspring fitness is always training-only.

For subtree output `s`, donor output `t`, root sensitivity `J`, residual `r`:

`delta = J * (t-s)`; `score = mean(2*r*delta - delta**2)`.

This is a **local prediction of MSE reduction**, exact only for affine contexts away from numerical guards. In `sin(g(x))`, `J=cos(g(x))`; raw correlation between a proposed inner expression and the final residual is insufficient. We never assume a predicted positive change guarantees actual improvement.

## Budget

Budget is the total number of training-set vector passes. Charge one unit for each complete-expression fitness evaluation, each donor subtree semantics probe, and each full context/sensitivity pass. Cached parent predictions can be reused; duplicate donor probes still cost a unit. Initialization is charged. Guided operations reserve the final offspring fitness evaluation. At the tail, reduce probe count or fall back to ordinary crossover. Every run consumes exactly the configured budget, even if an exact solution appears early.

This definition is conservative for small donor subtrees, but it does not equate FLOPs. Report separately: fitness evaluations, probes, context passes, expanded node × training-sample evaluations (includes repeated work), search time, guidance time, duplicate offspring and improvements. Context derivative/scoring operations are not captured by node counts; their cost is captured in search/guidance time. Variation time includes guidance time and must not be added to it. No uncharged train evaluation is performed by tracing. Final two holdout passes are outside search budget and never feed back into search.

## Data and reporting

Five preselected synthetic tasks: polynomial `x0³+x0²+x0`; additive `sin(x0)+x1²`; interaction `x0*x1+sin(x0)`; composition `sin(x0²)`; additive with iid Gaussian training noise, standard deviation 0.1. Train and interpolation test inputs uniform [-1,1]^d; extrapolation uses every coordinate in [-2,-1] or [1,2]. Holdouts are noise-free latent targets. These are modest, custom tasks, not official benchmark reproductions. The narrow in-domain interval makes composition relatively easy; wider oscillatory domains are a later independent test.

Default: fixed data seed 20261104, 128 training samples, 512 samples per holdout, ten search seeds 0..9, population 64, budget 3000. Runs are sequential on one CPU process; arm order deterministically shuffled per task/seed. This evaluates optimizer variability on one split, not variability over data splits.

Primary descriptive outcome: paired test-MSE difference `method - gp` per seed, median and wins/ties/losses (absolute tolerance 1e-12). Also report train/test/extrapolation error, IQR, expression complexity, time and work. No statistical significance or global superiority claim. Selection uses training MSE + parsimony; the training MSE of the incumbent need not decrease monotonically, but its objective does.

## Execution sequence and decision rule

1. Verify numerical semantics, budget accounting, deterministic seeds and absence of holdout leakage.
2. Run baseline-only pilot (5 tasks × 3 seeds, budget 1200). Inspect duplicate/improving-offspring rates and fitness/variation times to establish measured search behavior.
3. Run the fixed benchmark across all three arms without tuning on pilot/holdout outcomes. If core correctness changes are required, retain old evidence and rerun with a new manifest.
4. If guidance loses, report it. If it wins on some tasks, retain the task-specific conclusion and inspect the shuffled control, work and runtime. Do not generalize to modern SR systems or claim novelty.
5. Next: independently test grammar adaptation, working memory, and serial symbolic modules with factorial ablations; expand datasets/splits, baselines and compute budgets before a paper claim.
