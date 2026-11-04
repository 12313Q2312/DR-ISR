# Neural comparison protocol v1 — fixed before measured training

This extends the GP-only v0.1 into an end-to-end synthetic comparison. It does not reproduce the paper's 16 real-data results or claim a new neural architecture.

## Methods and provenance

- **TN-VSR (controlled reproduction)**: the author's polynomial-neuron parameterization `sum_k W_k x^k+b`, Sigmoid hidden activation, discovered polynomial powers. Independent safe implementation of `Task_based_neurons/utils/nolinear_neuron.py`; coefficients found by search become independent trainable weights. Homogeneous polynomial structure search is reimplemented with bounded trees and explicit accounting rather than running the author's original script unchanged.
- **TN-DR (experimental)**: identical parameterization, grammar and search engine; residual/context-guided donor selection changes only crossover choice. This version remains polynomial; adaptive grammar, working memory and general nonpolynomial serial neuron discovery are not included.
- **KAN**: unmodified `Blealtan/efficient-kan/src/efficient_kan/kan.py`, Git blob `f27c1b7649e7ada4e329c35a7c14ba677a09cb1d`, with adjacent MIT license. Cubic B-splines, grid size 5, SiLU base, vendor default scaling; fixed uniform grids, no grid updates, symbolic branch, pruning or regularization. This is not a full pykan reproduction and not necessarily the implementation in the TN paper.
- **MLP**: Linear–Sigmoid–Linear sanity baseline.

All models have one hidden layer and scalar output. TN powers apply to both layers. A shared parameter **ceiling** of 256 sets each model's largest allowable hidden width analytically. Actual counts and widths are reported; counts are not exactly equal. KAN edge functions and TN aggregation functions have different inductive biases. No claim of identical FLOPs or optimal architecture tuning is made.

## Data, search and optimization

Use all 10 groups in the author's existing epsilon_low.h5 and epsilon_high.h5: 20 synthetic tasks, each 600 examples, 10 features. Preserve/import raw data CSVs and original HDF5 hashes. Their existing `x_train/y_train` names do not define the benchmark split: split before all processing with fixed seed 20261104 into 360 training, 120 validation, 120 test. Each group uses the same index permutation for every method and search seed. Features map training min/max to [-1,1]; labels use training mean and population standard deviation. Validation/test never determine scalers or searched powers.

Search space: a single scalar polynomial syntax applied identically to each input coordinate, prediction `mean_j p(x_j)`. Grammar `+,-,*`, variable x, integer constants [-2,2]. A mean instead of sum changes coefficient scale, not polynomial support. Maximum degree 8, depth 5, nodes 47; nonconstant expressions only; initial depth up to 3; population 48; tournament 5; crossover .3, mutation .6, reproduction .1; training MSE + 1e-5 per tree node. Constants and bounds differ from the author's original GP; this is a controlled mechanism study, not a literal rerun. Same initial population and budget for both TN searches.

1500 charged training-set vector evaluations per search, including initialization, repeated probes and full context passes. A pass operates across all training samples and feature coordinates; count node×sample×feature work too. Up to four donor proposals; score `mean(2*r*delta-delta²)`, `delta=sum_j J_j*(t_j-s_j)` with `J` including the output mean factor. No local coefficient fitting. The selected nonzero polynomial powers define the neurons; original numerical coefficients are not reused as fitted weights.

Seeds 10,20,30,40,50. Reset model RNG to the paired seed; same discovered powers yield same TN initialization. CPU, two PyTorch threads. Adam lr=.01, full batch, 800 updates for all models; no early stopping or test-based tuning. Check validation at update 0, every 25 updates and final update. Keep the checkpoint with lowest validation MSE. Retain/report any nonfinite failures; never silently drop seeds. Deterministic algorithms enabled; small cross-platform floating-point/lstsq variation may remain.

Total 20 tasks × 5 seeds × 4 methods = **400** independent training runs, plus 200 structure searches. Search, model construction and training/validation time are separately measured; their sum is the reported end-to-end time. Common preprocessing and final holdout inference excluded. KAN/MLP have zero search cost, not fictitious matched search work. Equal update/parameter ceilings do not create equal total compute.

## Reporting, verification and interpretation

Primary metric: held-out normalized MSE (raw MSE divided by training-label variance); also retain raw MSE, parameter count, selected powers, validation step, all training curves, predictions, final model weights and every search counter. These targets contain author-provided noise; no clean-target or out-of-domain extrapolation claim. Per-task median/IQR and paired wins/ties/losses, absolute normalized-MSE tie tolerance 1e-8. Summaries descriptive, not significance tests.

Tests must verify polynomial parameterization/gradients, B-spline partition/finite gradients, parameter limits, vector context derivatives, budget tails, shared initialization, train-only normalization and test-independent checkpoint selection. Archived weights must reproduce predictions/errors on archived data. Record environment, source hashes, source data hashes, split indices and file checksums.

Run smoke for correctness only, then execute the fixed main configuration without tuning from validation/test outcomes. If a correctness bug is fixed, retain the old run and rerun under a new output name. A weak score is a reported result, not grounds for changing the fixed comparison. Future multi-split, equal-time, grid/tuning sweeps and real-data replication require separate protocols.
