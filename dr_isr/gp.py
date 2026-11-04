"""Steady-state GP and context-aware residual-guided donor selection."""

from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from .tree import Node, context, evaluate, random_tree

METHODS = ("gp", "residual_gp", "shuffled_gp")


@dataclass(frozen=True)
class Config:
    budget: int = 3000
    population: int = 64
    tournament: int = 5
    max_depth: int = 6
    max_nodes: int = 63
    init_depth: int = 3
    mutation_depth: int = 2
    crossover_rate: float = 0.75
    guide_candidates: int = 4
    parsimony: float = 1e-4
    trace_every: int = 100

    def __post_init__(self):
        ints = (self.budget, self.population, self.tournament, self.max_depth,
                self.max_nodes, self.init_depth, self.mutation_depth,
                self.guide_candidates, self.trace_every)
        if any(type(v) is not int or v < 1 for v in ints):
            raise ValueError("Integer GP settings must be positive")
        if self.population < 2 or self.budget < self.population:
            raise ValueError("Budget must cover the population (at least 2)")
        if self.init_depth > self.max_depth or 2 ** (self.init_depth + 1) - 1 > self.max_nodes:
            raise ValueError("Initial trees may exceed size/depth limits")
        if not 0 <= self.crossover_rate <= 1 or not np.isfinite(self.parsimony) or self.parsimony < 0:
            raise ValueError("Invalid crossover probability/parsimony")


@dataclass
class Ledger:
    limit: int
    fitness_evaluations: int = 0
    semantic_probes: int = 0
    context_passes: int = 0
    node_sample_evaluations: int = 0

    @property
    def used(self):
        return self.fitness_evaluations + self.semantic_probes + self.context_passes

    @property
    def remaining(self):
        return self.limit - self.used

    def charge(self, kind, tree, n_samples):
        if self.remaining <= 0:
            raise RuntimeError("Evaluation budget exhausted")
        setattr(self, kind, getattr(self, kind) + 1)
        self.node_sample_evaluations += tree.size * n_samples


@dataclass
class Individual:
    tree: Node
    prediction: np.ndarray
    mse: float
    objective: float


def residual_score(residual, subtree_values, donor_values, sensitivity):
    """Predicted decrease in training MSE under a first-order root-output model.

    delta = J * (t-s), score = mean(2*r*delta - delta^2).
    This is exact in affine contexts away from numerical guards, approximate otherwise.
    No fitted coefficient or extra grammar operator is introduced.
    """
    delta = sensitivity * (donor_values - subtree_values)
    return float(np.mean(2 * residual * delta - delta * delta))


def valid(tree, config):
    return tree.depth <= config.max_depth and tree.size <= config.max_nodes


def legal_donor(parent, path, donor, rng, config):
    """Same uniform-over-valid-path proposal distribution for all methods."""
    paths = [p for p in donor.paths() if valid(parent.replace(path, donor.at(p)), config)]
    if not paths:
        return parent.at(path)
    return donor.at(paths[int(rng.integers(len(paths)))])


def search(train_x, train_y, method, seed, config):
    """Search sees training data only; holdouts cannot influence selection."""
    if method not in METHODS or seed < 0:
        raise ValueError("Unknown method or invalid seed")
    if train_x.ndim != 2 or train_y.shape != (len(train_x),) or not len(train_x) or not train_x.shape[1]:
        raise ValueError("Expected nonempty X[n,d] and y[n]")
    if not np.all(np.isfinite(train_x)) or not np.all(np.isfinite(train_y)):
        raise ValueError("Nonfinite training data")
    start = perf_counter()
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    ledger = Ledger(config.budget)
    diagnostics = dict(offspring=0, improving_offspring=0, accepted_offspring=0,
                       duplicate_offspring=0, guided_events=0, predicted_positive=0,
                       guided_actual_positive=0, guided_predicted_positive_actual_positive=0,
                       variation_seconds=0.0, fitness_seconds=0.0,
                       guidance_seconds=0.0)

    def assess(tree):
        begin = perf_counter()
        ledger.charge("fitness_evaluations", tree, len(train_y))
        pred = evaluate(tree, train_x)
        mse = float(np.mean((train_y - pred) ** 2))
        diagnostics["fitness_seconds"] += perf_counter() - begin
        return Individual(tree, pred, mse, mse + config.parsimony * tree.size)

    population = [assess(random_tree(init_rng, train_x.shape[1],
                                    int(init_rng.integers(1, config.init_depth + 1)),
                                    full=i % 2 == 0)) for i in range(config.population)]
    best = min(population, key=lambda p: p.objective)
    initial = [p.tree.to_dict() for p in population]
    trace = []

    def record():
        trace.append({"budget_used": ledger.used, "fitness_evaluations": ledger.fitness_evaluations,
                      "node_sample_evaluations": ledger.node_sample_evaluations,
                      "best_train_mse": best.mse, "best_objective": best.objective,
                      "elapsed_seconds": perf_counter() - start})

    record()
    next_trace = ledger.used + config.trace_every

    def tournament():
        candidates = rng.integers(len(population), size=config.tournament)
        return min((population[int(i)] for i in candidates), key=lambda p: p.objective)

    while ledger.remaining:
        begin = perf_counter()
        parent = tournament()
        paths = list(parent.tree.paths())
        path = paths[int(rng.integers(len(paths)))]
        predicted = None
        if rng.random() < config.crossover_rate:
            donor_parent = tournament().tree
            if method != "gp" and ledger.remaining >= 3:
                guide_start = perf_counter()
                ledger.charge("context_passes", parent.tree, len(train_y))
                subtree, sensitivity = context(parent.tree, path, train_x)
                residual = train_y - parent.prediction
                if method == "shuffled_gp":
                    residual = rng.permutation(residual)
                best_score = -np.inf
                chosen = parent.tree.at(path)
                # Reserve one complete fitness evaluation for the selected child.
                for _ in range(min(config.guide_candidates, ledger.remaining - 1)):
                    candidate = legal_donor(parent.tree, path, donor_parent, rng, config)
                    ledger.charge("semantic_probes", candidate, len(train_y))
                    score = residual_score(residual, subtree, evaluate(candidate, train_x), sensitivity)
                    if score > best_score:
                        chosen, best_score = candidate, score
                predicted = best_score
                diagnostics["guided_events"] += 1
                diagnostics["predicted_positive"] += int(best_score > 1e-12)
                diagnostics["guidance_seconds"] += perf_counter() - guide_start
                child = parent.tree.replace(path, chosen)
            else:
                child = parent.tree.replace(path, legal_donor(parent.tree, path, donor_parent, rng, config))
        else:
            subtree = random_tree(rng, train_x.shape[1], config.mutation_depth)
            child = parent.tree.replace(path, subtree)
            if not valid(child, config):
                child = parent.tree
        diagnostics["variation_seconds"] += perf_counter() - begin
        offspring = assess(child)
        diagnostics["offspring"] += 1
        diagnostics["duplicate_offspring"] += int(child == parent.tree)
        improved = offspring.mse < parent.mse - 1e-12
        diagnostics["improving_offspring"] += int(improved)
        if predicted is not None:
            diagnostics["guided_actual_positive"] += int(improved)
            diagnostics["guided_predicted_positive_actual_positive"] += int(improved and predicted > 1e-12)
        worst = max(range(len(population)), key=lambda i: population[i].objective)
        if offspring.objective < population[worst].objective:
            population[worst] = offspring
            diagnostics["accepted_offspring"] += 1
        if offspring.objective < best.objective:
            best = offspring
        if ledger.used >= next_trace:
            record()
            next_trace = ledger.used + config.trace_every
    if trace[-1]["budget_used"] != ledger.used:
        record()
    return {"method": method, "seed": seed, "tree": best.tree.to_dict(),
            "expression": str(best.tree), "train_mse": best.mse,
            "objective": best.objective, "nodes": best.tree.size,
            "budget_used": ledger.used, **asdict(ledger), **diagnostics,
            "elapsed_seconds": perf_counter() - start, "trace": trace,
            "initial_population": initial}
