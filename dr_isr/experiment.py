"""Paired multi-seed experiments, raw evidence, descriptive summaries and plots."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import numpy as np
from .data import TASKS, make_dataset
from .gp import Config, METHODS, search
from .tree import Node, evaluate


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git_state():
    def command(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[1], check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {"commit": command("rev-parse", "HEAD"), "dirty": command("status", "--porcelain")}


def run(config_path, output):
    specification = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = Config(**specification["gp"])
    seeds = specification["seeds"]
    tasks = specification["tasks"]
    methods = specification.get("methods", list(METHODS))
    if (not seeds or len(seeds) != len(set(seeds)) or
            any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("Use distinct, nonnegative integer seeds")
    if not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks):
        raise ValueError("Use distinct known tasks")
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Use distinct known methods")
    out = Path(output)
    # Never overwrite an earlier experiment, including an incomplete run.
    out.mkdir(parents=True, exist_ok=False)
    (out / "datasets").mkdir()
    source_files = [*Path(__file__).parent.glob("*.py"), Path(__file__).parents[1] / "run.py"]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(),
                "specification": specification, "git": git_state(),
                "python": sys.version, "numpy": np.__version__,
                "platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                "budget_unit": "one training-set vector evaluation: full fitness, donor probe, or context pass",
                "timing": "perf_counter inside search; excludes data generation, final holdouts, reporting, tests",
                "datasets": {}}
    write_json(out / "manifest.json", manifest)
    rows = []
    traces = []
    for task in tasks:
        data = make_dataset(task, **specification["data"])
        manifest["datasets"][task] = data.fingerprint
        for split in ("train", "test", "extrap"):
            x, y = getattr(data, split + "_x"), getattr(data, split + "_y")
            np.savetxt(out / "datasets" / f"{task}_{split}.csv", np.column_stack([x, y]),
                       delimiter=",", header=",".join([f"x{i}" for i in range(x.shape[1])] + ["y"]),
                       comments="", fmt="%.17g")
        for seed in seeds:
            order = list(methods)
            random.Random(f"{task}/{seed}/order-v1").shuffle(order)
            pair_initial_hash = None
            for method in order:
                result = search(data.train_x, data.train_y, method, seed, config)
                initial = json.dumps(result.pop("initial_population"), sort_keys=True).encode()
                initial_hash = hashlib.sha256(initial).hexdigest()
                if pair_initial_hash is not None and pair_initial_hash != initial_hash:
                    raise RuntimeError("Paired initial populations differ")
                pair_initial_hash = initial_hash
                result["initial_population_sha256"] = initial_hash
                tree = Node.from_dict(result["tree"])
                # Exactly one post-search evaluation on each holdout. No selection here.
                for split in ("test", "extrap"):
                    y = getattr(data, split + "_y")
                    error = y - evaluate(tree, getattr(data, split + "_x"))
                    result[split + "_mse"] = float(np.mean(error ** 2))
                    result[split + "_rmse"] = float(np.sqrt(np.mean(error ** 2)))
                result["task"] = task
                result["data_sha256"] = data.fingerprint
                result["run_order"] = order.index(method)
                for point in result.pop("trace"):
                    traces.append({"task": task, "method": method, "seed": seed, **point})
                rows.append(result)
                with (out / "runs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, allow_nan=False) + "\n")
                print(f"{task:16} seed={seed:2} {method:12} "
                      f"test MSE={result['test_mse']:.5g} budget={result['budget_used']} "
                      f"time={result['elapsed_seconds']:.2f}s", flush=True)
    with (out / "traces.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "manifest.json", manifest)
    summarize(rows, out)
    plot(traces, tasks, methods, config.budget, out)
    from .demo import build_demo
    build_demo(out)
    checksums = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "checksums.json", checksums)
    return rows


def summarize(rows, out):
    summaries = []
    paired = []
    lines = ["# Measured results", "", "Descriptive pilot evidence; no statistical or novelty claim.", "",
             "Selected expression minimizes training MSE + node penalty. Holdouts are never used for search.", "",
             "Noisy task: training labels contain N(0,0.1²) noise; both holdouts use the clean target.", "",
             "| Task | Method | Seeds | Train MSE median | Test MSE median [Q1,Q3] | Extrap MSE median | Seconds median | Fitness / probes / context median |", 
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for task in dict.fromkeys(r["task"] for r in rows):
        for method in METHODS:
            group = [r for r in rows if r["task"] == task and r["method"] == method]
            if not group:
                continue
            summary = {"task": task, "method": method, "n": len(group)}
            for metric in ("train_mse", "test_mse", "test_rmse", "extrap_mse", "elapsed_seconds",
                           "fitness_evaluations", "semantic_probes", "context_passes",
                           "node_sample_evaluations", "nodes", "improving_offspring",
                           "duplicate_offspring", "guidance_seconds"):
                values = [r[metric] for r in group]
                summary[metric] = dict(zip(("q1", "median", "q3"), map(float, np.quantile(values, [.25,.5,.75]))))
            summaries.append(summary)
            m = lambda key: summary[key]["median"]
            q = summary["test_mse"]
            lines.append(f"| {task} | {method} | {len(group)} | {m('train_mse'):.5g} | "
                         f"{q['median']:.5g} [{q['q1']:.4g},{q['q3']:.4g}] | {m('extrap_mse'):.5g} | "
                         f"{m('elapsed_seconds'):.3f} | {m('fitness_evaluations'):g} / "
                         f"{m('semantic_probes'):g} / {m('context_passes'):g} |")
        base = {r["seed"]: r for r in rows if r["task"] == task and r["method"] == "gp"}
        for method in ("residual_gp", "shuffled_gp"):
            group = [r for r in rows if r["task"] == task and r["method"] == method]
            pairs = [(r, base[r["seed"]]) for r in group if r["seed"] in base]
            if not pairs:
                continue
            deltas = [r["test_mse"] - b["test_mse"] for r, b in pairs]
            paired.append({"task": task, "method": method, "paired_seeds": len(pairs),
                           "median_test_mse_difference": float(np.median(deltas)),
                           "wins": sum(d < -1e-12 for d in deltas),
                           "ties": sum(abs(d) <= 1e-12 for d in deltas),
                           "losses": sum(d > 1e-12 for d in deltas)})
    lines += ["", "## Paired comparisons against GP", "", "Negative median difference favors the compared method. A difference of medians is not a median paired difference.", "",
              "| Task | Method | Paired seeds | Median test MSE difference | Wins / ties / losses |",
              "|---|---|---:|---:|---|"]
    for p in paired:
        lines.append(f"| {p['task']} | {p['method']} | {p['paired_seeds']} | "
                     f"{p['median_test_mse_difference']:.6g} | {p['wins']} / {p['ties']} / {p['losses']} |")
    lines += ["", "## Budget and limitations", "",
              "Every training vector pass is charged, including donor probes and context passes. "
              "These units differ in tree size and derivative overhead; inspect node-sample work and wall time too. "
              "This is a conservative accounting protocol, not a claim of identical FLOPs or equal runtime.", "",
              "Guidance uses a local linearization. Multiplication/composition can invalidate its prediction. "
              "The shuffled-residual control measures whether per-sample residual alignment adds value at similar overhead. "
              "All budgets are exhausted even after a solution is found. Duplicate probes and unchanged offspring are charged.", "",
              "Timing is machine-dependent. Small synthetic tasks and a small seed sample cannot establish general superiority. "
              "No hyperparameter tuning or selection is performed using holdout metrics."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "summary.json", {"groups": summaries, "paired": paired})


def plot(traces, tasks, methods, budget, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(min(t["budget_used"] for t in traces), budget, 60)
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for method in methods:
            subset = [t for t in traces if t["task"] == task and t["method"] == method]
            series = []
            for seed in sorted({t["seed"] for t in subset}):
                points = [p for p in subset if p["seed"] == seed]
                xs = np.array([p["budget_used"] for p in points])
                ys = np.array([p["best_train_mse"] for p in points])
                # Previous observed incumbent, without interpolation from future checkpoints.
                ids = np.clip(np.searchsorted(xs, grid, side="right") - 1, 0, len(xs) - 1)
                series.append(ys[ids])
            q1, med, q3 = np.quantile(series, [.25,.5,.75], axis=0)
            ax.plot(grid, np.maximum(med, 1e-14), label=method)
            ax.fill_between(grid, np.maximum(q1, 1e-14), np.maximum(q3, 1e-14), alpha=.14)
        ax.set_title(task)
        ax.set_xlabel("Charged training vector passes")
        ax.set_yscale("log")
        ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Incumbent training MSE (median, IQR)")
    axes[0,-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.svg")
    fig.savefig(out / "learning_curves.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or "results/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run(args.config, output)
    print(f"Evidence saved in {output}")


if __name__ == "__main__":
    main()
