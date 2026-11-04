"""Offline acceptance demo built exclusively from recorded runs (no live training)."""

import csv
import json
from pathlib import Path
import numpy as np
from .data import target
from .tree import Node, evaluate


def build_demo(out):
    out = Path(out)
    runs = [json.loads(s) for s in (out / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    traces = list(csv.DictReader((out / "traces.csv").open(encoding="utf-8")))
    x = np.linspace(-2, 2, 161)
    tasks = list(dict.fromkeys(r["task"] for r in runs))
    payload = {"x": x.tolist(), "targets": {}, "runs": []}
    for task in tasks:
        features = 1 if task in ("composition", "polynomial") else 2
        data = np.full((len(x), features), .5)
        data[:, 0] = x
        payload["targets"][task] = target(task, data).tolist()
        for r in [r for r in runs if r["task"] == task]:
            row = {k: r[k] for k in ("task", "method", "seed", "expression", "train_mse", "test_mse",
                    "extrap_mse", "elapsed_seconds", "fitness_evaluations", "semantic_probes", "context_passes", "budget_used")}
            row["curve"] = evaluate(Node.from_dict(r["tree"]), data).tolist()
            row["trace"] = [[int(t["budget_used"]), float(t["best_train_mse"])] for t in traces
                            if t["task"] == task and t["method"] == r["method"] and int(t["seed"]) == r["seed"]]
            payload["runs"].append(row)
    template = Path(__file__).with_name("demo_template.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    (out / "demo.html").write_text(template.replace("__DATA__", encoded), encoding="utf-8")


if __name__ == "__main__":
    import sys
    build_demo(sys.argv[1])
