"""Recompute every stored model's errors and verify paired budgets and file hashes."""

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from dr_isr.tree import Node, evaluate
from dr_isr.data import Dataset


def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    specification = manifest["specification"]
    config = specification["gp"]
    rows = [json.loads(s) for s in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {(t, s, m) for t in specification["tasks"] for s in specification["seeds"]
                for m in specification["methods"]}
    actual = [(r["task"], r["seed"], r["method"]) for r in rows]
    assert len(actual) == len(set(actual)) and set(actual) == expected, "Missing/duplicate runs"
    initial_hashes = {}
    for task in specification["tasks"]:
        splits = [np.loadtxt(folder / "datasets" / f"{task}_{split}.csv", delimiter=",", skiprows=1, ndmin=2)
                  for split in ("train", "test", "extrap")]
        values = [value for split in splits for value in (split[:,:-1], split[:,-1])]
        assert Dataset(task,*values).fingerprint == manifest["datasets"][task], "Dataset fingerprint mismatch"
    for row in rows:
        assert row["budget_used"] == config["budget"], "Unequal budget"
        assert row["budget_used"] == sum(row[k] for k in ("fitness_evaluations", "semantic_probes", "context_passes"))
        pair = row["task"], row["seed"]
        assert initial_hashes.setdefault(pair, row["initial_population_sha256"]) == row["initial_population_sha256"]
        assert row["data_sha256"] == manifest["datasets"][row["task"]]
        tree = Node.from_dict(row["tree"])
        for split in ("train", "test", "extrap"):
            data = np.loadtxt(folder / "datasets" / f"{row['task']}_{split}.csv", delimiter=",", skiprows=1, ndmin=2)
            mse = float(np.mean((evaluate(tree, data[:,:-1])-data[:,-1]) ** 2))
            assert np.isclose(mse,row[split+"_mse"],rtol=1e-11,atol=1e-25), (pair,split,mse,row[split+"_mse"])
    checksums = json.loads((folder / "checksums.json").read_text(encoding="utf-8"))
    for path, expected_hash in checksums.items():
        assert hashlib.sha256((folder / path).read_bytes()).hexdigest() == expected_hash, path
    print(f"PASS: {len(rows)} runs, all budgets/initializations, {len(rows)*3} model-error replays, {len(checksums)} file checksums")


if __name__ == "__main__":
    verify(sys.argv[1])
