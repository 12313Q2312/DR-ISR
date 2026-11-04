"""Immutable expression trees; protected numerical semantics are shared by all arms."""

from dataclasses import dataclass
import math
import numpy as np

OPS = {"add": 2, "sub": 2, "mul": 2, "div": 2, "sin": 1, "cos": 1}
LIMIT = 1e6
EPS = 1e-6


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple = ()
    value: float = 0.0

    def __post_init__(self):
        if self.op not in (*OPS, "x", "const"):
            raise ValueError(f"Unknown operator: {self.op}")
        if len(self.children) != OPS.get(self.op, 0):
            raise ValueError("Wrong operator arity")
        if not math.isfinite(self.value):
            raise ValueError("Non-finite terminal")
        if self.op == "x" and (self.value < 0 or int(self.value) != self.value):
            raise ValueError("Variable index must be a nonnegative integer")

    @property
    def size(self):
        return 1 + sum(c.size for c in self.children)

    @property
    def depth(self):
        return 0 if not self.children else 1 + max(c.depth for c in self.children)

    def paths(self, prefix=()):
        yield prefix
        for i, child in enumerate(self.children):
            yield from child.paths(prefix + (i,))

    def at(self, path):
        return self if not path else self.children[path[0]].at(path[1:])

    def replace(self, path, subtree):
        if not path:
            return subtree
        children = list(self.children)
        children[path[0]] = children[path[0]].replace(path[1:], subtree)
        return Node(self.op, tuple(children), self.value)

    def to_dict(self):
        return {"op": self.op, "value": self.value,
                "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, data):
        return cls(data["op"], tuple(cls.from_dict(c) for c in data["children"]),
                   data["value"])

    def __str__(self):
        if self.op == "x":
            return f"x{int(self.value)}"
        if self.op == "const":
            return repr(self.value)
        if len(self.children) == 1:
            return f"{self.op}({self.children[0]})"
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "pdiv"}[self.op]
        return f"({self.children[0]} {symbol} {self.children[1]})"


def raw_apply(op, values):
    a = values[0]
    if op == "sin":
        return np.sin(a)
    if op == "cos":
        return np.cos(a)
    b = values[1]
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    out = np.ones_like(a)
    np.divide(a, b, out=out, where=np.abs(b) >= EPS)
    return out


def evaluate(tree, x, cache=None, path=()):
    """Each node produces one length-n vector; optional cache is per-call only."""
    if tree.op == "x":
        out = x[:, int(tree.value)]
    elif tree.op == "const":
        out = np.full(x.shape[0], tree.value, dtype=float)
    else:
        args = [evaluate(c, x, cache, path + (i,)) for i, c in enumerate(tree.children)]
        out = np.clip(raw_apply(tree.op, args), -LIMIT, LIMIT)
    if cache is not None:
        cache[path] = out
    return out


def context(tree, path, x):
    """Return selected subtree output and root sensitivity df/ds (one forward pass).

    Local derivatives use the same protected division and clipping as evaluate.
    Clipping/threshold kinks use derivative zero. No test data is used here.
    """
    cache = {}
    evaluate(tree, x, cache)
    derivative = np.ones(x.shape[0])
    cursor = ()
    node = tree
    for child_index in path:
        args = [cache[cursor + (i,)] for i in range(len(node.children))]
        a = args[0]
        if node.op == "sin":
            local = np.cos(a)
        elif node.op == "cos":
            local = -np.sin(a)
        elif node.op == "add":
            local = np.ones_like(a)
        elif node.op == "sub":
            local = np.full_like(a, 1.0 if child_index == 0 else -1.0)
        elif node.op == "mul":
            local = args[1 - child_index]
        else:
            b = args[1]
            safe = np.abs(b) >= EPS
            local = np.zeros_like(a)
            if child_index == 0:
                np.divide(1.0, b, out=local, where=safe)
            else:
                np.divide(-a, b * b, out=local, where=safe)
        raw = raw_apply(node.op, args)
        local = np.where(np.abs(raw) < LIMIT, local, 0.0)
        # Bound the heuristic's sensitivity only, never the evaluated expression.
        derivative = np.clip(derivative * local, -1e12, 1e12)
        cursor += (child_index,)
        node = node.children[child_index]
    return cache[path], derivative


def random_tree(rng, n_features, max_depth, full=False):
    if max_depth <= 0 or (not full and rng.random() < 0.3):
        if rng.random() < 0.75:
            return Node("x", value=int(rng.integers(n_features)))
        return Node("const", value=float(rng.uniform(-2, 2)))
    op = str(rng.choice(list(OPS)))
    return Node(op, tuple(random_tree(rng, n_features, max_depth - 1, full)
                          for _ in range(OPS[op])))
