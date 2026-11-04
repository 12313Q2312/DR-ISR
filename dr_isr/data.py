"""Small synthetic tasks, not official Nguyen/SRBench replications."""

from dataclasses import dataclass
import hashlib
import numpy as np

TASKS = ("polynomial", "additive", "interaction", "composition", "noisy_additive")


def target(name, x):
    if name == "polynomial":
        return x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0]
    if name in ("additive", "noisy_additive"):
        return np.sin(x[:, 0]) + x[:, 1] ** 2
    if name == "interaction":
        return x[:, 0] * x[:, 1] + np.sin(x[:, 0])
    if name == "composition":
        return np.sin(x[:, 0] ** 2)
    raise ValueError(f"Unknown task: {name}")


@dataclass
class Dataset:
    name: str
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    extrap_x: np.ndarray
    extrap_y: np.ndarray

    @property
    def fingerprint(self):
        digest = hashlib.sha256()
        for value in vars(self).values():
            digest.update(value.encode() if isinstance(value, str)
                          else np.ascontiguousarray(value, dtype="<f8").tobytes())
        return digest.hexdigest()


def make_dataset(name, seed, n_train=128, n_test=512):
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}")
    if n_train < 2 or n_test < 2 or seed < 0:
        raise ValueError("Invalid data sizes/seed")
    n_features = 1 if name in ("polynomial", "composition") else 2
    # Distinct train/test/extrapolation/noise streams, independent of search RNG.
    streams = np.random.SeedSequence([seed, TASKS.index(name), 9201]).spawn(4)
    train_x = np.random.default_rng(streams[0]).uniform(-1, 1, (n_train, n_features))
    test_x = np.random.default_rng(streams[1]).uniform(-1, 1, (n_test, n_features))
    rng = np.random.default_rng(streams[2])
    extrap_x = rng.uniform(1, 2, (n_test, n_features)) * rng.choice([-1, 1], (n_test, n_features))
    train_y = target(name, train_x)
    if name == "noisy_additive":
        train_y = train_y + np.random.default_rng(streams[3]).normal(0, 0.1, n_train)
    # Holdouts are clean latent targets, including for the noisy-training task.
    return Dataset(name, train_x, train_y, test_x, target(name, test_x),
                   extrap_x, target(name, extrap_x))
