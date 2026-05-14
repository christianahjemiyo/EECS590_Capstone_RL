from __future__ import annotations

"""Lightweight Bayesian-style hyperparameter tuning helpers.

This module keeps the tuning logic simple and interpretable for the capstone.
It does not depend on an external optimization package. Instead, it uses a
small radial-basis surrogate and an upper-confidence-style acquisition rule to
propose promising next candidates.

That is enough to demonstrate Bayesian hyperparameter tuning as a tool while
remaining aligned with the compact tabular setting of the project.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SearchBounds:
    """Continuous search box for a four-parameter reward-design candidate."""

    no_reward: tuple[float, float]
    late_penalty: tuple[float, float]
    early_penalty: tuple[float, float]
    cost_scale: tuple[float, float]

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        mins = np.array(
            [
                self.no_reward[0],
                self.late_penalty[0],
                self.early_penalty[0],
                self.cost_scale[0],
            ],
            dtype=float,
        )
        maxs = np.array(
            [
                self.no_reward[1],
                self.late_penalty[1],
                self.early_penalty[1],
                self.cost_scale[1],
            ],
            dtype=float,
        )
        return rng.uniform(mins, maxs, size=(n, 4))

    def clip(self, x: np.ndarray) -> np.ndarray:
        mins = np.array(
            [
                self.no_reward[0],
                self.late_penalty[0],
                self.early_penalty[0],
                self.cost_scale[0],
            ],
            dtype=float,
        )
        maxs = np.array(
            [
                self.no_reward[1],
                self.late_penalty[1],
                self.early_penalty[1],
                self.cost_scale[1],
            ],
            dtype=float,
        )
        return np.clip(x, mins, maxs)


def as_tag(x: np.ndarray) -> dict[str, float]:
    """Convert one search vector into the project's reward-tuning tag shape."""

    vals = np.asarray(x, dtype=float)
    return {
        "no_reward": float(vals[0]),
        "late_penalty": float(vals[1]),
        "early_penalty": float(vals[2]),
        "cost_scale": float(vals[3]),
    }


class RBFSurrogate:
    """Small radial-basis surrogate for Bayesian-style tuning.

    This is intentionally modest. The capstone does not need a large surrogate
    library to demonstrate that hyperparameter search can be guided by prior
    observations rather than brute-force enumeration.
    """

    def __init__(self, length_scale: float = 0.75, ridge: float = 1e-6) -> None:
        self.length_scale = float(length_scale)
        self.ridge = float(ridge)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.k_xx_inv: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFSurrogate":
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        K = self._kernel(self.X, self.X)
        K = K + self.ridge * np.eye(len(self.X), dtype=float)
        self.k_xx_inv = np.linalg.inv(K)
        self.alpha = self.k_xx_inv @ self.y
        return self

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.X is None or self.alpha is None or self.k_xx_inv is None:
            raise RuntimeError("Surrogate must be fit before prediction.")
        X_star = np.asarray(X_star, dtype=float)
        K_star = self._kernel(X_star, self.X)
        mean = K_star @ self.alpha
        prior = np.ones(len(X_star), dtype=float)
        variance = prior - np.sum((K_star @ self.k_xx_inv) * K_star, axis=1)
        variance = np.maximum(variance, 1e-9)
        return mean, np.sqrt(variance)

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        diff = A[:, None, :] - B[None, :, :]
        sqdist = np.sum(diff * diff, axis=2)
        return np.exp(-0.5 * sqdist / (self.length_scale ** 2))


def propose_next(
    X: np.ndarray,
    y: np.ndarray,
    bounds: SearchBounds,
    *,
    rng: np.random.Generator,
    n_candidates: int = 256,
    exploration_weight: float = 1.5,
) -> np.ndarray:
    """Propose the next candidate using an upper-confidence-style score."""

    if len(X) < 3:
        return bounds.sample(rng, n=1)[0]

    surrogate = RBFSurrogate().fit(X, y)
    candidates = bounds.sample(rng, n=n_candidates)
    mean, std = surrogate.predict(candidates)
    acquisition = mean + exploration_weight * std
    return candidates[int(np.argmax(acquisition))]
