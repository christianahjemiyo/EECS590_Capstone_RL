from __future__ import annotations

"""Tabular world model for hospital readmission planning experiments.

The project already models hospital readmission planning as a compact tabular
Markov decision process. This module keeps the Version 3 extension aligned with
that setup: it learns or reconstructs transition dynamics and expected rewards
in the same discrete state-action space.

This is meant for decision-support simulation around modeled readmission risk.
It is intentionally simple and interpretable. It should not be read as a claim
that the learned transition model is a faithful clinical process model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    """Normalize a 3D state-action-next_state array along the last axis.

    A small helper keeps the main fitting methods easier to read. It also
    protects against division by zero if a state-action row is entirely empty.
    """

    totals = values.sum(axis=2, keepdims=True)
    safe_totals = np.where(totals > 0.0, totals, 1.0)
    return values / safe_totals


def _expected_rewards_from_transition_rewards(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Collapse transition-level rewards into expected state-action rewards."""

    if R.ndim == 2:
        return R.astype(float, copy=True)
    if R.ndim != 3:
        raise ValueError("Reward array must have shape (S, A) or (S, A, S).")
    return np.sum(P * R, axis=2)


def _coerce_transition(record: object) -> tuple[int, int, int, float, bool]:
    """Parse one trajectory item into a standard tabular transition tuple.

    Supported formats:
    - mapping with keys: state, action, next_state, reward, done(optional)
    - tuple/list: (state, action, next_state, reward) or
                  (state, action, next_state, reward, done)
    """

    if isinstance(record, Mapping):
        return (
            int(record["state"]),
            int(record["action"]),
            int(record["next_state"]),
            float(record["reward"]),
            bool(record.get("done", False)),
        )

    if isinstance(record, (tuple, list)):
        if len(record) == 4:
            s, a, s_next, reward = record
            done = False
        elif len(record) == 5:
            s, a, s_next, reward, done = record
        else:
            raise ValueError("Trajectory tuples must have length 4 or 5.")
        return int(s), int(a), int(s_next), float(reward), bool(done)

    raise TypeError("Unsupported trajectory record type.")


@dataclass
class TabularWorldModel:
    """Learned tabular transition and reward model.

    The learned model approximates:
    - P_hat(s' | s, a): next-state probabilities
    - R_hat(s, a): expected reward for taking action a in state s

    This is appropriate for the current capstone because the existing hospital
    readmission environment is itself a compact tabular MDP with modeled
    decision-support dynamics.
    """

    smoothing: float = 1.0
    transition_probs: np.ndarray | None = None
    reward_table: np.ndarray | None = None
    terminal_states: list[int] = field(default_factory=list)
    n_states: int = 0
    n_actions: int = 0
    fit_source: str = "unfit"

    def fit_from_mdp(
        self,
        mdp_path: str | Path | None = None,
        *,
        P: np.ndarray | None = None,
        R: np.ndarray | None = None,
        terminal_states: Sequence[int] | None = None,
    ) -> "TabularWorldModel":
        """Fit the world model from an existing tabular MDP artifact.

        This path is useful when the project already has a reference MDP and
        Version 3 needs a lightweight simulator interface without requiring
        logged trajectories first.
        """

        if mdp_path is None and (P is None or R is None):
            raise ValueError("Provide either mdp_path or both P and R arrays.")

        if mdp_path is not None:
            data = np.load(Path(mdp_path))
            P = data["P"]
            R = data["R"]

        assert P is not None
        assert R is not None

        if P.ndim != 3:
            raise ValueError("Transition array P must have shape (S, A, S).")

        # Even if the original MDP already contains probabilities, a tiny amount
        # of additive smoothing guards against brittle zero-probability rows.
        smoothed = P.astype(float, copy=True) + (float(self.smoothing) * 1e-12)
        self.transition_probs = _normalize_rows(smoothed)
        self.reward_table = _expected_rewards_from_transition_rewards(self.transition_probs, R)
        self.n_states = int(self.transition_probs.shape[0])
        self.n_actions = int(self.transition_probs.shape[1])
        self.terminal_states = sorted({int(s) for s in (terminal_states or [])})
        self.fit_source = "mdp"
        return self

    def fit_from_trajectories(
        self,
        trajectories: Iterable[object],
        *,
        n_states: int | None = None,
        n_actions: int | None = None,
        terminal_states: Sequence[int] | None = None,
    ) -> "TabularWorldModel":
        """Fit the world model from trajectory-like transitions.

        This route is useful when the project has logged or simulated
        transitions from the decision-support environment and needs to estimate
        a learned simulator rather than directly copying the reference MDP.
        """

        parsed = [_coerce_transition(item) for item in trajectories]
        if not parsed:
            raise ValueError("At least one trajectory transition is required.")

        if n_states is None:
            n_states = max(max(s, s_next) for s, _, s_next, _, _ in parsed) + 1
        if n_actions is None:
            n_actions = max(a for _, a, _, _, _ in parsed) + 1

        counts = np.full((n_states, n_actions, n_states), float(self.smoothing), dtype=float)
        reward_sums = np.zeros((n_states, n_actions), dtype=float)
        reward_counts = np.zeros((n_states, n_actions), dtype=float)
        observed_terminal_states = set(int(s) for s in (terminal_states or []))

        for state, action, next_state, reward, done in parsed:
            counts[state, action, next_state] += 1.0
            reward_sums[state, action] += reward
            reward_counts[state, action] += 1.0
            if done:
                observed_terminal_states.add(next_state)

        self.transition_probs = _normalize_rows(counts)
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.terminal_states = sorted(observed_terminal_states)

        # Unseen state-action pairs fall back to the global mean reward rather
        # than zero so the simulator stays numerically stable and less brittle.
        global_mean_reward = float(np.mean([reward for _, _, _, reward, _ in parsed]))
        self.reward_table = np.full((n_states, n_actions), global_mean_reward, dtype=float)
        seen_mask = reward_counts > 0.0
        self.reward_table[seen_mask] = reward_sums[seen_mask] / reward_counts[seen_mask]
        self.fit_source = "trajectories"
        return self

    def predict_next_state_distribution(self, state: int, action: int) -> np.ndarray:
        """Return P_hat(s' | s, a) for one state-action pair."""

        self._require_fit()
        return self.transition_probs[int(state), int(action), :].copy()

    def predict_reward(self, state: int, action: int) -> float:
        """Return the expected reward estimate R_hat(s, a)."""

        self._require_fit()
        return float(self.reward_table[int(state), int(action)])

    def sample_next_state(
        self,
        state: int,
        action: int,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Sample one next state from the learned transition model."""

        self._require_fit()
        generator = rng if rng is not None else np.random.default_rng()
        probs = self.transition_probs[int(state), int(action), :]
        return int(generator.choice(self.n_states, p=probs))

    def save(self, path: str | Path) -> None:
        """Persist the learned world model to a compressed NumPy archive."""

        self._require_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            transition_probs=self.transition_probs,
            reward_table=self.reward_table,
            terminal_states=np.array(self.terminal_states, dtype=int),
            smoothing=np.array([self.smoothing], dtype=float),
            n_states=np.array([self.n_states], dtype=int),
            n_actions=np.array([self.n_actions], dtype=int),
            fit_source=np.array([self.fit_source]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TabularWorldModel":
        """Load a saved world model from disk."""

        data = np.load(Path(path), allow_pickle=True)
        model = cls(smoothing=float(data["smoothing"][0]))
        model.transition_probs = data["transition_probs"].astype(float)
        model.reward_table = data["reward_table"].astype(float)
        model.terminal_states = [int(x) for x in data["terminal_states"].tolist()]
        model.n_states = int(data["n_states"][0])
        model.n_actions = int(data["n_actions"][0])
        model.fit_source = str(data["fit_source"][0])
        return model

    def _require_fit(self) -> None:
        if self.transition_probs is None or self.reward_table is None:
            raise RuntimeError("World model has not been fit yet.")
