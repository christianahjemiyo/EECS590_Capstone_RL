from __future__ import annotations

"""Evaluation helpers for learned world models.

These functions compare a learned tabular world model against the original MDP
 and evaluate fixed policies inside the learned simulator. The purpose is to
 quantify how closely the learned decision-support simulator reproduces the
 modeled readmission-risk dynamics already present in the repository.
"""

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .simulator import WorldModelSimulator
from .tabular_world_model import TabularWorldModel


def _load_mdp_arrays(
    mdp_path: str | Path | None = None,
    *,
    P: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if mdp_path is None and (P is None or R is None):
        raise ValueError("Provide either mdp_path or both P and R arrays.")
    if mdp_path is not None:
        data = np.load(Path(mdp_path))
        return data["P"], data["R"]
    assert P is not None
    assert R is not None
    return P, R


def _expected_reward_table(P: np.ndarray, R: np.ndarray) -> np.ndarray:
    if R.ndim == 2:
        return R.astype(float, copy=True)
    return np.sum(P * R, axis=2)


def compare_world_model_to_mdp(
    world_model: TabularWorldModel,
    mdp_path: str | Path | None = None,
    *,
    P: np.ndarray | None = None,
    R: np.ndarray | None = None,
    epsilon: float = 1e-12,
    include_kl: bool = True,
) -> pd.DataFrame:
    """Compare learned transitions and rewards against the original MDP.

    The result is one row per state-action pair so downstream scripts can
    summarize or visualize the error structure.
    """

    true_P, true_R = _load_mdp_arrays(mdp_path, P=P, R=R)
    true_rewards = _expected_reward_table(true_P, true_R)

    rows: list[dict[str, float | int]] = []
    for state in range(world_model.n_states):
        for action in range(world_model.n_actions):
            learned = world_model.predict_next_state_distribution(state, action)
            target = true_P[state, action, :]
            diff = learned - target
            row: dict[str, float | int] = {
                "state": state,
                "action": action,
                "transition_mae": float(np.mean(np.abs(diff))),
                "transition_mse": float(np.mean(diff ** 2)),
                "reward_mae": float(abs(world_model.predict_reward(state, action) - true_rewards[state, action])),
            }
            if include_kl:
                row["transition_kl"] = float(
                    np.sum((target + epsilon) * np.log((target + epsilon) / (learned + epsilon)))
                )
            rows.append(row)

    return pd.DataFrame(rows)


def evaluate_policy_in_world_model(
    world_model: TabularWorldModel,
    policy: Mapping[str, int] | Mapping[int, int],
    *,
    episodes: int = 200,
    seed: int = 7,
    max_steps: int = 30,
    start_state: int | None = None,
) -> dict[str, float]:
    """Evaluate a fixed policy in the learned simulator.

    This gives a lightweight safety-check view of how a policy behaves under
    learned modeled dynamics before comparing it to the original tabular MDP.
    """

    simulator = WorldModelSimulator(
        world_model,
        seed=seed,
        max_steps=max_steps,
        start_state=start_state,
    )
    terminal_lookup = set(world_model.terminal_states)

    returns: list[float] = []
    lengths: list[int] = []
    terminal_hits = 0

    for _ in range(int(episodes)):
        trajectory = simulator.simulate_trajectory(policy, start_state=start_state, max_steps=max_steps)
        total_reward = float(sum(float(step["reward"]) for step in trajectory))
        if trajectory:
            final_step = trajectory[-1]
            terminal_hit = bool(final_step["next_state"] in terminal_lookup)
        else:
            terminal_hit = False
        returns.append(total_reward)
        lengths.append(len(trajectory))
        if terminal_hit:
            terminal_hits += 1

    arr = np.array(returns, dtype=float)
    len_arr = np.array(lengths, dtype=float) if lengths else np.array([0.0], dtype=float)
    return {
        "episodes": float(episodes),
        "avg_return": float(arr.mean()) if len(arr) else 0.0,
        "std_return": float(arr.std()) if len(arr) else 0.0,
        "avg_length": float(len_arr.mean()),
        "terminal_rate": float(terminal_hits / episodes) if episodes else 0.0,
    }


def evaluate_policy_in_mdp(
    policy: Mapping[str, int] | Mapping[int, int],
    *,
    mdp_path: str | Path | None = None,
    P: np.ndarray | None = None,
    R: np.ndarray | None = None,
    terminal_states: list[int] | None = None,
    episodes: int = 200,
    seed: int = 7,
    max_steps: int = 30,
    start_state: int | None = None,
) -> dict[str, float]:
    """Evaluate a fixed policy in the original tabular MDP.

    This creates the direct baseline needed to judge whether the learned world
    model is accurate enough for offline policy evaluation.
    """

    true_P, true_R = _load_mdp_arrays(mdp_path, P=P, R=R)
    n_states = int(true_P.shape[0])
    rng = np.random.default_rng(seed)
    terminal_lookup = set(terminal_states or [])

    returns: list[float] = []
    lengths: list[int] = []
    terminal_hits = 0

    for _ in range(int(episodes)):
        if start_state is None:
            candidates = [s for s in range(n_states) if s not in terminal_lookup] or list(range(n_states))
            state = int(rng.choice(candidates))
        else:
            state = int(start_state)

        total_reward = 0.0
        done = False
        steps = 0

        while steps < max_steps and not done:
            action = int(policy.get(str(state), policy.get(state, 0)))  # type: ignore[arg-type]
            probs = true_P[state, action, :]
            next_state = int(rng.choice(n_states, p=probs))
            reward = float(true_R[state, action, next_state]) if true_R.ndim == 3 else float(true_R[state, action])
            total_reward += reward
            steps += 1
            done = next_state in terminal_lookup or steps >= max_steps
            state = next_state

        returns.append(total_reward)
        lengths.append(steps)
        if done and state in terminal_lookup:
            terminal_hits += 1

    arr = np.array(returns, dtype=float)
    len_arr = np.array(lengths, dtype=float) if lengths else np.array([0.0], dtype=float)
    return {
        "episodes": float(episodes),
        "avg_return": float(arr.mean()) if len(arr) else 0.0,
        "std_return": float(arr.std()) if len(arr) else 0.0,
        "avg_length": float(len_arr.mean()),
        "terminal_rate": float(terminal_hits / episodes) if episodes else 0.0,
    }


def save_metrics_csv(metrics: pd.DataFrame, path: str | Path) -> None:
    """Save comparison metrics to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
