from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ApproxTrainResult:
    policy: Dict[str, int]
    V: Dict[str, float]
    train_info: Dict[str, float]
    episode_returns: list[float]
    Q: Dict[str, list[float]] | None = None


def epsilon_schedule(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return eps_end
    frac = min(step / decay_steps, 1.0)
    return float(eps_start + frac * (eps_end - eps_start))


def state_features(state: int, n_states: int) -> np.ndarray:
    denom = max(1, n_states - 1)
    x = float(state) / float(denom)
    return np.array(
        [
            1.0,
            x,
            x * x,
            np.sin(np.pi * x),
            np.cos(np.pi * x),
        ],
        dtype=float,
    )


def value_from_weights(weights: np.ndarray, state: int, n_states: int) -> float:
    return float(np.dot(weights, state_features(state, n_states)))


def q_values_from_weights(weights: np.ndarray, state: int, n_states: int) -> np.ndarray:
    feats = state_features(state, n_states)
    return weights @ feats


def select_action(weights: np.ndarray, state: int, n_states: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, weights.shape[0]))
    q_vals = q_values_from_weights(weights, state, n_states)
    return int(np.argmax(q_vals))


def linear_value_function_approximation(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    seed: int,
) -> ApproxTrainResult:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=5)
    returns = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = int(rng.integers(0, env.n_actions))
            step = env.step(action)
            phi = state_features(state, env.n_states)
            target = step.reward + gamma * value_from_weights(w, step.state, env.n_states) * float(not step.done)
            pred = float(np.dot(w, phi))
            w += alpha * (target - pred) * phi
            state = step.state
            done = step.done
            ep_return += step.reward
        returns.append(float(ep_return))

    values = {str(s): value_from_weights(w, s, env.n_states) for s in range(env.n_states)}
    policy = {str(s): 0 for s in range(env.n_states)}
    return ApproxTrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "feature_dim": float(w.shape[0])},
        episode_returns=returns,
    )


def semi_gradient_td(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    seed: int,
) -> ApproxTrainResult:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=5)
    returns = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = int(rng.integers(0, env.n_actions))
            step = env.step(action)
            phi = state_features(state, env.n_states)
            v_s = float(np.dot(w, phi))
            v_next = value_from_weights(w, step.state, env.n_states) if not step.done else 0.0
            delta = step.reward + gamma * v_next - v_s
            w += alpha * delta * phi
            state = step.state
            done = step.done
            ep_return += step.reward
        returns.append(float(ep_return))

    values = {str(s): value_from_weights(w, s, env.n_states) for s in range(env.n_states)}
    policy = {str(s): 0 for s in range(env.n_states)}
    return ApproxTrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "feature_dim": float(w.shape[0])},
        episode_returns=returns,
    )


def gradient_td(
    env,
    episodes: int,
    alpha: float,
    beta: float,
    gamma: float,
    seed: int,
) -> ApproxTrainResult:
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=5)
    h = np.zeros_like(w)
    returns = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = int(rng.integers(0, env.n_actions))
            step = env.step(action)
            phi = state_features(state, env.n_states)
            phi_next = state_features(step.state, env.n_states)
            v_s = float(np.dot(w, phi))
            v_next = float(np.dot(w, phi_next)) if not step.done else 0.0
            delta = step.reward + gamma * v_next - v_s
            h += beta * (delta - float(np.dot(h, phi))) * phi
            correction = gamma * float(np.dot(phi_next, h)) * phi_next if not step.done else 0.0
            w += alpha * (delta * phi - correction)
            state = step.state
            done = step.done
            ep_return += step.reward
        returns.append(float(ep_return))

    values = {str(s): value_from_weights(w, s, env.n_states) for s in range(env.n_states)}
    policy = {str(s): 0 for s in range(env.n_states)}
    return ApproxTrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "feature_dim": float(w.shape[0]), "beta": float(beta)},
        episode_returns=returns,
    )


def approximate_q_learning(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> ApproxTrainResult:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(env.n_actions, 5))
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = select_action(weights, state, env.n_states, epsilon, rng)
            step = env.step(action)
            phi = state_features(state, env.n_states)
            next_q = q_values_from_weights(weights, step.state, env.n_states)
            target = step.reward + gamma * float(np.max(next_q)) * float(not step.done)
            pred = float(np.dot(weights[action], phi))
            weights[action] += alpha * (target - pred) * phi
            state = step.state
            done = step.done
            ep_return += step.reward
        returns.append(float(ep_return))

    q_json = {}
    policy = {}
    values = {}
    for s in range(env.n_states):
        q_vals = q_values_from_weights(weights, s, env.n_states)
        q_json[str(s)] = q_vals.tolist()
        policy[str(s)] = int(np.argmax(q_vals))
        values[str(s)] = float(np.max(q_vals))
    return ApproxTrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "feature_dim": float(weights.shape[1])},
        episode_returns=returns,
        Q=q_json,
    )
