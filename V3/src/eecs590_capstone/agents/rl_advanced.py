from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class TrainResult:
    policy: Dict[str, int]
    V: Dict[str, float]
    train_info: Dict[str, float]
    episode_returns: list[float]
    Q: Dict[str, list[float]] | None = None
    checkpoint: dict | None = None


def state_features(state: int, n_states: int) -> np.ndarray:
    denom = max(1, n_states - 1)
    x = float(state) / float(denom)
    return np.array([1.0, x, x * x, np.sin(np.pi * x), np.cos(np.pi * x)], dtype=float)


def action_scalar_features(state: int, n_states: int, action_value: float, n_actions: int) -> np.ndarray:
    a_norm = float(action_value) / max(1.0, float(n_actions - 1))
    return np.concatenate([state_features(state, n_states), np.array([a_norm, a_norm * a_norm], dtype=float)])


def continuous_actor_output(weights: np.ndarray, state: int, n_states: int, n_actions: int) -> float:
    raw = float(np.dot(weights, state_features(state, n_states)))
    squashed = np.tanh(raw)
    return 0.5 * (squashed + 1.0) * float(n_actions - 1)


def to_discrete_action(action_value: float, n_actions: int) -> int:
    return int(np.clip(np.rint(action_value), 0, n_actions - 1))


def q_linear(weights: np.ndarray, state: int, n_states: int, action_value: float, n_actions: int) -> float:
    return float(np.dot(weights, action_scalar_features(state, n_states, action_value, n_actions)))


def ddpg(
    env,
    episodes: int,
    gamma: float,
    actor_lr: float,
    critic_lr: float,
    noise_std: float,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    actor = rng.normal(0.0, 0.05, size=5)
    critic = rng.normal(0.0, 0.05, size=7)
    returns = []

    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            mu = continuous_actor_output(actor, s, env.n_states, env.n_actions)
            a_cont = float(np.clip(mu + rng.normal(0.0, noise_std), 0.0, env.n_actions - 1))
            a = to_discrete_action(a_cont, env.n_actions)
            step = env.step(a)

            next_mu = continuous_actor_output(actor, step.state, env.n_states, env.n_actions)
            target = step.reward + gamma * q_linear(critic, step.state, env.n_states, next_mu, env.n_actions) * float(not step.done)
            phi = action_scalar_features(s, env.n_states, a_cont, env.n_actions)
            pred = float(np.dot(critic, phi))
            critic += critic_lr * (target - pred) * phi

            eps = 1e-3
            q_plus = q_linear(critic, s, env.n_states, min(env.n_actions - 1, mu + eps), env.n_actions)
            q_minus = q_linear(critic, s, env.n_states, max(0.0, mu - eps), env.n_actions)
            dq_da = (q_plus - q_minus) / (2.0 * eps)
            x = float(np.dot(actor, state_features(s, env.n_states)))
            dmu_dx = 0.5 * (env.n_actions - 1) * (1.0 - np.tanh(x) ** 2)
            actor += actor_lr * dq_da * dmu_dx * state_features(s, env.n_states)

            s = step.state
            done = step.done
            ep_return += step.reward
        returns.append(float(ep_return))

    policy = {}
    values = {}
    for s in range(env.n_states):
        mu = continuous_actor_output(actor, s, env.n_states, env.n_actions)
        policy[str(s)] = to_discrete_action(mu, env.n_actions)
        values[str(s)] = q_linear(critic, s, env.n_states, mu, env.n_actions)
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes)},
        episode_returns=returns,
        checkpoint={"actor": actor, "critic": critic},
    )


def td3(
    env,
    episodes: int,
    gamma: float,
    actor_lr: float,
    critic_lr: float,
    noise_std: float,
    policy_noise: float,
    delay: int,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    actor = rng.normal(0.0, 0.05, size=5)
    critic1 = rng.normal(0.0, 0.05, size=7)
    critic2 = rng.normal(0.0, 0.05, size=7)
    returns = []
    step_count = 0

    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            mu = continuous_actor_output(actor, s, env.n_states, env.n_actions)
            a_cont = float(np.clip(mu + rng.normal(0.0, noise_std), 0.0, env.n_actions - 1))
            a = to_discrete_action(a_cont, env.n_actions)
            step = env.step(a)

            next_mu = continuous_actor_output(actor, step.state, env.n_states, env.n_actions)
            next_mu = float(np.clip(next_mu + rng.normal(0.0, policy_noise), 0.0, env.n_actions - 1))
            q1_next = q_linear(critic1, step.state, env.n_states, next_mu, env.n_actions)
            q2_next = q_linear(critic2, step.state, env.n_states, next_mu, env.n_actions)
            target = step.reward + gamma * min(q1_next, q2_next) * float(not step.done)
            phi = action_scalar_features(s, env.n_states, a_cont, env.n_actions)

            pred1 = float(np.dot(critic1, phi))
            pred2 = float(np.dot(critic2, phi))
            critic1 += critic_lr * (target - pred1) * phi
            critic2 += critic_lr * (target - pred2) * phi

            if step_count % max(1, delay) == 0:
                eps = 1e-3
                q_plus = q_linear(critic1, s, env.n_states, min(env.n_actions - 1, mu + eps), env.n_actions)
                q_minus = q_linear(critic1, s, env.n_states, max(0.0, mu - eps), env.n_actions)
                dq_da = (q_plus - q_minus) / (2.0 * eps)
                x = float(np.dot(actor, state_features(s, env.n_states)))
                dmu_dx = 0.5 * (env.n_actions - 1) * (1.0 - np.tanh(x) ** 2)
                actor += actor_lr * dq_da * dmu_dx * state_features(s, env.n_states)

            s = step.state
            done = step.done
            ep_return += step.reward
            step_count += 1
        returns.append(float(ep_return))

    policy = {}
    values = {}
    for s in range(env.n_states):
        mu = continuous_actor_output(actor, s, env.n_states, env.n_actions)
        policy[str(s)] = to_discrete_action(mu, env.n_actions)
        values[str(s)] = min(
            q_linear(critic1, s, env.n_states, mu, env.n_actions),
            q_linear(critic2, s, env.n_states, mu, env.n_actions),
        )
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "delay": float(delay)},
        episode_returns=returns,
        checkpoint={"actor": actor, "critic1": critic1, "critic2": critic2},
    )


def sac(env, episodes: int, alpha: float, lr: float, gamma: float, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    logits = np.zeros((env.n_states, env.n_actions), dtype=float)
    q_values = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = []

    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            probs = np.exp(logits[s] - np.max(logits[s]))
            probs /= probs.sum()
            a = int(rng.choice(env.n_actions, p=probs))
            step = env.step(a)

            next_logits = logits[step.state]
            next_probs = np.exp(next_logits - np.max(next_logits))
            next_probs /= next_probs.sum()
            soft_value = float(np.sum(next_probs * (q_values[step.state] - alpha * np.log(np.maximum(next_probs, 1e-8)))))
            target = step.reward + gamma * soft_value * float(not step.done)
            q_values[s, a] += lr * (target - q_values[s, a])

            advantages = q_values[s] - alpha * np.log(np.maximum(probs, 1e-8))
            logits[s] = advantages - np.max(advantages)
            s = step.state
            done = step.done
            ep_return += step.reward
        returns.append(float(ep_return))

    policy = {str(s): int(np.argmax(logits[s])) for s in range(env.n_states)}
    values = {str(s): float(np.max(q_values[s])) for s in range(env.n_states)}
    q_json = {str(s): q_values[s].tolist() for s in range(env.n_states)}
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "alpha": float(alpha)},
        episode_returns=returns,
        Q=q_json,
        checkpoint={"logits": logits, "q_values": q_values},
    )


def learned_environment_model(
    mdp,
    episodes: int,
    max_steps: int,
    gamma: float,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    counts = np.zeros_like(mdp.P)
    rewards = np.zeros_like(mdp.R)

    for _ in range(episodes):
        s = int(rng.integers(0, mdp.n_states))
        for _ in range(max_steps):
            a = int(rng.integers(0, mdp.n_actions))
            sn = int(rng.choice(mdp.n_states, p=mdp.P[s, a]))
            r = float(mdp.R[s, a, sn])
            counts[s, a, sn] += 1.0
            rewards[s, a, sn] += r
            s = sn

    p_hat = counts / np.maximum(counts.sum(axis=2, keepdims=True), 1.0)
    r_hat = rewards / np.maximum(counts, 1.0)
    V = np.zeros(mdp.n_states, dtype=float)
    for _ in range(300):
        new_v = V.copy()
        for s in range(mdp.n_states):
            q_vals = []
            for a in range(mdp.n_actions):
                q_vals.append(float(np.sum(p_hat[s, a] * (r_hat[s, a] + gamma * V))))
            new_v[s] = max(q_vals)
        if np.max(np.abs(new_v - V)) < 1e-8:
            V = new_v
            break
        V = new_v

    policy = {}
    q_json = {}
    for s in range(mdp.n_states):
        q_vals = []
        for a in range(mdp.n_actions):
            q_vals.append(float(np.sum(p_hat[s, a] * (r_hat[s, a] + gamma * V))))
        q_arr = np.array(q_vals, dtype=float)
        q_json[str(s)] = q_arr.tolist()
        policy[str(s)] = int(np.argmax(q_arr))
    values = {str(s): float(V[s]) for s in range(mdp.n_states)}
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "max_steps": float(max_steps)},
        episode_returns=[],
        Q=q_json,
        checkpoint={"p_hat": p_hat, "r_hat": r_hat},
    )


def make_observation_model(n_states: int, n_obs: int) -> np.ndarray:
    obs_model = np.full((n_states, n_obs), 0.1 / max(1, n_obs - 1), dtype=float)
    for s in range(n_states):
        obs_model[s, s % n_obs] = 0.9
    return obs_model


def belief_state_update(
    mdp,
    episodes: int,
    max_steps: int,
    gamma: float,
    seed: int,
    n_obs: int = 3,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    obs_model = make_observation_model(mdp.n_states, n_obs)
    q_values = np.zeros((mdp.n_states, mdp.n_actions), dtype=float)
    returns = []

    for _ in range(episodes):
        belief = np.full(mdp.n_states, 1.0 / mdp.n_states, dtype=float)
        s = int(rng.integers(0, mdp.n_states))
        ep_return = 0.0
        for _ in range(max_steps):
            expected_q = belief @ q_values
            a = int(np.argmax(expected_q))
            sn = int(rng.choice(mdp.n_states, p=mdp.P[s, a]))
            r = float(mdp.R[s, a, sn])
            obs = int(rng.choice(n_obs, p=obs_model[sn]))
            pred_belief = belief @ mdp.P[:, a, :]
            likelihood = obs_model[:, obs]
            belief = pred_belief * likelihood
            belief /= np.maximum(belief.sum(), 1e-8)
            q_values[s, a] += 0.1 * (r + gamma * np.max(q_values[sn]) - q_values[s, a])
            s = sn
            ep_return += r
        returns.append(ep_return)

    policy = {str(s): int(np.argmax(q_values[s])) for s in range(mdp.n_states)}
    values = {str(s): float(np.max(q_values[s])) for s in range(mdp.n_states)}
    q_json = {str(s): q_values[s].tolist() for s in range(mdp.n_states)}
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "n_obs": float(n_obs)},
        episode_returns=[float(x) for x in returns],
        Q=q_json,
        checkpoint={"q_values": q_values, "obs_model": obs_model},
    )


def rnn_based_rl(
    mdp,
    episodes: int,
    max_steps: int,
    gamma: float,
    alpha: float,
    seed: int,
    n_obs: int = 3,
    hidden_size: int = 8,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    obs_model = make_observation_model(mdp.n_states, n_obs)
    w_h = rng.normal(0.0, 0.1, size=(hidden_size, hidden_size + n_obs))
    w_q = rng.normal(0.0, 0.1, size=(mdp.n_actions, hidden_size))
    returns = []

    for _ in range(episodes):
        s = int(rng.integers(0, mdp.n_states))
        h = np.zeros(hidden_size, dtype=float)
        ep_return = 0.0
        for _ in range(max_steps):
            obs = int(rng.choice(n_obs, p=obs_model[s]))
            x = np.zeros(n_obs, dtype=float)
            x[obs] = 1.0
            h_in = np.concatenate([h, x])
            h = np.tanh(w_h @ h_in)
            q = w_q @ h
            if rng.random() < 0.1:
                a = int(rng.integers(0, mdp.n_actions))
            else:
                a = int(np.argmax(q))

            sn = int(rng.choice(mdp.n_states, p=mdp.P[s, a]))
            r = float(mdp.R[s, a, sn])
            next_obs = int(rng.choice(n_obs, p=obs_model[sn]))
            x_next = np.zeros(n_obs, dtype=float)
            x_next[next_obs] = 1.0
            h_next = np.tanh(w_h @ np.concatenate([h, x_next]))
            td_error = r + gamma * float(np.max(w_q @ h_next)) - float(q[a])
            w_q[a] += alpha * td_error * h
            s = sn
            ep_return += r
        returns.append(float(ep_return))

    policy = {}
    values = {}
    q_json = {}
    for s in range(mdp.n_states):
        obs = s % n_obs
        x = np.zeros(n_obs, dtype=float)
        x[obs] = 1.0
        h = np.tanh(w_h @ np.concatenate([np.zeros(hidden_size, dtype=float), x]))
        q = w_q @ h
        policy[str(s)] = int(np.argmax(q))
        values[str(s)] = float(np.max(q))
        q_json[str(s)] = q.tolist()
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "hidden_size": float(hidden_size)},
        episode_returns=returns,
        Q=q_json,
        checkpoint={"w_h": w_h, "w_q": w_q, "obs_model": obs_model},
    )
