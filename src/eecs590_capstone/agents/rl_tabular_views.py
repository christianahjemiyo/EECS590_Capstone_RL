from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from eecs590_capstone.agents.rl_tabular import (
    epsilon_schedule,
    greedy_policy_from_Q,
    greedy_policy_from_V,
    select_action,
)
from eecs590_capstone.mdp.definitions import TabularMDP


@dataclass
class TrainResult:
    policy: Dict[str, int]
    V: Dict[str, float]
    train_info: Dict[str, float]
    episode_returns: list[float]
    Q: Dict[str, list[float]] | None = None


def _lambda_return(rewards: list[float], next_values: list[float], gamma: float, lam: float, t: int) -> float:
    horizon = len(rewards) - 1
    g_lambda = 0.0
    lam_pow = 1.0
    for n in range(1, horizon - t + 1):
        g_n = 0.0
        for k in range(1, n + 1):
            g_n += (gamma ** (k - 1)) * rewards[t + k]
        bootstrap_idx = t + n
        if bootstrap_idx < len(next_values):
            g_n += (gamma ** n) * next_values[bootstrap_idx]
        weight = (1.0 - lam) * lam_pow if n < horizon - t else lam_pow
        g_lambda += weight * g_n
        lam_pow *= lam
    return float(g_lambda)


def td_n_forward(
    env,
    mdp: TabularMDP,
    episodes: int,
    n: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    from eecs590_capstone.agents.rl_tabular import td_n

    result = td_n(env, mdp, episodes, n, alpha, gamma, eps_start, eps_end, decay_steps, seed)
    result.train_info["view"] = "forward"
    return result


def td_n_backward(
    env,
    mdp: TabularMDP,
    episodes: int,
    n: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    V = np.zeros(env.n_states, dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        e_trace = np.zeros(env.n_states, dtype=float)
        state = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            greedy_policy = greedy_policy_from_V(mdp, V, gamma)
            action = int(rng.integers(0, env.n_actions)) if rng.random() < epsilon else int(greedy_policy[str(state)])
            step = env.step(action)
            ep_return += step.reward
            delta = step.reward + gamma * V[step.state] * float(not step.done) - V[state]
            e_trace *= gamma
            e_trace[state] += 1.0
            if n > 0:
                cutoff = gamma ** max(1, n)
                e_trace[np.abs(e_trace) < cutoff] = 0.0
            V += alpha * delta * e_trace
            state = step.state
            done = step.done
        returns.append(float(ep_return))

    policy = greedy_policy_from_V(mdp, V, gamma)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes), "n": float(n), "view": "backward", "trace_cutoff": float(n)},
        episode_returns=returns,
    )


def td_lambda_forward(
    env,
    mdp: TabularMDP,
    episodes: int,
    alpha: float,
    gamma: float,
    lam: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    V = np.zeros(env.n_states, dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        states = [env.reset()]
        rewards = [0.0]
        done = False
        ep_return = 0.0
        while not done:
            greedy_policy = greedy_policy_from_V(mdp, V, gamma)
            action = int(rng.integers(0, env.n_actions)) if rng.random() < epsilon else int(greedy_policy[str(states[-1])])
            step = env.step(action)
            rewards.append(float(step.reward))
            states.append(step.state)
            ep_return += step.reward
            done = step.done

        next_values = [float(V[s]) for s in states]
        for t in range(len(states) - 1):
            target = _lambda_return(rewards, next_values, gamma, lam, t)
            s_t = states[t]
            V[s_t] += alpha * (target - V[s_t])
        returns.append(float(ep_return))

    policy = greedy_policy_from_V(mdp, V, gamma)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes), "lambda": float(lam), "view": "forward"},
        episode_returns=returns,
    )


def td_lambda_backward(
    env,
    mdp: TabularMDP,
    episodes: int,
    alpha: float,
    gamma: float,
    lam: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    from eecs590_capstone.agents.rl_tabular import td_lambda

    result = td_lambda(env, mdp, episodes, alpha, gamma, lam, eps_start, eps_end, decay_steps, seed)
    result.train_info["view"] = "backward"
    return result


def sarsa_n_forward(
    env,
    episodes: int,
    n: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    from eecs590_capstone.agents.rl_tabular import sarsa_n

    result = sarsa_n(env, episodes, n, alpha, gamma, eps_start, eps_end, decay_steps, seed)
    result.train_info["view"] = "forward"
    return result


def sarsa_n_backward(
    env,
    episodes: int,
    n: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        e_trace = np.zeros_like(Q)
        state = env.reset()
        action = select_action(Q, state, epsilon, rng)
        ep_return = 0.0
        done = False
        while not done:
            step = env.step(action)
            next_action = select_action(Q, step.state, epsilon, rng) if not step.done else 0
            ep_return += step.reward
            target = step.reward + gamma * Q[step.state, next_action] * float(not step.done)
            delta = target - Q[state, action]
            e_trace *= gamma
            e_trace[state, action] += 1.0
            if n > 0:
                cutoff = gamma ** max(1, n)
                e_trace[np.abs(e_trace) < cutoff] = 0.0
            Q += alpha * delta * e_trace
            state = step.state
            action = next_action
            done = step.done
        returns.append(float(ep_return))

    policy = greedy_policy_from_Q(Q)
    V = np.max(Q, axis=1)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes), "n": float(n), "view": "backward", "trace_cutoff": float(n)},
        episode_returns=returns,
        Q={str(i): Q[i].tolist() for i in range(Q.shape[0])},
    )


def sarsa_lambda_forward(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    lam: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        states, actions, rewards = [], [], [0.0]
        state = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action = select_action(Q, state, epsilon, rng)
            step = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(float(step.reward))
            ep_return += step.reward
            state = step.state
            done = step.done

        next_q_values = [float(np.max(Q[s])) for s in states + [state]]
        for t in range(len(states)):
            target = _lambda_return(rewards, next_q_values, gamma, lam, t)
            Q[states[t], actions[t]] += alpha * (target - Q[states[t], actions[t]])
        returns.append(float(ep_return))

    policy = greedy_policy_from_Q(Q)
    V = np.max(Q, axis=1)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes), "lambda": float(lam), "view": "forward"},
        episode_returns=returns,
        Q={str(i): Q[i].tolist() for i in range(Q.shape[0])},
    )


def sarsa_lambda_backward(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    lam: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    from eecs590_capstone.agents.rl_tabular import sarsa_lambda

    result = sarsa_lambda(env, episodes, alpha, gamma, lam, eps_start, eps_end, decay_steps, seed)
    result.train_info["view"] = "backward"
    return result
