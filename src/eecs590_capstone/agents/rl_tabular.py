from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from eecs590_capstone.mdp.definitions import TabularMDP


@dataclass
class TrainResult:
    policy: Dict[str, int]
    V: Dict[str, float]
    train_info: Dict[str, float]
    episode_returns: list[float]
    Q: Dict[str, list] | None = None


def epsilon_schedule(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return eps_end
    frac = min(step / decay_steps, 1.0)
    return float(eps_start + frac * (eps_end - eps_start))


def select_action(Q: np.ndarray, s: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, Q.shape[1]))
    return int(np.argmax(Q[s]))


def greedy_policy_from_Q(Q: np.ndarray) -> Dict[str, int]:
    return {str(i): int(np.argmax(Q[i])) for i in range(Q.shape[0])}


def greedy_policy_from_V(mdp: TabularMDP, V: np.ndarray, gamma: float) -> Dict[str, int]:
    policy = {}
    for s in range(mdp.n_states):
        best_a = 0
        best_q = -1e18
        for a in range(mdp.n_actions):
            q = 0.0
            for s_next in range(mdp.n_states):
                p = mdp.P[s, a, s_next]
                r = mdp.R[s, a, s_next]
                q += p * (r + gamma * V[s_next])
            if q > best_q:
                best_q = q
                best_a = a
        policy[str(s)] = int(best_a)
    return policy


def mc_control(env, episodes: int, gamma: float, eps_start: float, eps_end: float, decay_steps: int, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    N = np.zeros_like(Q)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        state = env.reset()
        episode = []
        ep_return = 0.0
        done = False
        while not done:
            action = select_action(Q, state, epsilon, rng)
            step = env.step(action)
            episode.append((state, action, step.reward))
            ep_return += step.reward
            state = step.state
            done = step.done

        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            N[s, a] += 1.0
            Q[s, a] += (G - Q[s, a]) / N[s, a]
        returns.append(float(ep_return))

    policy = greedy_policy_from_Q(Q)
    V = np.max(Q, axis=1)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes)},
        episode_returns=returns,
        Q={str(i): Q[i].tolist() for i in range(Q.shape[0])},
    )


def td_n(env, mdp: TabularMDP, episodes: int, n: int, alpha: float, gamma: float, eps_start: float, eps_end: float, decay_steps: int, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    V = np.zeros(env.n_states, dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        state = env.reset()
        states = [state]
        rewards = [0.0]
        T = 10**9
        t = 0

        while True:
            if t < T:
                greedy_policy = greedy_policy_from_V(mdp, V, gamma)
                a_greedy = greedy_policy[str(states[t])]
                if rng.random() < epsilon:
                    action = int(rng.integers(0, env.n_actions))
                else:
                    action = int(a_greedy)
                step = env.step(action)
                rewards.append(step.reward)
                states.append(step.state)
                if step.done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma ** n) * V[states[tau + n]]
                V[states[tau]] += alpha * (G - V[states[tau]])
            if tau == T - 1:
                break
            t += 1
        returns.append(float(sum(rewards[1:])))

    policy = greedy_policy_from_V(mdp, V, gamma)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes), "n": float(n)},
        episode_returns=returns,
    )


def td_lambda(env, mdp: TabularMDP, episodes: int, alpha: float, gamma: float, lam: float, eps_start: float, eps_end: float, decay_steps: int, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    V = np.zeros(env.n_states, dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        E = np.zeros(env.n_states, dtype=float)
        state = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            greedy_policy = greedy_policy_from_V(mdp, V, gamma)
            a_greedy = greedy_policy[str(state)]
            if rng.random() < epsilon:
                action = int(rng.integers(0, env.n_actions))
            else:
                action = int(a_greedy)
            step = env.step(action)
            ep_return += step.reward
            td_error = step.reward + gamma * V[step.state] - V[state]
            E[state] += 1.0
            V += alpha * td_error * E
            E *= gamma * lam
            state = step.state
            done = step.done
        returns.append(float(ep_return))

    policy = greedy_policy_from_V(mdp, V, gamma)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes), "lambda": float(lam)},
        episode_returns=returns,
    )


def sarsa_n(env, episodes: int, n: int, alpha: float, gamma: float, eps_start: float, eps_end: float, decay_steps: int, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        state = env.reset()
        action = select_action(Q, state, epsilon, rng)
        states = [state]
        actions = [action]
        rewards = [0.0]
        T = 10**9
        t = 0

        while True:
            if t < T:
                step = env.step(actions[t])
                rewards.append(step.reward)
                states.append(step.state)
                if step.done:
                    T = t + 1
                else:
                    actions.append(select_action(Q, step.state, epsilon, rng))
            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * rewards[i]
                if tau + n < T:
                    G += (gamma ** n) * Q[states[tau + n], actions[tau + n]]
                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])
            if tau == T - 1:
                break
            t += 1
        returns.append(float(sum(rewards[1:])))

    policy = greedy_policy_from_Q(Q)
    V = np.max(Q, axis=1)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes), "n": float(n)},
        episode_returns=returns,
        Q={str(i): Q[i].tolist() for i in range(Q.shape[0])},
    )


def sarsa_lambda(env, episodes: int, alpha: float, gamma: float, lam: float, eps_start: float, eps_end: float, decay_steps: int, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        E = np.zeros_like(Q)
        state = env.reset()
        action = select_action(Q, state, epsilon, rng)
        ep_return = 0.0
        done = False
        while not done:
            step = env.step(action)
            next_action = select_action(Q, step.state, epsilon, rng) if not step.done else 0
            ep_return += step.reward
            td_error = step.reward + gamma * Q[step.state, next_action] - Q[state, action]
            E[state, action] += 1.0
            Q += alpha * td_error * E
            E *= gamma * lam
            state = step.state
            action = next_action
            done = step.done
        returns.append(float(ep_return))

    policy = greedy_policy_from_Q(Q)
    V = np.max(Q, axis=1)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes), "lambda": float(lam)},
        episode_returns=returns,
        Q={str(i): Q[i].tolist() for i in range(Q.shape[0])},
    )


def q_learning(env, episodes: int, alpha: float, gamma: float, eps_start: float, eps_end: float, decay_steps: int, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        state = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action = select_action(Q, state, epsilon, rng)
            step = env.step(action)
            ep_return += step.reward
            td_error = step.reward + gamma * np.max(Q[step.state]) - Q[state, action]
            Q[state, action] += alpha * td_error
            state = step.state
            done = step.done
        returns.append(float(ep_return))

    policy = greedy_policy_from_Q(Q)
    V = np.max(Q, axis=1)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes)},
        episode_returns=returns,
        Q={str(i): Q[i].tolist() for i in range(Q.shape[0])},
    )


def double_q_learning(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    decay_steps: int,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    Q1 = np.zeros((env.n_states, env.n_actions), dtype=float)
    Q2 = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = []

    for ep in range(episodes):
        epsilon = epsilon_schedule(ep, eps_start, eps_end, decay_steps)
        state = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            Qsum = Q1 + Q2
            action = select_action(Qsum, state, epsilon, rng)
            step = env.step(action)
            ep_return += step.reward

            if rng.random() < 0.5:
                next_a = int(np.argmax(Q1[step.state]))
                td_target = step.reward + gamma * Q2[step.state, next_a]
                Q1[state, action] += alpha * (td_target - Q1[state, action])
            else:
                next_a = int(np.argmax(Q2[step.state]))
                td_target = step.reward + gamma * Q1[step.state, next_a]
                Q2[state, action] += alpha * (td_target - Q2[state, action])

            state = step.state
            done = step.done
        returns.append(float(ep_return))

    Q = Q1 + Q2
    policy = greedy_policy_from_Q(Q)
    V = np.max(Q, axis=1)
    return TrainResult(
        policy=policy,
        V={str(i): float(v) for i, v in enumerate(V)},
        train_info={"episodes": float(episodes)},
        episode_returns=returns,
        Q={str(i): Q[i].tolist() for i in range(Q.shape[0])},
    )
