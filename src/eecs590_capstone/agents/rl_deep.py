from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np


@dataclass
class TrainResult:
    policy: Dict[str, int]
    V: Dict[str, float]
    train_info: Dict[str, float]
    episode_returns: list[float]
    Q: Dict[str, list[float]] | None = None
    checkpoint: dict | None = None


def epsilon_schedule(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return eps_end
    frac = min(step / decay_steps, 1.0)
    return float(eps_start + frac * (eps_end - eps_start))


@dataclass
class QNet:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    v_head: np.ndarray | None = None
    v_bias: np.ndarray | None = None
    a_head: np.ndarray | None = None
    a_bias: np.ndarray | None = None


def init_qnet(
    n_states: int,
    n_actions: int,
    hidden: int,
    rng: np.random.Generator,
    dueling: bool,
) -> QNet:
    w1 = rng.normal(loc=0.0, scale=0.1, size=(n_states, hidden))
    b1 = np.zeros(hidden, dtype=float)
    if dueling:
        return QNet(
            w1=w1,
            b1=b1,
            w2=np.empty((0, 0)),
            b2=np.empty(0),
            v_head=rng.normal(loc=0.0, scale=0.1, size=(hidden, 1)),
            v_bias=np.zeros(1, dtype=float),
            a_head=rng.normal(loc=0.0, scale=0.1, size=(hidden, n_actions)),
            a_bias=np.zeros(n_actions, dtype=float),
        )
    return QNet(
        w1=w1,
        b1=b1,
        w2=rng.normal(loc=0.0, scale=0.1, size=(hidden, n_actions)),
        b2=np.zeros(n_actions, dtype=float),
    )


def copy_qnet(src: QNet) -> QNet:
    return QNet(
        w1=src.w1.copy(),
        b1=src.b1.copy(),
        w2=src.w2.copy(),
        b2=src.b2.copy(),
        v_head=None if src.v_head is None else src.v_head.copy(),
        v_bias=None if src.v_bias is None else src.v_bias.copy(),
        a_head=None if src.a_head is None else src.a_head.copy(),
        a_bias=None if src.a_bias is None else src.a_bias.copy(),
    )


def forward_q(net: QNet, states: np.ndarray, dueling: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_pre = net.w1[states] + net.b1
    h = np.maximum(0.0, h_pre)
    if dueling:
        value = h @ net.v_head + net.v_bias
        advantage = h @ net.a_head + net.a_bias
        q = value + (advantage - advantage.mean(axis=1, keepdims=True))
    else:
        q = h @ net.w2 + net.b2
    return q, h, h_pre


def qnet_values(net: QNet, n_states: int, dueling: bool) -> tuple[Dict[str, int], Dict[str, float], Dict[str, list[float]]]:
    states = np.arange(n_states, dtype=np.int64)
    q_all, _, _ = forward_q(net, states, dueling=dueling)
    policy = {str(s): int(np.argmax(q_all[s])) for s in range(n_states)}
    values = {str(s): float(np.max(q_all[s])) for s in range(n_states)}
    q_json = {str(s): q_all[s].tolist() for s in range(n_states)}
    return policy, values, q_json


def select_action(net: QNet, state: int, eps: float, rng: np.random.Generator, n_actions: int, dueling: bool) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, n_actions))
    q, _, _ = forward_q(net, np.array([state], dtype=np.int64), dueling=dueling)
    return int(np.argmax(q[0]))


def train_dqn_variant(
    env,
    episodes: int,
    gamma: float,
    lr: float,
    hidden: int,
    batch_size: int,
    replay_size: int,
    warmup_steps: int,
    target_update: int,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    seed: int,
    variant: Literal["dqn", "double_dqn", "dueling_dqn"],
) -> TrainResult:
    dueling = variant == "dueling_dqn"
    double = variant == "double_dqn"
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.n_states, env.n_actions
    online = init_qnet(n_states, n_actions, hidden, rng, dueling=dueling)
    target = copy_qnet(online)

    s_buf = np.zeros(replay_size, dtype=np.int64)
    a_buf = np.zeros(replay_size, dtype=np.int64)
    r_buf = np.zeros(replay_size, dtype=float)
    sn_buf = np.zeros(replay_size, dtype=np.int64)
    d_buf = np.zeros(replay_size, dtype=np.float64)
    size = 0
    ptr = 0
    total_steps = 0
    losses = []
    episode_returns = []

    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            eps = epsilon_schedule(total_steps, eps_start, eps_end, eps_decay)
            a = select_action(online, s, eps, rng, n_actions, dueling=dueling)
            step = env.step(a)
            sn, r, done = step.state, float(step.reward), step.done

            s_buf[ptr], a_buf[ptr], r_buf[ptr], sn_buf[ptr], d_buf[ptr] = s, a, r, sn, 1.0 if done else 0.0
            ptr = (ptr + 1) % replay_size
            size = min(size + 1, replay_size)
            s = sn
            ep_return += r
            total_steps += 1

            if size >= max(batch_size, warmup_steps):
                idx = rng.integers(0, size, size=batch_size)
                bs, ba, br, bsn, bd = s_buf[idx], a_buf[idx], r_buf[idx], sn_buf[idx], d_buf[idx]
                q, h, h_pre = forward_q(online, bs, dueling=dueling)
                qt, _, _ = forward_q(target, bsn, dueling=dueling)
                if double:
                    qo_next, _, _ = forward_q(online, bsn, dueling=dueling)
                    next_actions = np.argmax(qo_next, axis=1)
                    target_next = qt[np.arange(batch_size), next_actions]
                else:
                    target_next = np.max(qt, axis=1)
                tgt = br + gamma * target_next * (1.0 - bd)

                pred = q[np.arange(batch_size), ba]
                err = pred - tgt
                losses.append(0.5 * float(np.mean(err * err)))

                grad_q = np.zeros_like(q)
                grad_q[np.arange(batch_size), ba] = err / float(batch_size)

                if dueling:
                    grad_v = grad_q.sum(axis=1, keepdims=True)
                    grad_a = grad_q - grad_q.mean(axis=1, keepdims=True)
                    d_v_head = h.T @ grad_v
                    d_v_bias = grad_v.sum(axis=0)
                    d_a_head = h.T @ grad_a
                    d_a_bias = grad_a.sum(axis=0)
                    d_h = grad_v @ online.v_head.T + grad_a @ online.a_head.T
                else:
                    d_w2 = h.T @ grad_q
                    d_b2 = grad_q.sum(axis=0)
                    d_h = grad_q @ online.w2.T

                d_h[h_pre <= 0.0] = 0.0
                d_w1 = np.zeros_like(online.w1)
                np.add.at(d_w1, bs, d_h)
                d_b1 = d_h.sum(axis=0)

                online.w1 -= lr * d_w1
                online.b1 -= lr * d_b1
                if dueling:
                    online.v_head -= lr * d_v_head
                    online.v_bias -= lr * d_v_bias
                    online.a_head -= lr * d_a_head
                    online.a_bias -= lr * d_a_bias
                else:
                    online.w2 -= lr * d_w2
                    online.b2 -= lr * d_b2

            if total_steps % target_update == 0:
                target = copy_qnet(online)

        episode_returns.append(float(ep_return))

    policy, values, q_json = qnet_values(online, n_states, dueling=dueling)
    return TrainResult(
        policy=policy,
        V=values,
        train_info={
            "episodes": float(episodes),
            "total_steps": float(total_steps),
            "final_epsilon": float(epsilon_schedule(total_steps, eps_start, eps_end, eps_decay)),
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
        },
        episode_returns=episode_returns,
        Q=q_json,
        checkpoint={
            "online": {
                "w1": online.w1,
                "b1": online.b1,
                "w2": online.w2,
                "b2": online.b2,
                "v_head": online.v_head if online.v_head is not None else np.array([]),
                "v_bias": online.v_bias if online.v_bias is not None else np.array([]),
                "a_head": online.a_head if online.a_head is not None else np.array([]),
                "a_bias": online.a_bias if online.a_bias is not None else np.array([]),
            },
            "target": {
                "w1": target.w1,
                "b1": target.b1,
                "w2": target.w2,
                "b2": target.b2,
                "v_head": target.v_head if target.v_head is not None else np.array([]),
                "v_bias": target.v_bias if target.v_bias is not None else np.array([]),
                "a_head": target.a_head if target.a_head is not None else np.array([]),
                "a_bias": target.a_bias if target.a_bias is not None else np.array([]),
            },
        },
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / np.sum(exp)


def sample_action(theta: np.ndarray, state: int, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    probs = softmax(theta[state])
    action = int(rng.choice(theta.shape[1], p=probs))
    return action, probs


def policy_value_tables(theta: np.ndarray, values: np.ndarray) -> tuple[Dict[str, int], Dict[str, float]]:
    policy = {str(s): int(np.argmax(theta[s])) for s in range(theta.shape[0])}
    value_map = {str(s): float(values[s]) for s in range(theta.shape[0])}
    return policy, value_map


def reinforce(env, episodes: int, alpha: float, gamma: float, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions), dtype=float)
    baseline = np.zeros(env.n_states, dtype=float)
    returns = []

    for _ in range(episodes):
        states, actions, rewards, probs_trace = [], [], [], []
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            a, probs = sample_action(theta, s, rng)
            step = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(float(step.reward))
            probs_trace.append(probs)
            s = step.state
            done = step.done
            ep_return += step.reward

        G = 0.0
        for t in reversed(range(len(states))):
            G = rewards[t] + gamma * G
            s_t, a_t, probs = states[t], actions[t], probs_trace[t]
            advantage = G - baseline[s_t]
            baseline[s_t] += 0.1 * advantage
            grad = -probs
            grad[a_t] += 1.0
            theta[s_t] += alpha * (gamma ** t) * advantage * grad
        returns.append(float(ep_return))

    policy, values = policy_value_tables(theta, baseline)
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes)},
        episode_returns=returns,
        checkpoint={"theta": theta, "baseline": baseline},
    )


def a2c(env, episodes: int, alpha_actor: float, alpha_critic: float, gamma: float, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions), dtype=float)
    value = np.zeros(env.n_states, dtype=float)
    returns = []

    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            a, probs = sample_action(theta, s, rng)
            step = env.step(a)
            v_next = value[step.state] if not step.done else 0.0
            delta = step.reward + gamma * v_next - value[s]
            value[s] += alpha_critic * delta
            grad = -probs
            grad[a] += 1.0
            theta[s] += alpha_actor * delta * grad
            s = step.state
            done = step.done
            ep_return += step.reward
        returns.append(float(ep_return))

    policy, values = policy_value_tables(theta, value)
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes)},
        episode_returns=returns,
        checkpoint={"theta": theta, "value": value},
    )


def a3c(env, episodes: int, alpha_actor: float, alpha_critic: float, gamma: float, seed: int, workers: int = 4) -> TrainResult:
    theta = np.zeros((env.n_states, env.n_actions), dtype=float)
    value = np.zeros(env.n_states, dtype=float)
    returns = []

    for worker_idx in range(workers):
        rng = np.random.default_rng(seed + worker_idx)
        local_episodes = max(1, episodes // workers)
        for _ in range(local_episodes):
            s = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                a, probs = sample_action(theta, s, rng)
                step = env.step(a)
                v_next = value[step.state] if not step.done else 0.0
                delta = step.reward + gamma * v_next - value[s]
                value[s] += alpha_critic * delta
                grad = -probs
                grad[a] += 1.0
                theta[s] += alpha_actor * delta * grad
                s = step.state
                done = step.done
                ep_return += step.reward
            returns.append(float(ep_return))

    policy, values = policy_value_tables(theta, value)
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "workers": float(workers)},
        episode_returns=returns,
        checkpoint={"theta": theta, "value": value},
    )


def ppo(
    env,
    episodes: int,
    alpha_actor: float,
    alpha_critic: float,
    gamma: float,
    clip_eps: float,
    seed: int,
) -> TrainResult:
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions), dtype=float)
    value = np.zeros(env.n_states, dtype=float)
    returns = []

    for _ in range(episodes):
        traj = []
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            a, probs = sample_action(theta, s, rng)
            step = env.step(a)
            traj.append((s, a, float(step.reward), probs[a], step.state, step.done))
            s = step.state
            done = step.done
            ep_return += step.reward

        G = 0.0
        for t in reversed(range(len(traj))):
            s_t, a_t, r_t, old_prob, sn, done_t = traj[t]
            G = r_t + gamma * G
            advantage = G - value[s_t]
            value[s_t] += alpha_critic * advantage
            probs = softmax(theta[s_t])
            ratio = probs[a_t] / max(old_prob, 1e-8)
            clipped_ratio = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            scale = clipped_ratio if ratio * advantage != clipped_ratio * advantage else ratio
            grad = -probs
            grad[a_t] += 1.0
            theta[s_t] += alpha_actor * scale * advantage * grad
        returns.append(float(ep_return))

    policy, values = policy_value_tables(theta, value)
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "clip_eps": float(clip_eps)},
        episode_returns=returns,
        checkpoint={"theta": theta, "value": value},
    )


def trpo(env, episodes: int, alpha_actor: float, alpha_critic: float, gamma: float, max_kl: float, seed: int) -> TrainResult:
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions), dtype=float)
    value = np.zeros(env.n_states, dtype=float)
    returns = []

    for _ in range(episodes):
        states, actions, rewards = [], [], []
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            a, _ = sample_action(theta, s, rng)
            step = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(float(step.reward))
            s = step.state
            done = step.done
            ep_return += step.reward

        G = 0.0
        for t in reversed(range(len(states))):
            G = rewards[t] + gamma * G
            s_t, a_t = states[t], actions[t]
            advantage = G - value[s_t]
            value[s_t] += alpha_critic * advantage
            old_probs = softmax(theta[s_t])
            grad = -old_probs
            grad[a_t] += 1.0
            step_size = alpha_actor
            for _ in range(8):
                candidate = theta[s_t] + step_size * advantage * grad
                new_probs = softmax(candidate)
                kl = float(np.sum(old_probs * (np.log(np.maximum(old_probs, 1e-8)) - np.log(np.maximum(new_probs, 1e-8)))))
                if kl <= max_kl:
                    theta[s_t] = candidate
                    break
                step_size *= 0.5
        returns.append(float(ep_return))

    policy, values = policy_value_tables(theta, value)
    return TrainResult(
        policy=policy,
        V=values,
        train_info={"episodes": float(episodes), "max_kl": float(max_kl)},
        episode_returns=returns,
        checkpoint={"theta": theta, "value": value},
    )
