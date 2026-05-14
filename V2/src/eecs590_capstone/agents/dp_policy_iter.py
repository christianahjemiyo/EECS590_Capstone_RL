from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from eecs590_capstone.mdp.definitions import TabularMDP


def policy_evaluation(mdp: TabularMDP, policy: np.ndarray, gamma: float, theta: float, max_iter: int) -> Tuple[np.ndarray, int]:
    V = np.zeros(mdp.n_states, dtype=float)

    it = 0
    while it < max_iter:
        delta = 0.0
        for s in range(mdp.n_states):
            if s in mdp.terminal_states:
                continue
            a = int(policy[s])
            v_old = V[s]
            v_new = 0.0
            for s_next in range(mdp.n_states):
                p = mdp.P[s, a, s_next]
                r = mdp.R[s, a, s_next]
                v_new += p * (r + gamma * V[s_next])
            V[s] = v_new
            delta = max(delta, abs(v_old - v_new))

        it += 1
        if delta < theta:
            break

    return V, it


def greedy_policy_from_V(mdp: TabularMDP, V: np.ndarray, gamma: float) -> np.ndarray:
    policy = np.zeros(mdp.n_states, dtype=int)

    for s in range(mdp.n_states):
        if s in mdp.terminal_states:
            policy[s] = 0
            continue

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

        policy[s] = best_a

    return policy


def policy_iteration(mdp: TabularMDP, gamma: float = 0.99, theta: float = 1e-10, max_iter: int = 10_000) -> Dict[str, object]:
    policy = np.zeros(mdp.n_states, dtype=int)  # start with all zeros

    eval_iters_total = 0
    improve_steps = 0

    while True:
        V, eval_iters = policy_evaluation(mdp, policy, gamma=gamma, theta=theta, max_iter=max_iter)
        eval_iters_total += eval_iters

        new_policy = greedy_policy_from_V(mdp, V, gamma=gamma)
        improve_steps += 1

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

        if improve_steps > 1000:  # safety
            break

    return {
        "policy": {str(i): int(a) for i, a in enumerate(policy)},
        "V": {str(i): float(v) for i, v in enumerate(V)},
        "train_info": {
            "eval_iters_total": int(eval_iters_total),
            "improve_steps": int(improve_steps),
        },
    }
