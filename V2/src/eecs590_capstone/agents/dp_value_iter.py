from __future__ import annotations

from typing import Dict

import numpy as np

from eecs590_capstone.mdp.definitions import TabularMDP


def value_iteration(mdp: TabularMDP, gamma: float = 0.99, theta: float = 1e-10, max_iter: int = 10_000) -> Dict[str, object]:
    V = np.zeros(mdp.n_states, dtype=float)
    it = 0
    while it < max_iter:
        delta = 0.0
        for s in range(mdp.n_states):
            if s in mdp.terminal_states:
                continue
            v_old = V[s]
            q_vals = []
            for a in range(mdp.n_actions):
                q = 0.0
                for s_next in range(mdp.n_states):
                    p = mdp.P[s, a, s_next]
                    r = mdp.R[s, a, s_next]
                    q += p * (r + gamma * V[s_next])
                q_vals.append(q)
            V[s] = float(np.max(q_vals))
            delta = max(delta, abs(v_old - V[s]))
        it += 1
        if delta < theta:
            break

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

    return {
        "policy": {str(i): int(a) for i, a in enumerate(policy)},
        "V": {str(i): float(v) for i, v in enumerate(V)},
        "train_info": {
            "iterations": int(it),
        },
    }
