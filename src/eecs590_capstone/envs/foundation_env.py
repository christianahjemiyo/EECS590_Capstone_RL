from __future__ import annotations

import numpy as np
from eecs590_capstone.mdp.definitions import TabularMDP


def build_foundation_mdp(seed: int = 7) -> TabularMDP:
    """A small, reproducible tabular MDP.

    States: 0..5
    Actions: 0..1
    Terminal: 5

    Interpretation (toy):
    - 0 start
    - 5 terminal success
    - some transitions give negative reward to encourage planning
    """
    rng = np.random.default_rng(seed)

    S, A = 6, 2
    P = np.zeros((S, A, S), dtype=float)
    R = np.zeros((S, A, S), dtype=float)

    terminal = [5]

    # Action 0: conservative move
    # Action 1: risky move

    # From 0:
    P[0, 0, 1] = 0.85; P[0, 0, 2] = 0.15
    P[0, 1, 2] = 0.70; P[0, 1, 3] = 0.30
    R[0, :, :] -= 0.1

    # From 1:
    P[1, 0, 2] = 0.80; P[1, 0, 4] = 0.20
    P[1, 1, 3] = 0.75; P[1, 1, 4] = 0.25
    R[1, :, :] -= 0.2

    # From 2:
    P[2, 0, 3] = 0.85; P[2, 0, 1] = 0.15
    P[2, 1, 4] = 0.80; P[2, 1, 3] = 0.20
    R[2, :, :] -= 0.15

    # From 3:
    P[3, 0, 4] = 0.90; P[3, 0, 2] = 0.10
    P[3, 1, 5] = 0.55; P[3, 1, 4] = 0.45
    R[3, :, :] -= 0.25

    # From 4:
    P[4, 0, 5] = 0.70; P[4, 0, 3] = 0.30
    P[4, 1, 5] = 0.85; P[4, 1, 2] = 0.15
    R[4, :, :] -= 0.3

    # Terminal 5 loops to itself
    for a in range(A):
        P[5, a, 5] = 1.0
        R[5, a, 5] = 0.0

    # Big positive reward for reaching terminal 5
    for s in range(S):
        for a in range(A):
            R[s, a, 5] += 5.0

    return TabularMDP(P=P, R=R, terminal_states=terminal)


class FoundationEnv:
    def __init__(self, seed: int = 7):
        self.mdp = build_foundation_mdp(seed=seed)
