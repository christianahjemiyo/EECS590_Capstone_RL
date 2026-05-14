from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class TabularMDP:
    """A tiny tabular MDP container.

    P[s,a,s'] = transition probability
    R[s,a,s'] = reward
    """
    P: np.ndarray  # shape (S, A, S)
    R: np.ndarray  # shape (S, A, S)
    terminal_states: List[int]

    @property
    def n_states(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.P.shape[1])


def rollout_policy(mdp: TabularMDP, policy: Dict[str, int], episodes: int = 2000, seed: int = 7, max_steps: int = 200) -> Dict[str, float]:
    rng = np.random.default_rng(seed)

    returns: List[float] = []
    terminal_hits = 0

    for _ in range(episodes):
        s = 0  # start state
        G = 0.0
        steps = 0

        while steps < max_steps and s not in mdp.terminal_states:
            a = int(policy.get(str(s), 0))
            probs = mdp.P[s, a, :]
            s_next = int(rng.choice(mdp.n_states, p=probs))
            r = float(mdp.R[s, a, s_next])
            G += r
            s = s_next
            steps += 1

        if s in mdp.terminal_states:
            terminal_hits += 1

        returns.append(G)

    arr = np.array(returns, dtype=float)
    return {
        "episodes": float(episodes),
        "avg_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "terminal_rate": float(terminal_hits / episodes),
    }


def rollout_policy_array(mdp: TabularMDP, policy: np.ndarray, episodes: int = 2000, seed: int = 7, max_steps: int = 200) -> Dict[str, float]:
    pol = {str(i): int(a) for i, a in enumerate(policy)}
    return rollout_policy(mdp, pol, episodes=episodes, seed=seed, max_steps=max_steps)
