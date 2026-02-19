from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from eecs590_capstone.utils.data_load import encode_tabular, load_dataframe


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class DataDrivenEnv:
    """Data-driven environment skeleton.

    This environment does NOT model causal effects of actions because the dataset
    has no intervention actions. It provides a reproducible interface and reward
    shaping scaffold for future extensions.
    """

    def __init__(
        self,
        data_path: str,
        seed: int = 7,
        action_costs: List[float] | None = None,
        reward_map: Dict[str, float] | None = None,
        label_col: str = "readmitted",
        max_steps: int = 1,
        terminal_on_readmit: bool = True,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.data_path = data_path
        self.df = load_dataframe(data_path)

        self.label_col = label_col
        self.X, self.feature_names, self.y = encode_tabular(self.df, label_col=label_col)
        self.n_samples = self.X.shape[0]

        self.action_costs = action_costs or [0.0, 0.2, 0.5]
        self.reward_map = reward_map or {"<30": -10.0, ">30": -2.0, "NO": 2.0}
        self.n_actions = len(self.action_costs)
        self.max_steps = max_steps
        self.terminal_on_readmit = terminal_on_readmit

        self._step_count = 0
        self._current_idx = 0

    def reset(self) -> np.ndarray:
        self._step_count = 0
        self._current_idx = int(self.rng.integers(0, self.n_samples))
        return self.X[self._current_idx]

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Invalid action {action}. Expected 0..{self.n_actions - 1}.")

        label = str(self.y[self._current_idx])
        reward = self._outcome_reward(label) - float(self.action_costs[action])

        self._step_count += 1
        done = self._step_count >= self.max_steps
        if self.terminal_on_readmit and label == "<30":
            done = True

        # Transition is a proxy (independent resampling).
        self._current_idx = int(self.rng.integers(0, self.n_samples))
        next_state = self.X[self._current_idx]

        info = {"label": label}
        return StepResult(state=next_state, reward=reward, done=done, info=info)

    def _outcome_reward(self, label: str) -> float:
        return float(self.reward_map.get(label, 0.0))
