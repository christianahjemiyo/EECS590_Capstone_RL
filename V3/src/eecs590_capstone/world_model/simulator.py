from __future__ import annotations

"""Environment-like simulator built from a learned tabular world model.

This wrapper makes the learned dynamics easier to evaluate like the existing
project environments. It is intended for safe policy testing in a modeled
decision-support simulator, not for real-world clinical use.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .tabular_world_model import TabularWorldModel


@dataclass
class WorldModelStep:
    """Single simulator transition output."""

    state: int
    reward: float
    done: bool
    info: dict[str, Any]


class WorldModelSimulator:
    """Environment-like simulator driven by a learned tabular world model."""

    def __init__(
        self,
        world_model: TabularWorldModel,
        *,
        seed: int = 7,
        max_steps: int = 30,
        start_state: int | None = None,
    ) -> None:
        self.world_model = world_model
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.max_steps = int(max_steps)
        self.default_start_state = start_state
        self._state = 0
        self._step_count = 0

    def reset(self, start_state: int | None = None) -> int:
        """Reset the simulator.

        If no explicit start state is provided, the simulator samples a start
        state from the non-terminal portion of the state space when possible.
        """

        self._step_count = 0
        chosen = start_state if start_state is not None else self.default_start_state
        if chosen is None:
            non_terminal = [
                s for s in range(self.world_model.n_states) if s not in set(self.world_model.terminal_states)
            ]
            candidates = non_terminal if non_terminal else list(range(self.world_model.n_states))
            chosen = int(self.rng.choice(candidates))
        self._state = int(chosen)
        return self._state

    def step(self, action: int) -> WorldModelStep:
        """Advance one step using the learned transition and reward model."""

        if action < 0 or action >= self.world_model.n_actions:
            raise ValueError(f"Invalid action {action}. Expected 0..{self.world_model.n_actions - 1}.")

        next_state = self.world_model.sample_next_state(self._state, action, rng=self.rng)
        reward = self.world_model.predict_reward(self._state, action)
        self._step_count += 1
        done = next_state in self.world_model.terminal_states or self._step_count >= self.max_steps
        self._state = next_state
        return WorldModelStep(
            state=next_state,
            reward=reward,
            done=done,
            info={"terminal_state": next_state in self.world_model.terminal_states},
        )

    def simulate_trajectory(
        self,
        policy: Mapping[str, int] | Mapping[int, int],
        *,
        start_state: int | None = None,
        max_steps: int | None = None,
    ) -> list[dict[str, float | int | bool]]:
        """Simulate one policy rollout and return transition records."""

        state = self.reset(start_state=start_state)
        limit = self.max_steps if max_steps is None else int(max_steps)
        transitions: list[dict[str, float | int | bool]] = []

        for _ in range(limit):
            action = int(policy.get(str(state), policy.get(state, 0)))  # type: ignore[arg-type]
            step = self.step(action)
            transitions.append(
                {
                    "state": state,
                    "action": action,
                    "next_state": step.state,
                    "reward": step.reward,
                    "done": step.done,
                }
            )
            state = step.state
            if step.done:
                break

        return transitions

    def simulate_trajectories(
        self,
        policy: Mapping[str, int] | Mapping[int, int],
        *,
        episodes: int = 100,
        start_state: int | None = None,
        max_steps: int | None = None,
    ) -> list[list[dict[str, float | int | bool]]]:
        """Simulate multiple policy rollouts."""

        return [
            self.simulate_trajectory(policy, start_state=start_state, max_steps=max_steps)
            for _ in range(int(episodes))
        ]
