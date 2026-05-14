from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BaselinePolicy:
    name: str
    n_actions: int

    def act(self, rng: np.random.Generator) -> int:
        if self.name == "random":
            return int(rng.integers(0, self.n_actions))
        if self.name == "conservative":
            return 0
        if self.name == "aggressive":
            return self.n_actions - 1
        raise ValueError(f"Unknown policy '{self.name}'.")


def available_policies() -> List[str]:
    return ["random", "conservative", "aggressive"]
