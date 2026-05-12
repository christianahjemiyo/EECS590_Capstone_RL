"""World-model utilities for decision-support simulation.

This package adds a lightweight Version 3 extension to the capstone project.
The implementation is intentionally tabular so it matches the existing clinical
MDP abstraction already used for modeled readmission risk experiments.

The goal is not to claim a realistic clinical simulator. Instead, these tools
learn or reuse compact transition dynamics so policies can be stress-tested in
an interpretable learned simulator before making comparisons against the
original tabular MDP.
"""

from .evaluation import (
    compare_world_model_to_mdp,
    evaluate_policy_in_world_model,
    save_metrics_csv,
)
from .simulator import WorldModelSimulator, WorldModelStep
from .tabular_world_model import TabularWorldModel

__all__ = [
    "TabularWorldModel",
    "WorldModelSimulator",
    "WorldModelStep",
    "compare_world_model_to_mdp",
    "evaluate_policy_in_world_model",
    "save_metrics_csv",
]
