from __future__ import annotations

from copy import deepcopy
from typing import Any


def materialize_reward_candidate(
    base_cfg: dict[str, Any],
    no_reward: float,
    late_penalty: float,
    early_penalty: float,
    cost_scale: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    cfg = deepcopy(base_cfg)
    cfg["reward_map"] = {
        "<30": float(early_penalty),
        ">30": float(late_penalty),
        "NO": float(no_reward),
    }
    cfg["action_costs"] = [float(cost_scale * x) for x in cfg["action_costs"]]
    tag = {
        "no_reward": float(no_reward),
        "late_penalty": float(late_penalty),
        "early_penalty": float(early_penalty),
        "cost_scale": float(cost_scale),
    }
    return cfg, tag


def candidate_name(tag: dict[str, float]) -> str:
    return (
        f"no_{tag['no_reward']:.2f}_late_{tag['late_penalty']:.2f}_"
        f"early_{tag['early_penalty']:.2f}_cost_{tag['cost_scale']:.2f}"
    ).replace("-", "m")
