from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eecs590_capstone.agents.baseline_policy import BaselinePolicy, available_policies
from eecs590_capstone.envs.data_env import DataDrivenEnv
from eecs590_capstone.utils.io import load_json, save_json


def run_episodes(env: DataDrivenEnv, policy_name: str, episodes: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    policy = BaselinePolicy(name=policy_name, n_actions=env.n_actions)
    returns = []
    readmit_hits = 0

    for _ in range(episodes):
        _ = env.reset()
        done = False
        G = 0.0
        while not done:
            action = policy.act(rng)
            step = env.step(action)
            G += step.reward
            done = step.done
            if step.info.get("label") == "<30":
                readmit_hits += 1
        returns.append(G)

    arr = np.asarray(returns, dtype=float)
    return {
        "episodes": float(episodes),
        "avg_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "readmission_rate": float(readmit_hits / episodes),
        "policy": policy_name,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies on the data-driven env.")
    parser.add_argument("--config", type=str, default="configs/data_env.json")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="outputs/data_eval/baselines.json")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    env = DataDrivenEnv(
        data_path=cfg["data_path"],
        seed=args.seed,
        action_costs=cfg.get("action_costs"),
        reward_map=cfg.get("reward_map"),
        label_col=cfg.get("label_col", "readmitted"),
        max_steps=cfg.get("max_steps", 1),
        terminal_on_readmit=cfg.get("terminal_on_readmit", True),
    )

    results = {}
    for policy in available_policies():
        results[policy] = run_episodes(env, policy, args.episodes, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, results)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
