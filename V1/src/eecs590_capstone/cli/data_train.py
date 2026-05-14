from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eecs590_capstone.agents.baseline_policy import BaselinePolicy, available_policies
from eecs590_capstone.envs.data_env import DataDrivenEnv
from eecs590_capstone.utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a baseline policy config for the data-driven env.")
    parser.add_argument("--config", type=str, default="configs/data_env.json")
    parser.add_argument("--policy", type=str, default="random", choices=available_policies())
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="outputs/data_train")
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

    rng = np.random.default_rng(args.seed)
    policy = BaselinePolicy(name=args.policy, n_actions=env.n_actions)

    returns = []
    readmit_hits = 0
    for _ in range(args.episodes):
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
    train_info = {
        "episodes": float(args.episodes),
        "avg_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "readmission_rate": float(readmit_hits / args.episodes),
        "policy": args.policy,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(outdir / "policy.json", {"policy": args.policy})
    save_json(outdir / "train_args.json", vars(args))
    save_json(outdir / "train_info.json", train_info)

    print("Data policy saved.")
    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
