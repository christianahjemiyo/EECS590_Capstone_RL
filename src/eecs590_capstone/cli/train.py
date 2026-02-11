from __future__ import annotations

import argparse
from pathlib import Path

from eecs590_capstone.envs.foundation_env import FoundationEnv
from eecs590_capstone.agents.dp_policy_iter import policy_iteration
from eecs590_capstone.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DP agent using policy iteration")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    env = FoundationEnv(seed=7)
    result = policy_iteration(env.mdp, gamma=args.gamma, theta=args.theta, max_iter=args.max_iter)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(outdir / "train_args.json", vars(args))
    save_json(outdir / "policy.json", result["policy"])
    save_json(outdir / "value_function.json", result["V"])
    save_json(outdir / "train_info.json", result["train_info"])

    print("✅ Training complete.")
    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
