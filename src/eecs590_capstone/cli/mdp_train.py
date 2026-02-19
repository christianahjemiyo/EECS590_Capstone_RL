from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eecs590_capstone.agents.dp_policy_iter import policy_iteration
from eecs590_capstone.agents.dp_value_iter import value_iteration
from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.utils.io import save_json


def load_mdp(mdp_path: Path) -> TabularMDP:
    data = np.load(mdp_path)
    P = data["P"]
    R = data["R"]
    return TabularMDP(P=P, R=R, terminal_states=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DP algorithms on the simulated MDP.")
    parser.add_argument("--mdp", type=str, default="outputs/mdp/mdp.npz")
    parser.add_argument("--algo", type=str, default="policy_iter", choices=["policy_iter", "value_iter"])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="outputs/mdp")
    args = parser.parse_args()

    mdp = load_mdp(Path(args.mdp))
    if args.algo == "policy_iter":
        result = policy_iteration(mdp, gamma=args.gamma, theta=args.theta, max_iter=args.max_iter)
    else:
        result = value_iteration(mdp, gamma=args.gamma, theta=args.theta, max_iter=args.max_iter)

    policy = result["policy"]
    metrics = rollout_policy(mdp, policy, episodes=args.episodes, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "dp_train_args.json", vars(args))
    save_json(outdir / f"{args.algo}_policy.json", policy)
    save_json(outdir / f"{args.algo}_value_function.json", result["V"])
    save_json(outdir / f"{args.algo}_train_info.json", result["train_info"])
    save_json(outdir / f"{args.algo}_eval.json", metrics)

    print(f"DP training complete: {args.algo}")
    print(metrics)


if __name__ == "__main__":
    main()
