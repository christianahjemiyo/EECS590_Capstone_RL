from __future__ import annotations

import argparse
from pathlib import Path

from eecs590_capstone.envs.foundation_env import FoundationEnv
from eecs590_capstone.mdp.definitions import rollout_policy
from eecs590_capstone.utils.io import load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved policy")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-path", type=str, default="outputs/policy.json")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    env = FoundationEnv(seed=args.seed)
    policy = load_json(Path(args.policy_path))

    results = rollout_policy(env.mdp, policy, episodes=args.episodes, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(outdir / "eval_args.json", vars(args))
    save_json(outdir / "eval_results.json", results)

    print("✅ Eval complete.")
    print(f"Saved to: {outdir.resolve()}")
    print(results)


if __name__ == "__main__":
    main()
