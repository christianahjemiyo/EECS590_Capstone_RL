from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eecs590_capstone.agents.rl_approx import (
    approximate_q_learning,
    gradient_td,
    linear_value_function_approximation,
    semi_gradient_td,
)
from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv
from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.utils.io import save_json


def load_mdp(mdp_path: Path) -> TabularMDP:
    data = np.load(mdp_path)
    return TabularMDP(P=data["P"], R=data["R"], terminal_states=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train approximate RL algorithms on the simulated MDP.")
    parser.add_argument("--mdp", type=str, default="outputs/V2/mdp/mdp.npz")
    parser.add_argument(
        "--algo",
        type=str,
        default="approx_q_learning",
        choices=["linear_fa", "semi_gradient_td", "gradient_td", "approx_q_learning"],
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="outputs/V2/approx")
    args = parser.parse_args()

    env = MDPSimEnv(mdp_path=args.mdp, seed=args.seed, max_steps=args.max_steps)
    mdp = load_mdp(Path(args.mdp))

    if args.algo == "linear_fa":
        result = linear_value_function_approximation(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed,
        )
    elif args.algo == "semi_gradient_td":
        result = semi_gradient_td(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed,
        )
    elif args.algo == "gradient_td":
        result = gradient_td(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            seed=args.seed,
        )
    else:
        result = approximate_q_learning(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=args.seed,
        )

    eval_metrics = rollout_policy(mdp, result.policy, episodes=2000, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "train_args.json", vars(args))
    save_json(outdir / "policy.json", result.policy)
    save_json(outdir / "value_function.json", result.V)
    save_json(outdir / "train_info.json", result.train_info)
    save_json(outdir / "learning_curve.json", {"episode_returns": result.episode_returns})
    save_json(outdir / "eval_results.json", eval_metrics)
    if result.Q is not None:
        save_json(outdir / "q_values.json", result.Q)

    print(f"Approximate RL training complete: {args.algo}")
    print(eval_metrics)


if __name__ == "__main__":
    main()
