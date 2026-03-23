from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eecs590_capstone.agents.rl_advanced import (
    belief_state_update,
    ddpg,
    learned_environment_model,
    rnn_based_rl,
    sac,
    td3,
)
from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv
from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.utils.io import save_json


def load_mdp(mdp_path: Path) -> TabularMDP:
    data = np.load(mdp_path)
    return TabularMDP(P=data["P"], R=data["R"], terminal_states=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train advanced RL algorithms and wrappers on the simulated MDP.")
    parser.add_argument("--mdp", type=str, default="outputs/V2/mdp/mdp.npz")
    parser.add_argument(
        "--algo",
        type=str,
        default="ddpg",
        choices=["ddpg", "td3", "sac", "learned_model", "belief_update", "rnn_rl"],
    )
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-lr", type=float, default=0.02)
    parser.add_argument("--critic-lr", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--noise-std", type=float, default=0.35)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-obs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="outputs/V2/advanced")
    args = parser.parse_args()

    mdp = load_mdp(Path(args.mdp))
    env = MDPSimEnv(mdp_path=args.mdp, seed=args.seed, max_steps=args.max_steps)

    if args.algo == "ddpg":
        result = ddpg(
            env,
            episodes=args.episodes,
            gamma=args.gamma,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            noise_std=args.noise_std,
            seed=args.seed,
        )
    elif args.algo == "td3":
        result = td3(
            env,
            episodes=args.episodes,
            gamma=args.gamma,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            noise_std=args.noise_std,
            policy_noise=args.policy_noise,
            delay=args.delay,
            seed=args.seed,
        )
    elif args.algo == "sac":
        result = sac(env, episodes=args.episodes, alpha=args.alpha, lr=args.lr, gamma=args.gamma, seed=args.seed)
    elif args.algo == "learned_model":
        result = learned_environment_model(
            mdp,
            episodes=args.episodes,
            max_steps=args.max_steps,
            gamma=args.gamma,
            seed=args.seed,
        )
    elif args.algo == "belief_update":
        result = belief_state_update(
            mdp,
            episodes=args.episodes,
            max_steps=args.max_steps,
            gamma=args.gamma,
            seed=args.seed,
            n_obs=args.n_obs,
        )
    else:
        result = rnn_based_rl(
            mdp,
            episodes=args.episodes,
            max_steps=args.max_steps,
            gamma=args.gamma,
            alpha=args.alpha,
            seed=args.seed,
            n_obs=args.n_obs,
            hidden_size=args.hidden_size,
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

    print(f"Advanced RL training complete: {args.algo}")
    print(eval_metrics)


if __name__ == "__main__":
    main()
