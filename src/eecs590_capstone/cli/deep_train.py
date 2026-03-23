from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eecs590_capstone.agents.rl_deep import a2c, a3c, ppo, reinforce, train_dqn_variant, trpo
from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv
from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.utils.io import save_json


def load_mdp(mdp_path: Path) -> TabularMDP:
    data = np.load(mdp_path)
    return TabularMDP(P=data["P"], R=data["R"], terminal_states=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep and policy-gradient RL algorithms on the simulated MDP.")
    parser.add_argument("--mdp", type=str, default="outputs/V2/mdp/mdp.npz")
    parser.add_argument(
        "--algo",
        type=str,
        default="double_dqn",
        choices=["double_dqn", "dueling_dqn", "reinforce", "a2c", "a3c", "ppo", "trpo"],
    )
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--target-update", type=int, default=200)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=3000)
    parser.add_argument("--actor-lr", type=float, default=0.05)
    parser.add_argument("--critic-lr", type=float, default=0.1)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--max-kl", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="outputs/V2/deep")
    args = parser.parse_args()

    env = MDPSimEnv(mdp_path=args.mdp, seed=args.seed, max_steps=args.max_steps)
    mdp = load_mdp(Path(args.mdp))

    if args.algo in {"double_dqn", "dueling_dqn"}:
        result = train_dqn_variant(
            env=env,
            episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr,
            hidden=args.hidden,
            batch_size=args.batch_size,
            replay_size=args.replay_size,
            warmup_steps=args.warmup_steps,
            target_update=args.target_update,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            seed=args.seed,
            variant=args.algo,
        )
    elif args.algo == "reinforce":
        result = reinforce(env, episodes=args.episodes, alpha=args.actor_lr, gamma=args.gamma, seed=args.seed)
    elif args.algo == "a2c":
        result = a2c(
            env,
            episodes=args.episodes,
            alpha_actor=args.actor_lr,
            alpha_critic=args.critic_lr,
            gamma=args.gamma,
            seed=args.seed,
        )
    elif args.algo == "a3c":
        result = a3c(
            env,
            episodes=args.episodes,
            alpha_actor=args.actor_lr,
            alpha_critic=args.critic_lr,
            gamma=args.gamma,
            seed=args.seed,
            workers=args.workers,
        )
    elif args.algo == "ppo":
        result = ppo(
            env,
            episodes=args.episodes,
            alpha_actor=args.actor_lr,
            alpha_critic=args.critic_lr,
            gamma=args.gamma,
            clip_eps=args.clip_eps,
            seed=args.seed,
        )
    else:
        result = trpo(
            env,
            episodes=args.episodes,
            alpha_actor=args.actor_lr,
            alpha_critic=args.critic_lr,
            gamma=args.gamma,
            max_kl=args.max_kl,
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

    print(f"Deep RL training complete: {args.algo}")
    print(eval_metrics)


if __name__ == "__main__":
    main()
