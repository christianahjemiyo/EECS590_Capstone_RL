from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eecs590_capstone.agents.rl_deep import a2c, a3c, ppo, reinforce, train_dqn_variant, trpo
from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv
from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.utils.io import save_json
from viz_theme import annotate_bars, apply_v2_theme, colors_for


def load_mdp(path: Path) -> TabularMDP:
    data = np.load(path)
    return TabularMDP(P=data["P"], R=data["R"], terminal_states=[])


def parse_seeds(seed_arg: str) -> list[int]:
    return [int(tok.strip()) for tok in seed_arg.split(",") if tok.strip()]


def ci95(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * float(std) / np.sqrt(n)


def run_algo(algo: str, env: MDPSimEnv, args: argparse.Namespace, seed: int):
    if algo == "double_dqn":
        return train_dqn_variant(env, args.episodes, args.gamma, args.lr, args.hidden, args.batch_size, args.replay_size, args.warmup_steps, args.target_update, args.eps_start, args.eps_end, args.eps_decay, seed, "double_dqn")
    if algo == "dueling_dqn":
        return train_dqn_variant(env, args.episodes, args.gamma, args.lr, args.hidden, args.batch_size, args.replay_size, args.warmup_steps, args.target_update, args.eps_start, args.eps_end, args.eps_decay, seed, "dueling_dqn")
    if algo == "reinforce":
        return reinforce(env, args.episodes, args.actor_lr, args.gamma, seed)
    if algo == "a2c":
        return a2c(env, args.episodes, args.actor_lr, args.critic_lr, args.gamma, seed)
    if algo == "a3c":
        return a3c(env, args.episodes, args.actor_lr, args.critic_lr, args.gamma, seed, workers=args.workers)
    if algo == "ppo":
        return ppo(env, args.episodes, args.actor_lr, args.critic_lr, args.gamma, args.clip_eps, seed)
    if algo == "trpo":
        return trpo(env, args.episodes, args.actor_lr, args.critic_lr, args.gamma, args.max_kl, seed)
    raise ValueError(f"Unsupported algo: {algo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2 deep/policy-gradient suite with multi-seed summaries.")
    parser.add_argument("--mdp", default="outputs/V2/mdp/mdp.npz")
    parser.add_argument("--outdir", default="outputs/V2/deep_suite")
    parser.add_argument("--episodes", type=int, default=2500)
    parser.add_argument("--eval-episodes", type=int, default=2000)
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
    parser.add_argument("--seeds", type=str, default="7,11,19")
    args = parser.parse_args()

    algos = ["double_dqn", "dueling_dqn", "reinforce", "a2c", "a3c", "ppo", "trpo"]
    seeds = parse_seeds(args.seeds)
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    mdp = load_mdp(Path(args.mdp))

    per_seed_rows = []
    for algo in algos:
        for seed in seeds:
            env = MDPSimEnv(mdp_path=args.mdp, seed=seed, max_steps=args.max_steps)
            result = run_algo(algo, env, args, seed)
            metrics = rollout_policy(mdp, result.policy, episodes=args.eval_episodes, seed=seed, max_steps=args.max_steps)
            per_seed_rows.append({"algo": algo, "seed": seed, "rollout_mean": float(metrics["avg_return"])})

    with (outdir / "per_seed_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "seed", "rollout_mean"])
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary_rows = []
    for algo in algos:
        vals = np.array([r["rollout_mean"] for r in per_seed_rows if r["algo"] == algo], dtype=float)
        summary_rows.append({"algo": algo, "rollout_mean": float(vals.mean()), "rollout_std": float(vals.std(ddof=0)), "rollout_ci95": float(ci95(float(vals.std(ddof=0)), len(vals)))})

    with (outdir / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "rollout_mean", "rollout_std", "rollout_ci95"])
        writer.writeheader()
        writer.writerows(summary_rows)

    labels = [r["algo"] for r in summary_rows]
    means = [r["rollout_mean"] for r in summary_rows]
    errs = [r["rollout_ci95"] for r in summary_rows]
    apply_v2_theme()
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(np.arange(len(labels)), means, yerr=errs, capsize=5, color=colors_for(labels), alpha=0.93)
    annotate_bars(plt.gca(), means, fmt="{:.2f}")
    plt.xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("Average Return")
    plt.title("V2 Deep and Policy-Gradient Suite")
    plt.tight_layout()
    plt.savefig(figdir / "deep_rollout_comparison.png", dpi=180)
    plt.close()

    save_json(outdir / "suite_config.json", vars(args))
    print(f"Wrote deep suite outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
