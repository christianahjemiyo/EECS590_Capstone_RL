from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eecs590_capstone.agents.rl_tabular import (
    double_q_learning,
    mc_control,
    q_learning,
    sarsa,
    sarsa_lambda,
    sarsa_n,
    td0,
    td_lambda,
    td_n,
)
from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv
from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.utils.io import save_json
from viz_theme import apply_v2_theme, annotate_bars, colors_for


def load_mdp(mdp_path: Path) -> TabularMDP:
    data = np.load(mdp_path)
    return TabularMDP(P=data["P"], R=data["R"], terminal_states=[])


def parse_seeds(seed_arg: str) -> list[int]:
    out = []
    for tok in seed_arg.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    if not out:
        raise ValueError("At least one seed is required.")
    return out


def run_algo(algo: str, env: MDPSimEnv, mdp: TabularMDP, args: argparse.Namespace, seed: int):
    if algo == "mc":
        return mc_control(
            env,
            episodes=args.episodes,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    if algo == "td_n":
        return td_n(
            env,
            mdp,
            episodes=args.episodes,
            n=args.n,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    if algo == "td_lambda":
        return td_lambda(
            env,
            mdp,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            lam=args.lam,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    if algo == "td0":
        return td0(
            env,
            mdp,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    if algo == "sarsa_n":
        return sarsa_n(
            env,
            episodes=args.episodes,
            n=args.n,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    if algo == "sarsa":
        return sarsa(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    if algo == "sarsa_lambda":
        return sarsa_lambda(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            lam=args.lam,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    if algo == "q_learning":
        return q_learning(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    if algo == "double_q_learning":
        return double_q_learning(
            env,
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_start=args.eps_start,
            eps_end=args.eps_end,
            decay_steps=args.eps_decay,
            seed=seed,
        )
    raise ValueError(f"Unsupported algo: {algo}")


def ci95(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * float(std) / np.sqrt(n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V2 tabular algorithm suite (MC/TD/SARSA/Q variants) with multi-seed summaries."
    )
    parser.add_argument("--mdp", default="outputs/V2/mdp/mdp.npz")
    parser.add_argument("--outdir", default="outputs/V2/tabular_suite")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.8)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=2000)
    parser.add_argument("--seeds", type=str, default="7,11,19,23,29")
    args = parser.parse_args()

    algos = ["mc", "td0", "td_n", "td_lambda", "sarsa", "sarsa_n", "sarsa_lambda", "q_learning", "double_q_learning"]
    seeds = parse_seeds(args.seeds)
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    mdp = load_mdp(Path(args.mdp))
    per_seed_rows = []
    curve_store: dict[str, list[np.ndarray]] = {a: [] for a in algos}

    for algo in algos:
        for seed in seeds:
            env = MDPSimEnv(mdp_path=args.mdp, seed=seed, max_steps=args.max_steps)
            result = run_algo(algo, env, mdp, args, seed)
            metrics = rollout_policy(mdp, result.policy, episodes=args.eval_episodes, seed=seed, max_steps=args.max_steps)
            per_seed_rows.append(
                {
                    "algo": algo,
                    "seed": seed,
                    "rollout_mean": float(metrics["avg_return"]),
                    "rollout_std": float(metrics["std_return"]),
                }
            )
            curve_store[algo].append(np.asarray(result.episode_returns, dtype=float))

    per_seed_path = outdir / "per_seed_metrics.csv"
    with per_seed_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "seed", "rollout_mean", "rollout_std"])
        w.writeheader()
        w.writerows(per_seed_rows)

    summary_rows = []
    for algo in algos:
        vals = np.array([r["rollout_mean"] for r in per_seed_rows if r["algo"] == algo], dtype=float)
        row = {
            "algo": algo,
            "rollout_mean": float(vals.mean()),
            "rollout_std": float(vals.std(ddof=0)),
            "rollout_ci95": float(ci95(float(vals.std(ddof=0)), len(vals))),
        }
        summary_rows.append(row)

    summary_path = outdir / "summary_metrics.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "rollout_mean", "rollout_std", "rollout_ci95"])
        w.writeheader()
        w.writerows(summary_rows)

    style_colors = {k: v for k, v in zip(algos, colors_for(algos))}

    labels = [r["algo"] for r in summary_rows]
    means = [r["rollout_mean"] for r in summary_rows]
    errs = [r["rollout_ci95"] for r in summary_rows]
    colors = [style_colors[x] for x in labels]
    x = np.arange(len(labels))
    apply_v2_theme()
    plt.figure(figsize=(10, 4.6))
    plt.bar(x, means, yerr=errs, color=colors, capsize=5, alpha=0.9)
    annotate_bars(plt.gca(), means, fmt="{:.2f}")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Average Return (higher is better)")
    plt.title("V2 Tabular Suite: Rollout Performance (mean +/- 95% CI)")
    plt.tight_layout()
    plt.savefig(figdir / "tabular_rollout_comparison.png", dpi=190)
    plt.close()

    apply_v2_theme()
    plt.figure(figsize=(10.5, 5.0))
    for algo in algos:
        curves = curve_store[algo]
        if not curves:
            continue
        min_len = int(min(len(c) for c in curves))
        if min_len == 0:
            continue
        mat = np.stack([c[:min_len] for c in curves], axis=0)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        xs = np.arange(min_len)
        c = style_colors[algo]
        plt.plot(xs, mean, label=algo, color=c, linewidth=1.8)
        plt.fill_between(xs, mean - std, mean + std, color=c, alpha=0.15)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("V2 Tabular Suite: Learning Curves (mean +/- 1 std)")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(figdir / "tabular_learning_curves.png", dpi=190)
    plt.close()

    save_json(
        outdir / "suite_config.json",
        {
            "mdp": args.mdp,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "n": args.n,
            "lambda": args.lam,
            "eps_start": args.eps_start,
            "eps_end": args.eps_end,
            "eps_decay": args.eps_decay,
            "seeds": seeds,
        },
    )

    best = max(summary_rows, key=lambda r: r["rollout_mean"])
    lines = [
        "# V2 Tabular Suite Interpretation",
        "",
        "## What this run adds",
        "- Adds MC/TD/SARSA family beside Q-Learning and Double Q-Learning.",
        "- Reports uncertainty across seeds (95% CI).",
        "- Produces learning-curve and final-performance figures for class-algorithm coverage.",
        "",
        "## Key result",
        f"- Best mean rollout return: **{best['algo']}** ({best['rollout_mean']:.3f}).",
        "",
        "## Files",
        "- `summary_metrics.csv`",
        "- `per_seed_metrics.csv`",
        "- `figures/tabular_rollout_comparison.png`",
        "- `figures/tabular_learning_curves.png`",
    ]
    (outdir / "INTERPRETATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote suite outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
