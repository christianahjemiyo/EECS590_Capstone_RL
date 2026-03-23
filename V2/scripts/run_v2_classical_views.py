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

from eecs590_capstone.agents.rl_tabular_views import (
    sarsa_lambda_backward,
    sarsa_lambda_forward,
    sarsa_n_backward,
    sarsa_n_forward,
    td_lambda_backward,
    td_lambda_forward,
    td_n_backward,
    td_n_forward,
)
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


def run_algo(algo: str, env: MDPSimEnv, mdp: TabularMDP, args: argparse.Namespace, seed: int):
    common = dict(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        decay_steps=args.eps_decay,
        seed=seed,
    )
    if algo == "td_n_forward":
        return td_n_forward(env, mdp, n=args.n, **common)
    if algo == "td_n_backward":
        return td_n_backward(env, mdp, n=args.n, **common)
    if algo == "td_lambda_forward":
        return td_lambda_forward(env, mdp, lam=args.lam, **common)
    if algo == "td_lambda_backward":
        return td_lambda_backward(env, mdp, lam=args.lam, **common)
    if algo == "sarsa_n_forward":
        return sarsa_n_forward(env, n=args.n, **common)
    if algo == "sarsa_n_backward":
        return sarsa_n_backward(env, n=args.n, **common)
    if algo == "sarsa_lambda_forward":
        return sarsa_lambda_forward(env, lam=args.lam, **common)
    if algo == "sarsa_lambda_backward":
        return sarsa_lambda_backward(env, lam=args.lam, **common)
    raise ValueError(f"Unsupported algorithm: {algo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit forward-view and backward-view classical RL variants for V2.")
    parser.add_argument("--mdp", default="outputs/V2/mdp/mdp.npz")
    parser.add_argument("--outdir", default="outputs/V2/classical_views")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.8)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=2000)
    parser.add_argument("--seeds", type=str, default="7,11,19")
    args = parser.parse_args()

    algos = [
        "td_n_forward",
        "td_n_backward",
        "td_lambda_forward",
        "td_lambda_backward",
        "sarsa_n_forward",
        "sarsa_n_backward",
        "sarsa_lambda_forward",
        "sarsa_lambda_backward",
    ]
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
            result = run_algo(algo, env, mdp, args, seed)
            metrics = rollout_policy(mdp, result.policy, episodes=args.eval_episodes, seed=seed, max_steps=args.max_steps)
            per_seed_rows.append({"algo": algo, "seed": seed, "rollout_mean": float(metrics["avg_return"])})

    with (outdir / "per_seed_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "seed", "rollout_mean"])
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary_rows = []
    for algo in algos:
        vals = np.array([r["rollout_mean"] for r in per_seed_rows if r["algo"] == algo], dtype=float)
        summary_rows.append(
            {
                "algo": algo,
                "rollout_mean": float(vals.mean()),
                "rollout_std": float(vals.std(ddof=0)),
                "rollout_ci95": float(ci95(float(vals.std(ddof=0)), len(vals))),
            }
        )

    with (outdir / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "rollout_mean", "rollout_std", "rollout_ci95"])
        writer.writeheader()
        writer.writerows(summary_rows)

    labels = [r["algo"] for r in summary_rows]
    means = [r["rollout_mean"] for r in summary_rows]
    errs = [r["rollout_ci95"] for r in summary_rows]
    apply_v2_theme()
    plt.figure(figsize=(11, 5))
    plt.bar(np.arange(len(labels)), means, yerr=errs, capsize=5, color=colors_for(labels), alpha=0.93)
    annotate_bars(plt.gca(), means, fmt="{:.2f}")
    plt.xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("Average Return")
    plt.title("V2 Classical RL Forward-View vs Backward-View Comparison")
    plt.tight_layout()
    plt.savefig(figdir / "classical_views_comparison.png", dpi=180)
    plt.close()

    save_json(outdir / "suite_config.json", vars(args))
    (outdir / "INTERPRETATION.md").write_text(
        "\n".join(
            [
                "# V2 Classical View Comparison",
                "",
                "- This suite compares explicit forward-view and backward-view implementations of TD and SARSA family methods.",
                "- Forward-view methods use explicit return targets.",
                "- Backward-view methods use eligibility traces or truncated trace-style updates.",
                "- These outputs exist to satisfy the view-based classical RL requirements and make the distinction visible in V2.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote classical view outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
