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

from eecs590_capstone.agents.dp_policy_iter import policy_iteration
from eecs590_capstone.agents.dp_value_iter import value_iteration
from eecs590_capstone.agents.rl_advanced import belief_state_update, ddpg, learned_environment_model, rnn_based_rl, sac, td3
from eecs590_capstone.agents.rl_approx import approximate_q_learning, gradient_td, linear_value_function_approximation, semi_gradient_td
from eecs590_capstone.agents.rl_deep import a2c, a3c, ppo, reinforce, train_dqn_variant, trpo
from eecs590_capstone.agents.rl_tabular import dyna_q, expected_sarsa, mc_control, q_lambda, q_learning, sarsa, sarsa_lambda, td0, td_lambda
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


def run_online_algo(name: str, mdp: TabularMDP, args: argparse.Namespace, seed: int):
    env = MDPSimEnv(mdp_path=args.mdp, seed=seed, max_steps=args.max_steps)
    if name == "policy_iteration":
        result = policy_iteration(mdp, gamma=args.gamma)
        return result["policy"]
    if name == "value_iteration":
        result = value_iteration(mdp, gamma=args.gamma)
        return result["policy"]
    if name == "mc_control":
        return mc_control(env, args.episodes, args.gamma, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "td0":
        return td0(env, mdp, args.episodes, 0.1, args.gamma, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "td_lambda":
        return td_lambda(env, mdp, args.episodes, 0.1, args.gamma, 0.8, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "sarsa":
        return sarsa(env, args.episodes, 0.1, args.gamma, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "expected_sarsa":
        return expected_sarsa(env, args.episodes, 0.1, args.gamma, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "sarsa_lambda":
        return sarsa_lambda(env, args.episodes, 0.1, args.gamma, 0.8, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "q_learning":
        return q_learning(env, args.episodes, 0.1, args.gamma, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "q_lambda":
        return q_lambda(env, args.episodes, 0.1, args.gamma, 0.8, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "dyna_q":
        return dyna_q(env, args.episodes, 0.1, args.gamma, 10, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "linear_fa":
        return linear_value_function_approximation(env, args.episodes, 0.05, args.gamma, seed).policy
    if name == "semi_gradient_td":
        return semi_gradient_td(env, args.episodes, 0.05, args.gamma, seed).policy
    if name == "gradient_td":
        return gradient_td(env, args.episodes, 0.05, 0.02, args.gamma, seed).policy
    if name == "approx_q_learning":
        return approximate_q_learning(env, args.episodes, 0.05, args.gamma, 1.0, 0.05, max(500, args.episodes // 2), seed).policy
    if name == "dqn":
        return train_dqn_variant(env, args.episodes, args.gamma, 1e-3, 32, 64, 20000, 300, 200, 1.0, 0.05, max(500, args.episodes), seed, "dqn").policy
    if name == "double_dqn":
        return train_dqn_variant(env, args.episodes, args.gamma, 1e-3, 32, 64, 20000, 300, 200, 1.0, 0.05, max(500, args.episodes), seed, "double_dqn").policy
    if name == "dueling_dqn":
        return train_dqn_variant(env, args.episodes, args.gamma, 1e-3, 32, 64, 20000, 300, 200, 1.0, 0.05, max(500, args.episodes), seed, "dueling_dqn").policy
    if name == "reinforce":
        return reinforce(env, args.episodes, 0.05, args.gamma, seed).policy
    if name == "a2c":
        return a2c(env, args.episodes, 0.05, 0.1, args.gamma, seed).policy
    if name == "a3c":
        return a3c(env, args.episodes, 0.05, 0.1, args.gamma, seed, workers=4).policy
    if name == "ppo":
        return ppo(env, args.episodes, 0.05, 0.1, args.gamma, 0.2, seed).policy
    if name == "trpo":
        return trpo(env, args.episodes, 0.05, 0.1, args.gamma, 0.02, seed).policy
    if name == "ddpg":
        return ddpg(env, args.episodes, args.gamma, 0.02, 0.05, 0.35, seed).policy
    if name == "td3":
        return td3(env, args.episodes, args.gamma, 0.02, 0.05, 0.35, 0.2, 2, seed).policy
    if name == "sac":
        return sac(env, args.episodes, 0.1, 0.05, args.gamma, seed).policy
    if name == "learned_model":
        return learned_environment_model(mdp, args.episodes, args.max_steps, args.gamma, seed).policy
    if name == "belief_update":
        return belief_state_update(mdp, args.episodes, args.max_steps, args.gamma, seed, n_obs=3).policy
    if name == "rnn_rl":
        return rnn_based_rl(mdp, args.episodes, args.max_steps, args.gamma, 0.1, seed, n_obs=3, hidden_size=8).policy
    raise ValueError(f"Unsupported algorithm: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single V2 comparison across all implemented algorithms.")
    parser.add_argument("--mdp", default="outputs/V2/mdp/mdp.npz")
    parser.add_argument("--outdir", default="outputs/V2/all_algorithms")
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seeds", type=str, default="7,11,19")
    args = parser.parse_args()

    all_algos = [
        "policy_iteration",
        "value_iteration",
        "mc_control",
        "td0",
        "td_lambda",
        "sarsa",
        "expected_sarsa",
        "sarsa_lambda",
        "q_learning",
        "q_lambda",
        "dyna_q",
        "linear_fa",
        "semi_gradient_td",
        "gradient_td",
        "approx_q_learning",
        "dqn",
        "double_dqn",
        "dueling_dqn",
        "reinforce",
        "a2c",
        "a3c",
        "ppo",
        "trpo",
        "ddpg",
        "td3",
        "sac",
        "learned_model",
        "belief_update",
        "rnn_rl",
    ]

    seeds = parse_seeds(args.seeds)
    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    mdp = load_mdp(Path(args.mdp))

    per_seed_rows = []
    for algo in all_algos:
        for seed in seeds:
            policy = run_online_algo(algo, mdp, args, seed)
            metrics = rollout_policy(mdp, policy, episodes=args.eval_episodes, seed=seed, max_steps=args.max_steps)
            per_seed_rows.append({"algo": algo, "seed": seed, "rollout_mean": float(metrics["avg_return"]), "rollout_std": float(metrics["std_return"])})

    with (outdir / "per_seed_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "seed", "rollout_mean", "rollout_std"])
        writer.writeheader()
        writer.writerows(per_seed_rows)

    summary_rows = []
    for algo in all_algos:
        vals = np.array([r["rollout_mean"] for r in per_seed_rows if r["algo"] == algo], dtype=float)
        summary_rows.append(
            {
                "algo": algo,
                "rollout_mean": float(vals.mean()),
                "rollout_std": float(vals.std(ddof=0)),
                "rollout_ci95": float(ci95(float(vals.std(ddof=0)), len(vals))),
            }
        )
    summary_rows.sort(key=lambda r: r["rollout_mean"], reverse=True)

    with (outdir / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "rollout_mean", "rollout_std", "rollout_ci95"])
        writer.writeheader()
        writer.writerows(summary_rows)

    labels = [r["algo"] for r in summary_rows]
    means = [r["rollout_mean"] for r in summary_rows]
    errs = [r["rollout_ci95"] for r in summary_rows]

    apply_v2_theme()
    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(labels)), means, yerr=errs, capsize=4, color=colors_for(labels), alpha=0.93)
    plt.xticks(np.arange(len(labels)), labels, rotation=55, ha="right")
    plt.ylabel("Average Return")
    plt.title("V2 All Algorithms Comparison")
    plt.tight_layout()
    plt.savefig(figdir / "all_algorithms_rollout_comparison.png", dpi=180)
    plt.close()

    top10 = summary_rows[:10]
    apply_v2_theme()
    plt.figure(figsize=(11.5, 5))
    plt.bar(np.arange(len(top10)), [r["rollout_mean"] for r in top10], yerr=[r["rollout_ci95"] for r in top10], capsize=4, color=colors_for([r["algo"] for r in top10]), alpha=0.93)
    annotate_bars(plt.gca(), [r["rollout_mean"] for r in top10], fmt="{:.2f}")
    plt.xticks(np.arange(len(top10)), [r["algo"] for r in top10], rotation=35, ha="right")
    plt.ylabel("Average Return")
    plt.title("V2 Top 10 Algorithms")
    plt.tight_layout()
    plt.savefig(figdir / "top10_algorithms_rollout.png", dpi=180)
    plt.close()

    best = summary_rows[0]
    lines = [
        "# V2 All Algorithms Interpretation",
        "",
        "## Main result",
        f"- Best mean rollout return: **{best['algo']}** ({best['rollout_mean']:.3f} +/- {best['rollout_ci95']:.3f}).",
        "",
        "## Files",
        "- `summary_metrics.csv`",
        "- `per_seed_metrics.csv`",
        "- `figures/all_algorithms_rollout_comparison.png`",
        "- `figures/top10_algorithms_rollout.png`",
        "",
        "## Note",
        "- Advanced continuous-control and partial-observability methods are adapted to the discrete V2 MDP so they can be compared in one pipeline.",
    ]
    (outdir / "INTERPRETATION_ALL_ALGOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(outdir / "all_algorithms_config.json", vars(args))
    print(f"Wrote all-algorithm comparison outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
