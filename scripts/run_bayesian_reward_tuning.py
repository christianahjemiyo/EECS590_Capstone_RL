from __future__ import annotations

"""Run lightweight Bayesian hyperparameter tuning for reward design.

This script is intentionally targeted at the existing capstone structure. It
does not tune an unrelated deep architecture. Instead, it tunes reward-design
and action-cost parameters that materially affect policy behavior in the
hospital readmission MDP benchmark.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np

from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.agents.dp_policy_iter import policy_iteration
from eecs590_capstone.utils.bayesian_tuning import SearchBounds, as_tag, propose_next
from eecs590_capstone.utils.io import load_json, save_json
from eecs590_capstone.utils.reward_search import candidate_name, materialize_reward_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Version 3 Bayesian hyperparameter tuning for reward and action-cost settings."
    )
    parser.add_argument("--base-config", default="v2_pipeline/configs/mdp_sim_mimic.json")
    parser.add_argument("--outdir", default="outputs/V3/bayesian_tuning")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--rollout-episodes", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=30)
    return parser.parse_args()


def build_mdp_for_candidate(cfg_path: Path, outdir: Path) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [sys.executable, "scripts/build_mdp.py", "--config", str(cfg_path), "--outdir", str(outdir)],
        cwd=ROOT,
        check=True,
        env=env,
    )
    return outdir / "mdp.npz"


def score_candidate(
    mdp_path: Path,
    *,
    gamma: float,
    rollout_episodes: int,
    seed: int,
) -> tuple[float, dict[str, float], dict[str, int]]:
    data = np.load(mdp_path)
    mdp = TabularMDP(P=data["P"], R=data["R"], terminal_states=[])
    result = policy_iteration(mdp, gamma=gamma)
    metrics = rollout_policy(mdp, result["policy"], episodes=rollout_episodes, seed=seed)
    score = float(metrics["avg_return"])
    policy = {str(k): int(v) for k, v in result["policy"].items()}
    return score, metrics, policy


def save_trials_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_progress_plot(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        return
    best_so_far = []
    current_best = -1e18
    for row in rows:
        current_best = max(current_best, float(row["score"]))
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, marker="o")
    ax.set_title("Bayesian Tuning Progress")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best avg rollout return so far")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    outdir = ROOT / Path(args.outdir)
    configs_dir = outdir / "configs"
    mdp_root = outdir / "mdp_candidates"
    outdir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    mdp_root.mkdir(parents=True, exist_ok=True)

    base_cfg = load_json(ROOT / args.base_config)
    bounds = SearchBounds(
        no_reward=(3.0, 6.0),
        late_penalty=(-3.5, -0.5),
        early_penalty=(-14.0, -6.0),
        cost_scale=(0.4, 1.8),
    )

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    table_rows: list[dict[str, float | str]] = []
    best_row: dict[str, float | str] | None = None

    for trial_idx in range(args.n_trials):
        if trial_idx < args.n_init:
            x = bounds.sample(rng, n=1)[0]
            source = "random_init"
        else:
            x = propose_next(np.vstack(X_rows), np.array(y_rows, dtype=float), bounds, rng=rng)
            source = "bayesian_proposal"

        tag = as_tag(x)
        cfg, _ = materialize_reward_candidate(base_cfg, **tag)
        name = candidate_name(tag)
        cfg_path = configs_dir / f"trial_{trial_idx:02d}_{name}.json"
        candidate_outdir = mdp_root / f"trial_{trial_idx:02d}_{name}"
        save_json(cfg_path, cfg)

        mdp_path = build_mdp_for_candidate(cfg_path, candidate_outdir)
        score, metrics, policy = score_candidate(
            mdp_path,
            gamma=args.gamma,
            rollout_episodes=args.rollout_episodes,
            seed=args.seed + trial_idx,
        )

        row: dict[str, float | str] = {
            "trial": float(trial_idx),
            "source": source,
            "candidate": name,
            "no_reward": tag["no_reward"],
            "late_penalty": tag["late_penalty"],
            "early_penalty": tag["early_penalty"],
            "cost_scale": tag["cost_scale"],
            "score": score,
            "std_return": float(metrics["std_return"]),
            "terminal_rate": float(metrics["terminal_rate"]),
        }
        table_rows.append(row)
        X_rows.append(x.astype(float))
        y_rows.append(score)

        save_json(candidate_outdir / "policy_iter_policy.json", policy)
        save_json(candidate_outdir / "policy_eval.json", metrics)

        if best_row is None or float(row["score"]) > float(best_row["score"]):
            best_row = row

        print(
            f"Trial {trial_idx + 1}/{args.n_trials}: {name} | "
            f"score={score:.4f} | source={source}"
        )

    assert best_row is not None
    save_trials_csv(table_rows, outdir / "trials.csv")
    save_progress_plot(table_rows, outdir / "tuning_progress.png")
    save_json(outdir / "best_config.json", best_row)

    summary_lines = [
        "# Bayesian Tuning Summary",
        "",
        "This search applies lightweight Bayesian hyperparameter tuning to the capstone's reward design.",
        "The objective is policy-iteration average rollout return under the rebuilt tabular MDP.",
        "",
        "## Best candidate",
        f"- candidate: {best_row['candidate']}",
        f"- score: {float(best_row['score']):.4f}",
        f"- no_reward: {float(best_row['no_reward']):.4f}",
        f"- late_penalty: {float(best_row['late_penalty']):.4f}",
        f"- early_penalty: {float(best_row['early_penalty']):.4f}",
        f"- cost_scale: {float(best_row['cost_scale']):.4f}",
        "",
        "## Why this is relevant",
        "- Reward design and intervention cost scaling materially affect policy behavior in this readmission-planning MDP.",
        "- Bayesian tuning is a better fit for this capstone than forcing unrelated multi-agent or hierarchy methods.",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved Bayesian tuning outputs to: {outdir}")
    print(f"Best score: {float(best_row['score']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
