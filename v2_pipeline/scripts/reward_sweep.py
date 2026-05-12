from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eecs590_capstone.utils.io import load_json, save_json
from eecs590_capstone.utils.reward_search import candidate_name, materialize_reward_candidate


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_cmd(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def read_summary_mean(path: Path, algo: str) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["algo"] == algo:
                return float(row["rollout_mean"])
    raise ValueError(f"Missing algo '{algo}' in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep reward maps and action-cost scales for the V2 benchmark.")
    parser.add_argument("--base-config", default="v2_pipeline/configs/mdp_sim_mimic.json")
    parser.add_argument("--outdir", default="outputs/v2_outputs/reward_sweep")
    parser.add_argument("--no-rewards", default="3.0,4.0,5.0")
    parser.add_argument("--late-penalties", default="-1.5,-2.0,-3.0")
    parser.add_argument("--early-penalties", default="-8.0,-10.0,-12.0")
    parser.add_argument("--cost-scales", default="0.5,1.0,1.5")
    parser.add_argument("--max-candidates", type=int, default=9)
    parser.add_argument("--benchmark-seeds", type=int, default=3)
    parser.add_argument("--benchmark-online-episodes", type=int, default=1200)
    parser.add_argument("--benchmark-offline-episodes", type=int, default=1200)
    parser.add_argument("--benchmark-rollout-episodes", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=30)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    configs_dir = outdir / "configs"
    mdp_dir = outdir / "mdp"
    bench_dir = outdir / "benchmarks"
    outdir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    mdp_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_json(Path(args.base_config))
    combos: list[tuple[float, float, float, float]] = []
    for no_reward in parse_csv_floats(args.no_rewards):
        for late_penalty in parse_csv_floats(args.late_penalties):
            for early_penalty in parse_csv_floats(args.early_penalties):
                for cost_scale in parse_csv_floats(args.cost_scales):
                    combos.append((no_reward, late_penalty, early_penalty, cost_scale))
    combos = combos[: args.max_candidates]

    summary_rows: list[dict[str, str | float]] = []
    learned_algos = ["DP_PolicyIter", "DP_ValueIter", "Q_Learning", "Double_Q", "Offline_CQL", "Offline_IQL"]
    for no_reward, late_penalty, early_penalty, cost_scale in combos:
        cfg, tag = materialize_reward_candidate(
            base_cfg,
            no_reward=no_reward,
            late_penalty=late_penalty,
            early_penalty=early_penalty,
            cost_scale=cost_scale,
        )
        name = candidate_name(tag)
        cfg_path = configs_dir / f"{name}.json"
        candidate_mdp_dir = mdp_dir / name
        candidate_bench_dir = bench_dir / name
        save_json(cfg_path, cfg)

        run_cmd([sys.executable, "scripts/build_mdp.py", "--config", str(cfg_path), "--outdir", str(candidate_mdp_dir)])
        run_cmd(
            [
                sys.executable,
                "v2_pipeline/scripts/run_v2_benchmark.py",
                "--mdp",
                str(candidate_mdp_dir / "mdp.npz"),
                "--outdir",
                str(candidate_bench_dir),
                "--seeds",
                str(args.benchmark_seeds),
                "--online-episodes",
                str(args.benchmark_online_episodes),
                "--offline-episodes",
                str(args.benchmark_offline_episodes),
                "--rollout-episodes",
                str(args.benchmark_rollout_episodes),
                "--max-steps",
                str(args.max_steps),
            ]
        )
        summary_path = candidate_bench_dir / "summary_metrics.csv"
        learned_best = max(read_summary_mean(summary_path, algo) for algo in learned_algos)
        behavior = read_summary_mean(summary_path, "Behavior_Action0")
        summary_rows.append(
            {
                "candidate": name,
                "no_reward": no_reward,
                "late_penalty": late_penalty,
                "early_penalty": early_penalty,
                "cost_scale": cost_scale,
                "best_learned_rollout": learned_best,
                "behavior_rollout": behavior,
                "delta_vs_behavior": learned_best - behavior,
            }
        )

    summary_rows.sort(key=lambda row: float(row["delta_vs_behavior"]), reverse=True)
    with (outdir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "no_reward",
                "late_penalty",
                "early_penalty",
                "cost_scale",
                "best_learned_rollout",
                "behavior_rollout",
                "delta_vs_behavior",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = ["# Reward Sweep Interpretation", ""]
    if summary_rows:
        best = summary_rows[0]
        lines.append("## Best candidate")
        lines.append(
            f"- `{best['candidate']}` improved best-learned rollout over behavior by {float(best['delta_vs_behavior']):.3f}."
        )
        lines.append(
            f"- Reward map: NO={float(best['no_reward']):.2f}, >30={float(best['late_penalty']):.2f}, <30={float(best['early_penalty']):.2f}."
        )
        lines.append(f"- Action-cost scale: {float(best['cost_scale']):.2f}.")
    lines.append("")
    lines.append("## What this sweep does")
    lines.append("- Treats reward design as a tunable modeling choice instead of a fixed assumption.")
    lines.append("- Rebuilds the MDP and reruns the compact benchmark for each candidate.")
    lines.append("- Ranks candidates by how much the best learned policy beats the behavior baseline.")
    (outdir / "INTERPRETATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (outdir / "sweep_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reward sweep outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()

