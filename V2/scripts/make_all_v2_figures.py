from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all V2 figures and interpretation outputs in one command.")
    parser.add_argument("--mdp", default="outputs/V2/mdp/mdp.npz")
    parser.add_argument("--data", default="data/processed/train.csv")
    parser.add_argument("--out-root", default="outputs/V2")
    parser.add_argument("--benchmark-seeds", type=int, default=5)
    parser.add_argument("--suite-seeds", default="7,11,19,23,29")
    parser.add_argument("--quick", action="store_true", help="Faster run for sanity checks.")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--skip-tabular-suite", action="store_true")
    parser.add_argument("--skip-approx-suite", action="store_true")
    parser.add_argument("--skip-deep-suite", action="store_true")
    parser.add_argument("--skip-advanced-suite", action="store_true")
    parser.add_argument("--skip-all-algorithms", action="store_true")
    parser.add_argument("--skip-dqn", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    py = sys.executable
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    out_root = Path(args.out_root)
    mdp = args.mdp
    data = args.data

    if not args.skip_dqn:
        dqn_cmd = [
            py,
            "V2/scripts/train_dqn.py",
            "--mdp",
            mdp,
            "--outdir",
            str(out_root / "rl_dqn"),
        ]
        if args.quick:
            dqn_cmd += ["--episodes", "1200", "--warmup-steps", "150", "--target-update", "100"]
        run_cmd(dqn_cmd, cwd=repo_root, env=env)

    run_cmd([py, "V2/scripts/plot_v2_results.py"], cwd=repo_root, env=env)
    run_cmd([py, "V2/scripts/write_v2_interpretation.py"], cwd=repo_root, env=env)
    run_cmd(
        [
            py,
            "V2/scripts/plot_v2_saliency.py",
            "--data",
            data,
            "--outdir",
            str(out_root / "figures"),
        ],
        cwd=repo_root,
        env=env,
    )

    if not args.skip_offline:
        offline_cmd = [
            py,
            "V2/scripts/offline_rl_benchmark.py",
            "--mdp",
            mdp,
            "--outdir",
            str(out_root / "offline"),
        ]
        if args.quick:
            offline_cmd += ["--episodes", "1200", "--fqi-iters", "40", "--cql-epochs", "30"]
        run_cmd(offline_cmd, cwd=repo_root, env=env)

    if not args.skip_benchmark:
        bench_seeds = 2 if args.quick else args.benchmark_seeds
        bench_cmd = [
            py,
            "V2/scripts/run_v2_benchmark.py",
            "--mdp",
            mdp,
            "--outdir",
            str(out_root / "benchmark"),
            "--seeds",
            str(bench_seeds),
        ]
        if args.quick:
            bench_cmd += ["--online-episodes", "1000", "--offline-episodes", "1000", "--rollout-episodes", "1000"]
        run_cmd(bench_cmd, cwd=repo_root, env=env)

    if not args.skip_tabular_suite:
        suite_cmd = [
            py,
            "V2/scripts/run_v2_tabular_suite.py",
            "--mdp",
            mdp,
            "--outdir",
            str(out_root / "tabular_suite"),
            "--seeds",
            args.suite_seeds,
        ]
        if args.quick:
            suite_cmd += ["--episodes", "1000", "--eval-episodes", "1000"]
        run_cmd(suite_cmd, cwd=repo_root, env=env)

    if not args.skip_approx_suite:
        approx_cmd = [
            py,
            "V2/scripts/run_v2_approx_suite.py",
            "--mdp",
            mdp,
            "--outdir",
            str(out_root / "approx_suite"),
        ]
        if args.quick:
            approx_cmd += ["--episodes", "1000", "--eval-episodes", "1000", "--seeds", "7,11"]
        run_cmd(approx_cmd, cwd=repo_root, env=env)

    if not args.skip_deep_suite:
        deep_cmd = [
            py,
            "V2/scripts/run_v2_deep_suite.py",
            "--mdp",
            mdp,
            "--outdir",
            str(out_root / "deep_suite"),
        ]
        if args.quick:
            deep_cmd += ["--episodes", "800", "--eval-episodes", "1000", "--seeds", "7,11"]
        run_cmd(deep_cmd, cwd=repo_root, env=env)

    if not args.skip_advanced_suite:
        advanced_cmd = [
            py,
            "V2/scripts/run_v2_advanced_suite.py",
            "--mdp",
            mdp,
            "--outdir",
            str(out_root / "advanced_suite"),
        ]
        if args.quick:
            advanced_cmd += ["--episodes", "800", "--eval-episodes", "1000", "--seeds", "7,11"]
        run_cmd(advanced_cmd, cwd=repo_root, env=env)

    if not args.skip_all_algorithms:
        all_cmd = [
            py,
            "V2/scripts/run_v2_all_algorithms.py",
            "--mdp",
            mdp,
            "--outdir",
            str(out_root / "all_algorithms"),
        ]
        if args.quick:
            all_cmd += ["--episodes", "600", "--eval-episodes", "1000", "--seeds", "7,11"]
        run_cmd(all_cmd, cwd=repo_root, env=env)

    print("\nAll requested V2 figures and interpretation files are generated.")
    print(f"See: {(repo_root / out_root).resolve()}")


if __name__ == "__main__":
    main()
