from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict | None = None) -> None:
    print(f"==> {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)


def main() -> int:
    python = sys.executable
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{src}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    run([python, "scripts/preprocess.py"], env=env)
    run([python, "scripts/data_profile.py"], env=env)
    run([python, "scripts/build_mdp.py"], env=env)
    run([python, "-m", "eecs590_capstone.cli.mdp_train", "--algo", "policy_iter"], env=env)
    run([python, "-m", "eecs590_capstone.cli.mdp_train", "--algo", "value_iter"], env=env)
    run([python, "scripts/plot_mdp_results.py", "--algo", "policy_iter"], env=env)
    run([python, "scripts/plot_mdp_results.py", "--algo", "value_iter"], env=env)
    run([python, "scripts/render_mdp_html.py", "--policy", "outputs/mdp/policy_iter_policy.json"], env=env)
    run([python, "scripts/run_all_rl.py", "--runs", "5"], env=env)
    run([python, "scripts/plot_learning_curves.py"], env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
