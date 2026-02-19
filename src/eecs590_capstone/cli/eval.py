from __future__ import annotations

import argparse
from pathlib import Path

from eecs590_capstone.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved policy")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-path", type=str, default="outputs/policy.json")
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(outdir / "eval_args.json", vars(args))
    save_json(
        outdir / "eval_results.json",
        {
            "status": "disabled",
            "reason": "Toy foundation MDP moved to sandbox/. Use data-driven CLI instead.",
        },
    )

    print("Eval disabled.")
    print("Toy foundation MDP was moved to sandbox/. Use data-driven CLI instead.")


if __name__ == "__main__":
    main()
