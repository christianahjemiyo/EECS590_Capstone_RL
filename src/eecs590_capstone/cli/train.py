from __future__ import annotations

import argparse
from pathlib import Path

from eecs590_capstone.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DP agent using policy iteration")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--theta", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--outdir", type=str, default="outputs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(outdir / "train_args.json", vars(args))
    save_json(
        outdir / "train_info.json",
        {
            "status": "disabled",
            "reason": "Toy foundation MDP moved to sandbox/. Use data-driven CLI instead.",
        },
    )

    print("Training disabled.")
    print("Toy foundation MDP was moved to sandbox/. Use data-driven CLI instead.")


if __name__ == "__main__":
    main()
