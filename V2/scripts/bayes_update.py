from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(probs: np.ndarray) -> np.ndarray:
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive value.")
    return probs / total


def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    posterior = prior * likelihood
    return normalize(posterior)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic Bayes update helper for V2 belief-state work.")
    parser.add_argument("--prior", required=True, help="Comma-separated prior probabilities.")
    parser.add_argument("--likelihood", required=True, help="Comma-separated likelihood values.")
    parser.add_argument("--out", default="outputs/V2/bayes_update_example.json")
    args = parser.parse_args()

    prior = normalize(np.array([float(x.strip()) for x in args.prior.split(",") if x.strip()], dtype=float))
    likelihood = np.array([float(x.strip()) for x in args.likelihood.split(",") if x.strip()], dtype=float)
    if prior.shape != likelihood.shape:
        raise ValueError("Prior and likelihood must have the same length.")

    posterior = bayes_update(prior, likelihood)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "prior": prior.tolist(),
                "likelihood": likelihood.tolist(),
                "posterior": posterior.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Bayes update example to: {out.resolve()}")


if __name__ == "__main__":
    main()
