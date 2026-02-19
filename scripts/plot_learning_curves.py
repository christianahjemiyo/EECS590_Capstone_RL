from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt


def load_curve(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data.get("episode_returns", []), dtype=float)


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) == 0:
        return y
    w = np.ones(window) / window
    return np.convolve(y, w, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves from RL outputs.")
    parser.add_argument("--root", default="outputs/rl")
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--out", default="outputs/rl/learning_curves.png")
    args = parser.parse_args()

    root = Path(args.root)
    curves = {}
    for algo_dir in root.iterdir():
        if not algo_dir.is_dir():
            continue
        run_curves = []
        for run_dir in algo_dir.glob("run_*"):
            curve_path = run_dir / "learning_curve.json"
            if curve_path.exists():
                run_curves.append(load_curve(curve_path))
        if run_curves:
            curves[algo_dir.name] = run_curves

    if not curves:
        raise FileNotFoundError("No learning_curve.json files found under outputs/rl/*/run_*/.")

    plt.figure(figsize=(7, 4))
    for name, runs in curves.items():
        min_len = min(len(r) for r in runs)
        if min_len == 0:
            continue
        trimmed = np.stack([r[:min_len] for r in runs], axis=0)
        mean = trimmed.mean(axis=0)
        std = trimmed.std(axis=0)
        mean_s = smooth(mean, args.window)
        std_s = smooth(std, args.window)
        xs = np.arange(len(mean_s))
        plt.plot(xs, mean_s, label=name)
        plt.fill_between(xs, mean_s - std_s, mean_s + std_s, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Return (smoothed)")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
