from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def smooth(y: np.ndarray, window: int = 100) -> np.ndarray:
    if len(y) == 0 or window <= 1:
        return y
    if len(y) < window:
        window = max(2, len(y) // 2)
    w = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, w, mode="valid")


def plot_learning_curve(rl_series: dict[str, Path], outdir: Path) -> None:
    plt.figure(figsize=(7, 4))
    plotted = 0
    for name, rl_dir in rl_series.items():
        curve_path = rl_dir / "learning_curve.json"
        if not curve_path.exists():
            continue
        data = load_json(curve_path)
        returns = np.array(data.get("episode_returns", []), dtype=float)
        if len(returns) == 0:
            continue
        sm = smooth(returns, window=100)
        xs = np.arange(len(sm))
        plt.plot(xs, sm, label=f"{name} (smoothed)")
        plotted += 1

    if plotted == 0:
        raise FileNotFoundError("No learning_curve.json found for V2 RL outputs.")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("V2 RL Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "rl_learning_curve.png", dpi=180)
    plt.close()


def plot_algo_comparison(mdp_dir: Path, rl_series: dict[str, Path], outdir: Path) -> None:
    pi = load_json(mdp_dir / "policy_iter_eval.json")
    vi = load_json(mdp_dir / "value_iter_eval.json")
    labels = ["Policy Iter", "Value Iter"]
    means = [float(pi.get("avg_return", 0.0)), float(vi.get("avg_return", 0.0))]
    stds = [float(pi.get("std_return", 0.0)), float(vi.get("std_return", 0.0))]

    for name, rl_dir in rl_series.items():
        eval_path = rl_dir / "eval_results.json"
        if not eval_path.exists():
            continue
        ev = load_json(eval_path)
        labels.append(name)
        means.append(float(ev.get("avg_return", 0.0)))
        stds.append(float(ev.get("std_return", 0.0)))

    plt.figure(figsize=(7, 4))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.ylabel("Average Return")
    plt.title("V2 Algorithm Comparison")
    plt.tight_layout()
    plt.savefig(outdir / "algo_avg_return_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    outdir = Path("outputs/V2/figures")
    mdp_dir = Path("outputs/V2/mdp")
    rl_series = {
        "Q-Learning": Path("outputs/V2/rl"),
        "Double Q-Learning": Path("outputs/V2/rl_double_q"),
    }
    outdir.mkdir(parents=True, exist_ok=True)

    plot_learning_curve(rl_series, outdir)
    plot_algo_comparison(mdp_dir, rl_series, outdir)

    print(f"Wrote: {(outdir / 'rl_learning_curve.png').resolve()}")
    print(f"Wrote: {(outdir / 'algo_avg_return_comparison.png').resolve()}")


if __name__ == "__main__":
    main()
