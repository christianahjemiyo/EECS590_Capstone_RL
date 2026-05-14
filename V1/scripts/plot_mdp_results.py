from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_policy_table(policy: dict, out_path: Path) -> None:
    lines = []
    lines.append("State -> Action")
    for k in sorted(policy.keys(), key=lambda x: int(x)):
        lines.append(f"{k} -> {policy[k]}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_value_bar(values: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(values)), values)
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_policy_bar(policy: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(len(policy)), policy)
    plt.xlabel("State")
    plt.ylabel("Action")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_heatmap(values: np.ndarray, out_path: Path, title: str, cmap: str = "viridis") -> None:
    plt.figure(figsize=(6, 2))
    mat = values.reshape(1, -1)
    plt.imshow(mat, aspect="auto", cmap=cmap)
    plt.yticks([])
    plt.xticks(np.arange(len(values)), [str(i) for i in range(len(values))])
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DP results for the simulated MDP.")
    parser.add_argument("--outdir", default="outputs/mdp")
    parser.add_argument("--algo", default="policy_iter", choices=["policy_iter", "value_iter"])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    policy_path = outdir / f"{args.algo}_policy.json"
    value_path = outdir / f"{args.algo}_value_function.json"

    if not policy_path.exists() or not value_path.exists():
        raise FileNotFoundError("Run mdp_train.py first to generate policy/value JSON.")

    policy_dict = load_json(policy_path)
    value_dict = load_json(value_path)

    policy = np.array([policy_dict[str(i)] for i in sorted(map(int, policy_dict.keys()))], dtype=int)
    values = np.array([value_dict[str(i)] for i in sorted(map(int, value_dict.keys()))], dtype=float)

    plot_value_bar(values, outdir / f"{args.algo}_value_bar.png", f"{args.algo} value function")
    plot_policy_bar(policy, outdir / f"{args.algo}_policy_bar.png", f"{args.algo} policy")
    plot_heatmap(values, outdir / f"{args.algo}_value_heatmap.png", f"{args.algo} value heatmap")
    plot_heatmap(policy.astype(float), outdir / f"{args.algo}_policy_heatmap.png", f"{args.algo} policy heatmap")

    save_policy_table(policy_dict, outdir / f"{args.algo}_policy_human.txt")
    print(f"Wrote plots and human-readable policy to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
