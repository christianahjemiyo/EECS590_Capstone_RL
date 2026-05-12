from __future__ import annotations

"""Create Version 3 world-model interpretation figures.

These figures support interpretation of the lightweight world-model extension by
showing where the learned simulator differs from the reference tabular MDP and
how simple policies behave under the learned decision-support simulation.

The outputs are intentionally simple and readable so they fit the current
tabular hospital readmission planning project without adding heavy plotting
dependencies.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Version 3 world-model figures.")
    parser.add_argument(
        "--outdir",
        default="outputs/V3/world_model",
        help="Directory containing world-model CSV outputs and where figures will be saved.",
    )
    return parser.parse_args()


def make_transition_error_heatmap(outdir: Path) -> None:
    metrics_path = outdir / "world_model_metrics.csv"
    if not metrics_path.exists():
        print(f"Skipping transition error heatmap. Missing file: {metrics_path}")
        return

    df = pd.read_csv(metrics_path)
    if not {"state", "action", "transition_mae"}.issubset(df.columns):
        print("Skipping transition error heatmap. Required columns are missing from world_model_metrics.csv.")
        return

    heatmap_df = (
        df.pivot_table(index="action", columns="state", values="transition_mae", aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(heatmap_df.to_numpy(), aspect="auto")
    ax.set_title("World Model Transition Error by State and Action")
    ax.set_xlabel("State")
    ax.set_ylabel("Action")
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels([str(col) for col in heatmap_df.columns])
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels([str(idx) for idx in heatmap_df.index])
    fig.colorbar(image, ax=ax, label="Transition MAE")
    fig.tight_layout()

    path = outdir / "transition_error_heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


def make_policy_return_comparison(outdir: Path) -> None:
    policy_eval_path = outdir / "policy_eval_world_model.csv"
    if not policy_eval_path.exists():
        print(f"Skipping policy return comparison. Missing file: {policy_eval_path}")
        return

    df = pd.read_csv(policy_eval_path)
    required = {"policy_name", "avg_return"}
    if not required.issubset(df.columns):
        print("Skipping policy return comparison. Required columns are missing from policy_eval_world_model.csv.")
        return

    target_policies = ["random_policy", "conservative_policy", "aggressive_policy"]
    filtered = df[df["policy_name"].isin(target_policies)].copy()
    if filtered.empty:
        print("Skipping policy return comparison. No baseline policy rows were found.")
        return

    filtered["policy_name"] = pd.Categorical(filtered["policy_name"], categories=target_policies, ordered=True)
    filtered = filtered.sort_values("policy_name")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(filtered["policy_name"].astype(str), filtered["avg_return"])
    ax.set_title("Average Return in the World Model Simulator")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Average simulated return")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    path = outdir / "policy_return_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


def make_terminal_rate_comparison(outdir: Path) -> None:
    policy_eval_path = outdir / "policy_eval_world_model.csv"
    note_path = outdir / "readmission_or_terminal_rate_note.txt"
    if not policy_eval_path.exists():
        note_path.write_text(
            "Terminal-rate comparison was skipped because policy_eval_world_model.csv was not found.\n",
            encoding="utf-8",
        )
        print(f"Skipping terminal-rate comparison. Missing file: {policy_eval_path}")
        print(f"Wrote note: {note_path}")
        return

    df = pd.read_csv(policy_eval_path)
    required = {"policy_name", "terminal_rate"}
    if not required.issubset(df.columns):
        note_path.write_text(
            "Terminal-rate comparison was skipped because the terminal_rate column was not available.\n",
            encoding="utf-8",
        )
        print("Skipping terminal-rate comparison. terminal_rate column is missing.")
        print(f"Wrote note: {note_path}")
        return

    target_policies = ["random_policy", "conservative_policy", "aggressive_policy"]
    filtered = df[df["policy_name"].isin(target_policies)].copy()
    if filtered.empty:
        note_path.write_text(
            "Terminal-rate comparison was skipped because baseline policy rows were not available.\n",
            encoding="utf-8",
        )
        print("Skipping terminal-rate comparison. No baseline policy rows were found.")
        print(f"Wrote note: {note_path}")
        return

    filtered["policy_name"] = pd.Categorical(filtered["policy_name"], categories=target_policies, ordered=True)
    filtered = filtered.sort_values("policy_name")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(filtered["policy_name"].astype(str), filtered["terminal_rate"])
    ax.set_title("Terminal-State Rate in the World Model Simulator")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Terminal-state rate")
    ax.set_ylim(0.0, max(1.0, float(filtered["terminal_rate"].max()) * 1.1))
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()

    path = outdir / "readmission_or_terminal_rate_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")

    note_text = (
        "The world-model outputs did not include a direct clinical readmission indicator. "
        "This figure uses the available terminal-state rate from the learned simulator as a proxy summary.\n"
    )
    note_path.write_text(note_text, encoding="utf-8")
    print(f"Wrote note: {note_path}")


def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)

    if not outdir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {outdir}")

    print(f"Reading world-model outputs from: {outdir}")
    make_transition_error_heatmap(outdir)
    make_policy_return_comparison(outdir)
    make_terminal_rate_comparison(outdir)
    print("World-model figure generation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
