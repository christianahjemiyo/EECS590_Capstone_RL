from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz_theme import apply_v2_theme, annotate_bars, colors_for


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary_csv(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "algo": row["algo"],
                    "rollout_mean": float(row["rollout_mean"]),
                    "rollout_std": float(row["rollout_std"]),
                    "rollout_ci95": float(row["rollout_ci95"]),
                }
            )
    return rows


def family_of(algo: str) -> str:
    if algo in {"policy_iteration", "value_iteration"}:
        return "DP"
    if algo in {"mc_control", "td0", "td_lambda", "sarsa", "expected_sarsa", "sarsa_lambda", "q_learning", "q_lambda", "dyna_q"}:
        return "Tabular"
    if algo in {"linear_fa", "semi_gradient_td", "gradient_td", "approx_q_learning"}:
        return "Approximation"
    if algo in {"dqn", "double_dqn", "dueling_dqn"}:
        return "Deep Value"
    if algo in {"reinforce", "a2c", "a3c", "ppo", "trpo"}:
        return "Actor-Critic"
    return "Advanced"


def plot_family_panel(summary_rows: list[dict], out_path: Path) -> None:
    family_best: dict[str, tuple[str, float]] = {}
    for row in summary_rows:
        fam = family_of(str(row["algo"]))
        score = float(row["rollout_mean"])
        if fam not in family_best or score > family_best[fam][1]:
            family_best[fam] = (str(row["algo"]), score)

    families = list(family_best.keys())
    labels = [f"{fam}\n({family_best[fam][0]})" for fam in families]
    means = [family_best[fam][1] for fam in families]

    apply_v2_theme()
    plt.figure(figsize=(9.6, 4.8))
    plt.bar(np.arange(len(labels)), means, color=colors_for(families), alpha=0.93)
    annotate_bars(plt.gca(), means, fmt="{:.2f}")
    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel("Best Average Return")
    plt.title("V2 Algorithm Families: Best Result in Each Group")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_reward_cost_tradeoff(meta: dict, out_path: Path) -> None:
    action_strengths = np.array(meta.get("action_strengths", []), dtype=float)
    action_costs = np.array(meta.get("action_costs", []), dtype=float)
    if action_strengths.size == 0 or action_costs.size == 0:
        raise ValueError("Missing action strengths or costs in MDP metadata.")

    x = np.arange(len(action_strengths))
    width = 0.35
    apply_v2_theme()
    plt.figure(figsize=(8.2, 4.6))
    plt.bar(x - width / 2, action_strengths, width=width, label="Intervention strength", alpha=0.9)
    plt.bar(x + width / 2, action_costs, width=width, label="Intervention cost", alpha=0.9)
    plt.xticks(x, [f"A{i}" for i in range(len(action_strengths))])
    plt.ylabel("Value")
    plt.title("V2 Intervention Strength vs Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_policy_transition_diagram(meta: dict, policy: dict[str, int], out_path: Path) -> None:
    n_states = int(meta["n_states"])
    risk_bins = meta.get("risk_bins", [])
    action_strengths = meta.get("action_strengths", [])
    action_costs = meta.get("action_costs", [])

    xs = np.linspace(0.12, 0.88, n_states)
    ys = np.full(n_states, 0.55)

    apply_v2_theme()
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i in range(n_states - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.07, ys[i]), xytext=(xs[i] + 0.07, ys[i]), arrowprops=dict(arrowstyle="->", linewidth=2))

    for s in range(n_states):
        action = int(policy.get(str(s), 0))
        strength = action_strengths[action] if action < len(action_strengths) else float("nan")
        cost = action_costs[action] if action < len(action_costs) else float("nan")
        if s == 0:
            risk_text = "lowest risk"
        elif s == n_states - 1:
            risk_text = "highest risk"
        else:
            risk_text = "mid risk"
        if s < len(risk_bins):
            risk_text = f"<= {risk_bins[s]:.2f}"
        circle = plt.Circle((xs[s], ys[s]), 0.07, color="#dfe7ec", ec="#264653", lw=2)
        ax.add_patch(circle)
        ax.text(xs[s], ys[s] + 0.005, f"S{s}", ha="center", va="center", fontsize=11, weight="bold")
        ax.text(xs[s], ys[s] - 0.12, f"A{action}\nstr={strength:.2f}\ncost={cost:.2f}", ha="center", va="top", fontsize=8)
        ax.text(xs[s], ys[s] + 0.12, risk_text, ha="center", va="bottom", fontsize=8)

    ax.set_title("V2 Policy Transition Diagram")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create additional V2 visual summaries.")
    parser.add_argument("--summary", default="outputs/V2/all_algorithms/summary_metrics.csv")
    parser.add_argument("--policy-path", default="outputs/V2/rl/policy.json")
    parser.add_argument("--mdp-meta", default="outputs/V2/mdp/mdp_meta.json")
    parser.add_argument("--outdir", default="outputs/V2/figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = load_summary_csv(Path(args.summary))
    policy = load_json(Path(args.policy_path))
    meta = load_json(Path(args.mdp_meta))

    family_out = outdir / "algorithm_family_overview.png"
    tradeoff_out = outdir / "reward_cost_tradeoff.png"
    diagram_out = outdir / "policy_transition_diagram.png"

    plot_family_panel(summary_rows, family_out)
    plot_reward_cost_tradeoff(meta, tradeoff_out)
    plot_policy_transition_diagram(meta, policy, diagram_out)

    interp = [
        "# V2 Special Visualizations",
        "",
        "- `algorithm_family_overview.png`: best-performing method in each algorithm family.",
        "- `reward_cost_tradeoff.png`: side-by-side view of intervention strength and intervention cost.",
        "- `policy_transition_diagram.png`: simplified state progression view with the selected action in each state.",
    ]
    (outdir / "SPECIAL_VISUALS_INTERPRETATION.md").write_text("\n".join(interp) + "\n", encoding="utf-8")

    print(f"Wrote: {family_out.resolve()}")
    print(f"Wrote: {tradeoff_out.resolve()}")
    print(f"Wrote: {diagram_out.resolve()}")
    print(f"Wrote: {(outdir / 'SPECIAL_VISUALS_INTERPRETATION.md').resolve()}")


if __name__ == "__main__":
    main()
