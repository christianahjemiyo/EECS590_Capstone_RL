from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


OUTDIR = Path("outputs")
POLICY_PATH = OUTDIR / "policy.json"
VALUE_PATH = OUTDIR / "value_function.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(f"Missing {POLICY_PATH}. Run: python -m eecs590_capstone.cli.train")
    if not VALUE_PATH.exists():
        raise FileNotFoundError(f"Missing {VALUE_PATH}. Run: python -m eecs590_capstone.cli.train")

    policy = load_json(POLICY_PATH)          # dict: state(str) -> action(int)
    V = load_json(VALUE_PATH)                # dict: state(str) -> value(float)

    # Sort states numerically
    states = sorted([int(s) for s in policy.keys()])
    actions = [int(policy[str(s)]) for s in states]
    values = [float(V[str(s)]) for s in states]

    # ------------------------------------------------------------
    # (1) Text visualization
    # ------------------------------------------------------------
    print("\n=== (1) Policy (state -> action) ===")
    for s in states:
        a = int(policy[str(s)])
        print(f"State {s} -> Action {a}")

    print("\n=== Value function V(s) ===")
    for s in states:
        v = float(V[str(s)])
        print(f"V({s}) = {v:.4f}")

    # ------------------------------------------------------------
    # (2) Value bar plot
    # ------------------------------------------------------------
    plt.figure()
    plt.bar(states, values)
    plt.xlabel("State")
    plt.ylabel("V(s)")
    plt.title("Learned Value Function")
    plt.xticks(states)
    plt.tight_layout()
    plt.savefig(OUTDIR / "value_function_plot.png", dpi=200)

    # ------------------------------------------------------------
    # (3) Policy diagram (simple “timeline” view)
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 2))
    y = [0] * len(states)
    plt.scatter(states, y, s=350)

    for s, a in zip(states, actions):
        label = f"a={a}"
        plt.text(s, 0.08, label, ha="center", va="bottom")

    plt.yticks([])
    plt.xticks(states)
    plt.xlabel("State progression")
    plt.title("Policy Visualization (action per state)")
    plt.ylim(-0.2, 0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / "policy_plot.png", dpi=200)

    print("\nSaved plots:")
    print(f"- {OUTDIR / 'value_function_plot.png'}")
    print(f"- {OUTDIR / 'policy_plot.png'}")
    print()


if __name__ == "__main__":
    main()
