from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_eval(path: Path) -> dict | None:
    if not path.exists():
        return None
    return load_json(path)


def main() -> None:
    out_path = Path("outputs/V2/INTERPRETATION.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pi = maybe_eval(Path("outputs/V2/mdp/policy_iter_eval.json"))
    vi = maybe_eval(Path("outputs/V2/mdp/value_iter_eval.json"))
    ql = maybe_eval(Path("outputs/V2/rl/eval_results.json"))
    dql = maybe_eval(Path("outputs/V2/rl_double_q/eval_results.json"))

    lines = []
    lines.append("# V2 Interpretation (Presentation Notes)")
    lines.append("")
    lines.append("## 1) What these results mean in plain words")
    lines.append("- We are testing how well each algorithm learns a policy that avoids costly readmission outcomes.")
    lines.append("- In this setup, higher return is better (less negative is better).")
    lines.append("- Because readmission penalties are large, negative returns are expected; the key is relative performance.")
    lines.append("")
    lines.append("## 2) Quick metric snapshot")
    lines.append("| Algorithm | Avg Return | Std Return |")
    lines.append("| --- | ---: | ---: |")

    def row(name: str, ev: dict | None) -> str:
        if ev is None:
            return f"| {name} | n/a | n/a |"
        return f"| {name} | {float(ev.get('avg_return', 0.0)):.3f} | {float(ev.get('std_return', 0.0)):.3f} |"

    lines.append(row("Policy Iteration (DP)", pi))
    lines.append(row("Value Iteration (DP)", vi))
    lines.append(row("Q-Learning", ql))
    lines.append(row("Double Q-Learning", dql))
    lines.append("")
    lines.append("## 3) How to read the figures")
    lines.append("- `outputs/V2/mdp/*_value_bar.png`: how valuable each risk-state is under the learned plan.")
    lines.append("- `outputs/V2/mdp/*_policy_bar.png`: which action each state is assigned.")
    lines.append("- `outputs/V2/figures/rl_learning_curve.png`: how quickly RL improves with more episodes.")
    lines.append("- `outputs/V2/figures/algo_avg_return_comparison.png`: side-by-side comparison of final performance.")
    lines.append("")
    lines.append("## 4) Talking points for your instructor")
    lines.append("- DP methods are the benchmark here because they solve the known tabular MDP directly.")
    lines.append("- RL methods are useful because they scale to settings where transitions are not explicitly known.")
    lines.append("- Double Q-Learning is a stronger variant of Q-Learning because it reduces overestimation bias.")
    lines.append("- V2 is now fully reproducible: preprocessing, training, and figures are versioned with results.")
    lines.append("")
    lines.append("## 5) Honest limitation to mention")
    lines.append("- Current simulator uses proxy transitions derived from data, not causal treatment effects.")
    lines.append("- This is a strong engineering baseline, and a bridge toward offline RL with richer clinical actions.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
