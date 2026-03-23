from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from viz_theme import apply_v2_theme


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dqn_checkpoint(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def forward_q(ckpt: dict[str, np.ndarray], state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_pre = ckpt["online__w1"][state] + ckpt["online__b1"]
    h = np.maximum(0.0, h_pre)
    q = h @ ckpt["online__w2"] + ckpt["online__b2"]
    return q, h, h_pre


def input_gradient_saliency(ckpt: dict[str, np.ndarray], state: int, action: int) -> np.ndarray:
    w1 = ckpt["online__w1"]
    w2 = ckpt["online__w2"]
    _, _, h_pre = forward_q(ckpt, state)
    active = (h_pre > 0.0).astype(float)
    return (w1 * (active * w2[:, action])).sum(axis=1)


def hidden_contributions(ckpt: dict[str, np.ndarray], state: int, action: int) -> np.ndarray:
    _, h, _ = forward_q(ckpt, state)
    return h * ckpt["online__w2"][:, action]


def plot_input_saliency_heatmap(mat: np.ndarray, out_path: Path) -> None:
    apply_v2_theme()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    vmax = max(float(np.max(np.abs(mat))), 1e-9)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="dQ(selected action) / d(input state one-hot)")
    ax.set_xlabel("Input state dimension")
    ax.set_ylabel("Current state")
    ax.set_title("V2 DQN Neural Saliency: Input-State Sensitivity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_hidden_contrib_heatmap(mat: np.ndarray, out_path: Path) -> None:
    apply_v2_theme()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Hidden-unit contribution to selected Q")
    ax.set_xlabel("Hidden unit")
    ax.set_ylabel("Current state")
    ax.set_title("V2 DQN Neural Saliency: Hidden Contributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create neural saliency views from the saved V2 DQN checkpoint.")
    parser.add_argument("--checkpoint", default="V2/checkpoints/dqn/foundation_env/default/model_checkpoint.npz")
    parser.add_argument("--policy-path", default="outputs/V2/rl_dqn/policy.json")
    parser.add_argument("--outdir", default="outputs/V2/figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = load_dqn_checkpoint(Path(args.checkpoint))
    policy = load_json(Path(args.policy_path))

    n_states = ckpt["online__w1"].shape[0]
    hidden = ckpt["online__w1"].shape[1]
    input_sal = np.zeros((n_states, n_states), dtype=float)
    hidden_sal = np.zeros((n_states, hidden), dtype=float)
    top_rows = []

    for s in range(n_states):
        action = int(policy.get(str(s), 0))
        grad = input_gradient_saliency(ckpt, s, action)
        contrib = hidden_contributions(ckpt, s, action)
        input_sal[s] = grad
        hidden_sal[s] = contrib
        top_idx = np.argsort(np.abs(grad))[::-1][:3]
        top_rows.append(
            {
                "state": s,
                "selected_action": action,
                "top_input_dims": [int(i) for i in top_idx],
                "top_input_scores": [float(grad[i]) for i in top_idx],
            }
        )

    input_path = outdir / "nn_saliency_dqn_input_heatmap.png"
    hidden_path = outdir / "nn_saliency_dqn_hidden_heatmap.png"
    summary_path = outdir / "nn_saliency_dqn_summary.json"
    interp_path = outdir / "NN_SALIENCY_INTERPRETATION.md"

    plot_input_saliency_heatmap(input_sal, input_path)
    plot_hidden_contrib_heatmap(hidden_sal, hidden_path)
    summary_path.write_text(json.dumps(top_rows, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V2 Neural Saliency Interpretation",
        "",
        "## What these figures show",
        "- `nn_saliency_dqn_input_heatmap.png`: sensitivity of the selected DQN action value to each input state dimension under a one-hot state encoding.",
        "- `nn_saliency_dqn_hidden_heatmap.png`: hidden-unit contributions to the selected action value in each state.",
        "",
        "## How to read them",
        "- Larger absolute values indicate stronger influence on the DQN decision.",
        "- Positive values increase the selected action value, while negative values suppress it.",
        "- In this compact environment, the saliency is best interpreted as state-to-state sensitivity rather than image-style attention.",
        "",
        "## Example state summaries",
    ]
    for row in top_rows[: min(5, len(top_rows))]:
        dims = ", ".join(f"S{idx} ({score:.3f})" for idx, score in zip(row["top_input_dims"], row["top_input_scores"]))
        lines.append(f"- State S{row['state']} action A{row['selected_action']}: strongest input influences -> {dims}")
    interp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {input_path.resolve()}")
    print(f"Wrote: {hidden_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")
    print(f"Wrote: {interp_path.resolve()}")


if __name__ == "__main__":
    main()
