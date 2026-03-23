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


def main() -> None:
    parser = argparse.ArgumentParser(description="Rasterize V2 state-action values into a compact image.")
    parser.add_argument("--q-path", default="outputs/V2/rl/q_values.json")
    parser.add_argument("--out", default="outputs/V2/figures/state_action_raster.png")
    args = parser.parse_args()

    q_values = load_json(Path(args.q_path))
    keys = sorted(q_values.keys(), key=lambda k: int(k))
    mat = np.array([q_values[k] for k in keys], dtype=float)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    apply_v2_theme()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Q-value")
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title("V2 Rasterized State-Action Value Grid")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote rasterized state-action figure to: {out.resolve()}")


if __name__ == "__main__":
    main()
