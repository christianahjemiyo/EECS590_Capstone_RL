from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz_theme import apply_v2_theme


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _numeric_key_order(d: dict) -> list[str]:
    return sorted(d.keys(), key=lambda k: int(k))


def build_state_action_saliency(q_values: dict[str, list[float]]) -> np.ndarray:
    keys = _numeric_key_order(q_values)
    q = np.array([q_values[k] for k in keys], dtype=float)
    state_means = q.mean(axis=1, keepdims=True)
    saliency = q - state_means
    return saliency


def plot_state_action_saliency(
    saliency: np.ndarray, policy: dict[str, int], meta: dict, out_path: Path
) -> None:
    n_states, n_actions = saliency.shape
    risk_bins = meta.get("risk_bins", [])
    action_strengths = meta.get("action_strengths", [])
    action_costs = meta.get("action_costs", [])

    state_labels = []
    for s in range(n_states):
        if s == 0:
            state_labels.append(f"S{s} (lowest risk)")
        elif s == n_states - 1:
            state_labels.append(f"S{s} (highest risk)")
        else:
            state_labels.append(f"S{s}")

    action_labels = []
    for a in range(n_actions):
        strength = action_strengths[a] if a < len(action_strengths) else float("nan")
        cost = action_costs[a] if a < len(action_costs) else float("nan")
        action_labels.append(f"A{a}\nstr={strength:.2f}, cost={cost:.2f}")

    vmax = float(np.max(np.abs(saliency))) if saliency.size else 1.0
    vmax = max(vmax, 1e-9)

    apply_v2_theme()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(saliency, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Relative Action Advantage (Q - state mean)")

    ax.set_xticks(np.arange(n_actions))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(action_labels)
    ax.set_yticklabels(state_labels)
    ax.set_xlabel("Intervention Action")
    ax.set_ylabel("Risk State")
    title_suffix = f" | bins={risk_bins}" if risk_bins else ""
    ax.set_title(f"V2 Saliency Map: Action Impact by Risk State{title_suffix}")

    for s in range(n_states):
        for a in range(n_actions):
            marker = ""
            if int(policy.get(str(s), -1)) == a:
                marker = " *"
            ax.text(
                a,
                s,
                f"{saliency[s, a]:.2f}{marker}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def _feature_weighted_lifts(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in data.")

    y = df[label_col].astype(str)
    p_lt30 = float((y == "<30").mean())
    p_no = float((y == "NO").mean())
    n = float(len(df))

    skip = {label_col, "encounter_id", "patient_nbr", "hadm_id", "subject_id"}
    rows = []

    for col in df.columns:
        if col in skip:
            continue
        s = df[col]
        if s.isna().all():
            continue

        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.98 and num.nunique(dropna=True) > 12:
            binned = pd.qcut(num, q=5, duplicates="drop")
            groups = binned.astype(str).fillna("UNK")
        else:
            groups = s.astype(str).fillna("UNK")

        tmp = pd.DataFrame({"g": groups, "y": y})
        grp = tmp.groupby("g", dropna=False)
        counts = grp.size()
        lt30 = grp["y"].apply(lambda y_col: float((y_col == "<30").mean()))
        no = grp["y"].apply(lambda y_col: float((y_col == "NO").mean()))

        w = counts / n
        risk_lift = float(np.maximum(lt30 - p_lt30, 0.0).mul(w).sum())
        protect_lift = float(np.maximum(no - p_no, 0.0).mul(w).sum())
        total = float(risk_lift + protect_lift)

        rows.append(
            {
                "feature": col,
                "risk_lift_lt30": risk_lift,
                "protective_lift_no": protect_lift,
                "total_saliency": total,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("total_saliency", ascending=False).reset_index(drop=True)
    return out


def plot_feature_saliency_heatmap(df_saliency: pd.DataFrame, out_path: Path, top_k: int = 12) -> None:
    top = df_saliency.head(top_k).copy()
    if top.empty:
        raise ValueError("No feature saliency rows to plot.")

    mat = top[["risk_lift_lt30", "protective_lift_no"]].to_numpy(dtype=float)
    labels = top["feature"].tolist()
    cols = ["Risk Lift (<30)", "Protective Lift (NO)"]

    apply_v2_theme()
    fig, ax = plt.subplots(figsize=(8.8, 0.45 * len(labels) + 1.8))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Weighted Lift Contribution")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("V2 Saliency Map: Which Features Drive Readmission Risk vs Recovery")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create V2 saliency maps for policy/action and patient features.")
    parser.add_argument("--q-path", default="outputs/V2/rl/q_values.json")
    parser.add_argument("--policy-path", default="outputs/V2/rl/policy.json")
    parser.add_argument("--mdp-meta", default="outputs/V2/mdp/mdp_meta.json")
    parser.add_argument("--data", default="data/processed/train.csv")
    parser.add_argument("--label-col", default="readmitted")
    parser.add_argument("--outdir", default="outputs/V2/figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    q_values = load_json(Path(args.q_path))
    policy = load_json(Path(args.policy_path))
    meta = load_json(Path(args.mdp_meta)) if Path(args.mdp_meta).exists() else {}

    saliency = build_state_action_saliency(q_values)
    state_action_path = outdir / "saliency_state_action_heatmap.png"
    plot_state_action_saliency(saliency, policy, meta, state_action_path)

    df = pd.read_csv(args.data, low_memory=False)
    df_saliency = _feature_weighted_lifts(df, args.label_col)
    feat_csv = outdir / "saliency_feature_scores.csv"
    df_saliency.to_csv(feat_csv, index=False)
    feat_map_path = outdir / "saliency_feature_outcome_heatmap.png"
    plot_feature_saliency_heatmap(df_saliency, feat_map_path, top_k=12)

    top = df_saliency.head(5)
    lines = [
        "# V2 Saliency Interpretation",
        "",
        "## What these maps show",
        "- `saliency_state_action_heatmap.png`: how strongly each action is favored/disfavored by Q-values in each risk state.",
        "- `saliency_feature_outcome_heatmap.png`: which patient features most increase modeled readmission risk lift (`<30`) or protective lift (`NO`).",
        "",
        "## Top feature signals (by weighted lift)",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"- {row['feature']}: total={row['total_saliency']:.4f}, risk_lift={row['risk_lift_lt30']:.4f}, protective_lift={row['protective_lift_no']:.4f}"
        )
    (outdir / "SALIENCY_INTERPRETATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {state_action_path.resolve()}")
    print(f"Wrote: {feat_map_path.resolve()}")
    print(f"Wrote: {feat_csv.resolve()}")
    print(f"Wrote: {(outdir / 'SALIENCY_INTERPRETATION.md').resolve()}")


if __name__ == "__main__":
    main()
