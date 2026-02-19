from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from eecs590_capstone.utils.io import load_json, save_json


def parse_age(age_val: str) -> float:
    if not isinstance(age_val, str):
        return float("nan")
    s = age_val.strip().replace("[", "").replace(")", "")
    if "-" not in s:
        return float("nan")
    lo, hi = s.split("-", 1)
    try:
        return (float(lo) + float(hi)) / 2.0
    except ValueError:
        return float("nan")


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_medications",
        "num_procedures",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
    ]
    data = {}
    for col in cols:
        if col in df.columns:
            data[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            data[col] = pd.Series([np.nan] * len(df))

    age = pd.Series([parse_age(v) for v in df.get("age", pd.Series([np.nan] * len(df)))])
    data["age_mid"] = age

    mat = pd.DataFrame(data)
    mat = mat.fillna(mat.median(numeric_only=True))
    z = (mat - mat.mean()) / (mat.std(ddof=0) + 1e-6)
    score = z.sum(axis=1)
    return score


def bin_risk(score: pd.Series, bins: list[float]) -> pd.Series:
    quantiles = score.quantile(bins).to_numpy()
    return pd.Series(np.digitize(score.to_numpy(), quantiles, right=True))


def build_transition_counts(df: pd.DataFrame, state_col: str) -> Tuple[np.ndarray, int]:
    n_states = int(df[state_col].max() + 1)
    counts = np.zeros((n_states, n_states), dtype=float)

    if "patient_nbr" in df.columns and "encounter_id" in df.columns:
        df_sorted = df.sort_values(["patient_nbr", "encounter_id"])
        for _, grp in df_sorted.groupby("patient_nbr"):
            states = grp[state_col].to_numpy()
            if len(states) < 2:
                continue
            for s, s_next in zip(states[:-1], states[1:]):
                counts[int(s), int(s_next)] += 1.0
    else:
        states = df[state_col].to_numpy()
        for s, s_next in zip(states[:-1], states[1:]):
            counts[int(s), int(s_next)] += 1.0

    return counts, n_states


def build_base_transition(counts: np.ndarray, laplace: float) -> np.ndarray:
    counts = counts + laplace
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def improve_transition(P: np.ndarray) -> np.ndarray:
    n_states = P.shape[0]
    P_improve = np.zeros_like(P)
    for s in range(n_states):
        for s_next in range(n_states):
            target = max(0, s_next - 1)
            P_improve[s, target] += P[s, s_next]
    return P_improve


def build_action_transitions(P_base: np.ndarray, action_strengths: list[float]) -> np.ndarray:
    P_improve = improve_transition(P_base)
    n_states = P_base.shape[0]
    n_actions = len(action_strengths)
    P = np.zeros((n_states, n_actions, n_states), dtype=float)
    for a, alpha in enumerate(action_strengths):
        P[:, a, :] = (1.0 - alpha) * P_base + alpha * P_improve
    return P


def reward_by_state(df: pd.DataFrame, state_col: str, reward_map: Dict[str, float]) -> np.ndarray:
    n_states = int(df[state_col].max() + 1)
    rewards = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        subset = df[df[state_col] == s]
        if len(subset) == 0:
            rewards[s] = 0.0
            continue
        labels = subset["readmitted"].astype(str)
        r = 0.0
        for label, val in reward_map.items():
            r += float((labels == label).mean()) * float(val)
        rewards[s] = r
    return rewards


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a tabular MDP simulator from the dataset.")
    parser.add_argument("--config", default="configs/mdp_sim.json")
    parser.add_argument("--outdir", default="outputs/mdp")
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    df = pd.read_csv(cfg["data_path"], low_memory=False)
    df = df.replace("?", np.nan)

    score = compute_risk_score(df)
    df["risk_state"] = bin_risk(score, cfg["risk_bins"]).astype(int)

    counts, n_states = build_transition_counts(df, "risk_state")
    P_base = build_base_transition(counts, cfg.get("laplace", 1.0))
    P = build_action_transitions(P_base, cfg["action_strengths"])

    r_state = reward_by_state(df, "risk_state", cfg["reward_map"])
    n_actions = len(cfg["action_strengths"])
    R = np.zeros_like(P)
    for a in range(n_actions):
        for s in range(n_states):
            R[s, a, :] = r_state[s] - float(cfg["action_costs"][a])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "mdp.npz", P=P, R=R)
    save_json(
        outdir / "mdp_meta.json",
        {
            "n_states": n_states,
            "n_actions": n_actions,
            "risk_bins": cfg["risk_bins"],
            "action_strengths": cfg["action_strengths"],
            "action_costs": cfg["action_costs"],
            "reward_map": cfg["reward_map"],
        },
    )

    print(f"Wrote: {outdir / 'mdp.npz'}")
    print(f"Wrote: {outdir / 'mdp_meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
