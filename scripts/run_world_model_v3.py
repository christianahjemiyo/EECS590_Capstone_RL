from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from eecs590_capstone.utils.io import save_json
from eecs590_capstone.world_model import (
    TabularWorldModel,
    WorldModelSimulator,
    compare_world_model_to_mdp,
    evaluate_policy_in_world_model,
    save_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Version 3 world-model runner for modeled hospital readmission "
            "decision-support simulation."
        )
    )
    parser.add_argument(
        "--mdp",
        default=None,
        help="Path to the existing MDP .npz file. Defaults to a V2 MDP path if available.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/V3/world_model",
        help="Directory where Version 3 world-model outputs will be written.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=100,
        help="Number of simulated trajectories per policy.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Maximum trajectory length per rollout.",
    )
    return parser.parse_args()


def resolve_mdp_path(user_path: str | None) -> Path:
    """Resolve the MDP path while staying compatible with old and new layouts."""

    candidates: list[Path] = []
    if user_path:
        requested = Path(user_path)
        candidates.append(requested)

        # The repository was reorganized locally from outputs/V2 to
        # outputs/v2_outputs. This compatibility shim keeps the script runnable
        # with the older command shown in the project instructions.
        requested_str = str(requested).replace("\\", "/")
        if "outputs/V2/" in requested_str:
            remapped = Path(requested_str.replace("outputs/V2/", "outputs/v2_outputs/"))
            candidates.append(remapped)
    else:
        candidates.extend(
            [
                Path("outputs/V2/mdp/mdp.npz"),
                Path("outputs/v2_outputs/mdp/mdp.npz"),
                Path("outputs/mdp/mdp.npz"),
            ]
        )

    for candidate in candidates:
        absolute = candidate if candidate.is_absolute() else ROOT / candidate
        if absolute.exists():
            return absolute

    searched = "\n".join(f"- {c}" for c in candidates) if candidates else "- <none>"
    raise FileNotFoundError(f"Could not find an MDP file. Searched:\n{searched}")


def infer_terminal_states(P: np.ndarray) -> list[int]:
    """Infer terminal states conservatively from self-loop absorbing states."""

    terminal_states: list[int] = []
    n_states = int(P.shape[0])
    n_actions = int(P.shape[1])
    for state in range(n_states):
        absorbing = True
        for action in range(n_actions):
            probs = P[state, action, :]
            expected = np.zeros(n_states, dtype=float)
            expected[state] = 1.0
            if not np.allclose(probs, expected, atol=1e-10):
                absorbing = False
                break
        if absorbing:
            terminal_states.append(state)
    return terminal_states


def build_fixed_policies(n_states: int, n_actions: int, rng: np.random.Generator) -> dict[str, dict[str, int]]:
    """Create simple baseline policies for simulator stress tests."""

    low_action = 0
    high_action = max(0, n_actions - 1)
    random_policy = {str(state): int(rng.integers(0, n_actions)) for state in range(n_states)}
    conservative = {str(state): low_action for state in range(n_states)}
    aggressive = {str(state): high_action for state in range(n_states)}

    return {
        "random_policy": random_policy,
        "conservative_policy": conservative,
        "aggressive_policy": aggressive,
    }


def maybe_load_saved_policies() -> tuple[dict[str, dict[str, int]], list[str]]:
    """Load a small set of existing V2 policies when available.

    The world-model extension stays lightweight by only reading already-saved
    JSON policy artifacts. It does not retrain or depend on any large V2 run.
    """

    policy_candidates = {
        "v2_tabular_q_policy": ROOT / "outputs" / "v2_outputs" / "rl" / "policy.json",
        "v2_dqn_policy": ROOT / "outputs" / "v2_outputs" / "rl_dqn" / "policy.json",
        "v2_double_q_policy": ROOT / "outputs" / "v2_outputs" / "rl_double_q" / "policy.json",
    }

    loaded: dict[str, dict[str, int]] = {}
    notes: list[str] = []
    for label, path in policy_candidates.items():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded[label] = {str(k): int(v) for k, v in payload.items()}
        except Exception as exc:  # pragma: no cover - defensive logging path
            notes.append(f"Skipped {label} at {path}: {exc}")

    if not loaded:
        notes.append(
            "No saved V2 policies were loaded. The world-model extension remains lightweight and reproducible "
            "without requiring precomputed trained-policy artifacts."
        )

    return loaded, notes


def collect_trajectory_examples(
    world_model: TabularWorldModel,
    policies: Mapping[str, Mapping[str, int]],
    *,
    seed: int,
    horizon: int,
    n_trajectories: int,
) -> pd.DataFrame:
    """Generate a flat table of example transitions across policies."""

    rows: list[dict[str, float | int | str | bool]] = []
    episodes_to_store = min(5, int(n_trajectories))

    for offset, (policy_name, policy) in enumerate(policies.items()):
        simulator = WorldModelSimulator(
            world_model,
            seed=seed + offset,
            max_steps=horizon,
        )
        trajectories = simulator.simulate_trajectories(policy, episodes=episodes_to_store, max_steps=horizon)

        for episode_idx, trajectory in enumerate(trajectories):
            for step_idx, step in enumerate(trajectory):
                rows.append(
                    {
                        "policy_name": policy_name,
                        "episode": episode_idx,
                        "step": step_idx,
                        "state": int(step["state"]),
                        "action": int(step["action"]),
                        "next_state": int(step["next_state"]),
                        "reward": float(step["reward"]),
                        "done": bool(step["done"]),
                    }
                )

    return pd.DataFrame(rows)


def write_policy_eval_csv(
    metrics_by_policy: Mapping[str, Mapping[str, float]],
    path: Path,
) -> None:
    rows = []
    for policy_name, metrics in metrics_by_policy.items():
        row = {"policy_name": policy_name}
        row.update(metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary(
    path: Path,
    *,
    mdp_path: Path,
    outdir: Path,
    mean_transition_mae: float,
    mean_transition_mse: float,
    mean_reward_mae: float,
    mean_transition_kl: float | None,
    loaded_policy_names: list[str],
    notes: list[str],
) -> None:
    lines = [
        "Version 3 World Model Summary",
        "",
        f"MDP source: {mdp_path}",
        f"Output directory: {outdir}",
        "",
        "Main fit metrics against the original tabular MDP:",
        f"- Mean transition MAE: {mean_transition_mae:.6f}",
        f"- Mean transition MSE: {mean_transition_mse:.6f}",
        f"- Reward MAE: {mean_reward_mae:.6f}",
    ]
    if mean_transition_kl is not None:
        lines.append(f"- Mean transition KL divergence: {mean_transition_kl:.6f}")

    lines.extend(
        [
            "",
            "Included policies:",
            *[f"- {name}" for name in loaded_policy_names],
            "",
            "Notes:",
            *[f"- {note}" for note in notes],
            "",
            "Interpretation reminder:",
            "- This world model is a lightweight tabular decision-support simulator for modeled readmission risk.",
            "- It is intended for safe policy comparison inside the project MDP abstraction, not for clinical deployment.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    mdp_path = resolve_mdp_path(args.mdp)
    outdir = ROOT / Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(mdp_path)
    P = data["P"]
    R = data["R"]
    terminal_states = infer_terminal_states(P)

    world_model = TabularWorldModel(smoothing=1.0).fit_from_mdp(
        mdp_path=mdp_path,
        terminal_states=terminal_states,
    )
    world_model.save(outdir / "tabular_world_model.npz")

    metrics_df = compare_world_model_to_mdp(world_model, mdp_path=mdp_path)
    save_metrics_csv(metrics_df, outdir / "world_model_metrics.csv")

    fixed_policies = build_fixed_policies(world_model.n_states, world_model.n_actions, rng)
    saved_policies, notes = maybe_load_saved_policies()
    all_policies: dict[str, dict[str, int]] = {}
    all_policies.update(fixed_policies)
    all_policies.update(saved_policies)

    policy_metrics: dict[str, dict[str, float]] = {}
    for policy_name, policy in all_policies.items():
        policy_metrics[policy_name] = evaluate_policy_in_world_model(
            world_model,
            policy,
            episodes=args.n_trajectories,
            seed=args.seed,
            max_steps=args.horizon,
        )

    write_policy_eval_csv(policy_metrics, outdir / "policy_eval_world_model.csv")

    trajectory_examples = collect_trajectory_examples(
        world_model,
        all_policies,
        seed=args.seed,
        horizon=args.horizon,
        n_trajectories=args.n_trajectories,
    )
    trajectory_examples.to_csv(outdir / "trajectory_examples.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    fit_summary = {
        "mdp_path": str(mdp_path),
        "terminal_states": terminal_states,
        "mean_transition_mae": float(metrics_df["transition_mae"].mean()),
        "mean_transition_mse": float(metrics_df["transition_mse"].mean()),
        "mean_reward_mae": float(metrics_df["reward_mae"].mean()),
        "mean_transition_kl": float(metrics_df["transition_kl"].mean()) if "transition_kl" in metrics_df else None,
        "n_policies_evaluated": len(all_policies),
        "policies": sorted(all_policies.keys()),
    }
    save_json(outdir / "world_model_fit_summary.json", fit_summary)

    if saved_policies:
        notes.append(
            "Saved V2 policies were included only when their JSON artifacts were already present and safely readable."
        )
    write_summary(
        outdir / "world_model_summary.txt",
        mdp_path=mdp_path,
        outdir=outdir,
        mean_transition_mae=fit_summary["mean_transition_mae"],
        mean_transition_mse=fit_summary["mean_transition_mse"],
        mean_reward_mae=fit_summary["mean_reward_mae"],
        mean_transition_kl=fit_summary["mean_transition_kl"],
        loaded_policy_names=sorted(all_policies.keys()),
        notes=notes,
    )

    print(f"Loaded MDP from: {mdp_path}")
    print(f"Saved outputs to: {outdir}")
    print("Main evaluation metrics:")
    print(f"  Mean transition MAE: {fit_summary['mean_transition_mae']:.6f}")
    print(f"  Mean transition MSE: {fit_summary['mean_transition_mse']:.6f}")
    print(f"  Reward MAE: {fit_summary['mean_reward_mae']:.6f}")
    if fit_summary["mean_transition_kl"] is not None:
        print(f"  Mean transition KL: {fit_summary['mean_transition_kl']:.6f}")
    print(f"Policies evaluated: {', '.join(sorted(all_policies.keys()))}")
    print("World-model V3 run completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
