from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy


def load_mdp(path: Path) -> TabularMDP:
    data = np.load(path)
    return TabularMDP(P=data["P"], R=data["R"], terminal_states=[])


def collect_offline_data(
    mdp: TabularMDP,
    episodes: int,
    max_steps: int,
    behavior_eps: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    s_list, a_list, r_list, sn_list, d_list = [], [], [], [], []

    for _ in range(episodes):
        s = 0
        for t in range(max_steps):
            if rng.random() < behavior_eps:
                a = int(rng.integers(0, mdp.n_actions))
            else:
                a = 0
            sn = int(rng.choice(mdp.n_states, p=mdp.P[s, a, :]))
            r = float(mdp.R[s, a, sn])
            done = (t == max_steps - 1)
            s_list.append(s)
            a_list.append(a)
            r_list.append(r)
            sn_list.append(sn)
            d_list.append(done)
            s = sn

    return {
        "s": np.array(s_list, dtype=np.int64),
        "a": np.array(a_list, dtype=np.int64),
        "r": np.array(r_list, dtype=float),
        "sn": np.array(sn_list, dtype=np.int64),
        "done": np.array(d_list, dtype=bool),
    }


def greedy_policy(Q: np.ndarray) -> dict[str, int]:
    return {str(s): int(np.argmax(Q[s])) for s in range(Q.shape[0])}


def fitted_q_iteration(
    mdp: TabularMDP,
    data: dict[str, np.ndarray],
    gamma: float,
    iterations: int,
) -> tuple[np.ndarray, list[float]]:
    Q = np.zeros((mdp.n_states, mdp.n_actions), dtype=float)
    curve = []
    s, a, r, sn, done = data["s"], data["a"], data["r"], data["sn"], data["done"]

    for _ in range(iterations):
        Q_prev = Q.copy()
        target = r + gamma * np.max(Q_prev[sn], axis=1) * (~done)

        numer = np.zeros_like(Q)
        denom = np.zeros_like(Q)
        for i in range(len(s)):
            numer[s[i], a[i]] += target[i]
            denom[s[i], a[i]] += 1.0

        mask = denom > 0
        Q[mask] = numer[mask] / denom[mask]
        curve.append(float(np.mean(target)))

    return Q, curve


def conservative_q_learning(
    mdp: TabularMDP,
    data: dict[str, np.ndarray],
    gamma: float,
    alpha: float,
    lr: float,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    Q = np.zeros((mdp.n_states, mdp.n_actions), dtype=float)
    curve = []
    s, a, r, sn, done = data["s"], data["a"], data["r"], data["sn"], data["done"]
    idx = np.arange(len(s))

    for _ in range(epochs):
        rng.shuffle(idx)
        total_td = 0.0
        for i in idx:
            si, ai, ri, sni, di = s[i], a[i], r[i], sn[i], done[i]
            target = ri + gamma * np.max(Q[sni]) * (not di)
            lse = np.log(np.sum(np.exp(Q[si] - np.max(Q[si])))) + np.max(Q[si])
            conservative_penalty = alpha * (lse - Q[si, ai])
            td = target - Q[si, ai] - conservative_penalty
            Q[si, ai] += lr * td
            total_td += abs(td)
        curve.append(float(total_td / max(1, len(idx))))

    return Q, curve


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL benchmark (FQI vs CQL-style tabular).")
    parser.add_argument("--mdp", default="outputs/V2/mdp/mdp.npz")
    parser.add_argument("--outdir", default="outputs/V2/offline")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--behavior-eps", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--fqi-iters", type=int, default=120)
    parser.add_argument("--cql-epochs", type=int, default=80)
    parser.add_argument("--cql-alpha", type=float, default=0.15)
    parser.add_argument("--cql-lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    mdp = load_mdp(Path(args.mdp))
    data = collect_offline_data(
        mdp,
        episodes=args.episodes,
        max_steps=args.max_steps,
        behavior_eps=args.behavior_eps,
        seed=args.seed,
    )

    Q_fqi, fqi_curve = fitted_q_iteration(mdp, data, gamma=args.gamma, iterations=args.fqi_iters)
    Q_cql, cql_curve = conservative_q_learning(
        mdp,
        data,
        gamma=args.gamma,
        alpha=args.cql_alpha,
        lr=args.cql_lr,
        epochs=args.cql_epochs,
        seed=args.seed,
    )

    pi_fqi = greedy_policy(Q_fqi)
    pi_cql = greedy_policy(Q_cql)
    ev_fqi = rollout_policy(mdp, pi_fqi, episodes=2000, seed=args.seed)
    ev_cql = rollout_policy(mdp, pi_cql, episodes=2000, seed=args.seed)

    save_json(
        outdir / "dataset_stats.json",
        {
            "transitions": int(len(data["s"])),
            "episodes": int(args.episodes),
            "max_steps": int(args.max_steps),
            "behavior_eps": float(args.behavior_eps),
            "state_count": int(mdp.n_states),
            "action_count": int(mdp.n_actions),
        },
    )
    save_json(outdir / "fqi_policy.json", pi_fqi)
    save_json(outdir / "cql_policy.json", pi_cql)
    save_json(outdir / "fqi_eval.json", ev_fqi)
    save_json(outdir / "cql_eval.json", ev_cql)
    save_json(
        outdir / "learning_curves.json",
        {"fqi_mean_target": fqi_curve, "cql_mean_abs_td": cql_curve},
    )

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(fqi_curve)), fqi_curve, label="FQI (mean target)")
    plt.plot(np.arange(len(cql_curve)), cql_curve, label="CQL (mean abs TD)")
    plt.xlabel("Iteration / Epoch")
    plt.ylabel("Training Signal")
    plt.title("Offline RL Training Curves (V2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "offline_training_curves.png", dpi=180)
    plt.close()

    labels = ["FQI", "Conservative Q"]
    means = [float(ev_fqi["avg_return"]), float(ev_cql["avg_return"])]
    stds = [float(ev_fqi["std_return"]), float(ev_cql["std_return"])]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.ylabel("Average Return")
    plt.title("Offline RL Policy Comparison (V2)")
    plt.tight_layout()
    plt.savefig(figdir / "offline_return_comparison.png", dpi=180)
    plt.close()

    interp = []
    interp.append("# Offline RL Interpretation (V2)")
    interp.append("")
    interp.append("## What changed")
    interp.append("- We trained from a fixed replay dataset (offline), not from live interaction.")
    interp.append("- We compared FQI and a conservative Q-learning variant.")
    interp.append("")
    interp.append("## How to read this")
    interp.append("- Higher return (less negative) is better.")
    interp.append("- Conservative Q-learning is designed to avoid overly optimistic action values.")
    interp.append("- If conservative Q does not win, that still shows honest benchmarking.")
    interp.append("")
    interp.append("## Results")
    interp.append(f"- FQI avg return: {float(ev_fqi['avg_return']):.3f} +/- {float(ev_fqi['std_return']):.3f}")
    interp.append(f"- Conservative Q avg return: {float(ev_cql['avg_return']):.3f} +/- {float(ev_cql['std_return']):.3f}")
    interp.append("")
    interp.append("## Figures")
    interp.append("- `offline_training_curves.png`")
    interp.append("- `offline_return_comparison.png`")
    (outdir / "INTERPRETATION_OFFLINE.md").write_text("\n".join(interp) + "\n", encoding="utf-8")

    print(f"Wrote offline benchmark outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
