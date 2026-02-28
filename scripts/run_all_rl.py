from __future__ import annotations

import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

import numpy as np

from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv
from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.agents.rl_tabular import (
    mc_control,
    td_n,
    td_lambda,
    sarsa_n,
    sarsa_lambda,
    q_learning,
    double_q_learning,
)
from eecs590_capstone.utils.io import save_json


def load_mdp(mdp_path: Path) -> TabularMDP:
    data = np.load(mdp_path)
    return TabularMDP(P=data["P"], R=data["R"], terminal_states=[])


def save_result(outdir: Path, args: dict, result, eval_metrics: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "train_args.json", args)
    save_json(outdir / "policy.json", result.policy)
    save_json(outdir / "value_function.json", result.V)
    save_json(outdir / "train_info.json", result.train_info)
    save_json(outdir / "learning_curve.json", {"episode_returns": result.episode_returns})
    save_json(outdir / "eval_results.json", eval_metrics)
    if result.Q is not None:
        save_json(outdir / "q_values.json", result.Q)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all tabular RL algorithms and save results.")
    parser.add_argument("--mdp", type=str, default="outputs/mdp/mdp.npz")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--lambda", dest="lam", type=float, default=0.8)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="outputs/rl")
    args = parser.parse_args()

    env = MDPSimEnv(mdp_path=args.mdp, seed=args.seed, max_steps=args.max_steps)
    mdp = load_mdp(Path(args.mdp))

    configs = {
        "mc": lambda: mc_control(env, episodes=args.episodes, gamma=args.gamma,
                                 eps_start=args.eps_start, eps_end=args.eps_end,
                                 decay_steps=args.eps_decay, seed=args.seed),
        "td_n": lambda: td_n(env, mdp, episodes=args.episodes, n=args.n, alpha=args.alpha,
                             gamma=args.gamma, eps_start=args.eps_start, eps_end=args.eps_end,
                             decay_steps=args.eps_decay, seed=args.seed),
        "td_lambda": lambda: td_lambda(env, mdp, episodes=args.episodes, alpha=args.alpha,
                                        gamma=args.gamma, lam=args.lam, eps_start=args.eps_start,
                                        eps_end=args.eps_end, decay_steps=args.eps_decay, seed=args.seed),
        "sarsa_n": lambda: sarsa_n(env, episodes=args.episodes, n=args.n, alpha=args.alpha,
                                   gamma=args.gamma, eps_start=args.eps_start, eps_end=args.eps_end,
                                   decay_steps=args.eps_decay, seed=args.seed),
        "sarsa_lambda": lambda: sarsa_lambda(env, episodes=args.episodes, alpha=args.alpha,
                                             gamma=args.gamma, lam=args.lam, eps_start=args.eps_start,
                                             eps_end=args.eps_end, decay_steps=args.eps_decay, seed=args.seed),
        "q_learning": lambda: q_learning(env, episodes=args.episodes, alpha=args.alpha,
                                          gamma=args.gamma, eps_start=args.eps_start, eps_end=args.eps_end,
                                          decay_steps=args.eps_decay, seed=args.seed),
        "double_q_learning": lambda: double_q_learning(env, episodes=args.episodes, alpha=args.alpha,
                                                        gamma=args.gamma, eps_start=args.eps_start, eps_end=args.eps_end,
                                                        decay_steps=args.eps_decay, seed=args.seed),
    }

    for name, fn in configs.items():
        for run in range(args.runs):
            run_seed = args.seed + run
            env.rng = np.random.default_rng(run_seed)
            args_dict = vars(args) | {"algo": name, "run": run, "seed": run_seed}

            if name == "mc":
                result = mc_control(env, episodes=args.episodes, gamma=args.gamma,
                                    eps_start=args.eps_start, eps_end=args.eps_end,
                                    decay_steps=args.eps_decay, seed=run_seed)
            elif name == "td_n":
                result = td_n(env, mdp, episodes=args.episodes, n=args.n, alpha=args.alpha,
                              gamma=args.gamma, eps_start=args.eps_start, eps_end=args.eps_end,
                              decay_steps=args.eps_decay, seed=run_seed)
            elif name == "td_lambda":
                result = td_lambda(env, mdp, episodes=args.episodes, alpha=args.alpha,
                                   gamma=args.gamma, lam=args.lam, eps_start=args.eps_start,
                                   eps_end=args.eps_end, decay_steps=args.eps_decay, seed=run_seed)
            elif name == "sarsa_n":
                result = sarsa_n(env, episodes=args.episodes, n=args.n, alpha=args.alpha,
                                 gamma=args.gamma, eps_start=args.eps_start, eps_end=args.eps_end,
                                 decay_steps=args.eps_decay, seed=run_seed)
            elif name == "sarsa_lambda":
                result = sarsa_lambda(env, episodes=args.episodes, alpha=args.alpha,
                                      gamma=args.gamma, lam=args.lam, eps_start=args.eps_start,
                                      eps_end=args.eps_end, decay_steps=args.eps_decay, seed=run_seed)
            elif name == "q_learning":
                result = q_learning(env, episodes=args.episodes, alpha=args.alpha,
                                    gamma=args.gamma, eps_start=args.eps_start, eps_end=args.eps_end,
                                    decay_steps=args.eps_decay, seed=run_seed)
            else:
                result = double_q_learning(env, episodes=args.episodes, alpha=args.alpha,
                                           gamma=args.gamma, eps_start=args.eps_start, eps_end=args.eps_end,
                                           decay_steps=args.eps_decay, seed=run_seed)

            eval_metrics = rollout_policy(mdp, result.policy, episodes=2000, seed=run_seed)
            out = Path(args.outdir) / name / f"run_{run}"
            save_result(out, args_dict, result, eval_metrics)
            print(f"Wrote results for {name} run {run} -> {out}")


if __name__ == "__main__":
    main()
