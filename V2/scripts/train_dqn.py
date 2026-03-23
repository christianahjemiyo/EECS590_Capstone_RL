from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv
from eecs590_capstone.mdp.definitions import TabularMDP, rollout_policy
from eecs590_capstone.utils.checkpoint_io import save_npz_checkpoint
from eecs590_capstone.utils.io import save_json


def load_mdp(path: Path) -> TabularMDP:
    data = np.load(path)
    return TabularMDP(P=data["P"], R=data["R"], terminal_states=[])


def epsilon_schedule(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return eps_end
    frac = min(step / decay_steps, 1.0)
    return float(eps_start + frac * (eps_end - eps_start))


@dataclass
class Net:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def init_net(n_states: int, n_actions: int, hidden: int, rng: np.random.Generator) -> Net:
    w1 = rng.normal(loc=0.0, scale=0.1, size=(n_states, hidden))
    b1 = np.zeros(hidden, dtype=float)
    w2 = rng.normal(loc=0.0, scale=0.1, size=(hidden, n_actions))
    b2 = np.zeros(n_actions, dtype=float)
    return Net(w1=w1, b1=b1, w2=w2, b2=b2)


def forward(net: Net, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_pre = net.w1[states] + net.b1
    h = np.maximum(0.0, h_pre)
    q = h @ net.w2 + net.b2
    return q, h, h_pre


def copy_net(src: Net) -> Net:
    return Net(w1=src.w1.copy(), b1=src.b1.copy(), w2=src.w2.copy(), b2=src.b2.copy())


def select_action(net: Net, state: int, eps: float, rng: np.random.Generator, n_actions: int) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, n_actions))
    q, _, _ = forward(net, np.array([state], dtype=np.int64))
    return int(np.argmax(q[0]))


def train_dqn(
    env: MDPSimEnv,
    episodes: int,
    gamma: float,
    lr: float,
    hidden: int,
    batch_size: int,
    replay_size: int,
    warmup_steps: int,
    target_update: int,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    seed: int,
) -> tuple[dict[str, int], dict[str, float], dict[str, float], list[float], dict[str, list[float]], Net, Net]:
    rng = np.random.default_rng(seed)
    n_states, n_actions = env.n_states, env.n_actions
    online = init_net(n_states, n_actions, hidden, rng)
    target = copy_net(online)

    s_buf = np.zeros(replay_size, dtype=np.int64)
    a_buf = np.zeros(replay_size, dtype=np.int64)
    r_buf = np.zeros(replay_size, dtype=float)
    sn_buf = np.zeros(replay_size, dtype=np.int64)
    d_buf = np.zeros(replay_size, dtype=np.float64)
    size = 0
    ptr = 0

    total_steps = 0
    losses = []
    episode_returns = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            eps = epsilon_schedule(total_steps, eps_start, eps_end, eps_decay)
            a = select_action(online, s, eps, rng, n_actions)
            step = env.step(a)
            sn, r, done = step.state, float(step.reward), step.done

            s_buf[ptr] = s
            a_buf[ptr] = a
            r_buf[ptr] = r
            sn_buf[ptr] = sn
            d_buf[ptr] = 1.0 if done else 0.0
            ptr = (ptr + 1) % replay_size
            size = min(size + 1, replay_size)

            s = sn
            ep_return += r
            total_steps += 1

            if size >= max(batch_size, warmup_steps):
                idx = rng.integers(0, size, size=batch_size)
                bs = s_buf[idx]
                ba = a_buf[idx]
                br = r_buf[idx]
                bsn = sn_buf[idx]
                bd = d_buf[idx]

                q, h, h_pre = forward(online, bs)
                qt, _, _ = forward(target, bsn)
                tgt = br + gamma * np.max(qt, axis=1) * (1.0 - bd)

                pred = q[np.arange(batch_size), ba]
                err = pred - tgt
                loss = 0.5 * float(np.mean(err * err))
                losses.append(loss)

                grad_q = np.zeros_like(q)
                grad_q[np.arange(batch_size), ba] = err / float(batch_size)

                d_w2 = h.T @ grad_q
                d_b2 = grad_q.sum(axis=0)
                d_h = grad_q @ online.w2.T
                d_h[h_pre <= 0.0] = 0.0

                d_w1 = np.zeros_like(online.w1)
                np.add.at(d_w1, bs, d_h)
                d_b1 = d_h.sum(axis=0)

                online.w2 -= lr * d_w2
                online.b2 -= lr * d_b2
                online.w1 -= lr * d_w1
                online.b1 -= lr * d_b1

            if total_steps % target_update == 0:
                target = copy_net(online)

        episode_returns.append(float(ep_return))

    all_states = np.arange(n_states, dtype=np.int64)
    q_all, _, _ = forward(online, all_states)
    policy = {str(s): int(np.argmax(q_all[s])) for s in range(n_states)}
    values = {str(s): float(np.max(q_all[s])) for s in range(n_states)}
    train_info = {
        "episodes": float(episodes),
        "total_steps": float(total_steps),
        "final_epsilon": float(epsilon_schedule(total_steps, eps_start, eps_end, eps_decay)),
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
    }
    q_json = {str(s): q_all[s].tolist() for s in range(n_states)}
    return policy, values, train_info, episode_returns, q_json, online, target


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight DQN baseline on the tabular MDP.")
    parser.add_argument("--mdp", default="outputs/V2/mdp/mdp.npz")
    parser.add_argument("--outdir", default="outputs/V2/rl_dqn")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--target-update", type=int, default=200)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = MDPSimEnv(mdp_path=args.mdp, seed=args.seed, max_steps=args.max_steps)
    mdp = load_mdp(Path(args.mdp))
    policy, values, train_info, curve, q_json, online_net, target_net = train_dqn(
        env=env,
        episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        target_update=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        seed=args.seed,
    )
    eval_metrics = rollout_policy(mdp, policy, episodes=2000, seed=args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "train_args.json", vars(args))
    save_json(outdir / "policy.json", policy)
    save_json(outdir / "value_function.json", values)
    save_json(outdir / "train_info.json", train_info)
    save_json(outdir / "learning_curve.json", {"episode_returns": curve})
    save_json(outdir / "eval_results.json", eval_metrics)
    save_json(outdir / "q_values.json", q_json)
    ckpt_dir = Path("V2/checkpoints") / "dqn" / "foundation_env" / "default"
    save_npz_checkpoint(
        ckpt_dir / "model_checkpoint.npz",
        {
            "online": {"w1": online_net.w1, "b1": online_net.b1, "w2": online_net.w2, "b2": online_net.b2},
            "target": {"w1": target_net.w1, "b1": target_net.b1, "w2": target_net.w2, "b2": target_net.b2},
        },
    )
    save_json(ckpt_dir / "checkpoint_meta.json", {"algo": "dqn", "mdp": args.mdp, "outdir": str(outdir)})

    print("DQN training complete")
    print(eval_metrics)


if __name__ == "__main__":
    main()
