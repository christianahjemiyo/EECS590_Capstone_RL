from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv


ACTION_NAMES = {
    0: "Conservative",
    1: "Standard",
    2: "Intensive",
}


def load_policy(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def simulate_episode(env: MDPSimEnv, policy: dict, max_steps: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    env.rng = rng
    state = env.reset()
    steps = []
    for t in range(max_steps):
        action = int(policy.get(str(state), 0))
        step = env.step(action)
        steps.append(
            {
                "t": t,
                "state": int(state),
                "action": int(action),
                "reward": float(step.reward),
                "next_state": int(step.state),
                "done": bool(step.done),
            }
        )
        state = step.state
        if step.done:
            break
    return steps


def write_html(steps: list[dict], n_states: int, out_path: Path) -> None:
    data = json.dumps({"steps": steps, "n_states": n_states})
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MDP Policy Playback</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --card: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --accent: #2563eb;
      --good: #16a34a;
      --warn: #f59e0b;
      --bad: #dc2626;
    }}
    body {{
      font-family: "IBM Plex Serif", "Georgia", serif;
      margin: 24px;
      color: var(--ink);
      background: var(--bg);
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 16px;
      margin-top: 16px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid #e2e8f0;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }}
    .states {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .state {{
      border-radius: 12px;
      padding: 10px;
      text-align: center;
      border: 1px solid #e2e8f0;
      background: #f1f5f9;
      transition: transform 150ms ease, background 150ms ease, border 150ms ease;
    }}
    .state.active {{
      background: #dbeafe;
      border-color: var(--accent);
      transform: translateY(-2px);
    }}
    .state.next {{
      background: #dcfce7;
      border-color: var(--good);
    }}
    .badge {{
      display: inline-block;
      font-size: 12px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #e2e8f0;
      color: #0f172a;
      margin-left: 6px;
    }}
    .details {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 14px;
      margin-top: 10px;
      font-size: 14px;
      color: var(--muted);
    }}
    .details strong {{
      color: var(--ink);
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto auto 1fr;
      gap: 8px;
      margin-top: 12px;
      align-items: center;
    }}
    button {{
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid #cbd5f5;
      background: #ffffff;
      cursor: pointer;
    }}
    .timeline {{
      width: 100%;
    }}
    .slider {{
      width: 100%;
    }}
    .spark {{
      width: 100%;
      height: 120px;
      margin-top: 8px;
      border-top: 1px solid #e2e8f0;
      padding-top: 8px;
    }}
    .legend {{
      display: flex;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h2>MDP Policy Playback</h2>
    <div class="subtitle">Simulated episode under learned policy</div>
  </div>

  <div class="grid">
    <div class="card">
      <strong>State Space</strong>
      <div class="legend" style="margin-top:8px;">
        <span><span class="dot" style="background:#dbeafe;"></span> current</span>
        <span><span class="dot" style="background:#dcfce7;"></span> next</span>
      </div>
      <div id="states" class="states"></div>

      <div class="spark">
        <svg id="sparkline" width="100%" height="120" viewBox="0 0 640 120" preserveAspectRatio="none"></svg>
      </div>
    </div>

    <div class="card">
      <strong>Step</strong>
      <div id="details" class="details"></div>

      <div class="controls">
        <button onclick="play()">Play</button>
        <button onclick="pause()">Pause</button>
        <button onclick="stepOnce()">Step</button>
        <input id="speed" type="range" min="200" max="1500" value="800" class="timeline">
      </div>

      <div style="margin-top:12px;">
        <div class="subtitle">Timeline</div>
        <input id="scrub" type="range" min="0" max="0" value="0" class="slider" oninput="scrubTo(this.value)">
      </div>
    </div>
  </div>

<script>
const data = {data};
const ACTION_NAMES = {json.dumps(ACTION_NAMES)};
let idx = 0;
let timer = null;

function riskLabel(s) {{
  if (data.n_states === 4) {{
    return ["Low", "Medium", "High", "Very High"][s] || "Risk " + s;
  }}
  if (data.n_states === 3) {{
    return ["Low", "Medium", "High"][s] || "Risk " + s;
  }}
  return "Risk " + s;
}}

function renderStates(curr, next) {{
  const container = document.getElementById("states");
  container.innerHTML = "";
  for (let i = 0; i < data.n_states; i++) {{
    const div = document.createElement("div");
    div.className = "state";
    if (i === curr) div.classList.add("active");
    if (i === next) div.classList.add("next");
    div.innerHTML = `<div><strong>S${i}</strong><span class="badge">${riskLabel(i)}</span></div>`;
    container.appendChild(div);
  }}
}}

function renderDetails(step) {{
  const el = document.getElementById("details");
  const actionName = ACTION_NAMES[step.action] ?? step.action;
  el.innerHTML = `
    <div><strong>t</strong>: ${step.t}</div>
    <div><strong>done</strong>: ${step.done}</div>
    <div><strong>state</strong>: S${step.state}</div>
    <div><strong>next</strong>: S${step.next_state}</div>
    <div><strong>action</strong>: ${actionName}</div>
    <div><strong>reward</strong>: ${step.reward.toFixed(3)}</div>
  `;
}}

function renderSparkline() {{
  const svg = document.getElementById("sparkline");
  const w = 640;
  const h = 120;
  const rewards = data.steps.map(s => s.reward);
  const cum = rewards.reduce((acc, r) => {{
    acc.push((acc.length ? acc[acc.length - 1] : 0) + r);
    return acc;
  }}, []);
  const min = Math.min(...cum, 0);
  const max = Math.max(...cum, 0);
  const range = max - min || 1;

  let path = "";
  cum.forEach((v, i) => {{
    const x = (i / Math.max(cum.length - 1, 1)) * w;
    const y = h - ((v - min) / range) * h;
    path += (i === 0 ? "M" : "L") + x.toFixed(1) + "," + y.toFixed(1);
  }});

  svg.innerHTML = `
    <rect x="0" y="0" width="${w}" height="${h}" fill="#f8fafc"></rect>
    <path d="${path}" stroke="#2563eb" stroke-width="2" fill="none"></path>
  `;
}}

function renderFrame() {{
  const step = data.steps[idx];
  if (!step) return;
  renderStates(step.state, step.next_state);
  renderDetails(step);
  document.getElementById("scrub").value = idx;
}}

function stepOnce() {{
  renderFrame();
  idx = Math.min(idx + 1, data.steps.length - 1);
}}

function play() {{
  if (timer) return;
  timer = setInterval(() => {{
    renderFrame();
    if (idx >= data.steps.length - 1) {{
      pause();
      return;
    }}
    idx += 1;
  }}, parseInt(document.getElementById("speed").value, 10));
}}

function pause() {{
  if (timer) {{
    clearInterval(timer);
    timer = null;
  }}
}}

function scrubTo(val) {{
  idx = Math.max(0, Math.min(parseInt(val, 10), data.steps.length - 1));
  renderFrame();
}}

document.getElementById("scrub").max = data.steps.length - 1;
renderSparkline();
renderFrame();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an HTML animation for a simulated MDP episode.")
    parser.add_argument("--mdp", default="outputs/mdp/mdp.npz")
    parser.add_argument("--policy", default="outputs/mdp/policy_iter_policy.json")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="outputs/mdp/policy_playback.html")
    args = parser.parse_args()

    env = MDPSimEnv(mdp_path=args.mdp, seed=args.seed, max_steps=args.max_steps)
    policy = load_policy(Path(args.policy))
    steps = simulate_episode(env, policy, args.max_steps, args.seed)
    write_html(steps, env.n_states, Path(args.out))
    print(f"Wrote: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
