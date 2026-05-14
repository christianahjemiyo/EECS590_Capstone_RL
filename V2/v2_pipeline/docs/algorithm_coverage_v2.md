# V2 Algorithm Coverage

This V2 workflow now includes user-facing runners for all implemented algorithm families.

## V2 runners

- `v2_pipeline/scripts/run_v2_tabular_suite.py`
- `v2_pipeline/scripts/run_v2_classical_views.py`
- `v2_pipeline/scripts/run_v2_approx_suite.py`
- `v2_pipeline/scripts/run_v2_deep_suite.py`
- `v2_pipeline/scripts/run_v2_advanced_suite.py`
- `v2_pipeline/scripts/run_v2_all_algorithms.py`

## Coverage map

- Policy Iteration: `src/eecs590_capstone/agents/dp_policy_iter.py`
- Value Iteration: `src/eecs590_capstone/agents/dp_value_iter.py`
- Monte Carlo Control (epsilon-greedy): `src/eecs590_capstone/agents/rl_tabular.py`
- SARSA: `src/eecs590_capstone/agents/rl_tabular.py`
- Q-Learning: `src/eecs590_capstone/agents/rl_tabular.py`
- Expected SARSA: `src/eecs590_capstone/agents/rl_tabular.py`
- TD(0): `src/eecs590_capstone/agents/rl_tabular.py`
- TD(lambda): `src/eecs590_capstone/agents/rl_tabular.py`
- SARSA(lambda): `src/eecs590_capstone/agents/rl_tabular.py`
- Q(lambda): `src/eecs590_capstone/agents/rl_tabular.py`
- Linear Function Approximation: `src/eecs590_capstone/agents/rl_approx.py`
- Gradient TD / Semi-gradient TD: `src/eecs590_capstone/agents/rl_approx.py`
- Approximate Q-Learning: `src/eecs590_capstone/agents/rl_approx.py`
- DQN: `v2_pipeline/scripts/train_dqn.py`
- Double DQN: `src/eecs590_capstone/agents/rl_deep.py`
- Dueling DQN: `src/eecs590_capstone/agents/rl_deep.py`
- REINFORCE: `src/eecs590_capstone/agents/rl_deep.py`
- A2C: `src/eecs590_capstone/agents/rl_deep.py`
- A3C: `src/eecs590_capstone/agents/rl_deep.py`
- DDPG: `src/eecs590_capstone/agents/rl_advanced.py`
- TD3: `src/eecs590_capstone/agents/rl_advanced.py`
- SAC: `src/eecs590_capstone/agents/rl_advanced.py`
- TRPO: `src/eecs590_capstone/agents/rl_deep.py`
- PPO: `src/eecs590_capstone/agents/rl_deep.py`
- Dyna-Q: `src/eecs590_capstone/agents/rl_tabular.py`
- Learned environment models: `src/eecs590_capstone/agents/rl_advanced.py`
- Belief state update methods: `src/eecs590_capstone/agents/rl_advanced.py`
- RNN-based RL: `src/eecs590_capstone/agents/rl_advanced.py`

## Main comparison command

```powershell
$env:PYTHONPATH="src"
python v2_pipeline/scripts/run_v2_all_algorithms.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/all_algorithms --seeds 7,11,19
```

