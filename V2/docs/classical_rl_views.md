# Classical RL View Mapping

This note maps the V2 classical RL implementations to the view-based terminology used in the Version 2 requirements.

## Implemented classical methods

- Monte Carlo Control
- TD(0)
- TD(n)
- TD(lambda)
- SARSA
- SARSA(n)
- SARSA(lambda)
- Q-Learning
- Q(lambda)
- Dyna-Q

## Forward-view implementations

- `td_n_forward` in `src/eecs590_capstone/agents/rl_tabular_views.py`
  Uses explicit n-step return targets.

- `td_lambda_forward` in `src/eecs590_capstone/agents/rl_tabular_views.py`
  Uses explicit lambda-return targets over trajectories.

- `sarsa_n_forward` in `src/eecs590_capstone/agents/rl_tabular_views.py`
  Uses explicit n-step action-value targets.

- `sarsa_lambda_forward` in `src/eecs590_capstone/agents/rl_tabular_views.py`
  Uses explicit lambda-return style action-value updates.

- `mc_control` in `src/eecs590_capstone/agents/rl_tabular.py`
  Uses full-episode return targets.

## Backward-view implementations

- `td_n_backward` in `src/eecs590_capstone/agents/rl_tabular_views.py`
  Uses truncated trace-style backward updates with an n-cutoff interpretation.

- `td_lambda_backward` in `src/eecs590_capstone/agents/rl_tabular_views.py`
  Uses eligibility traces for value updates.

- `sarsa_n_backward` in `src/eecs590_capstone/agents/rl_tabular_views.py`
  Uses truncated trace-style backward action-value updates with an n-cutoff interpretation.

- `sarsa_lambda_backward` in `src/eecs590_capstone/agents/rl_tabular_views.py`
  Uses eligibility traces for action-value updates.

- `q_lambda` in `src/eecs590_capstone/agents/rl_tabular.py`
  Uses eligibility traces in an off-policy control setting.

## Notes

- The repository now explicitly names forward-view and backward-view TD and SARSA family variants in code.
- The backward-view `n` variants use truncated traces so that the n-cutoff idea is visible in implementation and documentation.
- Eligibility traces are part of the backward-view implementations and are visible directly in the update code.
- The `V2/scripts/run_v2_classical_views.py` suite exists to make these view distinctions visible in V2 outputs.
