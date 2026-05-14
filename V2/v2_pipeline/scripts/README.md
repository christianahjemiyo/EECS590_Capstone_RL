# V2 Scripts Guide

This index groups the `v2_pipeline/scripts/` entrypoints so the folder is easier to navigate.

## Core workflow
- `preprocess_mimic.py`: prepare the MIMIC-based dataset.
- `run_v2_benchmark.py`: main benchmark across DP, online RL, and offline RL.
- `make_all_v2_figures.py`: generate the major figures and interpretation outputs.
- `reward_sweep.py`: rerun the benchmark under alternate reward maps and cost scales.

## Offline and data-management tools
- `offline_rl_benchmark.py`: focused offline RL comparison.
- `manage_replay_buffer.py`: replay-buffer organization utilities.
- `bayes_update.py`: small belief-update helper for uncertainty-oriented experiments.

## Algorithm-family runners
- `run_v2_tabular_suite.py`
- `run_v2_classical_views.py`
- `run_v2_approx_suite.py`
- `run_v2_deep_suite.py`
- `run_v2_advanced_suite.py`
- `run_v2_all_algorithms.py`

## Plotting and interpretation
- `plot_v2_results.py`
- `plot_v2_saliency.py`
- `plot_v2_nn_saliency.py`
- `plot_v2_special_visuals.py`
- `rasterize_v2_state_action.py`
- `write_v2_interpretation.py`

## Model-specific utilities
- `train_dqn.py`: compact V2 DQN baseline.
- `viz_theme.py`: shared plotting theme helpers.

## Recommendation
In normal use, prefer the root-level wrappers:
- `scripts/run_benchmark.py`
- `scripts/run_offline_benchmark.py`
- `scripts/make_figures.py`
- `scripts/run_reward_sweep.py`

Those wrappers reduce the visible split between the repo root and the V2 implementation subtree.

