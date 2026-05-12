# Bayesian Tuning Summary

This search applies lightweight Bayesian hyperparameter tuning to the capstone's reward design.
The objective is policy-iteration average rollout return under the rebuilt tabular MDP.

## Best candidate
- candidate: no_5.29_late_m1.05_early_m8.16_cost_0.56
- score: 52.5299
- no_reward: 5.2921
- late_penalty: -1.0543
- early_penalty: -8.1601
- cost_scale: 0.5585

## Why this is relevant
- Reward design and intervention cost scaling materially affect policy behavior in this readmission-planning MDP.
- Bayesian tuning is a better fit for this capstone than forcing unrelated multi-agent or hierarchy methods.
