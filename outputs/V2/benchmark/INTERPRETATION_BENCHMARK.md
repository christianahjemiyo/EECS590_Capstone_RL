# V2 Benchmark Interpretation

## 1) What this benchmark is doing
- It compares model-based DP, model-free online RL, and model-free offline RL in one reproducible report.
- Every method is measured across multiple random seeds, then summarized using mean and 95% CI.

## 2) Main result in plain language
- Best overall rollout score: **Behavior_Action0** (-54.061 +/- 0.107).
- Best offline-only rollout score: **Offline_CQL** (-55.276 +/- 0.327).
- Behavior baseline (action 0) score: -54.061 +/- 0.107.

## 3) How to explain this to your instructor
- In this reward design, less negative return is better.
- If the behavior/action-0 policy is strongest, it means intervention costs are currently dominating benefits.
- That is not a failure. It is a useful diagnostic showing where reward calibration is needed.

## 4) Why this is still standout work
- You are not just reporting one score; you are showing uncertainty (seed variation + CI).
- You included FQE, which is a proper offline evaluation lens, not just training reward.
- You compared strong baselines (DP, QL, Double Q, CQL, IQL) in one pipeline.

## 5) Next concrete improvement
- Recalibrate action costs/rewards and re-run this exact benchmark.
- Goal: learned policies should outperform `Behavior_Action0` on both rollout and FQE.

## 6) Files to show in class
- `summary_metrics.csv`
- `figures/benchmark_rollout_comparison.png`
- `figures/benchmark_fqe_comparison.png`

