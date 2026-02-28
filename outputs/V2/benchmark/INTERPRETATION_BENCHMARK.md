# V2 Benchmark Interpretation

## 1) What this benchmark is doing
- It compares model-based DP, model-free online RL, and model-free offline RL in one reproducible report.
- Every method is measured across multiple random seeds, then summarized using mean and 95% CI.

## 2) Main result in plain language
- Best overall rollout score: **DP_PolicyIter** (-17.713 +/- 0.208).
- Best offline-only rollout score: **Offline_CQL** (-19.869 +/- 0.825).
- Behavior baseline (action 0) score: -21.762 +/- 0.114.

## 3) How to explain this to your instructor
- In this reward design, less negative return is better.
- Learned policies are now outperforming the behavior/action-0 baseline.
- This is the result we want: interventions have enough modeled benefit to justify their cost.

## 4) Why this is still standout work
- You are not just reporting one score; you are showing uncertainty (seed variation + CI).
- You included FQE, which is a proper offline evaluation lens, not just training reward.
- You compared strong baselines (DP, QL, Double Q, CQL, IQL) in one pipeline.

## 5) Next concrete improvement
- Stress test this calibrated setting with additional seeds and sensitivity sweeps.
- Goal: preserve gains over `Behavior_Action0` on both rollout and FQE across perturbations.

## 6) Files to show in class
- `summary_metrics.csv`
- `figures/benchmark_rollout_comparison.png`
- `figures/benchmark_fqe_comparison.png`

