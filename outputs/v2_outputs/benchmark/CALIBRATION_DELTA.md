# Calibration Delta (Before vs After)

## What was changed
- Increased action effect in transition model:
  - `action_strengths`: `[0.0, 0.08, 0.16]` -> `[0.0, 0.12, 0.25]`
- Reduced intervention cost:
  - `action_costs`: `[0.0, 0.15, 0.35]` -> `[0.0, 0.05, 0.10]`
- Rebalanced reward map:
  - `<30`: `-12.0` -> `-10.0`
  - `>30`: `-3.0` -> `-2.0`
  - `NO`: `2.0` -> `4.0`

## Why this matters
- Before calibration, the behavior/action-0 baseline tended to dominate.
- After calibration, learned policies now beat behavior baseline on rollout mean.

## Current benchmark headline
- `Behavior_Action0`: `-21.762`
- `Q_Learning`: `-18.726`
- `Offline_CQL`: `-19.869`
- `DP_PolicyIter`: `-17.713` (best overall)

Less negative is better under this reward design.
