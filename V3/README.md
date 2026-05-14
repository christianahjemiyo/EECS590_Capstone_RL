# V3 Overview

`V3/` is the root-level navigation entry for the world-model extension of the capstone.

Version 3 is layered onto the existing repository rather than stored as a separate code subtree. Its implementation is split across:

- `src/eecs590_capstone/world_model/`: world-model code.
- `scripts/run_world_model_v3.py`: main Version 3 runner.
- `scripts/make_world_model_figures.py`: Version 3 figure generation.
- `scripts/run_bayesian_reward_tuning.py`: lightweight Bayesian reward-tuning workflow.
- `docs/version3-decisions.md`: design and scope decisions.
- `docs/model-card-world-model.md`: model card and limitations.
- `outputs/V3/`: committed Version 3 artifacts.

## What V3 Adds

Version 3 extends the capstone with:

- a tabular world model,
- policy evaluation inside a learned simulator,
- trajectory-based analysis outputs,
- lightweight Bayesian tuning for reward and action-cost settings.

## Start Here

- Design notes: `docs/version3-decisions.md`
- Model card: `docs/model-card-world-model.md`
- World-model outputs: `outputs/V3/world_model/`
- Bayesian tuning outputs: `outputs/V3/bayesian_tuning/`
