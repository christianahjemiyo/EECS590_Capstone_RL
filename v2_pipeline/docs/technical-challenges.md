# Technical Challenges

This document records the main engineering and modeling challenges encountered in the capstone, what was tried, what worked, and what remains unresolved. The goal is to make the project easier to review and easier to extend.

## 1. Clinical data does not provide an action-labeled RL environment

### The problem
The motivating datasets provide patient outcomes and covariates, but they do not provide a clean sequence of intervention actions that can be used directly as an online RL environment.

### What I tried
- Built an initial data-driven environment around readmission outcomes and proxy actions.
- Used discrete intervention-intensity actions as modeling abstractions rather than literal observed actions.
- Converted processed patient data into a compact tabular MDP for controlled benchmarking.

### What worked
- The abstraction was sufficient to create a reproducible RL benchmark.
- It enabled comparison across DP, online RL, and offline RL in one environment.
- It made it possible to test policy behavior and reward sensitivity before building a richer clinical simulator.

### What did not fully work
- The action semantics are still modeled rather than directly observed.
- This limits causal interpretation and means strong empirical performance should not be overclaimed as clinical validity.

### Next step
Move toward richer logged trajectories or action-aware preprocessing so the policy-learning problem is closer to a true offline RL setting.

## 2. Algorithm breadth created organization pressure

### The problem
The project grew from a compact scaffold into a broad benchmark covering DP, tabular RL, approximation methods, deep RL, actor-critic methods, offline RL, and several adapted advanced methods.

### What I tried
- Kept reusable code in `src/eecs590_capstone/`.
- Kept orchestration and plotting in `scripts/` and `v2_pipeline/scripts/`.
- Committed outputs so reviewers could inspect results without rerunning everything.

### What worked
- The package code stayed reusable and script entrypoints remained simple to run.
- A shared benchmark environment made cross-family comparisons possible.
- Saved outputs made the project more auditable.

### What did not fully work
- The growth into `v2_pipeline/` made navigation harder than in the original scaffold.
- The distinction between reusable package code and experiment scripts was not always obvious from the README alone.
- Some legacy entrypoints remained after the workflow evolved.

### Next step
Keep one primary repository narrative in the root README and treat future changes as extensions of the same workflow rather than parallel top-level versions.

## 3. Some methods are natural fits and others are adaptation studies

### The problem
The benchmark environment is compact and tabular, which naturally favors DP and strong tabular RL methods. Some advanced methods in the repository were included for course coverage and comparative analysis, but they are not the native best fit for this MDP.

### What I tried
- Adapted some deep and continuous-control-style methods to the shared benchmark so they could be compared under one evaluation protocol.
- Kept the environment fixed so differences would mostly reflect algorithm behavior rather than environment drift.

### What worked
- The benchmark makes algorithm-environment fit visible.
- Simpler methods often performing strongly is informative rather than disappointing.
- The project can show both capability coverage and method-selection judgment.

### What did not fully work
- Some readers may assume every implemented method is equally appropriate for the task.
- Raw leaderboard comparisons can hide the fact that some methods are being stress-tested outside their ideal regime.

### Next step
Keep emphasizing method fit in the documentation and include more ablation-style summaries rather than only final ranking tables.

## 4. Reward design strongly changes conclusions

### The problem
Policy ranking is sensitive to the reward map and intervention-cost assumptions. In several runs, conservative policies looked strong because action costs outweighed modeled benefits.

### What I tried
- Used a clear reward shaping baseline to make the environment trainable and interpretable.
- Added reward-cost visualizations and benchmark interpretation files.
- Noted in the benchmark outputs that less negative return is better under the current reward design.

### What worked
- The reward function is explicit rather than hidden.
- It became possible to explain why seemingly strong policies were sometimes conservative.
- Reward sensitivity emerged as a meaningful result instead of a silent bug.

### What did not fully work
- Reward calibration is still only lightly explored.
- The new reward sweep is useful, but it is still a compact search rather than a fully principled hyperparameter-optimization study.

### Next step
Expand the current reward sweep into a broader multi-seed calibration study and connect the search space more directly to domain-motivated care tradeoffs.

## 5. Offline RL is important but still simplified

### The problem
The motivating use case is logged healthcare data, but the project began from a modeled environment. Offline RL was needed to make the project better aligned with the data reality.

### What I tried
- Added a fixed replay-style offline workflow.
- Implemented offline comparisons such as fitted-Q-style and conservative Q-style baselines.
- Included FQE-oriented evaluation in the broader benchmark.

### What worked
- Offline RL is now part of the main technical story, not just a future-work note.
- The benchmark can compare online and offline learning perspectives in one framework.
- This improves the realism of the capstone relative to a pure online-RL-only repo.

### What did not fully work
- The offline data is still generated within the project workflow rather than coming from a fully realistic treatment log.
- This means the offline benchmark is useful but still transitional.

### Next step
Use richer logged trajectories and evaluate whether offline methods remain robust when the data distribution is less idealized.

## 6. Interpretability in a non-image clinical setting required different tools

### The problem
Saliency and interpretation are more straightforward in image tasks than in a compact clinical state/action benchmark.

### What I tried
- Focused on action-preference views, feature-impact plots, and summary interpretation files.
- Added both classical saliency-style outputs and neural saliency views for the DQN family.

### What worked
- The project now has interpretable artifacts that can be discussed in a report or presentation.
- The visualizations help explain why policies prefer some actions over others.

### What did not fully work
- The interpretability tools are still lighter than what would be expected in a full clinical ML deployment.
- Gradient-based attribution and richer embedding-space analysis are still limited.

### Next step
Add stronger attribution methods and connect them more directly to clinical feature groups.

## 7. Representation learning is still limited

### The problem
The current state abstraction is mostly hand-constructed and discretized. That keeps the benchmark manageable, but it may hide useful structure in the data.

### What I tried
- Started with a compact risk-state representation to make the RL problem tractable.
- Kept the abstraction explicit so assumptions were auditable.
- Added a lightweight tokenizer/embedding prototype so categorical and discretized numeric clinical inputs can map into a learned continuous representation before state binning.

### What worked
- The resulting environment is stable and easy to benchmark.
- It allowed the capstone to cover the full RL pipeline without waiting for a more complex representation-learning subsystem.
- The tokenizer-style prototype now gives the project a concrete bridge from manual discretization toward learned state representations.

### What did not fully work
- Manual discretization can blur clinically meaningful distinctions.
- The current tokenizer/embedding path is still lightweight and experimental, not yet the main benchmark representation.
- Some advanced methods may still look less useful than they would with a richer learned state space.

### Next step
Decide whether the tokenizer path should become an ablation branch of the main benchmark, and if so, compare it directly against the current hand-built state abstraction.

## 8. What remains most confusing to me

The biggest open questions are not about whether the code runs. They are about model validity:
- how to define interventions that are both clinically meaningful and learnable,
- how to calibrate reward tradeoffs so policy ranking matches the intended care objective,
- how much benchmark performance survives once the representation and logged-data assumptions become more realistic.

Those questions are the main reason the next phase should focus less on adding more algorithms and more on representation quality, reward calibration, and offline evaluation realism.

