# EECS590 Capstone -- Reinforcement Learning for Hospital Readmission Planning

## Overview
This repository studies reinforcement learning for hospital readmission planning as a sequential decision problem. The project asks whether an RL policy can improve long-term recovery outcomes and reduce modeled 30-day readmissions relative to simple baseline strategies.

The repository now has one primary story:
- `v2_pipeline/` is the main experimental workflow and the version to review first.
- The root package under `src/eecs590_capstone/` contains the reusable environments, agents, CLIs, and utilities used by both the original scaffold and the expanded benchmark.
- `scripts/` contains dataset/MDP utilities, plotting, and orchestration scripts that sit on top of the reusable package code.
- `V1/`, `V2/`, and `V3/` are root-level navigation folders so each project version is visible directly from the repository root.

## Research Question
Can an RL agent learn a sequential discharge and follow-up policy that reduces 30-day readmissions compared with baseline strategies while maintaining or improving long-term recovery outcomes?

## Version 3
Version 3 adds a world-model extension to the hospital readmission reinforcement learning capstone. Earlier versions built a reproducible clinical MDP and compared several RL algorithms for sequential discharge and follow-up planning. The remaining gap is that healthcare RL cannot safely explore new policies directly on patients, and the current project relies on modeled and proxy patient transitions. Version 3 therefore focuses on learning and evaluating a simulator of patient progression.

The main Version 3 research question is:

**Can a learned world model approximate patient recovery and readmission dynamics well enough to support safer offline evaluation of discharge and follow-up policies?**

The world model learns estimated transition and reward behavior from trajectory data sampled out of the existing tabular MDP, then simulates patient trajectories under different policies. This does not make the project clinically deployable. Instead, it provides a transparent decision-support simulation for studying how policies may behave before any real-world use.

### Why world models?
World models fit this project because the core challenge is not simply choosing another RL algorithm. The main challenge is understanding how patient states may evolve under different discharge or follow-up strategies. A learned simulator directly addresses this gap by making the transition assumptions visible, testable, and reusable.

Version 3 adds:
- A tabular world model for learning and simulating patient transitions.
- A world-model simulator for decision-support simulation under modeled readmission risk.
- Policy evaluation inside the learned simulator.
- A lightweight Bayesian hyperparameter tuning workflow for reward and action-cost design.
- Metrics and plots saved under `outputs/V3/world_model/`.
- Documentation of implemented and not-implemented design choices.

### Why I did not implement safe hierarchical RL
I considered safe hierarchical RL but chose not to implement it in Version 3. Safe hierarchical RL is useful when the action space naturally separates into high-level goals and lower-level sub-actions. In this project, the current actions are already compact abstractions such as intervention intensity or care-management strategy. The available data do not reliably support detailed lower-level clinical sub-actions. Adding hierarchy now would risk creating artificial structure that is not supported by the dataset. For that reason, Version 3 prioritizes world modeling, because transition uncertainty and safe offline simulation are more central to the current capstone gap.

## Problem Setting
Hospital discharge planning is a sequence of interdependent decisions such as follow-up intensity, rehab referral, early intervention, and monitoring level. These decisions interact over time and affect readmission risk. Standard predictive models estimate risk, but they do not optimize a sequence of actions. RL is used here as a sequential decision framework for that optimization problem.

## Current Project Status
The repository started with a compact data-driven scaffold built from the Diabetes 130-US Hospitals dataset. The main capstone workflow first expanded into the V2 benchmark, which uses a shared clinical MDP, compares multiple RL families under the same reward design, and includes offline RL evaluation and interpretation outputs. Version 3 extends that work with a lightweight world-model layer so policy behavior can also be studied in a learned simulator rather than only in a fixed benchmark MDP.

If you are reviewing the project for the first time, start with:
1. `v2_pipeline/README.md`
2. `outputs/v2_outputs/benchmark/summary_metrics.csv`
3. `outputs/v2_outputs/all_algorithms/summary_metrics.csv`
4. `docs/version3-decisions.md`
5. `v2_pipeline/docs/technical-challenges.md`

## Versioning Notes
Versioning is documented here and also exposed through lightweight root-level navigation folders.

- V1: initial reproducible scaffold using a data-driven environment, baseline policies, and tabular/DP training utilities.
- V2: the main capstone workflow. Adds MIMIC-oriented preprocessing, broader algorithm coverage, multi-seed benchmarking, offline RL comparisons, saliency, checkpoint handling, and interpretation outputs.
- V3: a lightweight world-model extension that supports learned transition simulation and safer offline policy evaluation inside the existing project structure.

Root-level navigation:
- `V1/`: index for the original scaffold.
- `V2/`: index for the expanded benchmark workflow.
- `V3/`: index for the world-model extension.

## Repository Structure
- `V1/`, `V2/`, `V3/`: root-level version index folders for quick navigation on GitHub.
- `src/eecs590_capstone/`: reusable package code.
  Includes environments, agents, MDP definitions, command-line entrypoints, shared utilities, and the Version 3 world-model module under `src/eecs590_capstone/world_model/`.
- `scripts/`: top-level utilities and orchestration.
  Includes preprocessing, MDP construction, visualization, evaluation, end-to-end runners, and the Version 3 scripts `run_world_model_v3.py` and `make_world_model_figures.py`.
- `v2_pipeline/`: expanded benchmark workflow.
  Includes V2-specific configs, experiment scripts, documentation, architectures, checkpoints, and replay-buffer support.
- `docs/`: project documentation and decision notes, including `docs/version3-decisions.md`.
- `data/`: raw and processed datasets used by the scaffold and benchmark.
- `outputs/`: committed experiment outputs, metrics, figures, and interpretation files, including Version 3 artifacts under `outputs/V3/world_model/`.
- `tests/`: unit tests for core components, including lightweight Version 3 world-model tests.

## Data and MDP Formulation
The project uses clinical readmission data to define a sequential decision problem.

- States: abstracted patient risk or recovery groups.
- Actions: intervention intensity or care-management strategy.
- Transitions: stochastic patient progression under the chosen action.
- Rewards: tradeoff between recovery benefit, readmission avoidance, and intervention cost.
- Terminals: readmission or successful recovery.

The original scaffold uses proxy transitions because explicit intervention actions are not recorded in the source dataset. V2 keeps this limitation visible and treats reward design and representation quality as central modeling choices rather than hidden assumptions.

Version 3 keeps the same caution. The world model is a decision-support simulation layer built on modeled patient transitions, not a clinically deployable simulator.

## Main Workflow
Set the Python path from the repo root:

```powershell
$env:PYTHONPATH="src"
```

Core benchmark:

```powershell
python scripts/run_benchmark.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/benchmark --seeds 5
```

Offline RL comparison:

```powershell
python scripts/run_offline_benchmark.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/offline
```

All-algorithm comparison:

```powershell
python v2_pipeline/scripts/run_v2_all_algorithms.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/all_algorithms --seeds 7,11,19
```

Generate the major V2 figures and interpretation outputs:

```powershell
python scripts/make_figures.py --mdp outputs/v2_outputs/mdp/mdp.npz --data data/processed/train.csv
```

Reward tuning sweep:

```powershell
python scripts/run_reward_sweep.py --base-config v2_pipeline/configs/mdp_sim_mimic.json --outdir outputs/v2_outputs/reward_sweep
```

For the full operational runbook, see `v2_pipeline/README.md`.

## How To Run Version 3
Run the Version 3 world-model workflow from the repository root:

```powershell
python scripts/run_world_model_v3.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/V3/world_model
python scripts/make_world_model_figures.py
python scripts/run_bayesian_reward_tuning.py --base-config v2_pipeline/configs/mdp_sim_mimic.json --outdir outputs/V3/bayesian_tuning
python -m pytest tests/test_world_model.py
```

These commands fit the lightweight world model, generate decision-support simulation outputs and figures, and run the small synthetic tests for the V3 code.

## Key Outputs
- `outputs/v2_outputs/benchmark/`: multi-seed benchmark across DP, online RL, and offline RL.
- `outputs/v2_outputs/all_algorithms/`: single-table comparison across the broad implemented algorithm set.
- `outputs/v2_outputs/offline/`: offline RL outputs including FQI/CQL-style comparisons.
- `outputs/v2_outputs/figures/`: saliency, reward-cost, family-overview, and policy-flow visualizations.
- `outputs/V3/world_model/`: Version 3 world-model metrics, simulated policy evaluations, trajectory examples, summary notes, and interpretation figures.
- `outputs/V3/bayesian_tuning/`: Version 3 Bayesian reward-tuning trials, best configuration summaries, and tuning progress artifacts.

## Technical Notes
A few points matter for interpreting the results correctly:
- The environment is still a compact tabular clinical MDP, so exact and tabular methods have a structural advantage.
- Offline RL is included because the project is fundamentally motivated by logged healthcare data rather than unconstrained online interaction.
- Reward design materially changes rankings, so rewards should be treated as a tunable modeling choice rather than a fixed truth.
- Some advanced methods are intentionally included as adaptation studies to test algorithm-environment fit, not because they are always the most natural method for this MDP.
- The Version 3 world model is also lightweight and tabular, so it should be interpreted as a modeled decision-support simulator rather than a real clinical environment.
- The Bayesian tuning workflow is intentionally lightweight and applied to reward design because that is a more relevant post-V2 technique for this capstone than forcing unrelated architectures or multi-agent methods.

## Planned Next Improvements
- Reward calibration as a hyperparameter search problem.
- Stronger offline RL experiments on richer logged trajectories.
- Representation learning or learned embeddings/tokenization for richer state abstractions.
- Richer world models or stronger representation learning beyond the current lightweight tabular simulator.
- Better documentation of failed experiments, ablations, and debugging decisions.
- External validation on a second cohort or dataset split.

## Citations
Primary datasets and references used in the project:

- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., and Clore, J. N. "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." 2014.
  UCI dataset page: https://archive.ics.uci.edu/dataset/296/diabetic_readmission
- Diabetes 130-US Hospitals for Years 1999-2008.
  Kaggle mirror used for project setup: https://www.kaggle.com/datasets/ashikuzzamanshishir/diabetes-130-us-hospitals-for-years-1999-2008
- Johnson, A. E. W., Pollard, T. J., Shen, L., et al. "MIMIC-IV, a freely accessible electronic health record dataset." Scientific Data, 2023.
  Project workflow references MIMIC-based preprocessing in V2.
- Sutton, R. S., and Barto, A. G. Reinforcement Learning: An Introduction. Second edition.
  Used as the main RL reference for algorithm families implemented in the repository.

## Acknowledgments
- Author: Christianah Jemiyo, PhD Student in Artificial Intelligence, University of North Dakota.
- AI tooling: documentation drafting, code cleanup, and repository organization benefited from AI assistant support during development; all final project decisions, code review, and written claims were curated by the author.
- Human collaboration: no external code collaborators are currently listed in the repository. If future collaborators contribute materially, they should be acknowledged here and, where appropriate, added through GitHub collaboration history.

