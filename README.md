# EECS590 Capstone -- Reinforcement Learning for Hospital Readmission Planning

## Overview
This repository studies how reinforcement learning can support sequential clinical intervention planning for hospital readmission risk. The project asks whether an RL policy can improve long-term recovery outcomes and reduce modeled 30-day readmissions relative to simple baseline strategies.

The repository now has one primary story:
- `V2/` is the main experimental workflow and the version to review first.
- The root package under `src/eecs590_capstone/` contains the reusable environments, agents, CLIs, and utilities used by both the original scaffold and the expanded benchmark.
- `scripts/` contains dataset/MDP utilities, plotting, and orchestration scripts that sit on top of the reusable package code.

## Research Question
Can an RL agent learn a sequential discharge and follow-up policy that reduces 30-day readmissions compared with baseline strategies while maintaining or improving long-term recovery outcomes?

## Problem Setting
Hospital discharge planning is a sequence of interdependent decisions such as follow-up intensity, rehab referral, early intervention, and monitoring level. These decisions interact over time and affect readmission risk. Standard predictive models estimate risk, but they do not optimize a sequence of actions. RL is used here as a sequential decision framework for that optimization problem.

## Current Project Status
The repository started with a compact data-driven scaffold built from the Diabetes 130-US Hospitals dataset. The main capstone workflow is now the V2 benchmark, which uses a shared clinical MDP, compares multiple RL families under the same reward design, and includes offline RL evaluation and interpretation outputs.

If you are reviewing the project for the first time, start with:
1. `V2/README.md`
2. `outputs/V2/benchmark/summary_metrics.csv`
3. `outputs/V2/all_algorithms/summary_metrics.csv`
4. `V2/docs/technical-challenges.md`

## Versioning Notes
Versioning is documented here rather than spread across multiple top-level project names.

- V1: initial reproducible scaffold using a data-driven environment, baseline policies, and tabular/DP training utilities.
- V2: the main capstone workflow. Adds MIMIC-oriented preprocessing, broader algorithm coverage, multi-seed benchmarking, offline RL comparisons, saliency, checkpoint handling, and interpretation outputs.
- Future updates should extend the existing workflow and documentation rather than create a separate top-level `V3` tree unless the project scope fundamentally changes.

## Repository Structure
- `src/eecs590_capstone/`: reusable package code.
  Includes environments, agents, MDP definitions, command-line entrypoints, and shared utilities.
- `scripts/`: top-level utilities and orchestration.
  Includes preprocessing, MDP construction, visualization, evaluation, and end-to-end runners.
- `V2/`: expanded benchmark workflow.
  Includes V2-specific configs, experiment scripts, documentation, architectures, checkpoints, and replay-buffer support.
- `data/`: raw and processed datasets used by the scaffold and benchmark.
- `outputs/`: committed experiment outputs, metrics, figures, and interpretation files.
- `tests/`: unit tests for core components.

## Data and MDP Formulation
The project uses clinical readmission data to define a sequential decision problem.

- States: abstracted patient risk or recovery groups.
- Actions: intervention intensity or care-management strategy.
- Transitions: stochastic patient progression under the chosen action.
- Rewards: tradeoff between recovery benefit, readmission avoidance, and intervention cost.
- Terminals: readmission or successful recovery.

The original scaffold uses proxy transitions because explicit intervention actions are not recorded in the source dataset. V2 keeps this limitation visible and treats reward design and representation quality as central modeling choices rather than hidden assumptions.

## Main Workflow
Set the Python path from the repo root:

```powershell
$env:PYTHONPATH="src"
```

Core benchmark:

```powershell
python scripts/run_benchmark.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/benchmark --seeds 5
```

Offline RL comparison:

```powershell
python scripts/run_offline_benchmark.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/offline
```

All-algorithm comparison:

```powershell
python V2/scripts/run_v2_all_algorithms.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/all_algorithms --seeds 7,11,19
```

Generate the major V2 figures and interpretation outputs:

```powershell
python scripts/make_figures.py --mdp outputs/V2/mdp/mdp.npz --data data/processed/train.csv
```

Reward tuning sweep:

```powershell
python scripts/run_reward_sweep.py --base-config V2/configs/mdp_sim_mimic.json --outdir outputs/V2/reward_sweep
```

For the full operational runbook, see `V2/README.md`.

## Key Outputs
- `outputs/V2/benchmark/`: multi-seed benchmark across DP, online RL, and offline RL.
- `outputs/V2/all_algorithms/`: single-table comparison across the broad implemented algorithm set.
- `outputs/V2/offline/`: offline RL outputs including FQI/CQL-style comparisons.
- `outputs/V2/figures/`: saliency, reward-cost, family-overview, and policy-flow visualizations.

## Technical Notes
A few points matter for interpreting the results correctly:
- The environment is still a compact tabular clinical MDP, so exact and tabular methods have a structural advantage.
- Offline RL is included because the project is fundamentally motivated by logged healthcare data rather than unconstrained online interaction.
- Reward design materially changes rankings, so rewards should be treated as a tunable modeling choice rather than a fixed truth.
- Some advanced methods are intentionally included as adaptation studies to test algorithm-environment fit, not because they are always the most natural method for this MDP.

## Planned Next Improvements
- Reward calibration as a hyperparameter search problem.
- Stronger offline RL experiments on richer logged trajectories.
- Representation learning or learned embeddings/tokenization for richer state abstractions.
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
- AI tooling: documentation drafting, code cleanup, and repository organization benefited from AI assistant support during development; all final project decisions, code review, and written claims were checked and curated by the author.
- Human collaboration: no external code collaborators are currently listed in the repository. If future collaborators contribute materially, they should be acknowledged here and, where appropriate, added through GitHub collaboration history.
