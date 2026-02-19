# EECS590 Capstone -- Reinforcement Learning for Hospital Readmission Planning

## Project Overview
This capstone investigates how reinforcement learning (RL) can model and optimize sequential clinical decisions that influence hospital readmissions. The goal is to design agents that learn intervention strategies to reduce avoidable 30-day readmissions while supporting long-term patient recovery.

Version 1 establishes a clean, reproducible foundation: a data-driven environment scaffold, baseline policies, and evaluation tooling.

## Research Question
Can an RL agent learn a sequential discharge and follow-up policy that reduces 30-day readmissions compared to baseline strategies while maintaining or improving long-term recovery outcomes?

## Problem Statement
Hospital discharge planning is a sequence of interdependent decisions (discharge timing, medication reconciliation, follow-up intensity, rehab referrals, patient education). These choices interact over time and affect readmission risk. Traditional models describe risk factors but do not optimize sequential strategies. RL enables learning policies that optimize long-horizon outcomes under uncertainty.

## MDP Formulation (Assumptions)
- States: abstracted patient recovery/risk stages after discharge (e.g., stable, improving, high-risk deterioration).
- Actions: intervention strategies (e.g., conservative monitoring, intensified follow-up, rehab escalation, early clinical intervention).
- Transitions: stochastic evolution of recovery given intervention choice.
- Rewards: positive for sustained recovery, large penalty for readmission, and step costs for inefficient or overly aggressive care.
- Terminals: successful recovery or readmission events.

Version 1 uses a data-driven environment scaffold to validate learning and evaluation pipelines before integrating action-aware models.

## Foundational Environment (Primary)
A data-driven environment scaffold is included to plug into the Kaggle dataset.
- Implemented in `src/eecs590_capstone/envs/data_env.py`.
- Loads the processed dataset and provides a `reset()` / `step()` interface.
- Uses proxy transitions because actions are not recorded in the dataset.
- Provides a reward shaping baseline for readmission outcomes.


## Dataset (Kaggle)
Selected dataset: **Diabetes 130-US Hospitals for Years 1999-2008** (readmission label includes `<30`, `>30`, `NO`).

Kaggle dataset page (login may be required):
```text
https://www.kaggle.com/datasets/ashikuzzamanshishir/diabetes-130-us-hospitals-for-years-1999-2008
```
Primary source (UCI):
```text
https://archive.ics.uci.edu/dataset/296/diabetic_readmission
```

Planned pipeline:
- Raw ingest into `data/raw/`
- Cleaning + feature engineering into `data/processed/`
- Train/validation/test splits
- Conversion to trajectories or transition estimates for RL experiments

Data setup instructions:
- `data/README.md`
Data-driven config:
- `configs/data_env.json`
MDP simulator config:
- `configs/mdp_sim.json`
State/action mapping:
- `docs/state_action_mapping.md`

## Evaluation Metrics and Baselines
Metrics (current + future):
- Average return
- Terminal recovery rate
- Readmission rate (once mapped from dataset)
- Intervention cost
- Time-to-recovery

Baselines:
- Random policy
- Conservative policy (low-intervention)
- Aggressive policy (high-intervention)
- Risk-score threshold rule (data-driven baseline)

## Scope and Milestones
- Phase 1 (complete): data-driven env scaffold, baseline policies, evaluation CLI, dataset pipeline.
- Phase 2: integrate Kaggle dataset; define state/action mappings; generate trajectories or transition models.
- Phase 3: introduce model-free RL (MC, TD, SARSA, Q-learning).
- Phase 4: function approximation and richer state representations.

## Repository Structure
- `src/`: RL code (envs, agents, MDP definitions, CLI).
- `outputs/`: trained policies, value functions, metrics, plots.
- `scripts/`: utilities (e.g., visualization).
- `tests/`: reserved for validation tests.
- `requirements.txt`: dependencies.

## How to Run
Activate the virtual environment and set the Python path:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
```

Data-driven baseline policy (no learning, config-based):
```powershell
python -m eecs590_capstone.cli.data_train --policy random
python -m eecs590_capstone.cli.data_eval --policy-path outputs/data_train/policy.json
```

Evaluate all baselines:
```powershell
python scripts/eval_baselines.py
```

Build the MDP simulator from data:
```powershell
python scripts/build_mdp.py
```

Run DP on the simulated MDP:
```powershell
python -m eecs590_capstone.cli.mdp_train --algo policy_iter
python -m eecs590_capstone.cli.mdp_train --algo value_iter
```

Plot DP results (bars + heatmaps + human-readable policy):
```powershell
python scripts/plot_mdp_results.py --algo policy_iter
python scripts/plot_mdp_results.py --algo value_iter
```

Render an HTML animation (open in a browser):
```powershell
python scripts/render_mdp_html.py --policy outputs/mdp/policy_iter_policy.json
```

Train tabular RL algorithms on the simulated MDP:
```powershell
python -m eecs590_capstone.cli.rl_train --algo mc
python -m eecs590_capstone.cli.rl_train --algo td_n --n 3
python -m eecs590_capstone.cli.rl_train --algo td_lambda --lambda 0.8
python -m eecs590_capstone.cli.rl_train --algo sarsa_n --n 3
python -m eecs590_capstone.cli.rl_train --algo sarsa_lambda --lambda 0.8
python -m eecs590_capstone.cli.rl_train --algo q_learning
```

Run all RL algorithms and compare curves:
```powershell
python scripts/run_all_rl.py --runs 5
python scripts/plot_learning_curves.py
```

Run unit tests:
```powershell
python -m pytest -q
```

## Version 1 Outputs
- Policy kernel
- Value function estimates
- Training metadata
- Evaluation metrics
- Policy visualization plots

## Future Work
- DQN and other function-approximation methods
- More realistic transition modeling (causal or learned dynamics)
- Richer state representations and clinical feature grouping
- Offline policy evaluation and safety constraints

## Author
Christianah Jemiyo  
PhD Student, Artificial Intelligence  
University of North Dakota
