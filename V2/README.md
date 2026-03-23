# V2 Capstone Workflow

Version 2 is the main workflow for the capstone. It brings together the readmission-focused data pipeline, MDP construction, training scripts, interpretation outputs, and benchmark comparisons under `V2/` and `outputs/V2/`.

The goal of V2 is not just to produce one good policy. It also covers the full set of algorithms used in class, compares them in the same environment, and shows which methods fit this clinical decision problem best.

## Abstract

This version of the capstone studies how reinforcement learning can model sequential clinical intervention decisions related to hospital readmission risk. The workflow starts with MIMIC-based preprocessing, converts the resulting patient data into a tabular Markov Decision Process, and evaluates a broad set of Dynamic Programming, tabular RL, function-approximation, deep RL, actor-critic, model-based, and partial-observability methods. The main objective is to identify which algorithms learn the strongest long-term intervention policies under the current reward design while also showing full class algorithm coverage.

## 1) What V2 Is Solving

The project models sequential clinical intervention decisions related to hospital readmission risk. The core idea is:

1. Build a patient-state representation from clinical data.
2. Convert that representation into a tabular Markov Decision Process.
3. Train many reinforcement learning algorithms on the same environment.
4. Compare learned policies using shared evaluation metrics.
5. Interpret what the policy is doing and why.

In this project, the environment is fixed after the MDP is built. Algorithms do not change the environment itself. Instead, they learn:

- a policy: which intervention action to choose in each state
- a value function: how good each state is in long-term return
- action values: how good each action is in each state
- or an approximate environment model for planning

## 2) V2 Folder Structure

### Scripts

- `V2/scripts/preprocess_mimic.py`
  Prepares the MIMIC-based dataset and creates train/validation/test splits.

- `V2/scripts/train_dqn.py`
  Trains the V2 DQN baseline on the current tabular MDP.

- `V2/scripts/plot_v2_results.py`
  Produces core comparison plots for the standard V2 baseline runs.

- `V2/scripts/plot_v2_saliency.py`
  Produces V2 interpretability visualizations for state-action preference and feature impact.

- `V2/scripts/plot_v2_nn_saliency.py`
  Produces neural saliency views from the saved DQN checkpoint.

- `V2/scripts/plot_v2_special_visuals.py`
  Produces extra summary figures for algorithm families, reward/cost tradeoffs, and policy flow across states.

- `V2/scripts/rasterize_v2_state_action.py`
  Produces a rasterized state-action value image for quick inspection utilities next to saliency.

- `V2/scripts/write_v2_interpretation.py`
  Writes plain-language interpretation notes for the baseline V2 outputs.

- `V2/scripts/offline_rl_benchmark.py`
  Runs offline RL comparisons on a fixed replay-style dataset.

- `V2/scripts/run_v2_benchmark.py`
  Runs the broad V2 benchmark across DP, online RL, and offline RL with uncertainty across seeds.

- `V2/scripts/run_v2_tabular_suite.py`
  Runs the tabular RL family for class coverage.

- `V2/scripts/run_v2_classical_views.py`
  Runs explicit forward-view and backward-view classical RL variants for TD and SARSA family methods.

- `V2/scripts/run_v2_approx_suite.py`
  Runs the function-approximation family.

- `V2/scripts/run_v2_deep_suite.py`
  Runs deep RL and policy-gradient methods.

- `V2/scripts/run_v2_advanced_suite.py`
  Runs advanced adapted methods such as continuous-control-style, model-based, belief-state, and RNN-based methods.

- `V2/scripts/run_v2_all_algorithms.py`
  Runs a single unified comparison across all implemented algorithms.

- `V2/scripts/make_all_v2_figures.py`
  One-command pipeline that can generate most V2 results and figures.

### Configs

- `V2/configs/data_env_mimic.json`
- `V2/configs/mdp_sim_mimic.json`

### Docs

- `V2/docs/external_validation_plan.md`
- `V2/docs/algorithm_coverage_v2.md`
- `V2/docs/classical_rl_views.md`
- `V2/docs/technical-challenges.md`
- `V2/docs/pomdp_formalization.md`

### Architecture and checkpoint support

- `V2/architectures/`
- `V2/checkpoints/`
- `V2/replay_buffers/`

## Methodology

The V2 methodology follows a consistent experimental structure:

1. Data preparation
   Raw clinical data is cleaned, transformed, and split into train, validation, and test sets.

2. State and action design
   Patient information is compressed into a finite state representation, while intervention choices are mapped to discrete action levels.

3. MDP construction
   Transition probabilities and rewards are estimated to form a tabular MDP that represents the sequential decision problem.

4. Algorithm training
   Each algorithm family is trained on the same V2 environment or an adapted version of it.

5. Evaluation
   Learned policies are compared using rollout return, multi-seed summaries, and where appropriate offline evaluation.

6. Interpretation
   Saliency, feature-impact views, and written interpretation files are used to explain why certain policies behave as they do.

This setup keeps the comparison controlled. Every method is judged against the same V2 problem instead of being run on different tasks.

## 3) Algorithm Coverage in V2

V2 now includes the following algorithm families:

### Dynamic Programming

- Policy Iteration
- Value Iteration

### Tabular RL

- Monte Carlo Control (epsilon-greedy)
- TD(0)
- TD(lambda)
- SARSA
- Expected SARSA
- SARSA(lambda)
- Q-Learning
- Q(lambda)
- Dyna-Q

### Function Approximation

- Linear Function Approximation
- Semi-gradient TD
- Gradient TD
- Approximate Q-Learning

### Deep RL and Policy Gradient

- DQN
- Double DQN
- Dueling DQN
- REINFORCE
- A2C
- A3C
- PPO
- TRPO

### Advanced / Adapted Methods

- DDPG
- TD3
- SAC
- Learned environment models
- Belief state update methods
- RNN-based RL

Important note:

- Some advanced methods are adapted to the discrete V2 MDP so they can be compared in one benchmark.
- This is acceptable for course coverage and comparative analysis, but those methods should not be presented as the most natural fit for a small tabular clinical MDP.

## DRL Justification

The main V2 environment is a compact tabular clinical decision problem, so not every DRL algorithm is equally natural for it. The most appropriate deep methods for this setting are the value-based and actor-critic methods that can still work well with discrete actions and relatively small state representations.

The strongest practical choices for this environment are:

- `DQN`, because the action space is discrete and the environment can still benefit from nonlinear value estimation
- `Double DQN`, because it reduces overestimation bias compared with plain DQN
- `Dueling DQN`, because it can separate the value of being in a state from the advantage of taking a specific action
- `REINFORCE`, `A2C`, `A3C`, `PPO`, and `TRPO`, because they provide direct policy-learning baselines and useful actor-critic comparisons in the same environment

The less natural but still useful methods are:

- `DDPG`, `TD3`, and `SAC`, because they are designed for continuous-control settings and therefore require adaptation in V2

Why these choices still make sense:

- the pros are broad algorithm coverage, comparison across RL families, and a clear record of how discrete-clinical environments differ from continuous-control tasks
- the cons are manageable because the environment is stable, compact, and cheap enough to rerun while tuning or debugging
- even when these methods are not top performers, they still help show which algorithm families fit the environment well and which ones do not

## 4) Environment and Evaluation Meaning

The V2 environment is a tabular MDP derived from the project’s readmission setting. In practical terms:

- states represent patient risk or condition groupings
- actions represent intervention intensity or treatment choice
- rewards encode the tradeoff between intervention cost and clinical benefit
- returns summarize long-term policy quality

How to read the metrics:

- Higher return means better performance.
- In the current reward design, returns are often negative, so less negative is better.
- `policy.json` tells you which intervention the algorithm prefers in each state.
- `value_function.json` tells you how favorable each state is under that learned policy.
- `q_values.json` tells you how strongly each action is preferred in each state.
- `learning_curve.json` shows whether the algorithm improved during training.

## 5) Recommended End-to-End Workflow

Run from repo root.

```powershell
$env:PYTHONPATH="src"
```

### One-command workflow

```powershell
python V2/scripts/make_all_v2_figures.py --mdp outputs/V2/mdp/mdp.npz --data data/processed/train.csv
```

### Faster sanity run

```powershell
python V2/scripts/make_all_v2_figures.py --mdp outputs/V2/mdp/mdp.npz --data data/processed/train.csv --quick
```

## 6) Manual Step-by-Step Workflow

### A. Data preparation

```powershell
python V2/scripts/preprocess_mimic.py --mimic-zip "C:\Users\Christianah\OneDrive - North Dakota University System\Grad_applications\PhysioNet_Data\mimic-iv-3.1.zip"
python scripts/data_profile.py --data data/processed/mimic_data_clean.csv --out outputs/V2/data_profile_mimic.md
```

Optional faster preprocessing:

```powershell
python V2/scripts/preprocess_mimic.py --mimic-zip "<path-to-zip>" --skip-direct-counts
```

### B. Build the V2 MDP

```powershell
python scripts/build_mdp.py --config V2/configs/mdp_sim_mimic.json --outdir outputs/V2/mdp
```

### C. Train Dynamic Programming baselines

```powershell
python -m eecs590_capstone.cli.mdp_train --algo policy_iter --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/mdp
python -m eecs590_capstone.cli.mdp_train --algo value_iter --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/mdp
python scripts/plot_mdp_results.py --outdir outputs/V2/mdp --algo policy_iter
python scripts/plot_mdp_results.py --outdir outputs/V2/mdp --algo value_iter
```

### D. Train baseline RL methods

```powershell
python -m eecs590_capstone.cli.rl_train --algo q_learning --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl
python -m eecs590_capstone.cli.rl_train --algo double_q_learning --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl_double_q
python -m eecs590_capstone.cli.rl_train --algo td0 --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl_td0
python -m eecs590_capstone.cli.rl_train --algo sarsa --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl_sarsa
python V2/scripts/train_dqn.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl_dqn
python V2/scripts/plot_v2_results.py
python V2/scripts/write_v2_interpretation.py
```

### E. Generate interpretability outputs

```powershell
python V2/scripts/plot_v2_saliency.py --data data/processed/train.csv --outdir outputs/V2/figures
python V2/scripts/plot_v2_nn_saliency.py --outdir outputs/V2/figures
python V2/scripts/plot_v2_special_visuals.py --outdir outputs/V2/figures
python V2/scripts/rasterize_v2_state_action.py --q-path outputs/V2/rl/q_values.json --out outputs/V2/figures/state_action_raster.png
```

### F. Offline benchmark

```powershell
python V2/scripts/offline_rl_benchmark.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/offline
python V2/scripts/run_v2_benchmark.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/benchmark --seeds 5
```

### G. Full algorithm family coverage

```powershell
python V2/scripts/run_v2_tabular_suite.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/tabular_suite --seeds 7,11,19,23,29
python V2/scripts/run_v2_classical_views.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/classical_views --seeds 7,11,19
python V2/scripts/run_v2_approx_suite.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/approx_suite --seeds 7,11,19,23,29
python V2/scripts/run_v2_deep_suite.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/deep_suite --seeds 7,11,19
python V2/scripts/run_v2_advanced_suite.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/advanced_suite --seeds 7,11,19
```

### H. One comparison across all algorithms

```powershell
python V2/scripts/run_v2_all_algorithms.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/all_algorithms --seeds 7,11,19
```

### I. Replay buffer management

```powershell
python V2/scripts/manage_replay_buffer.py --root V2/replay_buffers/raw --algo dqn --task foundation_env --freshness fresh --max-gb 1.0 --keep-latest 10
python V2/scripts/bayes_update.py --prior 0.2,0.5,0.3 --likelihood 0.7,0.2,0.5 --out outputs/V2/bayes_update_example.json
```

## 7) Most Important Outputs

### Core benchmark outputs

- `outputs/V2/benchmark/summary_metrics.csv`
- `outputs/V2/benchmark/per_seed_metrics.csv`
- `outputs/V2/benchmark/figures/benchmark_rollout_comparison.png`
- `outputs/V2/benchmark/figures/benchmark_fqe_comparison.png`
- `outputs/V2/benchmark/INTERPRETATION_BENCHMARK.md`

### Saliency and interpretability

- `outputs/V2/figures/saliency_state_action_heatmap.png`
- `outputs/V2/figures/saliency_feature_outcome_heatmap.png`
- `outputs/V2/figures/saliency_feature_scores.csv`
- `outputs/V2/figures/SALIENCY_INTERPRETATION.md`
- `outputs/V2/figures/nn_saliency_dqn_input_heatmap.png`
- `outputs/V2/figures/nn_saliency_dqn_hidden_heatmap.png`
- `outputs/V2/figures/NN_SALIENCY_INTERPRETATION.md`
- `outputs/V2/figures/algorithm_family_overview.png`
- `outputs/V2/figures/reward_cost_tradeoff.png`
- `outputs/V2/figures/policy_transition_diagram.png`
- `outputs/V2/figures/SPECIAL_VISUALS_INTERPRETATION.md`

### Tabular family outputs

- `outputs/V2/tabular_suite/summary_metrics.csv`
- `outputs/V2/tabular_suite/per_seed_metrics.csv`
- `outputs/V2/tabular_suite/figures/tabular_rollout_comparison.png`
- `outputs/V2/tabular_suite/figures/tabular_learning_curves.png`

### Classical forward-view and backward-view outputs

- `outputs/V2/classical_views/summary_metrics.csv`
- `outputs/V2/classical_views/per_seed_metrics.csv`
- `outputs/V2/classical_views/figures/classical_views_comparison.png`

### All-algorithm comparison outputs

- `outputs/V2/all_algorithms/summary_metrics.csv`
- `outputs/V2/all_algorithms/per_seed_metrics.csv`
- `outputs/V2/all_algorithms/figures/all_algorithms_rollout_comparison.png`
- `outputs/V2/all_algorithms/figures/top10_algorithms_rollout.png`
- `outputs/V2/all_algorithms/INTERPRETATION_ALL_ALGOS.md`

### Supporting V2 infrastructure

- `V2/docs/technical-challenges.md`
- `V2/docs/classical_rl_views.md`
- `V2/docs/pomdp_formalization.md`
- `V2/architectures/`
- `V2/checkpoints/`
- `V2/replay_buffers/`

## 8) Reading the Results

The main rule for reading the V2 results is simple:

- less negative return is better
- higher value means the algorithm found a better long-term intervention strategy

What usually happens in V2:

- Dynamic Programming methods perform very strongly because the environment is compact and tabular.
- Strong tabular RL methods such as `TD(lambda)`, `Expected SARSA`, `SARSA(lambda)`, and `Q-Learning` often perform competitively.
- DQN-family methods can also perform well, especially when nonlinear action-value estimation helps.
- Adapted continuous-control and partial-observability methods are included for breadth and comparison, but they are not always the best natural fit for this exact environment.

This is not a weakness. It is a reasonable outcome for this kind of environment:

- algorithm performance depends on environment structure
- more complex methods do not automatically outperform simpler methods
- exact and tabular methods can remain strongest when the MDP is small and well structured

## 9) How Each Algorithm Helps the Project

The project goal is to learn intervention strategies that reduce modeled readmission risk while balancing action cost.

Different algorithm families help in different ways:

- Dynamic Programming tells us the best achievable policy when transitions are known.
- Tabular RL shows how policies can be learned directly from experience.
- Trace-based methods improve temporal credit assignment across sequences of care decisions.
- Function approximation shows how the project can scale when state descriptions become richer.
- Deep RL methods show nonlinear value learning and policy improvement.
- Actor-critic and policy-gradient methods show direct policy optimization.
- Model-based, belief-state, and RNN methods show how the project could extend to uncertainty and partial observability.

## 10) Key V2 Output Files

The following files are the easiest places to start when reviewing the main V2 results:

- `outputs/V2/all_algorithms/summary_metrics.csv`
- `outputs/V2/all_algorithms/figures/all_algorithms_rollout_comparison.png`
- `outputs/V2/all_algorithms/figures/top10_algorithms_rollout.png`
- `outputs/V2/all_algorithms/INTERPRETATION_ALL_ALGOS.md`
- `outputs/V2/benchmark/summary_metrics.csv`
- `outputs/V2/figures/saliency_state_action_heatmap.png`
- `outputs/V2/figures/saliency_feature_outcome_heatmap.png`

## 11) Why `outputs/V2` Should Stay Committed

- Results are reproducible.
- Figures and tables are auditable.
- Reviewers can inspect outputs without rerunning every experiment.
- V1 and V2 remain cleanly separated.
- The benchmark becomes a stable record of what was run and what performed best.

## Limitations and Future Work

The current V2 environment is still a compact tabular MDP. That makes it useful for benchmarking and course coverage, but it also creates some limits that are worth stating clearly.

### Current limitations

- Exact and tabular methods naturally have an advantage because the environment is relatively small and structured.
- Some advanced methods such as `DDPG`, `TD3`, `SAC`, belief-state methods, and RNN-based RL are implemented through adaptations so they can run in the same V2 framework.
- The current saliency pipeline is useful for interpretability, but it can still be extended further with stronger gradient-based neural attribution methods.
- Reward design strongly influences ranking. If intervention costs dominate benefits, even strong learning methods may prefer conservative policies.

### Next improvements

There are several natural ways to extend the project:

- build a richer or partially observed environment where belief-state and RNN methods become more natural
- extend the action space beyond the current discrete intervention setup
- strengthen offline evaluation with more realistic logged clinical trajectories
- calibrate the reward function to better reflect desired clinical tradeoffs
- continue improving neural interpretability for DQN-family and policy-gradient models
- perform external validation on a second dataset or cohort split
