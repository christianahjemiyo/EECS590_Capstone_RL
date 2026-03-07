# V2 Upgrade Pack (MIMIC-IV + Benchmarks + Saliency)

This folder is the Version 2 capstone workflow:
1. Build MIMIC-based readmission data
2. Build tabular MDP
3. Train/evaluate DP and RL algorithms
4. Generate interpretable and presentation-ready visualizations
5. Run benchmarks with uncertainty across seeds

## 1) What each script does
- `V2/scripts/preprocess_mimic.py`: creates `mimic_data_clean.csv` + train/val/test splits.
- `V2/scripts/plot_v2_results.py`: baseline V2 comparison plots (learning curves + algo bars).
- `V2/scripts/plot_v2_saliency.py`: saliency maps for action impact and feature impact.
- `V2/scripts/write_v2_interpretation.py`: writes plain-language V2 interpretation notes.
- `V2/scripts/offline_rl_benchmark.py`: offline RL comparison (FQI vs conservative variant).
- `V2/scripts/run_v2_benchmark.py`: broad benchmark (DP, online RL, offline RL, CIs).
- `V2/scripts/run_v2_tabular_suite.py`: class-family tabular suite (MC/TD/SARSA/Q variants).

Config files:
- `V2/configs/data_env_mimic.json`
- `V2/configs/mdp_sim_mimic.json`

Planning note:
- `V2/docs/external_validation_plan.md`

## 2) End-to-End Commands
Run from repo root:

```powershell
$env:PYTHONPATH="src"

# One-command figure pipeline (recommended)
python V2/scripts/make_all_v2_figures.py --mdp outputs/V2/mdp/mdp.npz --data data/processed/train.csv

# Optional fast sanity run
python V2/scripts/make_all_v2_figures.py --mdp outputs/V2/mdp/mdp.npz --data data/processed/train.csv --quick

# Manual step-by-step commands
```

```powershell
$env:PYTHONPATH="src"

# A. Data preparation
python V2/scripts/preprocess_mimic.py --mimic-zip "C:\Users\Christianah\OneDrive - North Dakota University System\Grad_applications\PhysioNet_Data\mimic-iv-3.1.zip"
python scripts/data_profile.py --data data/processed/mimic_data_clean.csv --out outputs/V2/data_profile_mimic.md

# B. Build MDP and train DP baselines
python scripts/build_mdp.py --config V2/configs/mdp_sim_mimic.json --outdir outputs/V2/mdp
python -m eecs590_capstone.cli.mdp_train --algo policy_iter --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/mdp
python -m eecs590_capstone.cli.mdp_train --algo value_iter --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/mdp
python scripts/plot_mdp_results.py --outdir outputs/V2/mdp --algo policy_iter
python scripts/plot_mdp_results.py --outdir outputs/V2/mdp --algo value_iter

# C. Train RL baselines
python -m eecs590_capstone.cli.rl_train --algo q_learning --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl
python -m eecs590_capstone.cli.rl_train --algo double_q_learning --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl_double_q
python V2/scripts/plot_v2_results.py
python V2/scripts/write_v2_interpretation.py

# D. Saliency maps (interpretability)
python V2/scripts/plot_v2_saliency.py --data data/processed/train.csv --outdir outputs/V2/figures

# E. Offline and multi-seed benchmarks
python V2/scripts/offline_rl_benchmark.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/offline
python V2/scripts/run_v2_benchmark.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/benchmark --seeds 5

# F. Class algorithm coverage: MC/TD/SARSA/Q suite
python V2/scripts/run_v2_tabular_suite.py --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/tabular_suite --seeds 7,11,19,23,29
```

## 3) Most Important Outputs to Show
Core performance:
- `outputs/V2/benchmark/summary_metrics.csv`
- `outputs/V2/benchmark/figures/benchmark_rollout_comparison.png`
- `outputs/V2/benchmark/figures/benchmark_fqe_comparison.png`
- `outputs/V2/benchmark/INTERPRETATION_BENCHMARK.md`

Saliency and interpretability:
- `outputs/V2/figures/saliency_state_action_heatmap.png`
- `outputs/V2/figures/saliency_feature_outcome_heatmap.png`
- `outputs/V2/figures/SALIENCY_INTERPRETATION.md`

Class algorithm suite:
- `outputs/V2/tabular_suite/summary_metrics.csv`
- `outputs/V2/tabular_suite/figures/tabular_rollout_comparison.png`
- `outputs/V2/tabular_suite/figures/tabular_learning_curves.png`

## 4) Explanation
- Higher return (less negative) is better in this reward design.
- DP is the tabular upper benchmark when transitions are known.
- RL methods matter because they scale when true dynamics are unknown.
- Saliency maps show:
1. Which intervention action is preferred by state (policy behavior)
2. Which features most influence modeled readmission risk/protection

## 5) Why `outputs/V2` is committed
- Results are reproducible and auditable.
- Reviewers can inspect plots/metrics without rerunning everything.
- V1 and V2 artifacts stay separated and clear.
