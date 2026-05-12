# V2 Workflow

`v2_pipeline/` is the main capstone workflow directory. The root `README.md` is the repository entry point; this document is the operational runbook for the expanded benchmark.

V2 exists to answer one controlled question:

How do different reinforcement-learning families behave when they are trained on the same clinical decision problem, under the same state representation, reward design, and evaluation protocol?

That design choice is the point of the workflow. The benchmark is meant to compare algorithm fit, not to make every method look equally appropriate.

## What V2 Adds
Relative to the initial scaffold, V2 adds:
- MIMIC-oriented preprocessing and a clearer data-to-MDP pipeline.
- Shared benchmarking across DP, tabular RL, approximation methods, deep RL, and selected adapted advanced methods.
- Offline RL experiments from fixed replay-style data.
- Multi-seed summaries, interpretation files, and figure generation.
- Better documentation of limitations, modeling assumptions, and partial-observability extensions.

## Folder Guide
- `v2_pipeline/scripts/`: benchmark runners, preprocessing, plotting, interpretation, offline RL, and experiment utilities.
- `v2_pipeline/configs/`: V2-specific environment and MDP configs.
- `v2_pipeline/docs/`: supporting notes on algorithm coverage, technical challenges, validation planning, and POMDP extensions.
- `v2_pipeline/architectures/`: architecture notes for neural components.
- `v2_pipeline/checkpoints/`: saved model checkpoints for selected runs.
- `v2_pipeline/replay_buffers/`: storage layout and notes for replay-style/offline experiments.

For navigation help inside this subtree, see:
- `v2_pipeline/scripts/README.md`
- `v2_pipeline/docs/version-log.md`

## Core Workflow
Run from the repository root:

```powershell
$env:PYTHONPATH="src"
```

### 1. Preprocess data
```powershell
python v2_pipeline/scripts/preprocess_mimic.py --mimic-zip "<path-to-mimic-zip>"
python scripts/data_profile.py --data data/processed/mimic_data_clean.csv --out outputs/v2_outputs/data_profile_mimic.md
```

### 2. Build the MDP
```powershell
python scripts/build_mdp.py --config v2_pipeline/configs/mdp_sim_mimic.json --outdir outputs/v2_outputs/mdp
```

### 3. Run the main benchmark
```powershell
python scripts/run_benchmark.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/benchmark --seeds 5
```

### 4. Run offline RL
```powershell
python scripts/run_offline_benchmark.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/offline
```

### 5. Run broad algorithm coverage
```powershell
python v2_pipeline/scripts/run_v2_tabular_suite.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/tabular_suite --seeds 7,11,19,23,29
python v2_pipeline/scripts/run_v2_approx_suite.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/approx_suite --seeds 7,11,19,23,29
python v2_pipeline/scripts/run_v2_deep_suite.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/deep_suite --seeds 7,11,19
python v2_pipeline/scripts/run_v2_advanced_suite.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/advanced_suite --seeds 7,11,19
python v2_pipeline/scripts/run_v2_all_algorithms.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/v2_outputs/all_algorithms --seeds 7,11,19
```

### 6. Generate figures and interpretation outputs
```powershell
python scripts/make_figures.py --mdp outputs/v2_outputs/mdp/mdp.npz --data data/processed/train.csv
```

### 7. Sweep reward settings
```powershell
python scripts/run_reward_sweep.py --base-config v2_pipeline/configs/mdp_sim_mimic.json --outdir outputs/v2_outputs/reward_sweep
```

### 8. Build an MDP with learned embedding states
The default V2 config still uses the compact hand-built risk-state representation. If you want to test the tokenizer-style representation prototype, add a `representation` block with `mode: "tokenizer_embedding"` to the MDP config and rebuild:

```powershell
python scripts/build_mdp.py --config <tokenizer-config.json> --outdir outputs/v2_outputs/mdp_tokenizer
```

## How To Read The Results
Important interpretation rules:
- Higher return is better.
- In the current reward design, many returns are negative, so less negative is better.
- Dynamic programming and strong tabular methods should be expected to do well because the benchmark environment is still compact and tabular.
- Offline RL results matter because the motivating use case is logged healthcare data, not unrestricted online experimentation.
- Reward sensitivity matters. If intervention costs dominate benefits, conservative policies can rank well even when they are clinically less ambitious.
- The reward sweep lets you measure that sensitivity directly instead of treating it as only a qualitative limitation.

## Recommended Review Order
If you only inspect a few outputs, start here:
- `outputs/v2_outputs/benchmark/summary_metrics.csv`
- `outputs/v2_outputs/benchmark/INTERPRETATION_BENCHMARK.md`
- `outputs/v2_outputs/all_algorithms/summary_metrics.csv`
- `outputs/v2_outputs/offline/INTERPRETATION_OFFLINE.md`
- `outputs/v2_outputs/figures/reward_cost_tradeoff.png`
- `outputs/v2_outputs/figures/SALIENCY_INTERPRETATION.md`

## Scope and Limitations
V2 is meant to be defensible, not overclaimed.

- The core environment is still a modeled clinical MDP rather than a full causal clinical simulator.
- Some advanced methods are included as adaptation studies and coverage exercises, not because they are the most natural method for a small discrete MDP.
- Offline RL is present, but the logged data is still simplified compared with true longitudinal treatment logs.
- Representation learning now includes a lightweight tokenizer/embedding prototype, but the main benchmark still relies on the simpler hand-constructed state abstraction.
- Reward design remains one of the most important open modeling choices in the project.

## Immediate Next Directions
The most useful next improvements are:
- systematic reward tuning or search over reward/cost weights,
- richer offline RL experiments,
- stronger learned state embeddings beyond the current lightweight tokenizer prototype,
- stronger experiment logging around failures, ablations, and debugging.

## Versioning Notes
V2 remains a named subtree because it is the main benchmark workflow developed after the initial scaffold, but versioning is documented rather than multiplied into new top-level project trees.

- The repository root is the primary entry point.
- Root-level wrapper scripts such as `scripts/run_benchmark.py` and `scripts/run_reward_sweep.py` reduce the need to navigate directly into `v2_pipeline/scripts/`.
- Additional workflow history is recorded in `v2_pipeline/docs/version-log.md`.

## Citations and Acknowledgments
Key references used directly in the V2 workflow:

- Johnson, A. E. W., Pollard, T. J., Shen, L., et al. "MIMIC-IV, a freely accessible electronic health record dataset." Scientific Data, 2023.
- Sutton, R. S., and Barto, A. G. Reinforcement Learning: An Introduction. Second edition.
- Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., and Clore, J. N. "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." 2014.

Acknowledgments:
- Author: Christianah Jemiyo.
- AI assistant support was used for code organization, documentation cleanup, and implementation assistance; final modeling decisions and project claims were reviewed and curated by the author.

