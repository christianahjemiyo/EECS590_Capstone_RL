# V2

`V2/` contains the main benchmark workflow.

## Contents

- `v2_pipeline/`: V2-specific scripts, configs, docs, checkpoints, replay-buffer notes, and architecture notes.
- `scripts/`: thin V2 entry-point wrappers plus shared build/profile utilities used by the V2 workflow.
- `src/`: V2-contained package code used by the benchmark.
- `tests/`: V2-specific tests.
- `outputs/v2_outputs/`: committed V2 benchmark outputs.

## Data

The canonical dataset is shared at the repository root under `../data/`.

## Start Here

- `v2_pipeline/README.md`
- `outputs/v2_outputs/benchmark/summary_metrics.csv`
- `outputs/v2_outputs/all_algorithms/summary_metrics.csv`
