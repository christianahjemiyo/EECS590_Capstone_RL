# V2 Overview

`V2/` is the root-level navigation entry for the main capstone benchmark.

The actual V2 workflow lives in:

- `v2_pipeline/`: V2-specific scripts, configs, docs, checkpoints, and related assets.
- `outputs/v2_outputs/`: committed benchmark outputs, summaries, figures, and interpretation files.

## What V2 Adds

Relative to V1, Version 2 adds:

- MIMIC-oriented preprocessing,
- broader algorithm coverage,
- shared benchmarking across RL families,
- offline RL comparisons,
- multi-seed summaries and interpretation outputs.

## Start Here

- Operational runbook: `v2_pipeline/README.md`
- Main benchmark outputs: `outputs/v2_outputs/benchmark/`
- Broad algorithm comparison: `outputs/v2_outputs/all_algorithms/`
