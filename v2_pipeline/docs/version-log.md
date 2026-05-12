# V2 Version Log

This file exists to keep versioning history in documentation rather than creating additional top-level project trees.

## V2.0
- Established the main benchmark workflow under `v2_pipeline/`.
- Added MIMIC-oriented preprocessing, a shared MDP pipeline, broad algorithm coverage, offline RL comparisons, figures, and interpretation outputs.

## V2.1
- Clarified the repository narrative so the root README is the primary entry point and V2 acts as the operational workflow.
- Added root-level wrapper scripts so the benchmark can be run without navigating directly into `v2_pipeline/scripts/`.
- Expanded `technical-challenges.md` into a more useful engineering notebook.
- Added reward-sweep tooling so reward design can be treated as a tunable modeling choice.
- Added a lightweight tokenizer/embedding representation prototype for state construction experiments.

## Guidance
- Extend this log for future workflow changes.
- Avoid creating a separate top-level `V3` tree unless the project truly becomes a different workflow with different assumptions.

