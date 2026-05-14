# V1 Overview

`V1/` is the root-level navigation entry for the original capstone scaffold.

Version 1 is not a separate code subtree in this repository. Its implementation is the reusable baseline project structure at the repository root:

- `src/eecs590_capstone/`: base environments, agents, CLIs, and utilities.
- `scripts/`: baseline data, MDP, training, evaluation, and plotting scripts.
- `configs/`: compact configuration files for the initial scaffold.
- `data/`: dataset layout used by the original workflow.
- `outputs/mdp/` and `outputs/rl/`: representative baseline artifacts from the initial tabular workflow.

## What V1 Represents

V1 is the initial reproducible reinforcement-learning scaffold for hospital readmission planning. It focuses on:

- a compact data-driven environment,
- baseline and tabular RL policies,
- dynamic-programming utilities,
- simple evaluation and visualization outputs.

## Start Here

- Repository entry point: `README.md`
- Baseline code: `src/eecs590_capstone/`
- Baseline scripts: `scripts/`
- Baseline outputs: `outputs/mdp/` and `outputs/rl/`
