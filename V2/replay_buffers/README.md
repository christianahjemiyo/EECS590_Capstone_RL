# V2 Replay Buffers

This directory contains the V2 replay buffer layout used for deep RL experiments and related storage management.

The replay-buffer structure is organized by:

- algorithm
- task
- policy freshness

## Layout

- `V2/replay_buffers/raw/<algorithm>/<task>/<policy_freshness>/`

Example directories:

- `V2/replay_buffers/raw/dqn/foundation_env/fresh/`
- `V2/replay_buffers/raw/dqn/foundation_env/stale/`

Replay archives are stored as `.npz` files with fields such as:

- `s`
- `a`
- `r`
- `sn`
- `done`

The replay-management utility in `V2/scripts/manage_replay_buffer.py` is used to:

- check total replay size against a limit
- keep only the newest files
- replace older experience with newer experience
- organize data by algorithm, task, and policy freshness

This keeps replay storage bounded, reproducible, and easier to audit when comparing runs across algorithms and configurations.
