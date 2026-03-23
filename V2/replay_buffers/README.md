# V2 Replay Buffers

This directory is reserved for raw replay buffer data used by deep RL experiments.

## Recommended layout

- `V2/replay_buffers/raw/<algorithm>/<task>/<policy_freshness>/`

Example:

- `V2/replay_buffers/raw/dqn/foundation_env/fresh/`
- `V2/replay_buffers/raw/dqn/foundation_env/stale/`

Replay files are intended to be stored as `.npz` archives with fields such as:

- `s`
- `a`
- `r`
- `sn`
- `done`

The replay-management utility in `V2/scripts/manage_replay_buffer.py` can be used to:

- check total replay size against a limit
- keep only the newest files
- replace older experience with newer experience
- organize data by algorithm, task, and policy freshness
