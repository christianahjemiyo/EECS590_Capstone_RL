# V2 Checkpoints

This directory stores checkpoint artifacts for neural-network-based V2 runs.

## Layout

- `V2/checkpoints/<algorithm>/foundation_env/default/`

Each algorithm directory contains:

- `model_checkpoint.npz`
  Stored weights or model-state arrays.

- `checkpoint_meta.json`
  Metadata such as algorithm name, MDP path, and output directory.

For algorithms with separate components, checkpoint payloads can include:

- online and target networks for DQN-style methods
- actor and critic weights for actor-critic methods
- multiple critic heads where applicable, such as TD3

The goal of this layout is to keep the saved model state separate from figures, metrics, and JSON summaries so that training artifacts are easy to inspect and compare.
