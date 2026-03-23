# V2 Checkpoints

This directory stores the best available checkpoint artifacts for neural-network-based V2 runs.

## Layout

- `V2/checkpoints/<algorithm>/foundation_env/default/`

Each algorithm directory can contain:

- `model_checkpoint.npz`
  Stored weights or model state arrays.

- `checkpoint_meta.json`
  Metadata such as algorithm name, MDP path, and output directory.

For algorithms with separate components, checkpoint payloads may contain:

- online and target networks for DQN-style methods
- actor and critic weights for actor-critic methods
- multiple critic heads where applicable, such as TD3
