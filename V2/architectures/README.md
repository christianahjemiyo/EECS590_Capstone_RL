# V2 Neural Architectures

This directory documents the main neural-network styles used for Version 2.

The V2 environment is based on compact tabular or feature-based state inputs, so the architectures here are lightweight MLP-style models rather than image CNNs.

## Files

- `dqn_mlp.md`
  Base value-network design for DQN-style methods.

- `actor_critic_mlp.md`
  Shared design notes for REINFORCE, A2C, A3C, PPO, and TRPO style models.

- `continuous_actor_critic.md`
  Lightweight actor/critic setup used for the adapted DDPG, TD3, and SAC workflows in V2.
