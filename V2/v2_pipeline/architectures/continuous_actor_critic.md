# Continuous Actor-Critic Setup

- Input type: compact tabular state features
- Actor output: continuous action proxy that is later mapped back to the discrete V2 action space
- Critic output: value or Q estimate for state-action pairs
- Main use cases:
  - DDPG
  - TD3
  - SAC

These are adapted versions for the discrete V2 environment, included mainly for algorithm coverage and comparative analysis.
