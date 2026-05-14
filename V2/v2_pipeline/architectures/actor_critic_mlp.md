# Actor-Critic MLP

- Input type: compact tabular state representation
- Actor output: action logits or policy probabilities
- Critic output: state-value estimate
- Main use cases:
  - REINFORCE
  - A2C
  - A3C
  - PPO
  - TRPO

For V2, these models stay intentionally lightweight so that policy-gradient comparisons remain easy to run and reproduce.
