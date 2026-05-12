# DQN MLP

- Input type: compact state index or feature-like tabular state encoding
- Hidden layers: single hidden MLP layer
- Nonlinearity: ReLU
- Output: one Q-value per action
- Special handling:
  - DQN keeps online and target networks
  - Double DQN separates action selection from target evaluation
  - Dueling DQN separates state value and action advantage

This architecture is appropriate for V2 because the environment does not require a heavy vision model.
