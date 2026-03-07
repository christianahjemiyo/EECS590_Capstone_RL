# Offline RL Interpretation (V2)

## What changed
- We trained from a fixed replay dataset (offline), not from live interaction.
- We compared FQI and a conservative Q-learning variant.

## How to read this
- Higher return (less negative) is better.
- Conservative Q-learning is designed to avoid overly optimistic action values.
- If conservative Q does not win, that still shows honest benchmarking.

## Results
- FQI avg return: -226.813 +/- 15.534
- Conservative Q avg return: -238.043 +/- 15.181

## Figures
- `offline_training_curves.png`
- `offline_return_comparison.png`
