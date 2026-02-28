# V2 Interpretation (Presentation Notes)

## 1) What these results mean 
- We are testing how well each algorithm learns a policy that avoids costly readmission outcomes.
- In this setup, higher return is better (less negative is better).
- Because readmission penalties are large, negative returns are expected; the key is relative performance.

## 2) Quick metric snapshot
| Algorithm | Avg Return | Std Return |
| --- | ---: | ---: |
| Policy Iteration (DP) | -570.295 | 18.808 |
| Value Iteration (DP) | -570.295 | 18.808 |
| Q-Learning | -571.626 | 19.184 |
| Double Q-Learning | -576.175 | 20.698 |

## 3) How to read the figures
- `outputs/V2/mdp/*_value_bar.png`: how valuable each risk-state is under the learned plan.
- `outputs/V2/mdp/*_policy_bar.png`: which action each state is assigned.
- `outputs/V2/figures/rl_learning_curve.png`: how quickly RL improves with more episodes.
- `outputs/V2/figures/algo_avg_return_comparison.png`: side-by-side comparison of final performance.

## 4) Notes
- DP methods are the benchmark here because they solve the known tabular MDP directly.
- RL methods are useful because they scale to settings where transitions are not explicitly known.
- Double Q-Learning is a stronger variant of Q-Learning because it reduces overestimation bias.
- V2 is now fully reproducible: preprocessing, training, and figures are versioned with results.

## 5) Limitation
- Current simulator uses proxy transitions derived from data, not causal treatment effects.
- This is a strong engineering baseline, and a bridge toward offline RL with richer clinical actions.

