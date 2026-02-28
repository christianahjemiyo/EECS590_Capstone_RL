# V2 Results Summary

## Dataset Build
- Source: MIMIC-IV v3.1 (local zip, PhysioNet)
- Output rows: 546,028 admissions
- Label distribution:
  - `NO`: 223,500
  - `>30`: 213,183
  - `<30`: 109,345

## Training Runs
- MDP config: `V2/configs/mdp_sim_mimic.json`
- DP (policy iteration):
  - `avg_return`: -570.2948
  - `std_return`: 18.8082
- DP (value iteration):
  - `avg_return`: -570.2948
  - `std_return`: 18.8082
- RL (Q-learning):
  - `avg_return`: -571.6260
  - `std_return`: 19.1843

Detailed artifacts are in:
- `outputs/V2/mdp/`
- `outputs/V2/rl/`
