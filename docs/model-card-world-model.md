# Model Card: Version 3 World Model

## Model Name
Version 3 Tabular World Model for Hospital Readmission RL Capstone

## Summary
This model card describes the Version 3 world-model component in the EECS590 capstone project on reinforcement learning for hospital readmission planning. The Version 3 model is a lightweight tabular world model that learns or reconstructs transition dynamics and expected rewards in the project's existing clinical MDP abstraction.

This is not a clinical deployment tool. It should only be interpreted as a decision-support simulation for a class capstone and research-style experimentation.

## Intended Use
- Support offline policy evaluation inside a learned simulator.
- Help compare simple and previously trained policies under modeled transition dynamics.
- Demonstrate how world models can extend a healthcare RL benchmark in a reproducible classroom project.
- Provide an interpretable bridge between a fixed tabular MDP and a learned simulation layer.

## Not Intended Use
- Real-world clinical deployment.
- Medical advice or patient-specific care recommendations.
- Autonomous discharge, follow-up, or intervention planning.
- Any claim of causal validity about real patient progression.
- Any safety-critical use without substantial external validation and domain review.

## Data / MDP Source
The Version 3 world model is built on top of the capstone's existing tabular MDP artifacts rather than directly on a hospital production system.

Primary source in the current implementation:
- `outputs/v2_outputs/mdp/mdp.npz`

Supporting project context:
- processed diabetes readmission data used in earlier project stages
- MIMIC-oriented preprocessing and modeling choices introduced in Version 2
- tabular state abstractions and proxy actions already defined by the capstone

The current Version 3 runner can also fit from trajectory-like transitions if available, but the main demonstrated workflow fits from the existing MDP arrays.

## Inputs and Outputs

### Inputs
- tabular transition probabilities `P[s, a, s']`
- tabular rewards `R[s, a, s']` or expected rewards by state-action
- optional terminal-state definitions
- optional trajectory-like transition records:
  - `state`
  - `action`
  - `next_state`
  - `reward`
  - `done`

### Outputs
- learned transition probabilities `P_hat(s' | s, a)`
- learned expected rewards `R_hat(s, a)`
- simulated trajectories in the learned world model
- policy evaluation summaries such as:
  - average return
  - return standard deviation
  - average rollout length
  - terminal-state rate
- comparison metrics against the original MDP
- figures and summary files under `outputs/V3/world_model/`

## Assumptions
- The patient decision problem can be represented as a compact tabular MDP.
- States are abstracted risk or recovery groups rather than full patient histories.
- Actions are proxy care-management or intervention-intensity decisions.
- Transition behavior can be approximated from the existing MDP or available transitions.
- Expected rewards summarize modeled tradeoffs among recovery, readmission avoidance, and intervention cost.
- The learned simulator is only as meaningful as the underlying state, action, reward, and transition abstractions.

## Evaluation Metrics
The current Version 3 evaluation code computes:
- mean absolute transition error
- mean squared transition error
- KL divergence with numerical safety when available
- reward mean absolute error
- policy-level rollout metrics inside the learned simulator:
  - average return
  - standard deviation of return
  - average rollout length
  - terminal-state rate

Associated artifacts are saved in `outputs/V3/world_model/`, including:
- `world_model_metrics.csv`
- `policy_eval_world_model.csv`
- `trajectory_examples.csv`
- `world_model_summary.txt`
- generated figures

## Ethical and Safety Considerations
- This project operates on modeled readmission risk, not real-time clinical judgment.
- The world model is a decision-support simulation, not a validated patient simulator.
- The action space is abstract and proxy-based, so it should not be mapped directly to clinical instructions.
- Offline evaluation is safer than direct online exploration, but that does not make the model clinically safe.
- Results should be interpreted as educational and research outputs, not operational guidance.

## Known Limitations
- The world model is tabular and intentionally simple.
- It inherits limitations from the underlying MDP and logged-data structure.
- Clinical actions are proxies rather than detailed validated treatment plans.
- Transition dynamics are modeled rather than causally identified.
- External validation has not been performed.
- A stronger representation model or richer longitudinal data could change conclusions.
- The current Version 3 system evaluates policies in simulation only; it does not establish clinical effectiveness.

## Reproducibility Commands
From the repository root:

```powershell
python scripts/run_world_model_v3.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/V3/world_model
python scripts/make_world_model_figures.py
python -m pytest tests/test_world_model.py
```

These commands run the Version 3 world-model pipeline, generate interpretation figures, and execute the lightweight test suite.
