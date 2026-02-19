# State and Action Mapping (Data-Driven Environment)

This document defines the **first-pass** mapping for the data-driven RL environment. It is a working
spec that will evolve as we learn from the dataset and refine assumptions.

## Goal
Create a compact state representation that approximates post-discharge risk and recovery status,
and define intervention actions that could be optimized in a sequential decision setting.

## State Representation (Initial)
We map each encounter into a discrete risk state using a **risk score** computed from a small set of
high-signal features. The risk score is then binned into 4 discrete states (by quantiles).

Candidate features (from the Kaggle dataset, used in the simulator):
- `time_in_hospital`
- `num_lab_procedures`
- `num_medications`
- `num_procedures`
- `number_inpatient`
- `number_emergency`
- `number_outpatient`
- `age`

Initial risk bins (example):
- State 0: Low risk
- State 1: Medium risk
- State 2: High risk
- State 3: Very high risk (optional)

## Actions (Initial)
Actions represent intervention intensity post-discharge:
- Action 0: Conservative monitoring
- Action 1: Standard follow-up
- Action 2: Intensive follow-up + care coordination

These are **not recorded** in the dataset, so in the current environment they are modeled
as actions that shift transition probabilities toward lower-risk states (a simulator assumption).
This provides an interface for future causal modeling.

## Reward Mapping (Initial)
Reward is shaped primarily by readmission outcome:
- `<30` readmission: -10
- `>30` readmission: -2
- `NO` readmission: +2

Action cost (per step):
- Action 0: 0.0
- Action 1: 0.2
- Action 2: 0.5

## Next Refinements
- Replace factorized encoding with explicit clinical grouping.
- Learn or estimate transitions from longitudinal data if available.
- Calibrate rewards using health economics or length-of-stay proxies.
