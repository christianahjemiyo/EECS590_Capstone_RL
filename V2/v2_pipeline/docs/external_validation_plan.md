# External Validation Dataset Plan (On Hold)

This is a planning note only. External dataset integration is intentionally deferred.

## Recommended Side Dataset
- **eICU Collaborative Research Database (eICU-CRD)** via PhysioNet.

## Why eICU for Validation
- Multi-center ICU data (different hospitals) gives stronger external validity than single-system training alone.
- Similar critical-care context and event structure to MIMIC makes feature mapping practical.
- Supports readmission-oriented and outcomes-oriented robustness checks.

## Minimal Validation Cohort Definition
- Adult patients (`age >= 18`).
- Keep one ICU stay per hospital admission in the validation table.
- Build `readmitted` labels with same classes used in training:
  - `<30`: next admission within 30 days from discharge.
  - `>30`: next admission after 30 days.
  - `NO`: no observed subsequent admission.

## Schema Alignment (to current pipeline)
Create these columns to match the existing training code:
- `encounter_id`
- `patient_nbr`
- `readmitted`
- `time_in_hospital`
- `num_lab_procedures`
- `num_medications`
- `num_procedures`
- `number_inpatient`
- `number_emergency`
- `number_outpatient`
- `age`
- Optional categorical context columns (`gender`, `insurance`, etc.)

## Evaluation Protocol
- Train on MIMIC-IV `train.csv`.
- Tune hyperparameters on MIMIC-IV `val.csv`.
- Final report:
  - In-domain test (MIMIC-IV `test.csv`)
  - External validation test (eICU-aligned table)
- Compare policy return and readmission rate deltas between in-domain and external sets.

## Risk Controls
- Keep label and feature definitions identical across datasets.
- Do not leak future admissions into current-state features.
- Report missingness and drift by feature before evaluation.
