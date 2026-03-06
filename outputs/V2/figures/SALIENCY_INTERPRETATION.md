# V2 Saliency Interpretation

## What these maps show
- `saliency_state_action_heatmap.png`: how strongly each action is favored/disfavored by Q-values in each risk state.
- `saliency_feature_outcome_heatmap.png`: which patient features most increase modeled readmission risk lift (`<30`) or protective lift (`NO`).

## Top feature signals (by weighted lift)
- number_inpatient: total=0.0477, risk_lift=0.0146, protective_lift=0.0330
- diag_3: total=0.0465, risk_lift=0.0143, protective_lift=0.0322
- diag_2: total=0.0463, risk_lift=0.0131, protective_lift=0.0332
- number_diagnoses: total=0.0306, risk_lift=0.0069, protective_lift=0.0237
- discharge_disposition_id: total=0.0264, risk_lift=0.0102, protective_lift=0.0162
