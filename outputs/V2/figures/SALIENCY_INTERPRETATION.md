# V2 Saliency Interpretation

## What these maps show
- `saliency_state_action_heatmap.png`: how strongly each action is favored/disfavored by Q-values in each risk state.
- `saliency_feature_outcome_heatmap.png`: which patient features most increase modeled readmission risk lift (`<30`) or protective lift (`NO`).

## Top feature signals (by weighted lift)
- number_inpatient: total=0.0923, risk_lift=0.0328, protective_lift=0.0595
- num_medications: total=0.0414, risk_lift=0.0180, protective_lift=0.0234
- insurance: total=0.0404, risk_lift=0.0111, protective_lift=0.0293
- number_diagnoses: total=0.0378, risk_lift=0.0151, protective_lift=0.0227
- discharge_location: total=0.0352, risk_lift=0.0148, protective_lift=0.0204
