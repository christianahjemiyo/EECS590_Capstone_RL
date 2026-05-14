# V2 Neural Saliency Interpretation

## What these figures show
- `nn_saliency_dqn_input_heatmap.png`: sensitivity of the selected DQN action value to each input state dimension under a one-hot state encoding.
- `nn_saliency_dqn_hidden_heatmap.png`: hidden-unit contributions to the selected action value in each state.

## How to read them
- Larger absolute values indicate stronger influence on the DQN decision.
- Positive values increase the selected action value, while negative values suppress it.
- In this compact environment, the saliency is best interpreted as state-to-state sensitivity rather than image-style attention.

## Example state summaries
- State S0 action A0: strongest input influences -> S3 (-6.017), S2 (-5.518), S1 (-4.195)
- State S1 action A0: strongest input influences -> S3 (-6.023), S2 (-5.525), S1 (-4.204)
- State S2 action A2: strongest input influences -> S3 (-5.840), S2 (-5.459), S1 (-4.363)
- State S3 action A2: strongest input influences -> S3 (-5.843), S2 (-5.466), S1 (-4.356)
