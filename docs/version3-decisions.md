# Version 3 Decisions

## 1. What I implemented
For Version 3 of this capstone, I implemented a world-model extension for my hospital readmission reinforcement learning project. Earlier versions built a reproducible clinical MDP and compared several RL algorithms for sequential discharge and follow-up planning. Version 3 extends that work by learning or simulating patient transition dynamics inside the tabular clinical RL setup I already built in Versions 1 and 2. In practice, this means I now have a learned simulator that can be used to evaluate policies inside a modeled environment instead of only relying on the fixed benchmark MDP outputs.

The Version 3 code adds a tabular world model, a simulator wrapper around that world model, evaluation utilities, a runnable script, figure generation, and lightweight tests. I kept the implementation simple and interpretable so it fits the scope of the capstone and stays aligned with the rest of the repository.

I also added a lightweight Bayesian hyperparameter tuning workflow for reward and action-cost design. I included that because it is one of the most directly applicable post-Version-2 techniques for this capstone. Reward design materially affects policy behavior in the modeled readmission setting, so Bayesian tuning is a much better fit here than forcing in algorithms that do not match the environment structure.

## 2. Why I chose world models
I chose world models because healthcare reinforcement learning cannot safely explore directly on patients. In my project, the underlying problem is sequential: discharge planning, follow-up intensity, and care-management choices interact over time. My existing work already models this as a tabular clinical MDP and compares multiple policies, but one of the biggest limitations is that patient transition dynamics and action effects are still modeled from limited logged data and proxy structure.

A world model directly addresses that gap. Instead of treating the transition structure as a fixed object in the background, it makes the patient progression model itself part of the Version 3 contribution. That is useful because it moves the project closer to safer offline policy evaluation under learned dynamics rather than only under a hand-specified benchmark structure.

World models fit this project because the core challenge is not simply choosing another RL algorithm. The main challenge is understanding how patient states may evolve under different discharge or follow-up strategies. A learned simulator directly addresses this gap by making the transition assumptions visible, testable, and reusable.

## 3. Main research question
My Version 3 research question is:

Can a learned world model approximate patient recovery and readmission dynamics well enough to support safer offline evaluation of discharge and follow-up policies?

I may refine the wording slightly in the final paper, but this is the main idea I wanted Version 3 to answer.

## 4. Gap addressed
A major gap in this space is that standard readmission models usually predict risk at one time point, but they do not evaluate sequences of care decisions. My V1 and V2 RL work addressed that by modeling sequential decision-making, but those versions still relied on a fixed MDP and proxy transition structure.

Version 3 adds a learned simulation layer so that policy behavior can be studied under modeled patient dynamics rather than only under a static benchmark formulation. I see this as an important conceptual bridge between simple prediction and sequential decision-support evaluation.

## 5. What I chose not to implement

### Safe hierarchical RL
I considered safe hierarchical RL, but I chose not to implement it in Version 3.

Safe hierarchical RL is useful when the action space naturally decomposes into high-level goals and lower-level sub-actions. In my current project, the actions are already compact abstractions such as intervention intensity or care-management strategy. The available data do not reliably support detailed lower-level clinical sub-actions. Adding hierarchy now would risk creating artificial structure that is not supported by the dataset.

For this capstone, I decided that the more important gap is uncertainty in patient transitions and safe offline simulation. That is why I prioritized world modeling instead of hierarchy.

### Foundation models
I also considered foundation-model-inspired representations. I did not implement a foundation model because training one would be beyond the scope of this capstone in both time and complexity. I still think this is an interesting future direction, especially for richer patient state representations, but it was not the right choice for a lightweight, defensible Version 3 extension.

## 6. Why this project is useful
I think this project is useful for a few practical reasons.

- It gives students and researchers a reproducible example of applying RL to a healthcare decision problem.
- It shows how policies can be compared before any real-world deployment is even considered.
- It documents limitations openly instead of hiding them behind performance numbers.
- It treats RL as decision support, not as automatic clinical instruction.

That last point matters to me. I do not want the project to imply that an RL model should replace clinical judgment. The right framing is that this is a structured way to study sequential decisions under modeled uncertainty.

## 7. Extra credit / advanced content connection
Version 3 connects to post-Version-2 class content because it applies world models to a clinical RL setting rather than stopping at policy comparison on a fixed benchmark. That makes the project more advanced than a standard tabular RL comparison because it asks whether learned dynamics can support safer offline evaluation.

I also considered safe hierarchical RL as another advanced direction, but I intentionally did not implement it because I do not think algorithm choice should be separated from environment structure. In this project, the environment is still compact and abstract, so world modeling was a better fit than forcing hierarchy into the action space.

## 8. Limitations
Version 3 still has important limitations.

- The learned world model is only as good as the MDP structure and logged or simulated data it is fit from.
- The clinical actions in this project are still proxies rather than real detailed treatment actions.
- The results should not be interpreted as medical advice.
- External validation would still be needed before making stronger claims.

So while Version 3 improves the project by adding learned simulation, it does not remove the underlying data and modeling limitations. I want that to stay explicit.

## 9. How to run
From the repository root, I can run Version 3 with:

```powershell
python scripts/run_world_model_v3.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/V3/world_model
python scripts/make_world_model_figures.py
python scripts/run_bayesian_reward_tuning.py --base-config v2_pipeline/configs/mdp_sim_mimic.json --outdir outputs/V3/bayesian_tuning
python -m pytest tests/test_world_model.py
```

These commands generate the Version 3 world-model outputs, create interpretation figures, and run the lightweight tests.
