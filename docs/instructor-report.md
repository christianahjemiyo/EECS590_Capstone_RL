# EECS590 Capstone Report Addendum: Version 3 World-Model Extension

## Project Context
This capstone studies reinforcement learning for hospital readmission planning as a sequential decision problem. The broader goal of the project is to examine whether reinforcement learning can support modeled discharge and follow-up planning more effectively than one-time risk prediction alone.

Earlier versions of the project established the core benchmark structure:
- Version 1 built a reproducible reinforcement learning scaffold around a healthcare-inspired environment.
- Version 2 expanded that work into a tabular clinical MDP benchmark with broader algorithm coverage, offline RL comparisons, and interpretation outputs.

Version 3 extends that sequence by introducing a world-model layer for safer offline simulation and policy evaluation.

## Why This Problem Matters
Traditional readmission prediction models estimate risk at one point in time, but they do not optimize sequences of discharge and follow-up decisions. That matters because hospital care planning is not a single prediction task. It is a sequential decision problem in which follow-up intensity, monitoring, intervention timing, and care-management strategy may change patient progression over time.

My earlier RL work addressed that gap by modeling the problem as a sequential decision process. However, Versions 1 and 2 still relied on a fixed MDP and proxy transition structure. The remaining limitation was that the project needed a safer way to study transition dynamics and policy behavior without any real-world experimentation on patients.

## Version 3 Research Question
The main Version 3 research question is:

**Can a learned world model approximate patient recovery and readmission dynamics well enough to support safer offline evaluation of discharge and follow-up policies?**

I chose this question because it directly addresses the main unresolved issue from the earlier versions: not just which policy performs better in a fixed benchmark, but how a learned simulator can be used to study policy behavior under modeled patient progression.

## What I Implemented
For Version 3, I implemented a lightweight world-model extension that fits the current structure of the capstone rather than replacing it.

The implementation includes:
- a tabular world model that learns transition probabilities and expected rewards from sampled trajectories,
- a simulator wrapper around that world model,
- policy evaluation inside the learned simulator,
- error metrics comparing learned dynamics against the original tabular MDP,
- trajectory generation under multiple policies,
- interpretation figures and summary outputs,
- lightweight automated tests.

All Version 3 outputs are saved under:

`outputs/V3/world_model/`

This design keeps the project reproducible and interpretable while extending the Version 2 benchmark in a meaningful way.

## Why I Chose World Models
I chose world models because healthcare reinforcement learning cannot safely explore directly on patients. In a real clinical context, it would be unacceptable to test arbitrary exploratory policies on vulnerable patient populations just to learn transition behavior online.

World models fit this project because the core challenge is not simply choosing another RL algorithm. The more important challenge is understanding how patient states may evolve under different discharge or follow-up strategies. A learned simulator directly addresses this gap by making the transition assumptions visible, testable, and reusable.

In that sense, Version 3 adds a decision-support simulation layer to the capstone. It does not claim clinical validity or deployment readiness. Instead, it creates a structured way to study policy behavior under learned modeled dynamics before any real-world use would even be considered.

## Why I Did Not Implement Safe Hierarchical RL
An important Version 3 design decision was the choice to implement a world-model extension rather than safe hierarchical reinforcement learning.

I considered safe hierarchical RL because it is a useful advanced framework when a problem naturally separates into high-level goals and lower-level sub-actions. However, in this capstone, the current action space is still compact and abstract, with actions represented as intervention intensity or care-management strategy rather than richly documented clinical sub-decisions. The available data do not reliably support a defensible lower-level action hierarchy, so adding hierarchy at this stage would risk imposing artificial structure that is not supported by the dataset.

By contrast, the more central limitation in the project was uncertainty in the modeled patient transition dynamics and the need for safer offline policy evaluation. For that reason, I selected a world-model extension for Version 3. This choice better matches the current tabular clinical MDP, keeps the implementation interpretable and reproducible, and directly addresses the project’s main gap: how to study policy behavior under learned decision-support simulation without real-world experimentation on patients.

## Why I Did Not Use a Deep or Latent World Model
I also considered whether Version 3 should use a deep or latent world model. I decided not to do that in the final capstone because the current project is still built on a compact tabular clinical MDP with abstract actions.

A more complex latent dynamics model would add architectural complexity, training instability, and interpretability challenges without being clearly justified by the available data structure. At this stage, the main gap was not model size, but transition uncertainty and safer offline policy evaluation. A tabular world model let me address that gap directly in a way that is reproducible, interpretable, and consistent with the rest of the project.

I see richer latent or foundation-model-inspired world models as possible future work, especially if the project later uses stronger sequential patient representations and more detailed action logging.

## Practical Usefulness
This project is useful in several practical ways even though it is not clinically deployable.

- It gives students and researchers a reproducible example of applying RL to a healthcare decision problem.
- It shows how policies can be compared before any real-world deployment is considered.
- It documents limitations openly instead of hiding them.
- It treats reinforcement learning as decision support rather than automatic clinical instruction.

That last point is especially important. The project should not be interpreted as recommending that RL replace clinical judgment. The practical value of the capstone is that it provides a transparent research and teaching framework for studying sequential healthcare decision problems under modeled uncertainty.

## Ethical and Safety Position
The Version 3 world model is not a clinical deployment tool. It should only be interpreted as a decision-support simulation for a class capstone.

The model:
- operates on modeled readmission risk,
- uses proxy actions rather than validated detailed treatment plans,
- depends on a tabular MDP abstraction,
- requires external validation before any stronger claim could be made.

For those reasons, the project should be read as an educational and research-oriented system, not a medical decision engine.

## Limitations
Version 3 still has important limitations.

- The learned world model is only as good as the underlying MDP structure and logged or simulated data.
- Clinical actions in the project are still proxies rather than real detailed treatment pathways.
- The current world model is lightweight and tabular, which helps interpretability but limits expressiveness.
- Results should not be interpreted as medical advice.
- External validation would still be required before any real-world use could be discussed.

These limitations do not invalidate the capstone. Instead, they define the project honestly and keep the claims appropriately scoped.

## Advanced Content Connection
Version 3 connects to more advanced course content by going beyond fixed-policy comparison and using world models in a clinical RL setting. This is important because it shows that algorithm choice should match the environment and the real project gap.

I also considered safe hierarchical RL as an advanced extension, but I intentionally did not implement it because I do not think algorithm choice should be separated from the structure of the environment and the structure of the data. In this capstone, world modeling was the more appropriate advanced extension.

## Reproducibility
From the repository root, Version 3 can be reproduced with:

```powershell
python scripts/run_world_model_v3.py --mdp outputs/v2_outputs/mdp/mdp.npz --outdir outputs/V3/world_model
python scripts/make_world_model_figures.py
python -m pytest tests/test_world_model.py
```

These commands run the world-model workflow, generate the Version 3 figures, and execute the lightweight tests.

## Closing Position
As a final capstone version, I believe Version 3 is sufficient because it extends the project in a coherent and defensible way. It does not pretend to solve the full healthcare RL problem, but it does show a clear progression:
- from basic RL framing,
- to structured benchmarking,
- to safer offline decision-support simulation through a learned world model.

That progression is the main intellectual contribution of the final project.
