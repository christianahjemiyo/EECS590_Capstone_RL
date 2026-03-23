# Technical Challenges

- The V2 environment is naturally tabular, so some deep RL and continuous-control methods had to be adapted to a discrete action setting instead of being used in their ideal native setup.
- Getting one benchmark pipeline to cover classical RL, deep RL, actor-critic, and adapted advanced methods required keeping evaluation consistent even when training dynamics were very different.
- Some algorithms converged to very similar policies, which made it harder to tell whether differences came from the algorithm itself or from the reward structure of the environment.
- Saliency in this environment is less visually obvious than in image-based tasks, so the interpretation utilities had to focus on action preference and feature impact rather than only pixel-style attention maps.
- Replay storage needed to stay lightweight because committing very large raw experience files would quickly make the repository hard to manage.
- The reward design strongly affects rankings. In some runs, conservative policies looked better simply because intervention costs dominated modeled benefits.
- Checkpoint organization was not part of the earlier workflow, so a separate V2 checkpoint structure had to be added after the core experiments were already in place.
- The forward-view versus backward-view distinction in classical RL needed clearer documentation than the original code layout provided.
