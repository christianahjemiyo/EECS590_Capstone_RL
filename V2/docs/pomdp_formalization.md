# POMDP and Belief Formalization

This note records the Version 2 extension path toward partially observed reinforcement learning.

## POMDP definition

A Partially Observable Markov Decision Process is defined by:

- states `S`
- actions `A`
- observations `O`
- transition model `P(s' | s, a)`
- observation model `P(o | s')`
- reward model `R(s, a, s')`
- discount factor `gamma`

In a POMDP, the agent does not directly observe the true state. Instead, it receives observations and maintains a belief distribution over states.

## Belief update

If the current belief is `b(s)`, the action taken is `a`, and the new observation is `o`, then the updated belief is:

`b'(s') propto P(o | s') * sum_s P(s' | s, a) b(s)`

After this proportional update, the belief is normalized so that it sums to 1.

## Why this matters for V2

- Clinical environments are often partially observed.
- Patient state is rarely known with certainty.
- A belief-state formulation is a natural next step when extending V2 beyond a compact tabular MDP.

## Temporal neural extensions

Neural architectures that can support temporal or partial-observability extensions include:

- recurrent actor-critic models
- recurrent Q-networks
- belief-conditioned policies
- sequence encoders over short patient histories

These extensions are not required for the base V2 benchmark, but they are natural preparation for Version 3 work.
