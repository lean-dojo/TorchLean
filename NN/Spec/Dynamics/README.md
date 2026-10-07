# Spec Dynamics

This folder defines spec-level interfaces for discrete-time and state-space dynamical systems. The
definitions are intentionally close to the mathematical object: a state, an optional input, a pure
transition rule, and the trajectories obtained by iteration.

The runtime is not part of this layer. A dynamics spec can later be used by Hopfield-style energy
arguments, state-space model semantics, controller examples, Lyapunov predicates, or RL boundary
statements without committing to a CUDA kernel, simulator, or training loop.

## Files

- `System.lean`: `DynamicalSystem`, `DrivenSystem`, iteration semantics, trajectories, and
  stability predicates parameterized by a supplied size function. These predicates do not assume
  norm laws; relating them to metric stability requires those laws separately.
- `StateSpace.lean`: channelwise/state-space recurrence structures used by state-space and
  sequence-model specifications.

## How To Use It

Use this folder when the claim is about repeated application of a transition rule:

```text
state₀
  -> step state₀
  -> step (step state₀)
  -> trajectory n
```

Proofs of global behavior belong in `NN/MLTheory/*`. The spec folder names the objects; MLTheory
proves stability, convergence, boundedness, or robustness facts about those objects.
