# Spec Dynamics

This folder defines a small spec level interface for discrete time dynamical systems: a system is a
pure transition function `step : SpecTensor s -> SpecTensor s`, plus basic semantics like
iteration/trajectories and named stability-style properties.

The intent is to keep definitions close to the spec layer, where they are straightforward to reuse across
models, while keeping proofs of global dynamics in `NN/MLTheory/*`. Hopfield networks are the motivating
example: convergence claims are about repeated application of an update rule, not about any
particular runtime implementation.

Files:

- `system.lean`: core interface (`DynamicalSystem`, `DrivenSystem`), iteration semantics, and
  stability style predicates wired to `NN.MLTheory.Robustness.Spec` and `NN.MLTheory.Stability.Spec`.
