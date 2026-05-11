/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Rounding.RoundingApprox

/-!
# Runtime Rounding Approximation

Scalar approximation lemmas for proof-relevant rounded arithmetic.

This layer reasons about a rounding model such as `neural_round`: one scalar operation is replaced
by a rounded scalar operation, and the proof records the resulting `ulp`-style error budget. Tensor
and graph modules lift these scalar facts to operators and end-to-end executions.

The public vocabulary is intentionally small:
- `scalarApprox` is the absolute-error predicate for one real scalar;
- `roundR` is one modeled rounding step;
- `roundedAdd` and `roundedMul` are the rounded scalar operations used by NF semantics;
- `scalarApprox_roundedAdd` and `scalarApprox_roundedMul` are the compositional error rules.
-/

@[expose] public section
