/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Scale.BackwardScale
public import NN.Proofs.RuntimeApprox.Scale.ForwardScale
public import NN.Proofs.RuntimeApprox.Scale.ScaleApprox

/-!
# Runtime Approximation Scale Bounds

Optional scale propagation for runtime-approximation proofs.

Plain approximation lemmas track absolute error budgets. This layer additionally tracks
nonnegative bounds on tensor magnitudes, usually `‖x‖∞ ≤ B`, so absolute error statements can be
reported as more readable absolute/relative tolerances.
-/

@[expose] public section

