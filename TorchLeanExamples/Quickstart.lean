/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.Autograd
public import TorchLean.NN.Tactic.Converges
public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Operations

/-!
# Quickstart: Proving Small TorchLean Facts

Many TorchLean guarantees are ordinary Lean theorems. These examples cover:

- compile-time guarantees from shape-indexed tensor types, and
- ordinary mathematical lemmas about the public API, and
- derivative formulas checked against mathlib with `autograd`.

The deeper proofs live next to the definitions they are about, in the `TorchLean.Tensor.*`,
`TorchLean.NN.*`, and `TorchLean.Verify.*` modules.
-/

@[expose] public section

namespace TorchLean.Examples.Quickstart

/-- A tensor's shape is part of its type. Lean checks that this literal has exactly two entries,
and elaboration rejects the same literal at type `Tensor Float [3]` before anything runs. -/
example : Tensor Float [2] :=
  [1.0, 2.0]

/-- ReLU fixes every nonnegative real number. -/
theorem relu_of_nonneg (x : ℝ) (hx : 0 ≤ x) :
    Activation.relu x = x := by
  simpa only [Activation.relu_eq_max] using max_eq_left hx

/-- ReLU clamps nonpositive real inputs to zero. -/
theorem relu_of_nonpos (x : ℝ) (hx : x ≤ 0) :
    Activation.relu x = 0 := by
  simpa only [Activation.relu_eq_max] using max_eq_right hx

/-- Differentiate one coordinate of mean squared error, holding the other terms fixed. -/
theorem hasDerivAt_mean_square (a b c : ℝ) :
    HasDerivAt (fun t : ℝ => (t ^ 2 + b + c) / 3) (2 * a / 3) a := by
  autograd

/-- The weight gradient is twice the residual times the corresponding input feature. -/
theorem hasDerivAt_affine_loss (w₁ w₂ b t x₁ x₂ : ℝ) :
    HasDerivAt (fun w : ℝ => (w * x₁ + w₂ * x₂ + b - t) ^ 2)
      (2 * (w₁ * x₁ + w₂ * x₂ + b - t) * x₁) w₁ := by
  autograd

/-! A new function can reuse the same rules. Registration lets later proofs use its derivative
without unfolding it. `local` keeps this tutorial's rule out of other modules' search sets. -/

/-- A smooth penalty combining a quadratic term with an exponential. -/
noncomputable def smoothPenalty (x : ℝ) : ℝ := x * x + Real.exp x

/-- Prove the new function's rule before registering it for composition. -/
@[local autograd] theorem hasDerivAt_smoothPenalty (x : ℝ) :
    HasDerivAt smoothPenalty (2 * x + Real.exp x) x := by
  unfold smoothPenalty
  autograd

end TorchLean.Examples.Quickstart
