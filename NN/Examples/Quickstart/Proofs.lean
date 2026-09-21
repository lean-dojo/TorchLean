/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tactic.Autograd
public import NN.Tactic.Converges
public import NN.Tensor

/-!
# Quickstart: Proving Small TorchLean Facts

Many TorchLean guarantees are ordinary Lean theorems. These examples cover:

- compile-time guarantees from shape-indexed tensor types, and
- ordinary mathematical lemmas about the public API, and
- derivative formulas checked against mathlib with `autograd`, and
- convergence of a quadratic's gradient-descent iteration with `converges`.

The deeper proof libraries live under `NN.Proofs.*`, `NN.Verification.*`, and `NN.MLTheory.*`.
-/

@[expose] public section

namespace NN.Examples.Quickstart.Proofs

open TorchLean

/--
A tensor's shape is part of its type.

If this definition compiles, Lean has already checked that the literal has exactly two entries and
therefore has type `Tensor Float [2]`. The commented shape mismatch below is the kind of bug Lean
catches before runtime:

```lean
-- def badTensor : Tensor Float [3] := [1.0, 2.0]
```
-/
def twoTensor : Tensor Float [2] :=
  [1.0, 2.0]

/-- ReLU fixes every nonnegative real number. -/
theorem relu_eq_self_of_nonnegative (x : ℝ) (hx : 0 ≤ x) :
    Activation.Math.reluSpec x = x := by
  simpa only [Activation.Math.reluSpec_eq_max] using max_eq_left hx

/-- ReLU clamps nonpositive real inputs to zero. -/
theorem relu_eq_zero_of_nonpositive (x : ℝ) (hx : x ≤ 0) :
    Activation.Math.reluSpec x = 0 := by
  simpa only [Activation.Math.reluSpec_eq_max] using max_eq_right hx

example : Activation.Math.reluSpec (3 : ℝ) = 3 := by
  exact relu_eq_self_of_nonnegative 3 (by norm_num)

example : Activation.Math.reluSpec (-2 : ℝ) = 0 := by
  exact relu_eq_zero_of_nonpositive (-2) (by norm_num)

/-- Differentiate one coordinate of mean squared error, holding the other terms fixed. -/
theorem mean_square_deriv (a b c : ℝ) :
    HasDerivAt (fun t : ℝ => (t ^ 2 + b + c) / 3) (2 * a / 3) a := by
  autograd

/-- The weight gradient is twice the residual times the corresponding input feature. -/
theorem affine_loss_deriv (w₁ w₂ b t x₁ x₂ : ℝ) :
    HasDerivAt (fun w : ℝ => (w * x₁ + w₂ * x₂ + b - t) ^ 2)
      (2 * (w₁ * x₁ + w₂ * x₂ + b - t) * x₁) w₁ := by
  autograd

/-! A new function can reuse the same rules. Registration lets later proofs use its derivative
without unfolding it. `local` keeps this tutorial's rule out of other modules' search sets. -/

/-- A smooth penalty combining a quadratic term with an exponential. -/
noncomputable def smoothPenalty (x : ℝ) : ℝ := x * x + Real.exp x

/-- Prove the new function's rule before registering it for composition. -/
@[local autograd] theorem smoothPenalty_deriv (x : ℝ) :
    HasDerivAt smoothPenalty (2 * x + Real.exp x) x := by
  unfold smoothPenalty
  autograd

example (x : ℝ) : HasDerivAt (fun y => Real.exp (smoothPenalty y))
    ((2 * x + Real.exp x) * Real.exp (smoothPenalty x)) x := by
  autograd

-- Domain assumptions remain part of the theorem, even when a tactic finds the proof.
example (x : ℝ) (hx : x ≠ 0) : HasDerivAt Real.log (1 / x) x := by
  autograd

/-! For the loss `x² / 2`, the gradient is `x`. Proving that derivative is one task;
proving that repeated gradient steps reach zero is another. Here the step size must lie in `(0, 2)`.
These are exact-real statements, not claims about rounded runtime iterates. -/

example (x : ℝ) : HasDerivAt (fun y : ℝ => y ^ 2 / 2) x x := by
  autograd

example (η x : ℝ) (hη : 0 < η) (hη2 : η < 2) :
    Filter.Tendsto (fun n : ℕ => (Optim.GD.step η id)^[n] x) Filter.atTop (nhds 0) := by
  have hmono : Optim.GD.StrongMonotone 1 (id : ℝ → ℝ) := by
    intro a b
    simp
  have hlip : LipschitzWith 1 (id : ℝ → ℝ) := LipschitzWith.id
  have hroot : (id : ℝ → ℝ) 0 = 0 := rfl
  have hstep : η * (1 : ℝ) ^ 2 < 2 * 1 := by simpa using hη2
  converges

end NN.Examples.Quickstart.Proofs
