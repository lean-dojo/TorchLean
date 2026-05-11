/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.GDLinearConvergence

import Mathlib.Logic.Function.Iterate
import Mathlib.Algebra.Order.GroupWithZero.Unbundled.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Gradient Descent Linear Convergence (Operator Form)

This file contains reusable gradient-descent convergence theorems.

The main theorems are stated for an operator `g : E → E` on a real inner product space. This is the
right abstraction boundary for TorchLean:

If `g` is

* **μ-strongly monotone** (a.k.a. `μ`-strongly accretive), and
* **L-Lipschitz**,

then the fixed-point iteration

`x_{k+1} = x_k - η g(x_k)`

contracts distances to any root `x⋆` of `g`, i.e. a point with `g x⋆ = 0`.

For gradients, the usual instantiation is `g = ∇f`. When `f` is `μ`-strongly convex and `L`-smooth,
one can prove (in calculus/convex-analysis land) that `∇f` is `μ`-strongly monotone and `L`-Lipschitz.
This file focuses on the convergence argument itself, keeping the assumptions minimal and reusable.

The final `ScalarGD` namespace keeps the one-dimensional quadratic facts as a compact sanity check:
they show the same contraction mechanism in the smallest possible setting and connect plain SGD,
L2 regularization, and decoupled weight decay algebraically.
-/

@[expose] public section

namespace Optim
namespace GD

open Real
open scoped RealInnerProductSpace

variable {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The squared-distance contraction factor from `step_norm_sq_le`. -/
def q (η μ : ℝ) (L : NNReal) : ℝ :=
  1 - 2 * η * μ + (η ^ 2) * (L : ℝ) ^ 2

theorem step_dist_sq_le (η μ : ℝ) (hη : 0 ≤ η) {L : NNReal} (g : E → E)
    (hmono : StrongMonotone (E := E) μ g) (hlip : LipschitzWith L g)
    {xStar x : E} (hxStar : g xStar = 0) :
    ‖step η g x - xStar‖ ^ 2 ≤ q η μ L * ‖x - xStar‖ ^ 2 := by
  -- Apply the two-point contraction to `(x, xStar)` and use `g xStar = 0`.
  simpa [q, step, hxStar] using
    (step_norm_sq_le (E := E) (η := η) (μ := μ) (hη := hη) (L := L) (g := g)
      hmono hlip x xStar)

/--
Iterated contraction bound in squared norm.

If `q η μ L ≥ 0`, then after `k` steps we have

`‖(step η g)^[k] x - x⋆‖² ≤ (q η μ L)^k * ‖x - x⋆‖²`.
-/
theorem dist_sq_iterate_le_of_q_nonneg (η μ : ℝ) (hη : 0 ≤ η) {L : NNReal} (g : E → E)
    (hmono : StrongMonotone (E := E) μ g) (hlip : LipschitzWith L g)
    {xStar x : E} (hxStar : g xStar = 0) (hq : 0 ≤ q η μ L) (k : Nat) :
    ‖(step η g)^[k] x - xStar‖ ^ 2 ≤ (q η μ L) ^ k * ‖x - xStar‖ ^ 2 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      have h1 :
          ‖(step η g)^[Nat.succ k] x - xStar‖ ^ 2
            ≤ q η μ L * ‖(step η g)^[k] x - xStar‖ ^ 2 := by
        -- Unfold one step of `^[k]` and apply `step_dist_sq_le`.
        simpa [Function.iterate_succ_apply'] using
          (step_dist_sq_le (E := E) (η := η) (μ := μ) (hη := hη) (L := L) (g := g)
            hmono hlip (x := (step η g)^[k] x) (xStar := xStar) hxStar)
      -- Multiply the induction hypothesis by `q` (using `hq`).
      have h2 :
          q η μ L * ‖(step η g)^[k] x - xStar‖ ^ 2
            ≤ q η μ L * ((q η μ L) ^ k * ‖x - xStar‖ ^ 2) := by
        exact mul_le_mul_of_nonneg_left ih hq
      -- Combine and rewrite powers.
      have := le_trans h1 h2
      -- `q * (q^k * a) = q^(k+1) * a`.
      simpa [pow_succ, mul_assoc, mul_left_comm, mul_comm, Function.iterate_succ_apply'] using this

/--
Linear convergence phrased as an exponentially decaying upper bound.

This is just `dist_sq_iterate_le_of_q_nonneg` plus the assumption `q < 1` which makes the RHS
actually shrink with `k`.
-/
theorem dist_sq_iterate_le_of_q_lt_one (η μ : ℝ) (hη : 0 ≤ η) {L : NNReal} (g : E → E)
    (hmono : StrongMonotone (E := E) μ g) (hlip : LipschitzWith L g)
    {xStar x : E} (hxStar : g xStar = 0) (hq : 0 ≤ q η μ L) (hq1 : q η μ L < 1) (k : Nat) :
    ‖(step η g)^[k] x - xStar‖ ^ 2 ≤ ‖x - xStar‖ ^ 2 := by
  -- We use the iterate bound and the fact that `0 ≤ q < 1` implies `q^k ≤ 1`.
  have hpow : (q η μ L) ^ k ≤ (1 : ℝ) := by
    have hqle : q η μ L ≤ 1 := le_of_lt hq1
    -- Use the `a ≤ 1` monotonicity lemma with `m = 0`, `n = k`.
    simpa using (pow_le_pow_of_le_one (a := q η μ L) hq hqle (m := 0) (n := k) (by simp))
  have hmain :=
    dist_sq_iterate_le_of_q_nonneg (E := E) (η := η) (μ := μ) (hη := hη) (L := L) (g := g)
      hmono hlip (xStar := xStar) (x := x) hxStar hq k
  -- Chain with `q^k ≤ 1`.
  have : (q η μ L) ^ k * ‖x - xStar‖ ^ 2 ≤ (1 : ℝ) * ‖x - xStar‖ ^ 2 := by
    exact mul_le_mul_of_nonneg_right hpow (sq_nonneg ‖x - xStar‖)
  have := le_trans hmain this
  simpa using this

end GD

namespace ScalarGD

/-!
## Scalar quadratic warm-up

These facts are deliberately small but not merely definitional. They prove algebraic behavior of
gradient descent on the one-dimensional quadratic objective

`L(x) = 1/2 * (x - target)^2`,

whose gradient is `x - target`. This is the simplest executable bridge from TorchLean's optimizer
equations to familiar convergence facts; the Hilbert-space operator theorem above is the reusable
version for tensor/vector models.
-/

/-- Gradient of `1/2 * (x - target)^2`. -/
def quadraticGrad {α : Type} [Sub α] (target x : α) : α :=
  x - target

/-- One scalar gradient-descent step on the quadratic objective. -/
def step {α : Type} [Sub α] [Mul α] (lr target x : α) : α :=
  x - lr * quadraticGrad target x

/-- The optimum is a fixed point of the scalar quadratic gradient-descent update. -/
theorem target_fixed {α : Type} [CommRing α] (lr target : α) :
    step lr target target = target := by
  unfold step quadraticGrad
  ring

/--
One scalar quadratic gradient-descent step multiplies the current error by `1 - lr`.

For ordered fields, this is the usual starting point for contraction proofs when `0 < lr < 2`.
-/
theorem error_after_step {α : Type} [CommRing α] (lr target x : α) :
    step lr target x - target = (1 - lr) * (x - target) := by
  unfold step quadraticGrad
  ring

/-- One step of SGD on a loss with L2 regularization term `λ/2 * x^2` adds `λ*x` to the gradient. -/
def stepL2 {α : Type} [Sub α] [Mul α] [Add α] (lr lambda grad x : α) : α :=
  x - lr * (grad + lambda * x)

/--
For plain SGD, L2 regularization and decoupled weight decay coincide at the update level.

This scalar statement is the common fact behind the regularization note: adding `λ x` to the
gradient produces the same update as multiplying parameters by `(1 - lr*λ)` and then taking the
plain gradient step. Adaptive optimizers need separate treatment; AdamW is checked in
`Optimization.FirstOrder`.
-/
theorem stepL2_eq_decoupledWeightDecay {α : Type} [CommRing α]
    (lr lambda grad x : α) :
    stepL2 lr lambda grad x = (1 - lr * lambda) * x - lr * grad := by
  unfold stepL2
  ring

/--
On the 1D quadratic, if `0 < lr < 2` then one GD step contracts the error in absolute value.

This is the scalar version of the operator-level contraction theorem above.
-/
theorem error_abs_contract_real (lr target x : ℝ) (h0 : 0 < lr) (h2 : lr < 2) (hne : x ≠ target) :
    |step lr target x - target| < |x - target| := by
  have herr : step lr target x - target = (1 - lr) * (x - target) :=
    error_after_step (α := ℝ) lr target x
  have habs : |1 - lr| < (1 : ℝ) := by
    have hlt1 : 1 - lr < 1 := by linarith
    have hgtm1 : -1 < 1 - lr := by linarith
    exact abs_lt.mpr ⟨hgtm1, hlt1⟩
  have hz : x - target ≠ 0 := by
    intro h
    apply hne
    linarith
  have hxpos : 0 < |x - target| := abs_pos.mpr hz
  calc
    |step lr target x - target|
        = |(1 - lr) * (x - target)| := by simp [herr]
    _ = |1 - lr| * |x - target| := by simp [abs_mul]
    _ < 1 * |x - target| := by
          have := (mul_lt_mul_of_pos_right habs hxpos)
          simpa [one_mul] using this
    _ = |x - target| := by simp

end ScalarGD
end Optim
