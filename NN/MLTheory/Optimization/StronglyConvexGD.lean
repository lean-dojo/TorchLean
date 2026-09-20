/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.GDLinearConvergence


/-!
# Gradient Descent Linear Convergence (Operator Form)

This file contains reusable gradient-descent convergence theorems.

The main theorems are stated for an operator $g:E\to E$ on a real inner product space. This is the
right abstraction boundary for TorchLean:

If $g$ is

* **$\mu$-strongly monotone** (a.k.a. $\mu$-strongly accretive), and
* **L-Lipschitz**,

then the fixed-point iteration

$$
x_{k+1}=x_k-\eta g(x_k)
$$

contracts distances to any root $x^\star$ of $g$, i.e. a point with $g(x^\star)=0$.

For gradients, the usual instantiation is $g=\nabla f$. When $f$ is $\mu$-strongly convex and
$L$-smooth, $\nabla f$ is $\mu$-strongly monotone and $L$-Lipschitz. Of these two facts,
`SmoothStrongConvexBridge` proves the strong-monotonicity half from first-order strong convexity;
the Lipschitz half is taken as a hypothesis there. This file focuses on the convergence argument
itself, keeping the assumptions minimal and reusable. The step-size lemmas at the end of the `GD`
namespace show when the contraction factor `q` lies in `[0, 1)`.

The final `ScalarGD` namespace keeps the one-dimensional quadratic facts as a compact reference
case: they show the same contraction mechanism in the smallest possible setting and connect plain
SGD, L2 regularization, and decoupled weight decay algebraically.
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

/-- One-step contraction of the squared distance to a root `xStar` of `g`. -/
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

If $q(\eta,\mu,L)\geq 0$, then after $k$ steps we have

$$
\left\lVert \operatorname{step}_\eta(g)^{\,k}(x)-x^\star\right\rVert^2
  \leq q(\eta,\mu,L)^k\lVert x-x^\star\rVert^2.
$$
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
Linear convergence: the squared distance to a root of `g` decays like `q ^ k`.

This is `dist_sq_iterate_le_of_q_nonneg` restated for the regime `0 ≤ q < 1`. The extra hypothesis
`q < 1` is what makes the right-hand side shrink geometrically in `k`; it is not used by the proof,
which is the same iterated contraction. Use `q_lt_one_of_mul_sq_lt` and `q_nonneg_of_le` to
discharge the two hypotheses on `q` from a step-size condition.
-/
theorem dist_sq_iterate_le_of_q_lt_one (η μ : ℝ) (hη : 0 ≤ η) {L : NNReal} (g : E → E)
    (hmono : StrongMonotone (E := E) μ g) (hlip : LipschitzWith L g)
    {xStar x : E} (hxStar : g xStar = 0) (hq : 0 ≤ q η μ L) (_hq1 : q η μ L < 1) (k : Nat) :
    ‖(step η g)^[k] x - xStar‖ ^ 2 ≤ (q η μ L) ^ k * ‖x - xStar‖ ^ 2 :=
  dist_sq_iterate_le_of_q_nonneg (E := E) (η := η) (μ := μ) (hη := hη) (L := L) (g := g)
    hmono hlip (xStar := xStar) (x := x) hxStar hq k

/--
The contraction factor is strictly below one when `0 < η` and `η * L ^ 2 < 2 * μ`.

Since `q - 1 = η * (η * L ^ 2 - 2 * μ)`, this is exactly the condition for `q < 1` once `η > 0`.
For `L > 0` it reads `η < 2 * μ / L ^ 2`; see `q_lt_one_of_lt_div`.
-/
theorem q_lt_one_of_mul_sq_lt (η μ : ℝ) (L : NNReal) (hη : 0 < η)
    (hstep : η * (L : ℝ) ^ 2 < 2 * μ) :
    q η μ L < 1 := by
  have hneg : η * (η * (L : ℝ) ^ 2 - 2 * μ) < 0 :=
    mul_neg_of_pos_of_neg hη (by linarith)
  have hq : q η μ L = 1 + η * (η * (L : ℝ) ^ 2 - 2 * μ) := by
    simp only [q]
    ring
  linarith

/--
Step-size form of `q_lt_one_of_mul_sq_lt`: for `L > 0`, any `η` with `0 < η < 2 * μ / L ^ 2`
gives `q η μ L < 1`.
-/
theorem q_lt_one_of_lt_div (η μ : ℝ) (L : NNReal) (hL : 0 < (L : ℝ)) (hη : 0 < η)
    (hstep : η < 2 * μ / (L : ℝ) ^ 2) :
    q η μ L < 1 := by
  apply q_lt_one_of_mul_sq_lt η μ L hη
  have hL2 : 0 < (L : ℝ) ^ 2 := by positivity
  exact (lt_div_iff₀ hL2).mp hstep

/--
The contraction factor is nonnegative whenever `0 ≤ μ ≤ L`.

This follows from the identity `q = (1 - η * μ) ^ 2 + η ^ 2 * (L ^ 2 - μ ^ 2)`, in which both
summands are nonnegative. No sign condition on `η` is needed.
-/
theorem q_nonneg_of_le (η μ : ℝ) (L : NNReal) (hμ : 0 ≤ μ) (hμL : μ ≤ (L : ℝ)) :
    0 ≤ q η μ L := by
  have hq : q η μ L = (1 - η * μ) ^ 2 + η ^ 2 * ((L : ℝ) ^ 2 - μ ^ 2) := by
    simp only [q]
    ring
  have hsq : μ ^ 2 ≤ (L : ℝ) ^ 2 := pow_le_pow_left₀ hμ hμL 2
  have h2 : 0 ≤ η ^ 2 * ((L : ℝ) ^ 2 - μ ^ 2) := mul_nonneg (sq_nonneg η) (by linarith)
  rw [hq]
  exact add_nonneg (sq_nonneg _) h2

/--
A strongly monotone and Lipschitz operator on a space with two distinct points has `μ ≤ L`.

This is the usual observation that the strong-monotonicity constant can never exceed the Lipschitz
constant; it lets `q_nonneg_of_le` be applied without assuming `μ ≤ L` separately.
-/
theorem StrongMonotone.le_lipschitz {μ : ℝ} {L : NNReal} {g : E → E}
    (hmono : StrongMonotone (E := E) μ g) (hlip : LipschitzWith L g)
    {x y : E} (hxy : x ≠ y) :
    μ ≤ (L : ℝ) := by
  have hpos : 0 < ‖x - y‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hxy)
  have h1 : μ * ‖x - y‖ ^ 2 ≤ ⟪x - y, g x - g y⟫ := hmono x y
  have h2 : ⟪x - y, g x - g y⟫ ≤ ‖x - y‖ * ‖g x - g y‖ := real_inner_le_norm _ _
  have h3 : ‖g x - g y‖ ≤ (L : ℝ) * ‖x - y‖ := by
    simpa [dist_eq_norm] using hlip.dist_le_mul x y
  have h4 : μ * ‖x - y‖ ^ 2 ≤ (L : ℝ) * ‖x - y‖ ^ 2 := by
    have h5 : ‖x - y‖ * ‖g x - g y‖ ≤ ‖x - y‖ * ((L : ℝ) * ‖x - y‖) :=
      mul_le_mul_of_nonneg_left h3 (norm_nonneg _)
    have h6 : ‖x - y‖ * ((L : ℝ) * ‖x - y‖) = (L : ℝ) * ‖x - y‖ ^ 2 := by ring
    linarith
  have hsq : 0 < ‖x - y‖ ^ 2 := by positivity
  exact le_of_mul_le_mul_right h4 hsq

/--
Linear convergence of gradient descent under an explicit step-size condition.

Assuming `0 ≤ μ ≤ L`, `0 < η`, and `η * L ^ 2 < 2 * μ`, the contraction factor satisfies
`0 ≤ q η μ L < 1` and the iterates converge linearly to any root `xStar` of `g`.
-/
theorem dist_sq_iterate_le_of_step_size (η μ : ℝ) {L : NNReal} (g : E → E)
    (hmono : StrongMonotone (E := E) μ g) (hlip : LipschitzWith L g)
    {xStar x : E} (hxStar : g xStar = 0)
    (hμ : 0 ≤ μ) (hμL : μ ≤ (L : ℝ)) (hη : 0 < η) (hstep : η * (L : ℝ) ^ 2 < 2 * μ) (k : Nat) :
    ‖(step η g)^[k] x - xStar‖ ^ 2 ≤ (q η μ L) ^ k * ‖x - xStar‖ ^ 2 ∧ q η μ L < 1 :=
  ⟨dist_sq_iterate_le_of_q_nonneg (E := E) (η := η) (μ := μ) (hη := le_of_lt hη) (L := L)
      (g := g) hmono hlip (xStar := xStar) (x := x) hxStar (q_nonneg_of_le η μ L hμ hμL) k,
    q_lt_one_of_mul_sq_lt η μ L hη hstep⟩

end GD

namespace ScalarGD

/-!
## Scalar quadratic warm-up

These facts are compact but not merely definitional. They prove algebraic behavior of
gradient descent on the one-dimensional quadratic objective

$$
L(x)=\frac12(x-\mathrm{target})^2,
$$

whose gradient is $x-\mathrm{target}$. This is the simplest executable bridge from TorchLean's
optimizer
equations to familiar convergence facts; the Hilbert-space operator theorem above is the reusable
version for tensor/vector models.
-/

/-- Gradient of $\frac12(x-\mathrm{target})^2$. -/
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
One scalar quadratic gradient-descent step multiplies the current error by $1-\mathrm{lr}$.

For ordered fields, this is the usual starting point for contraction proofs when
$0<\mathrm{lr}<2$.
-/
theorem error_after_step {α : Type} [CommRing α] (lr target x : α) :
    step lr target x - target = (1 - lr) * (x - target) := by
  unfold step quadraticGrad
  ring

/-- One SGD step with the L2 regularizer $\frac{\lambda}{2}x^2$ adds $\lambda x$ to the gradient. -/
def stepL2 {α : Type} [Sub α] [Mul α] [Add α] (lr lambda grad x : α) : α :=
  x - lr * (grad + lambda * x)

/--
For plain SGD, L2 regularization and decoupled weight decay coincide at the update level.

This scalar statement is the common fact behind the regularization note: adding $\lambda x$ to the
gradient produces the same update as multiplying parameters by $1-\mathrm{lr}\lambda$ and then
taking the plain gradient step. Adaptive optimizers need separate treatment; AdamW is checked in
`Optimization.FirstOrder`.
-/
theorem stepL2_eq_decoupledWeightDecay {α : Type} [CommRing α]
    (lr lambda grad x : α) :
    stepL2 lr lambda grad x = (1 - lr * lambda) * x - lr * grad := by
  unfold stepL2
  ring

/--
On the one-dimensional quadratic, if $0<\mathrm{lr}<2$, then one GD step contracts the error in
absolute value.

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
