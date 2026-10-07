/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.StronglyConvexGD
public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Mathlib.Analysis.Convex.Strong
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Strong Convexity and Strongly Monotone Gradients

TorchLean's GD convergence theorems are stated at the operator level: $g$ is $\mu$-strongly
monotone and $L$-Lipschitz. To apply them to gradient descent on an objective $f$, we instantiate
$g=\nabla f$.

Mathlib's `gradient f x` is total (it returns $0$ if the derivative does not exist), so we assume
differentiability and reason through the standard *first-order* characterization of strong
convexity.

Main results:

* `firstOrderStrongConvexAt_of_strongConvexOn_univ`: `StrongConvexOn univ μ f` plus
  differentiability at `x` gives the first-order inequality at `x`;
* `firstOrderStrongConvex_of_strongConvexOn_univ`: with differentiability everywhere, the global
  first-order premise `FirstOrderStrongConvex μ f`;
* `strongMonotone_gradient_of_firstOrderStrongConvex`: the first-order inequality makes `∇ f`
  $\mu$-strongly monotone, which is the hypothesis of `GD.step_norm_sq_le`;
* `dist_sq_iterate_le_of_firstOrderStrongConvex` and `dist_sq_iterate_le_of_strongConvexOn`:
  linear convergence of gradient descent to a critical point, assuming in addition that `∇ f` is
  Lipschitz.

The smoothness half is not proved here. Nothing in this file derives `LipschitzWith L (∇ f)` from
a second-order or `fderiv`-level smoothness hypothesis on `f`; the Lipschitz gradient is taken as
an explicit assumption.
-/

@[expose] public section

namespace Optim
namespace GD

open scoped RealInnerProductSpace Gradient

variable {E : Type} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/--
First-order strong convexity with parameter $\mu$, stated using the gradient.

This is the familiar inequality:

$$
f(y)\geq f(x)+\langle\nabla f(x),y-x\rangle
  +\frac{\mu}{2}\lVert y-x\rVert^2.
$$

It is a standard characterization of $\mu$-strong convexity for differentiable functions on
Euclidean/Hilbert spaces.
-/
def FirstOrderStrongConvex (μ : ℝ) (f : E → ℝ) : Prop :=
  ∀ x y, f y ≥ f x + ⟪(∇ f) x, y - x⟫ + (μ / 2) * ‖y - x‖ ^ 2

/--
First-order strong convexity at a fixed base point `x`.

This is the same inequality as `FirstOrderStrongConvex`, but quantified only over $y$.
It is the natural statement you get from convex analysis under a `DifferentiableAt` hypothesis at
$x$.
-/
def FirstOrderStrongConvexAt (μ : ℝ) (f : E → ℝ) (x : E) : Prop :=
  ∀ y, f y ≥ f x + ⟪(∇ f) x, y - x⟫ + (μ / 2) * ‖y - x‖ ^ 2

/--
`StrongConvexOn univ μ f` plus differentiability at $x$ implies the first-order strong convexity
inequality at $x$.

This is the standard convex-analysis “supporting hyperplane” argument applied to
$g(z)=f(z)-(\mu/2)\lVert z\rVert^2$, which is convex when $f$ is strongly convex.
-/
theorem firstOrderStrongConvexAt_of_strongConvexOn_univ (μ : ℝ) {f : E → ℝ}
    (hsc : StrongConvexOn (s := (Set.univ : Set E)) μ f) {x : E}
    (hdx : DifferentiableAt ℝ f x) :
    FirstOrderStrongConvexAt (E := E) μ f x := by
  intro y
  -- `g z = f z - (μ/2) ‖z‖²` is convex; restrict it to the segment from `x` to `y` and compare
  -- its derivative at `0` with the secant slope to `1`.
  let q : E → ℝ := fun z => (μ / 2) * ‖z‖ ^ 2
  have hq0 : DifferentiableAt ℝ (fun z : E => ‖z‖ ^ 2) x :=
    (hasStrictFDerivAt_norm_sq x).differentiableAt
  have hq : DifferentiableAt ℝ q x := hq0.const_mul (μ / 2)
  let g : E → ℝ := fun z => f z - q z
  have hg_conv : ConvexOn ℝ (Set.univ : Set E) g := by
    have := (strongConvexOn_iff_convex (s := (Set.univ : Set E)) (m := μ) (f := f)).1 hsc
    simpa [g, q] using this
  let h : ℝ →ᵃ[ℝ] E := (AffineMap.lineMap x y : ℝ →ᵃ[ℝ] E)
  have hφ_conv : ConvexOn ℝ (Set.univ : Set ℝ) (g ∘ h) := by
    simpa [Set.preimage_univ] using (hg_conv.comp_affineMap h)
  have hderiv_line : HasDerivAt (fun t => (g ∘ h) t) ((fderiv ℝ g x) (y - x)) (0 : ℝ) := by
    have hh : HasDerivAt (fun t => h t) (y - x) (0 : ℝ) := by
      simpa using (AffineMap.hasDerivAt_lineMap (a := x) (b := y) (x := (0 : ℝ)))
    have hgx : DifferentiableAt ℝ g x := by
      change DifferentiableAt ℝ (f - q) x
      exact hdx.sub hq
    exact HasFDerivAt.comp_hasDerivAt_of_eq (𝕜 := ℝ) (l := g)
      (l' := fderiv ℝ g x) (y := x) (f := fun t => h t)
      (f' := y - x) (x := (0 : ℝ)) hgx.hasFDerivAt hh (by simp [h])
  have hderiv0 : deriv (fun t => (g ∘ h) t) 0 = (fderiv ℝ g x) (y - x) := hderiv_line.deriv
  have hle_slope : deriv (fun t => (g ∘ h) t) 0 ≤ slope (fun t => (g ∘ h) t) 0 1 := by
    have hdiff0 : DifferentiableAt ℝ (fun t => (g ∘ h) t) (0 : ℝ) := hderiv_line.differentiableAt
    exact (hφ_conv.deriv_le_slope (x := (0 : ℝ)) (y := (1 : ℝ))
      (by simp) (by simp) (by linarith) hdiff0)
  have hslope : slope (fun t => (g ∘ h) t) 0 1 = (g y - g x) := by
    simp [slope_def_field, g, h, sub_eq_add_neg]
  have hgy : g y ≥ g x + (fderiv ℝ g x) (y - x) := by
    rw [hslope] at hle_slope
    linarith [hle_slope, hderiv0]
  -- The directional derivative of `g` at `x` is `⟪∇ f x, y - x⟫ - μ ⟪x, y - x⟫`.
  have hdir :
      (fderiv ℝ g x) (y - x) = ⟪(∇ f) x, y - x⟫ - μ * ⟪x, y - x⟫ := by
    change
      (fderiv ℝ (fun z => f z - (μ / 2) * ‖z‖ ^ 2) x) (y - x)
        = ⟪(∇ f) x, y - x⟫ - μ * ⟪x, y - x⟫
    have hf_apply : (fderiv ℝ f x) (y - x) = ⟪(∇ f) x, y - x⟫ :=
      (inner_gradient_left (f := f) (x := x) (y := y - x)).symm
    have hq_apply : (fderiv ℝ (fun z : E => (μ / 2) * ‖z‖ ^ 2) x) (y - x) = μ * ⟪x, y - x⟫ := by
      rw [fderiv_const_mul hq0 (μ / 2), fderiv_norm_sq_apply]
      simp
      ring
    have hsub :
        fderiv ℝ (fun z => f z - q z) x = fderiv ℝ f x - fderiv ℝ q x := by
      simpa [sub_eq_add_neg] using
        (fderiv_fun_sub (x := x) (f := f) (g := q) hdx hq)
    calc
      (fderiv ℝ (fun z => f z - (μ / 2) * ‖z‖ ^ 2) x) (y - x)
          = (fderiv ℝ (fun z => f z - q z) x) (y - x) := by
              simp [q]
      _ = (fderiv ℝ f x - fderiv ℝ q x) (y - x) := by
              rw [hsub]
      _ = (fderiv ℝ f x) (y - x) - (fderiv ℝ q x) (y - x) := by
              rfl
      _ = ⟪(∇ f) x, y - x⟫ - μ * ⟪x, y - x⟫ := by
              rw [hf_apply, hq_apply]
  -- `‖y‖² = ‖x‖² + 2⟪x, y - x⟫ + ‖y - x‖²` converts the quadratic terms of `g` back to `f`.
  have hnorm_sq :
      ‖y‖ ^ 2 = ‖x‖ ^ 2 + 2 * ⟪x, y - x⟫ + ‖y - x‖ ^ 2 := by
    have hy : y = x + (y - x) := by abel
    rw [hy]
    simpa [add_assoc, add_left_comm, add_comm, mul_assoc, mul_left_comm, mul_comm] using
      (norm_add_sq_real (x := x) (y := (y - x)))
  have : f y ≥ f x + ⟪(∇ f) x, y - x⟫ + (μ / 2) * ‖y - x‖ ^ 2 := by
    have hgy' : f y - (μ / 2) * ‖y‖ ^ 2 ≥ f x - (μ / 2) * ‖x‖ ^ 2 + (fderiv ℝ g x) (y - x) := by
      simpa [g, q, add_assoc, add_left_comm, add_comm, sub_eq_add_neg] using hgy
    have hgy'' :
        f y - (μ / 2) * ‖y‖ ^ 2
          ≥ f x - (μ / 2) * ‖x‖ ^ 2
            + (⟪(∇ f) x, y - x⟫ - μ * ⟪x, y - x⟫) := by
      simpa [hdir] using hgy'
    have hdiff :
        (μ / 2) * ‖y‖ ^ 2 - (μ / 2) * ‖x‖ ^ 2
          = μ * ⟪x, y - x⟫ + (μ / 2) * ‖y - x‖ ^ 2 := by
      have hdiffnorm :
          ‖y‖ ^ 2 - ‖x‖ ^ 2 = 2 * ⟪x, y - x⟫ + ‖y - x‖ ^ 2 := by
        linarith [hnorm_sq]
      calc
        (μ / 2) * ‖y‖ ^ 2 - (μ / 2) * ‖x‖ ^ 2
            = (μ / 2) * (‖y‖ ^ 2 - ‖x‖ ^ 2) := by ring
        _ = (μ / 2) * (2 * ⟪x, y - x⟫ + ‖y - x‖ ^ 2) := by
            rw [hdiffnorm]
        _ = μ * ⟪x, y - x⟫ + (μ / 2) * ‖y - x‖ ^ 2 := by
            ring
    linarith [hgy'', hdiff]
  simpa [FirstOrderStrongConvexAt] using this

/--
`StrongConvexOn univ μ f` plus differentiability everywhere gives the global first-order strong
convexity inequality, i.e. the premise of `strongMonotone_gradient_of_firstOrderStrongConvex`.
-/
theorem firstOrderStrongConvex_of_strongConvexOn_univ (μ : ℝ) {f : E → ℝ}
    (hsc : StrongConvexOn (s := (Set.univ : Set E)) μ f) (hd : Differentiable ℝ f) :
    FirstOrderStrongConvex (E := E) μ f := by
  intro x y
  exact firstOrderStrongConvexAt_of_strongConvexOn_univ (E := E) μ hsc (hd x) y

/--
`FirstOrderStrongConvex` implies the gradient is $\mu$-strongly monotone.

This is the exact operator-side fact needed to use `GD.step_norm_sq_le` with $g=\nabla f$.
-/
theorem strongMonotone_gradient_of_firstOrderStrongConvex (μ : ℝ) {f : E → ℝ}
    (h : FirstOrderStrongConvex (μ := μ) f) :
    StrongMonotone (μ := μ) (fun x => (∇ f) x) := by
  intro x y
  -- Add the first-order inequalities at `(x, y)` and `(y, x)`; the function values cancel and
  -- the two inner products combine into `-⟪x - y, ∇ f x - ∇ f y⟫`.
  have hxy' :
      f y - f x ≥ ⟪(∇ f) x, y - x⟫ + (μ / 2) * ‖y - x‖ ^ 2 := by
    linarith [h x y]
  have hyx' :
      f x - f y ≥ ⟪(∇ f) y, x - y⟫ + (μ / 2) * ‖x - y‖ ^ 2 := by
    linarith [h y x]
  have hadd :
      0 ≥ ⟪(∇ f) x, y - x⟫ + ⟪(∇ f) y, x - y⟫ + (μ / 2) * ‖y - x‖ ^ 2 + (μ / 2) * ‖x - y‖ ^ 2 := by
    linarith [hxy', hyx']
  have hinner :
      ⟪(∇ f) x, y - x⟫ + ⟪(∇ f) y, x - y⟫ = - ⟪x - y, (∇ f) x - (∇ f) y⟫ := by
    calc
      ⟪(∇ f) x, y - x⟫ + ⟪(∇ f) y, x - y⟫
          = ⟪y - x, (∇ f) x⟫ + ⟪x - y, (∇ f) y⟫ := by
              simp
      _ = -⟪x - y, (∇ f) x⟫ + ⟪x - y, (∇ f) y⟫ := by
              have hsub : y - x = -(x - y) := by
                abel
              rw [hsub]
              simp
      _ = - ⟪x - y, (∇ f) x - (∇ f) y⟫ := by
              simp [sub_eq_add_neg, inner_add_right, inner_neg_right, add_comm]
  have hnorm : ‖y - x‖ = ‖x - y‖ := norm_sub_rev y x
  have hmono0 :
      μ * ‖x - y‖ ^ 2 ≤ ⟪x - y, (∇ f) x - (∇ f) y⟫ := by
    rw [hinner, hnorm] at hadd
    nlinarith
  exact hmono0

/--
Linear convergence of gradient descent on a first-order strongly convex objective whose gradient
is assumed `L`-Lipschitz.

With `0 ≤ μ ≤ L`, `0 < η`, `η * L ^ 2 < 2 * μ`, and a critical point `xStar` of `f`, the iterates
of `step η (∇ f)` satisfy the geometric bound with factor `q η μ L < 1`. The Lipschitz hypothesis
on `∇ f` is not derived from smoothness of `f` in this repository; it must be supplied.
-/
theorem dist_sq_iterate_le_of_firstOrderStrongConvex (η μ : ℝ) {L : NNReal} {f : E → ℝ}
    (hsc : FirstOrderStrongConvex (μ := μ) f) (hlip : LipschitzWith L (fun x => (∇ f) x))
    {xStar x : E} (hxStar : (∇ f) xStar = 0)
    (hμ : 0 ≤ μ) (hμL : μ ≤ (L : ℝ)) (hη : 0 < η) (hstep : η * (L : ℝ) ^ 2 < 2 * μ) (k : Nat) :
    ‖(step η (fun x => (∇ f) x))^[k] x - xStar‖ ^ 2 ≤ (q η μ L) ^ k * ‖x - xStar‖ ^ 2 ∧
      q η μ L < 1 :=
  dist_sq_iterate_le_of_step_size (E := E) η μ (fun x => (∇ f) x)
    (strongMonotone_gradient_of_firstOrderStrongConvex (E := E) μ hsc) hlip hxStar hμ hμL hη
    hstep k

/--
Linear convergence of gradient descent for a differentiable `StrongConvexOn univ μ f` objective
whose gradient is assumed `L`-Lipschitz; see `dist_sq_iterate_le_of_firstOrderStrongConvex` for
the step-size regime.
-/
theorem dist_sq_iterate_le_of_strongConvexOn (η μ : ℝ) {L : NNReal} {f : E → ℝ}
    (hsc : StrongConvexOn (s := (Set.univ : Set E)) μ f) (hd : Differentiable ℝ f)
    (hlip : LipschitzWith L (fun x => (∇ f) x))
    {xStar x : E} (hxStar : (∇ f) xStar = 0)
    (hμ : 0 ≤ μ) (hμL : μ ≤ (L : ℝ)) (hη : 0 < η) (hstep : η * (L : ℝ) ^ 2 < 2 * μ) (k : Nat) :
    ‖(step η (fun x => (∇ f) x))^[k] x - xStar‖ ^ 2 ≤ (q η μ L) ^ k * ‖x - xStar‖ ^ 2 ∧
      q η μ L < 1 :=
  dist_sq_iterate_le_of_firstOrderStrongConvex (E := E) η μ
    (firstOrderStrongConvex_of_strongConvexOn_univ (E := E) μ hsc hd) hlip hxStar hμ hμL hη
    hstep k

end GD
end Optim
