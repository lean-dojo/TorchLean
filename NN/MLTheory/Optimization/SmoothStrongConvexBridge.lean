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

TorchLean's GD convergence theorems are stated at the operator level:

* $g$ is $\mu$-strongly monotone, and
* $g$ is $L$-Lipschitz.

To apply them to gradient descent on an objective $f$, we need to instantiate $g=\nabla f$.

Mathlib's global definition `Gradient.gradient f x` is total (it returns $0$ if the derivative does
not exist), so for optimization theory we typically assume differentiability and reason using the
standard *first-order* characterization of strong convexity.

This file provides the key local bridge lemma:

If $f$ satisfies the first-order strong convexity inequality (using the gradient), then $\nabla f$
is $\mu$-strongly monotone in the sense needed by `GDLinearConvergence`.

This file provides a concrete bridge from mathlib's `StrongConvexOn` definition
to a first-order inequality, under a `DifferentiableAt` assumption at the base point `x`.

The pointwise theorem gives the first-order inequality at one differentiability point.
To obtain global strong monotonicity, the first-order inequality must hold at every
base point. A strong-convexity hypothesis together with differentiability everywhere
provides that global premise.

The smoothness half is not proved here. Nothing in this file derives `LipschitzWith L (∇ f)` from
a second-order or `fderiv`-level smoothness hypothesis on `f`; the Lipschitz gradient is taken as
an explicit assumption. The closing theorem `dist_sq_iterate_le_of_firstOrderStrongConvex`
assembles the pieces that are proved: first-order strong convexity gives strong monotonicity of
`∇ f`, and together with an assumed Lipschitz gradient and a step-size condition this yields linear
convergence of gradient descent to a critical point.
-/

@[expose] public section

namespace Optim
namespace GD

open Real
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
  -- Let `g(z) = f(z) - (μ/2)‖z‖²`. Strong convexity of `f` means `g` is convex.
  let g : E → ℝ := fun z => f z - (μ / 2) * ‖z‖ ^ 2
  have hg_conv : ConvexOn ℝ (Set.univ : Set E) g := by
    -- Use the mathlib characterization.
    have := (strongConvexOn_iff_convex (s := (Set.univ : Set E)) (m := μ) (f := f)).1 hsc
    simpa [g] using this
  -- Restrict to the line segment between `x` and `y`.
  let h : ℝ →ᵃ[ℝ] E := (AffineMap.lineMap x y : ℝ →ᵃ[ℝ] E)
  have hφ_conv : ConvexOn ℝ (Set.univ : Set ℝ) (g ∘ h) := by
    simpa [Set.preimage_univ] using (hg_conv.comp_affineMap h)
  -- The 1D convex-function inequality: derivative at 0 is below the secant slope to 1.
  have hderiv_line : HasDerivAt (fun t => (g ∘ h) t) ((fderiv ℝ g x) (y - x)) (0 : ℝ) := by
    -- Compute the derivative of `h` and use the chain rule for `g ∘ h`.
    have hh : HasDerivAt (fun t => h t) (y - x) (0 : ℝ) := by
      simpa using (AffineMap.hasDerivAt_lineMap (a := x) (b := y) (x := (0 : ℝ)))
    -- `g` is differentiable at `x` because it is a difference of differentiable functions.
    have hgx : DifferentiableAt ℝ g x := by
      -- `‖·‖^2` is differentiable everywhere in an inner product space.
      have hq0 : DifferentiableAt ℝ (fun z : E => ‖z‖ ^ 2) x :=
        (hasStrictFDerivAt_norm_sq (x := x)).differentiableAt
      have hq : DifferentiableAt ℝ (fun z : E => (μ / 2) * ‖z‖ ^ 2) x := by
        simpa [mul_assoc, mul_left_comm, mul_comm] using hq0.const_mul (μ / 2)
      -- Combine.
      change DifferentiableAt ℝ (f - fun z : E => (μ / 2) * ‖z‖ ^ 2) x
      exact hdx.sub hq
    exact HasFDerivAt.comp_hasDerivAt_of_eq (𝕜 := ℝ) (l := g)
      (l' := fderiv ℝ g x) (y := x) (f := fun t => h t)
      (f' := y - x) (x := (0 : ℝ)) hgx.hasFDerivAt hh (by simp [h])
  have hderiv0 : deriv (fun t => (g ∘ h) t) 0 = (fderiv ℝ g x) (y - x) := by
    exact hderiv_line.deriv
  have hle_slope : deriv (fun t => (g ∘ h) t) 0 ≤ slope (fun t => (g ∘ h) t) 0 1 := by
    -- `ConvexOn.deriv_le_slope` at `t0=0`, `t1=1`.
    have hdiff0 : DifferentiableAt ℝ (fun t => (g ∘ h) t) (0 : ℝ) := hderiv_line.differentiableAt
    exact (hφ_conv.deriv_le_slope (x := (0 : ℝ)) (y := (1 : ℝ))
      (by simp) (by simp) (by linarith) hdiff0)
  -- Convert slope to a difference since `1 - 0 = 1`.
  have hslope : slope (fun t => (g ∘ h) t) 0 1 = (g y - g x) := by
    -- slope_def_field: (φ 1 - φ 0)/(1-0) = φ 1 - φ 0
    simp [slope_def_field, g, h, sub_eq_add_neg]
  -- Now: `deriv ≤ g y - g x`, hence `g y ≥ g x + deriv`.
  have hgy : g y ≥ g x + (fderiv ℝ g x) (y - x) := by
    -- Use `hle_slope` with `hslope`.
    -- `linarith` after rewriting `deriv` and `slope`.
    have : deriv (fun t => (g ∘ h) t) 0 ≤ g y - g x := by
      rw [hslope] at hle_slope
      exact hle_slope
    -- Rearrange.
    linarith [this, hderiv0]
  -- Expand `g` and rewrite the directional derivative using `∇ f x`.
  -- `fderiv g x (y-x) = ⟪∇ f x, y-x⟫ - μ * ⟪x, y-x⟫`.
  -- The quadratic difference also produces the `μ/2 * ‖y-x‖^2` term.
  -- Do the algebra in `ℝ`.
  have hdir :
      (fderiv ℝ g x) (y - x) = ⟪(∇ f) x, y - x⟫ - μ * ⟪x, y - x⟫ := by
    change
      (fderiv ℝ (fun z => f z - (μ / 2) * ‖z‖ ^ 2) x) (y - x)
        = ⟪(∇ f) x, y - x⟫ - μ * ⟪x, y - x⟫
    let q : E → ℝ := fun z => (μ / 2) * ‖z‖ ^ 2
    have hq0 : DifferentiableAt ℝ (fun z : E => ‖z‖ ^ 2) x :=
      (hasStrictFDerivAt_norm_sq (x := x)).differentiableAt
    have hq : DifferentiableAt ℝ q x := by
      simpa [q, mul_assoc, mul_comm, mul_left_comm] using hq0.const_mul (μ / 2)
    have hf_apply : (fderiv ℝ f x) (y - x) = ⟪(∇ f) x, y - x⟫ := by
      exact (inner_gradient_left (f := f) (x := x) (y := y - x)).symm
    have hq_apply : (fderiv ℝ (fun z : E => (μ / 2) * ‖z‖ ^ 2) x) (y - x) = μ * ⟪x, y - x⟫ := by
      have hfderiv :
          fderiv ℝ q x = (μ / 2) • fderiv ℝ (fun z : E => ‖z‖ ^ 2) x := by
        simpa [q, mul_comm, mul_left_comm, mul_assoc] using
          (fderiv_const_mul (x := x) (a := fun z : E => ‖z‖ ^ 2) hq0 (μ / 2))
      calc
        (fderiv ℝ (fun z : E => (μ / 2) * ‖z‖ ^ 2) x) (y - x)
            = (fderiv ℝ q x) (y - x) := by rfl
        _ = ((μ / 2) • fderiv ℝ (fun z : E => ‖z‖ ^ 2) x) (y - x) := by
            rw [hfderiv]
        _ = (μ / 2) * ((fderiv ℝ (fun z : E => ‖z‖ ^ 2) x) (y - x)) := by
            simp
        _ = (μ / 2) * ((2 • innerSL ℝ x) (y - x)) := by
            simp [fderiv_norm_sq_apply]
        _ = (μ / 2) * (2 * ⟪x, y - x⟫) := by
            simp [mul_left_comm, mul_comm]
        _ = μ * ⟪x, y - x⟫ := by
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
  -- Expand the `g` inequality back into `f`.
  -- Use `‖y‖^2 - ‖x‖^2` expansion:
  -- `‖y‖^2 = ‖x + (y-x)‖^2 = ‖x‖^2 + 2⟪x, y-x⟫ + ‖y-x‖^2`.
  have hnorm_sq :
      ‖y‖ ^ 2 = ‖x‖ ^ 2 + 2 * ⟪x, y - x⟫ + ‖y - x‖ ^ 2 := by
    -- `y = x + (y-x)`.
    have hy : y = x + (y - x) := by abel
    -- Apply `norm_add_sq_real`.
    -- `‖x + d‖^2 = ‖x‖^2 + 2⟪x,d⟫ + ‖d‖^2`.
    rw [hy]
    simpa [add_assoc, add_left_comm, add_comm, mul_assoc, mul_left_comm, mul_comm] using
      (norm_add_sq_real (x := x) (y := (y - x)))
  -- Finish by algebra.
  -- From `g y ≥ g x + fderiv g x (y-x)`, substitute `g` and `hdir`/`hnorm_sq`.
  -- Then rearrange to the desired inequality.
  -- This is scalar arithmetic, so `linarith` works once expanded.
  have : f y ≥ f x + ⟪(∇ f) x, y - x⟫ + (μ / 2) * ‖y - x‖ ^ 2 := by
    -- Expand `g` everywhere.
    -- Use `hnorm_sq` to eliminate `‖y‖^2 - ‖x‖^2` terms.
    -- Use `hdir` for the directional derivative.
    -- `linarith` for the final rearrangement.
    have hgy' : f y - (μ / 2) * ‖y‖ ^ 2 ≥ f x - (μ / 2) * ‖x‖ ^ 2 + (fderiv ℝ g x) (y - x) := by
      simpa [g, add_assoc, add_left_comm, add_comm, sub_eq_add_neg] using hgy
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
`FirstOrderStrongConvex` implies the gradient is $\mu$-strongly monotone.

This is the exact operator-side fact needed to use `GD.step_norm_sq_le` with $g=\nabla f$.
-/
theorem strongMonotone_gradient_of_firstOrderStrongConvex (μ : ℝ) {f : E → ℝ}
    (h : FirstOrderStrongConvex (μ := μ) f) :
    StrongMonotone (μ := μ) (fun x => (∇ f) x) := by
  intro x y
  -- Write the strong convexity inequality in both directions and add them.
  have hxy := h x y
  have hyx := h y x
  -- Expand the inner products in a symmetric form.
  -- After cancellation, we get the strong monotonicity inequality.
  -- (This is the standard “swap x/y and add” trick.)
  --
  -- `linarith` handles the scalar algebra once we rewrite `y - x` as `-(x - y)` and normalize
  -- inner products.
  have hxy' :
      f y - f x ≥ ⟪(∇ f) x, y - x⟫ + (μ / 2) * ‖y - x‖ ^ 2 := by
    linarith
  have hyx' :
      f x - f y ≥ ⟪(∇ f) y, x - y⟫ + (μ / 2) * ‖x - y‖ ^ 2 := by
    linarith
  -- Add and simplify.
  have hadd :
      0 ≥ ⟪(∇ f) x, y - x⟫ + ⟪(∇ f) y, x - y⟫ + (μ / 2) * ‖y - x‖ ^ 2 + (μ / 2) * ‖x - y‖ ^ 2 := by
    linarith [hxy', hyx']
  -- Rewrite `⟪∇f x, y - x⟫ + ⟪∇f y, x - y⟫` into `-⟪x - y, ∇f x - ∇f y⟫`.
  have hinner :
      ⟪(∇ f) x, y - x⟫ + ⟪(∇ f) y, x - y⟫ = - ⟪x - y, (∇ f) x - (∇ f) y⟫ := by
    -- Turn everything into `⟪x - y, …⟫` form using symmetry, then use bilinearity.
    calc
      ⟪(∇ f) x, y - x⟫ + ⟪(∇ f) y, x - y⟫
          = ⟪y - x, (∇ f) x⟫ + ⟪x - y, (∇ f) y⟫ := by
              simp
      _ = -⟪x - y, (∇ f) x⟫ + ⟪x - y, (∇ f) y⟫ := by
              -- `y - x = -(x - y)`, then `⟪-u, v⟫ = -⟪u, v⟫`.
              have hsub : y - x = -(x - y) := by
                abel
              rw [hsub]
              -- `inner_neg_left` gives `⟪-(x-y), ∇f x⟫ = -⟪x-y, ∇f x⟫`.
              simp
      _ = - ⟪x - y, (∇ f) x - (∇ f) y⟫ := by
              -- Rewrite subtraction as `a + -b` and use bilinearity in the right argument.
              simp [sub_eq_add_neg, inner_add_right, inner_neg_right, add_comm]
  -- Also `‖y-x‖ = ‖x-y‖`.
  have hnorm : ‖y - x‖ = ‖x - y‖ := by
    simpa using (norm_sub_rev y x)
  -- Combine into the strong monotonicity inequality.
  -- `hadd` gives a lower bound on `⟪x-y, ∇f x - ∇f y⟫` once we move terms.
  -- Rewrite `hadd` using `hinner`/`hnorm` and rearrange.
  have hmono0 :
      μ * ‖x - y‖ ^ 2 ≤ ⟪x - y, (∇ f) x - (∇ f) y⟫ := by
    -- From `hadd` we get:
    --   0 ≥ -(inner) + μ‖x-y‖²
    -- so `inner ≥ μ‖x-y‖²`.
    have hadd' := hadd
    rw [hinner, hnorm] at hadd'
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

end GD
end Optim
