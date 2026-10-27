/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.BatchNorm
public import NN.Spec.Core.Context.Real
public import NN.Spec.Layers.Loss
import NN.Proofs.Tensor.Basic.Folds

/-!
# Normalization analysis properties

This file proves algebraic properties of normalization and cosine loss over `ℝ`. The executable
definitions live in `NN.Spec.Layers.Normalization` and `NN.Spec.Layers.Loss`.

Current focus:
- **BatchNorm inference is affine** in the input `x` when the running statistics are fixed.
- Scaled cosine loss and its selected gradient preserve the epsilon-clipped real formulas.

That fact matters because inference-time BatchNorm can be folded into a preceding/following affine
layer for verification, bound propagation, and simplification. Training-mode BatchNorm has
batch-dependent statistics and is not claimed by this theorem.
The folding identity is over real arithmetic; it does not justify reordering finite-precision
division, multiplication, and addition without a separate rounding analysis.
-/

@[expose] public section

namespace Proofs

open Spec TorchLean TorchLean.Tensor

noncomputable section

namespace Normalization

/-!
## BatchNorm inference is affine

At inference time, BatchNorm uses fixed running mean/variance. The resulting function is affine:

$y=x\odot\operatorname{scale}+\operatorname{bias}$.
-/

/--
Scalar algebra behind inference-time BatchNorm folding.

For fixed mean $\mu$, scale $\gamma$, shift $\beta$, and standard deviation
$\operatorname{std}$, the expression

$\frac{x-\mu}{\operatorname{std}}\gamma+\beta$

is affine in $x$, with multiplicative coefficient $\gamma/\operatorname{std}$ and bias
$\beta-\mu\gamma/\operatorname{std}$.
-/
private theorem batchNorm_inference_affine_scalar (x μ γ β std : ℝ) :
    ((x - μ) / std) * γ + β = x * (γ / std) + (β - μ * (γ / std)) := by
  ring

/--
Shape-generic tensor version of `batchNorm_inference_affine_scalar`.

Coordinate extensionality reduces the pointwise tensor operations to the scalar identity. BatchNorm
parameters have already been broadcast to the input shape.
-/
private theorem batchNorm_inference_affine_tensor {s : Shape}
    (x mean gamma beta std : Tensor ℝ s) :
    Tensor.addSpec (Tensor.mulSpec (Tensor.divSpec (Tensor.subSpec x mean) std) gamma) beta =
      Tensor.addSpec (Tensor.mulSpec x (Tensor.divSpec gamma std))
        (Tensor.subSpec beta (Tensor.mulSpec mean (Tensor.divSpec gamma std))) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simpa [Tensor.addSpec, Tensor.subSpec, Tensor.mulSpec, Tensor.divSpec, Tensor.map2Spec] using
    batchNorm_inference_affine_scalar
      (x coordinate) (mean coordinate) (gamma coordinate) (beta coordinate) (std coordinate)

/--
Inference-time BatchNorm is affine in the input `x`.

`batchNormInference x runningMean runningVar gamma beta epsilon`

equals the pointwise affine map

$x\,\frac{\gamma}{\operatorname{std}}
+\left(\beta-\mu\frac{\gamma}{\operatorname{std}}\right)$

after broadcasting channel parameters to the input shape and clamping the running variance exactly
as the spec does.
-/
theorem batchNorm_inference_eq_mul_add
    {channels : Nat} {sSpatial : Shape}
    (x : Tensor ℝ (.dim channels sSpatial))
    (runningMean : Tensor ℝ [channels])
    (runningVar : Tensor ℝ [channels])
    (gamma : Tensor ℝ [channels])
    (beta : Tensor ℝ [channels])
    (epsilon : ℝ := TorchLean.normalizationEpsilon) :
    Spec.batchNormInference (α := ℝ) (channels := channels) (sSpatial := sSpatial)
        x runningMean runningVar gamma beta epsilon
      =
    let s : Shape := .dim channels sSpatial
    let runningVar := Tensor.maxSpec runningVar (Tensor.full (.dim channels .scalar) 0)
    let mean_b := Spec.broadcastChannel sSpatial runningMean
    let var_b := Spec.broadcastChannel sSpatial runningVar
    let gamma_b := Spec.broadcastChannel sSpatial gamma
    let beta_b := Spec.broadcastChannel sSpatial beta
    let std := Tensor.sqrtSpec (Tensor.addSpec var_b (Tensor.full s epsilon))
    Tensor.addSpec (Tensor.mulSpec x (Tensor.divSpec gamma_b std))
      (Tensor.subSpec beta_b (Tensor.mulSpec mean_b (Tensor.divSpec gamma_b std))) := by
  simp [Spec.batchNormInference, batchNorm_inference_affine_tensor]

/-! ## Scaled cosine normalization -/

private theorem cosine_normFactors_eq {s : Shape} (x : Tensor ℝ s) (epsilon : ℝ)
    (he : 0 < epsilon) :
    cosineSimilaritySpec.normFactors x epsilon =
      let scale := max (Tensor.maxAbs x) |epsilon|
      (scale, max (Real.sqrt (sumSpec (squareSpec x))) epsilon / scale) := by
  let scale := max (Tensor.maxAbs x) |epsilon|
  have hs : 0 < scale := lt_of_lt_of_le (by simpa [abs_of_pos he] using he)
    (le_max_right _ _)
  have hsum : sumSpec (Tensor.map (fun value => (value / scale) * (value / scale)) x) =
      sumSpec (squareSpec x) / scale ^ 2 := by
    rw [sum_spec_eq_coord_sum, sum_spec_eq_coord_sum, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    simp only [Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply, squareSpec]
    ring
  simp only [cosineSimilaritySpec.normFactors, Context.stopGradient, Context.stopGradient?,
    MathFunctions.sqrt, MathFunctions.abs]
  change (scale, if Real.sqrt (sumSpec (Tensor.map
    (fun value => (value / scale) * (value / scale)) x)) > epsilon / scale then
    Real.sqrt (sumSpec (Tensor.map (fun value => (value / scale) * (value / scale)) x))
    else epsilon / scale) = _
  rw [hsum, Real.sqrt_div' _ (sq_nonneg scale), Real.sqrt_sq hs.le]
  simp only [gt_iff_lt, div_lt_div_iff_of_pos_right hs]
  split_ifs with h
  · rw [max_eq_left h.le]
  · rw [max_eq_right (le_of_not_gt h)]

private theorem cosine_normalized_eq {s : Shape} (x : Tensor ℝ s) (epsilon : ℝ)
    (he : 0 < epsilon) :
    let f := cosineSimilaritySpec.normFactors x epsilon
    Tensor.map (fun value => (value / f.1) / f.2) x =
      Tensor.map (fun value => value / max (Real.sqrt (sumSpec (squareSpec x))) epsilon) x := by
  rw [cosine_normFactors_eq x epsilon he]
  let scale := max (Tensor.maxAbs x) |epsilon|
  have hs : 0 < scale := lt_of_lt_of_le (by simpa [abs_of_pos he] using he)
    (le_max_right _ _)
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply]
  exact div_div_div_cancel_right₀ hs.ne' _ _

private theorem cosine_dot_div {s : Shape} (x t : Tensor ℝ s) (a b : ℝ) :
    sumSpec (mulSpec (Tensor.map (fun value => value / a) x)
      (Tensor.map (fun value => value / b) t)) = sumSpec (mulSpec x t) / (a * b) := by
  conv_lhs => rw [sum_spec_eq_coord_sum]
  conv_rhs => rw [sum_spec_eq_coord_sum, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  simp only [mulSpec, map2Spec, TorchLean.Tensor.Internal.Rep.zipWith_apply,
    Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply]
  ring

/-- Scaling before normalization preserves the epsilon-clipped real cosine loss.
This identity does not assert bitwise equivalence of the two floating-point schedules. -/
theorem cosineSimilarity_eq {s : Shape} (x t : Tensor ℝ s) (epsilon : ℝ)
    (he : 0 < epsilon) :
    cosineSimilaritySpec x t epsilon =
      1 - sumSpec (mulSpec x t) /
        (max (Real.sqrt (sumSpec (squareSpec x))) epsilon *
          max (Real.sqrt (sumSpec (squareSpec t))) epsilon) := by
  unfold cosineSimilaritySpec
  change 1 - sumSpec (mulSpec
    (Tensor.map (fun value => (value / (cosineSimilaritySpec.normFactors x epsilon).1) /
      (cosineSimilaritySpec.normFactors x epsilon).2) x)
    (Tensor.map (fun value => (value / (cosineSimilaritySpec.normFactors t epsilon).1) /
      (cosineSimilaritySpec.normFactors t epsilon).2) t)) = _
  rw [cosine_normalized_eq x epsilon he, cosine_normalized_eq t epsilon he, cosine_dot_div]

/-- The scaled pullback preserves the clipped real formula, including its zero radial term
at and below the predicted-norm floor. No rounding-error bound is asserted here. -/
theorem cosineSimilarityDeriv_eq {s : Shape} (x t : Tensor ℝ s) (epsilon : ℝ)
    (he : 0 < epsilon) :
    cosineSimilarityDerivSpec x t epsilon =
      let xn := Real.sqrt (sumSpec (squareSpec x))
      let tn := Real.sqrt (sumSpec (squareSpec t))
      let xd := max xn epsilon
      let td := max tn epsilon
      map2Spec (fun p q =>
        (if xn > epsilon then sumSpec (mulSpec x t) / (xd ^ 3 * td) * p else 0) -
          q / (xd * td)) x t := by
  let xs := max (Tensor.maxAbs x) |epsilon|
  let ts := max (Tensor.maxAbs t) |epsilon|
  let xn := Real.sqrt (sumSpec (squareSpec x))
  let tn := Real.sqrt (sumSpec (squareSpec t))
  let xd := max xn epsilon
  let td := max tn epsilon
  have hxs : 0 < xs := lt_of_lt_of_le (by simpa [abs_of_pos he] using he)
    (le_max_right _ _)
  have hts : 0 < ts := lt_of_lt_of_le (by simpa [abs_of_pos he] using he)
    (le_max_right _ _)
  have hxd : 0 < xd := lt_of_lt_of_le he (le_max_right _ _)
  have htd : 0 < td := lt_of_lt_of_le he (le_max_right _ _)
  simp only [cosineSimilarityDerivSpec,
    cosine_normFactors_eq x epsilon he, cosine_normFactors_eq t epsilon he]
  dsimp only [xs] at hxs
  dsimp only [ts] at hts
  simp_rw [div_div_div_cancel_right₀ hxs.ne', div_div_div_cancel_right₀ hts.ne']
  change map2Spec (fun u v =>
    let radial := if xd / xs > epsilon / xs then
      sumSpec (mulSpec (Tensor.map (fun w => w / xd) x)
        (Tensor.map (fun w => w / td) t)) * u else 0
    ((radial - v) / (xd / xs)) / xs)
    (Tensor.map (fun w => w / xd) x) (Tensor.map (fun w => w / td) t) = _
  rw [cosine_dot_div]
  have hactive : epsilon / xs < xd / xs ↔ epsilon < xn := by
    rw [div_lt_div_iff_of_pos_right (show 0 < xs from hxs)]
    simp [xd, lt_max_iff]
  simp only [gt_iff_lt, hactive]
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [map2Spec, TorchLean.Tensor.Internal.Rep.zipWith_apply,
    Tensor.map, TorchLean.Tensor.Internal.Rep.map_apply]
  dsimp only [xs, xd, td, xn, tn] at *
  split_ifs <;> field_simp [hxs.ne', hxd.ne', htd.ne']
  ring

end Normalization

end

end Proofs
