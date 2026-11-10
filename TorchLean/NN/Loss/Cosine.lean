/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Normalization
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Layers.Loss.Axis
import TorchLean.Tensor.Reduce.Real

/-!
# Normalization analysis properties

Algebraic facts over `ℝ` about BatchNorm and the cosine loss, whose executable definitions live
under `TorchLean/Tensor/Layers/` and in `TorchLean.Tensor.Layers.Loss`.

Main results:
- **BatchNorm inference is affine** in the input `x` when the running statistics are fixed.
- For positive epsilon, the scaled cosine loss and its selected gradient equal the epsilon-clipped
  real formulas. For every epsilon, the half squared distance that the loss computes between unit
  normalized vectors is one minus their dot product (`fiberLoss_eq_one_sub_dot`).

Because inference-time BatchNorm is affine, it can be folded into an adjacent affine layer for
verification, bound propagation and simplification. Training-mode BatchNorm normalizes with batch
statistics and is outside these theorems. Every identity here holds over the reals, so reordering
the rounded division, multiplication and addition of a float execution needs its own rounding
analysis.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open Tensor

noncomputable section

/-!
## BatchNorm inference is affine

At inference time BatchNorm uses the fixed running mean and variance, so it is the affine map

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
private theorem Tensor.batchNorm_affine_scalar (x μ γ β std : ℝ) :
    ((x - μ) / std) * γ + β = x * (γ / std) + (β - μ * (γ / std)) := by
  ring

/--
Tensor form of `batchNorm_affine_scalar` for any shape, with the BatchNorm parameters
already broadcast to the input shape. Coordinate extensionality reduces it to the scalar identity.
-/
private theorem Tensor.batchNorm_affine {s : Shape}
    (x mean gamma beta std : Tensor ℝ s) :
    (((x - mean) / std) * gamma) + beta =
      (x * (gamma / std)) +
        (beta - (mean * (gamma / std))) := by
  apply Tensor.Rep.ext
  intro coordinate
  simpa [Tensor.add_def, Tensor.sub_def, Tensor.mul_def, Tensor.div_def] using
    batchNorm_affine_scalar
      (x coordinate) (mean coordinate) (gamma coordinate) (beta coordinate) (std coordinate)

/--
Inference-time BatchNorm is affine in the input `x`.

`Tensor.batchNorm x 0 gamma beta (.running runningMean runningVar) epsilon`

equals the pointwise affine map

$x\,\frac{\gamma}{\operatorname{std}}
+\left(\beta-\mu\frac{\gamma}{\operatorname{std}}\right)$

once the channel parameters are broadcast to the input shape. Here $\mu$ is the running mean and
$\operatorname{std}=\sqrt{\max(v,0)+\varepsilon}$ for the running variance $v$, which the spec
clamps at zero.
-/
theorem Tensor.batchNorm_running_eq_mul_add
    {channels : Nat} {sSpatial : Shape}
    (x : Tensor ℝ (.dim channels sSpatial))
    (runningMean : Tensor ℝ [channels])
    (runningVar : Tensor ℝ [channels])
    (gamma : Tensor ℝ [channels])
    (beta : Tensor ℝ [channels])
    (epsilon : ℝ := normalizationEpsilon) :
    Tensor.batchNorm x ⟨0, Nat.succ_pos _⟩ gamma beta (.running runningMean runningVar) epsilon
      =
    let s : Shape := .dim channels sSpatial
    let spread := fun t : Tensor ℝ [channels] =>
      Tensor.broadcast t s _ (s.broadcastDims_axis ⟨0, Nat.succ_pos _⟩)
    let runningVar := Tensor.maximum runningVar (Tensor.full (.dim channels .scalar) 0)
    let mean_b := spread runningMean
    let var_b := spread runningVar
    let gamma_b := spread gamma
    let beta_b := spread beta
    let std := Tensor.sqrt (var_b + Tensor.full s epsilon) .clampNegative
    (x * (gamma_b / std)) +
      (beta_b - (mean_b * (gamma_b / std))) := by
  exact (batchNorm_running_eq x ⟨0, Nat.succ_pos _⟩ gamma beta runningMean runningVar
    epsilon).trans (batchNorm_affine _ _ _ _ _)

/-! ## Scaled cosine normalization -/

private theorem CosineLoss.normFactors_eq {s : Shape} (x : Tensor ℝ s) (epsilon : ℝ)
    (he : 0 < epsilon) :
    CosineLoss.normFactors x epsilon =
      let scale := max (Tensor.maxAbs x) |epsilon|
      (scale, max (Real.sqrt (Tensor.sum (Tensor.square x))) epsilon / scale) := by
  let scale := max (Tensor.maxAbs x) |epsilon|
  have hs : 0 < scale := lt_of_lt_of_le (by simpa [abs_of_pos he] using he)
    (le_max_right _ _)
  have hsum : Tensor.sum (Tensor.map (fun value => (value / scale) * (value / scale)) x) =
      Tensor.sum (Tensor.square x) / scale ^ 2 := by
    rw [sum_eq_coord_sum, sum_eq_coord_sum, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    simp only [Tensor.map_apply, Tensor.square]
    ring
  simp only [CosineLoss.normFactors, Context.stopGradient, Context.stopGradient?,
    MathFunctions.sqrt, MathFunctions.abs]
  change (scale, if Real.sqrt (Tensor.sum (Tensor.map
    (fun value => (value / scale) * (value / scale)) x)) > epsilon / scale then
    Real.sqrt (Tensor.sum (Tensor.map (fun value => (value / scale) * (value / scale)) x))
    else epsilon / scale) = _
  rw [hsum, Real.sqrt_div' _ (sq_nonneg scale), Real.sqrt_sq hs.le]
  simp only [gt_iff_lt, div_lt_div_iff_of_pos_right hs]
  split_ifs with h
  · rw [max_eq_left h.le]
  · rw [max_eq_right (le_of_not_gt h)]

private theorem CosineLoss.normalized_eq {s : Shape} (x : Tensor ℝ s) (epsilon : ℝ)
    (he : 0 < epsilon) :
    let f := CosineLoss.normFactors x epsilon
    Tensor.map (fun value => (value / f.1) / f.2) x =
      Tensor.map (fun value => value / max (Real.sqrt (Tensor.sum (Tensor.square x))) epsilon)
        x := by
  rw [normFactors_eq x epsilon he]
  let scale := max (Tensor.maxAbs x) |epsilon|
  have hs : 0 < scale := lt_of_lt_of_le (by simpa [abs_of_pos he] using he)
    (le_max_right _ _)
  apply Tensor.Rep.ext
  intro coordinate
  simp only [Tensor.map_apply]
  exact div_div_div_cancel_right₀ hs.ne' _ _

private theorem CosineLoss.dot_div {s : Shape} (x t : Tensor ℝ s) (a b : ℝ) :
    Tensor.sum (Tensor.map (fun value => value / a) x *
      Tensor.map (fun value => value / b) t) = Tensor.sum (x * t) / (a * b) := by
  conv_lhs => rw [sum_eq_coord_sum]
  conv_rhs => rw [sum_eq_coord_sum, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  simp only [mul_def, Tensor.zipWith_apply,
    Tensor.map_apply]
  ring

/-- With a positive norm above its floor, the normalized vector of the cosine loss has unit
squared norm, for every epsilon. -/
private theorem CosineLoss.normalized_dot_self {s : Shape} (x : Tensor ℝ s) (epsilon : ℝ)
    (hpos : (CosineLoss.normFactors x epsilon).2 > 0)
    (hfloor : (CosineLoss.normFactors x epsilon).2 >
      epsilon / (CosineLoss.normFactors x epsilon).1) :
    Tensor.sum ((Tensor.map (fun value => (value / (CosineLoss.normFactors x epsilon).1) /
        (CosineLoss.normFactors x epsilon).2) x) *
      (Tensor.map (fun value => (value / (CosineLoss.normFactors x epsilon).1) /
        (CosineLoss.normFactors x epsilon).2) x)) = 1 := by
  set scale := (CosineLoss.normFactors x epsilon).1
  set squares := Tensor.sum (Tensor.map (fun value => (value / scale) * (value / scale)) x)
  have hsnd : (CosineLoss.normFactors x epsilon).2 =
      if Real.sqrt squares > epsilon / scale then Real.sqrt squares else epsilon / scale := rfl
  have hnorm : (CosineLoss.normFactors x epsilon).2 = Real.sqrt squares := by
    rw [hsnd] at hfloor ⊢
    split_ifs at hfloor ⊢
    · rfl
    · exact absurd hfloor (lt_irrefl _)
  rw [hnorm] at hpos ⊢
  have hnonneg : 0 ≤ squares := by
    simp only [squares]
    rw [sum_eq_coord_sum]
    exact Finset.sum_nonneg fun c _ => by
      simp only [Tensor.map_apply]
      exact mul_self_nonneg _
  have hsq : Real.sqrt squares ^ 2 = squares := Real.sq_sqrt hnonneg
  have hsum : Tensor.sum (Tensor.map (fun value => (value / scale) / Real.sqrt squares) x *
      Tensor.map (fun value => (value / scale) / Real.sqrt squares) x) =
      squares / Real.sqrt squares ^ 2 := by
    simp only [squares]
    rw [sum_eq_coord_sum, sum_eq_coord_sum, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro c _
    simp only [mul_def, Tensor.zipWith_apply,
      Tensor.map_apply]
    ring
  rw [hsum, hsq]
  exact div_self (ne_of_gt (lt_of_lt_of_le (pow_pos hpos 2) hsq.le))

/-- Half the squared distance of two unit vectors is one minus their dot product. -/
private theorem CosineLoss.dot_sub_self_div_two {s : Shape} (a b : Tensor ℝ s)
    (ha : Tensor.sum (a * a) = 1) (hb : Tensor.sum (b * b) = 1) :
    Tensor.sum ((a - b) * (a - b)) / 2 = 1 - Tensor.sum (a * b) := by
  rw [sum_eq_coord_sum] at ha hb ⊢
  rw [sum_eq_coord_sum]
  simp only [mul_def, sub_def, Tensor.zipWith_apply] at ha hb ⊢
  have expand : ∑ c : s.Coord, (a c - b c) * (a c - b c) =
      ∑ c : s.Coord, a c * a c + ∑ c : s.Coord, b c * b c - 2 * ∑ c : s.Coord, a c * b c := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun c _ => by ring
  rw [expand, ha, hb]
  ring

/-- Over the reals the cosine loss of two tensors is one minus the dot product of the normalized
tensors, in both of its branches and for every epsilon. -/
theorem CosineLoss.fiberLoss_eq_one_sub_dot {s : Shape} (x t : Tensor ℝ s) (epsilon : ℝ) :
    CosineLoss.fiberLoss x t epsilon =
      1 - Tensor.sum ((Tensor.map (fun value => (value / (CosineLoss.normFactors x epsilon).1) /
          (CosineLoss.normFactors x epsilon).2) x) *
        (Tensor.map (fun value => (value / (CosineLoss.normFactors t epsilon).1) /
          (CosineLoss.normFactors t epsilon).2) t)) := by
  unfold CosineLoss.fiberLoss
  set f := CosineLoss.normFactors x epsilon
  set g := CosineLoss.normFactors t epsilon
  set p := Tensor.map (fun value => (value / f.1) / f.2) x
  set q := Tensor.map (fun value => (value / g.1) / g.2) t
  change @ite ℝ (f.2 > 0 ∧ f.2 > epsilon / f.1 ∧ g.2 > 0 ∧ g.2 > epsilon / g.1) _
      (Tensor.sum ((p - q) * (p - q)) / 2) (1 - Tensor.sum (p * q)) =
    1 - Tensor.sum (p * q)
  split_ifs with h
  · obtain ⟨hx₀, hx, ht₀, ht⟩ := h
    exact dot_sub_self_div_two p q (normalized_dot_self x epsilon hx₀ hx)
      (normalized_dot_self t epsilon ht₀ ht)
  · rfl

/-- Over the reals the cosine loss over every axis is one minus the dot product of the normalized
tensors, in both of its branches. -/
theorem Tensor.cosineLoss_eq_one_sub_dot {s : Shape} (x t : Tensor ℝ s) (keepdim : Bool)
    (epsilon : ℝ) (o : (s.reduce (List.finRange s.rank) keepdim).Coord) :
    cosineLoss x t (List.finRange s.rank) keepdim epsilon o =
      1 - Tensor.sum ((Tensor.map (fun value => (value / (CosineLoss.normFactors x epsilon).1) /
          (CosineLoss.normFactors x epsilon).2) x) *
        (Tensor.map (fun value => (value / (CosineLoss.normFactors t epsilon).1) /
          (CosineLoss.normFactors t epsilon).2) t)) := by
  rw [cosineLoss_finRange, CosineLoss.fiberLoss_eq_one_sub_dot]

/-- For positive `epsilon`, scaling before normalization leaves the real cosine loss at its
epsilon-clipped formula. In floating point the scaled and unscaled schedules can round
differently. -/
theorem Tensor.cosineLoss_eq {s : Shape} (x t : Tensor ℝ s) (keepdim : Bool) (epsilon : ℝ)
    (he : 0 < epsilon) (o : (s.reduce (List.finRange s.rank) keepdim).Coord) :
    cosineLoss x t (List.finRange s.rank) keepdim epsilon o =
      1 - Tensor.sum (x * t) /
        (max (Real.sqrt (Tensor.sum (Tensor.square x))) epsilon *
          max (Real.sqrt (Tensor.sum (Tensor.square t))) epsilon) := by
  rw [cosineLoss_eq_one_sub_dot, CosineLoss.normalized_eq x epsilon he,
    CosineLoss.normalized_eq t epsilon he, CosineLoss.dot_div]

/-- For positive `epsilon`, the scaled pullback equals the clipped real gradient, whose radial term
is zero when the predicted norm is at or below `epsilon`. -/
theorem CosineLoss.fiberLossDeriv_eq {s : Shape} (x t : Tensor ℝ s) (epsilon : ℝ)
    (he : 0 < epsilon) :
    CosineLoss.fiberLossDeriv x t epsilon =
      let xn := Real.sqrt (Tensor.sum (Tensor.square x))
      let tn := Real.sqrt (Tensor.sum (Tensor.square t))
      let xd := max xn epsilon
      let td := max tn epsilon
      zipWith (fun p q =>
        (if xn > epsilon then Tensor.sum (x * t) / (xd ^ 3 * td) * p else 0) -
          q / (xd * td)) x t := by
  let xs := max (Tensor.maxAbs x) |epsilon|
  let ts := max (Tensor.maxAbs t) |epsilon|
  let xn := Real.sqrt (Tensor.sum (Tensor.square x))
  let tn := Real.sqrt (Tensor.sum (Tensor.square t))
  let xd := max xn epsilon
  let td := max tn epsilon
  have hxs : 0 < xs := lt_of_lt_of_le (by simpa [abs_of_pos he] using he)
    (le_max_right _ _)
  have hts : 0 < ts := lt_of_lt_of_le (by simpa [abs_of_pos he] using he)
    (le_max_right _ _)
  have hxd : 0 < xd := lt_of_lt_of_le he (le_max_right _ _)
  have htd : 0 < td := lt_of_lt_of_le he (le_max_right _ _)
  simp only [CosineLoss.fiberLossDeriv,
    normFactors_eq x epsilon he, normFactors_eq t epsilon he]
  dsimp only [xs] at hxs
  dsimp only [ts] at hts
  simp_rw [div_div_div_cancel_right₀ hxs.ne', div_div_div_cancel_right₀ hts.ne']
  change zipWith (fun u v =>
    let radial := if xd / xs > epsilon / xs then
      Tensor.sum (Tensor.map (fun w => w / xd) x *
        Tensor.map (fun w => w / td) t) * u else 0
    ((radial - v) / (xd / xs)) / xs)
    (Tensor.map (fun w => w / xd) x) (Tensor.map (fun w => w / td) t) = _
  rw [dot_div]
  have hactive : epsilon / xs < xd / xs ↔ epsilon < xn := by
    rw [div_lt_div_iff_of_pos_right (show 0 < xs from hxs)]
    simp [xd, lt_max_iff]
  simp only [gt_iff_lt, hactive]
  apply Tensor.Rep.ext
  intro coordinate
  simp only [Tensor.zipWith_apply,
    Tensor.map_apply]
  dsimp only [xs, xd, td, xn, tn] at *
  split_ifs <;> field_simp [hxs.ne', hxd.ne', htd.ne']
  ring

end

end TorchLean
