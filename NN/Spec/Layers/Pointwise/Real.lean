/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context.Real
public import NN.Spec.Layers.Pointwise

/-!
# Exact real meanings of pointwise operators

Over the reals, `Spec.PointwiseOp.meaning` is the mathematical function each PyTorch operator
approximates. It is defined except at poles and outside the real domain: the quotient and its floor
and truncation need a nonzero divisor, `1 / √x` a positive `x`, and `Real.rpow` the domain
described in `Context.realPointwise`, while `Spec.erf` is total. A real result beyond the range of
a binary format, such as a quotient that rounds to an infinity in binary32, keeps its real value.
This module proves that the exact GELU assembled in PyTorch's order of operations is
`x (1 + erf (x / √2)) / 2`, and that PyTorch's flooring-division schedule, which binary formats run
with rounded operations, computes `⌊a / b⌋` in exact arithmetic. `NN.Proofs.Probability.NormalCdf`
proves that `(1 + erf (x / √2)) / 2` is the standard normal distribution function `Φ(x)`.

## References

- Dan Hendrycks and Kevin Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415
-/

@[expose] public section

namespace Spec.PointwiseOp

/-- Over the reals, the exact GELU is `x (1 + erf (x / √2)) / 2`, PyTorch's formula for `x Φ(x)`. -/
theorem meaning_gelu_none_real :
    meaning (α := ℝ) (.gelu .none) =
      some (PointwiseFn.unary fun x => .ok (x * (1 + Spec.erf (x / √2)) / 2)) := by
  have hHalf : ((1 / 2 : Rat) : ℝ) = 1 / 2 := by norm_num
  have hScale (x : ℝ) : x * √(1 / 2) = x / √2 := by
    rw [one_div, Real.sqrt_inv, div_eq_mul_inv]
  simp only [meaning, Context.pointwise_real, Context.realPointwise]
  refine congrArg some (funext fun operands => ?_)
  simp only [PointwiseFn.unary, geluErf, hHalf]
  change Except.ok _ = Except.ok _
  congr 1
  change operands 0 * (1 / 2) * (1 + Spec.erf (operands 0 * √(1 / 2))) = _
  rw [hScale]
  ring

/-- Over the reals, the tanh GELU is `Activation.Math.geluSpec`, with `√(2 / π)` exact. -/
theorem meaning_gelu_tanh_real :
    meaning (α := ℝ) (.gelu .tanh) =
      some (PointwiseFn.unary fun x => .ok (Activation.Math.geluSpec x)) :=
  meaning_gelu_tanh rfl

end Spec.PointwiseOp

namespace Spec

/-- Truncation differs from the floor by one exactly when it overshoots the argument. -/
theorem floor_eq_truncReal (q : ℝ) :
    (q - Context.truncReal q < 0 → ⌊q⌋ = Context.truncReal q - 1) ∧
      (0 ≤ q - Context.truncReal q → ⌊q⌋ = Context.truncReal q) := by
  unfold Context.truncReal
  by_cases hq : 0 ≤ q
  · simp only [hq, ↓reduceIte]
    exact ⟨fun h => absurd (Int.floor_le q) (by linarith), fun _ => trivial⟩
  · simp only [hq, ↓reduceIte]
    have hCeil : q ≤ ⌈q⌉ := Int.le_ceil q
    have hBelow : (⌈q⌉ : ℝ) < q + 1 := Int.ceil_lt_add_one q
    constructor
    · intro h
      rw [Int.floor_eq_iff]
      push_cast
      constructor <;> linarith
    · intro h
      rw [Int.floor_eq_iff]
      constructor <;> linarith

/-- PyTorch's flooring-division schedule computes `⌊a / b⌋` in exact real arithmetic, with C's
`fmod` as `a - b * trunc (a / b)`. -/
theorem divFloorSchedule_real (a b : ℝ) (hb : b ≠ 0) :
    divFloorSchedule (fun x y => x - y * (Context.truncReal (x / y) : ℝ))
      (fun q => ((⌊q⌋ : ℤ) : ℝ)) (fun _ => 0) (1 / 2) a b = ⌊a / b⌋ := by
  set t : ℤ := Context.truncReal (a / b) with ht
  have hFloor := floor_eq_truncReal (a / b)
  rw [← ht] at hFloor
  have hQuotient : (a - (a - b * (t : ℝ))) / b = t := by
    field_simp
    ring
  have hRemainder : a - b * (t : ℝ) = b * (a / b - t) := by
    field_simp
  -- The corrected quotient is the floor of `a / b`.
  have hCorrected :
      (if !(a - b * (t : ℝ) == 0) && (decide (b < 0) != decide (a - b * (t : ℝ) < 0))
        then (a - (a - b * (t : ℝ))) / b - 1 else (a - (a - b * (t : ℝ))) / b) =
        ((⌊a / b⌋ : ℤ) : ℝ) := by
    rw [hQuotient, hRemainder]
    rcases lt_trichotomy (a / b - t) 0 with hNeg | hZero | hPos
    · rw [hFloor.1 hNeg]
      rcases lt_or_gt_of_ne hb with hbNeg | hbPos
      · have hm : 0 < b * (a / b - t) := mul_pos_of_neg_of_neg hbNeg hNeg
        simp [hbNeg, hm.ne', not_lt.mpr hm.le]
      · have hm : b * (a / b - t) < 0 := mul_neg_of_pos_of_neg hbPos hNeg
        simp [not_lt.mpr hbPos.le, hm.ne, hm]
    · rw [hFloor.2 hZero.ge, hZero]
      simp
    · rw [hFloor.2 hPos.le]
      rcases lt_or_gt_of_ne hb with hbNeg | hbPos
      · have hm : b * (a / b - t) < 0 := mul_neg_of_neg_of_pos hbNeg hPos
        simp [hbNeg, hm]
      · have hm : 0 < b * (a / b - t) := mul_pos hbPos hPos
        simp [not_lt.mpr hbPos.le, not_lt.mpr hm.le]
  unfold divFloorSchedule
  simp only [beq_iff_eq, hb, ↓reduceIte]
  rw [hCorrected]
  by_cases hk : ((⌊a / b⌋ : ℤ) : ℝ) = 0
  · simp [hk]
  · simp [hk, Int.floor_intCast]

end Spec
