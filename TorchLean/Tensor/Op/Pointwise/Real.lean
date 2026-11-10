/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Elementwise.Real
public import TorchLean.Tensor.Layers.Activation

/-!
# The real GELU and the real division schedules

Over the reals the two GELUs have their textbook formulas. The exact GELU, assembled in PyTorch's
order of operations, is the function `Tensor.gelu` maps and so the forward of `Op.gelu .none`
(`Op.rule?_gelu_none`), and `Context.geluErf_real` proves it is `x (1 + erf (x / √2)) / 2`. The tanh
GELU, the function that the forward of `Op.gelu .tanh` maps over a tensor (`Op.rule?_gelu`), is
`x (1 + tanh (√(2 / π) (x + 0.044715 x³))) / 2` by `Activation.geluTanh_real`.
`TorchLean.NN.NormalCDF` proves that `(1 + erf (x / √2)) / 2` is the standard normal distribution
function `Φ(x)`.

The two closed forms are those of Section 2 of Hendrycks and Gimpel, the exact `x Φ(x)` and their
first approximation. Their second, `x σ(1.702 x)`, has no `GeluApproximation`, and torch 2.8's
`gelu` raises an error for any `approximate` other than `'none'` and `'tanh'`.

Binary formats run the flooring-division schedule with rounded operations, and in exact arithmetic
it computes `⌊a / b⌋` (`divFloorSchedule_real`). Python's remainder schedule computes
`a - b ⌊a / b⌋` (`remainderSchedule_real`, `Context.remainder_real`). Both schedule theorems rest on
`Context.floor_eq_truncRound`, the floor through truncation.
`TorchLean.Tensor.copysign_floorDivide_abs_real` proves that the forward of `Op.div .trunc`
(`Op.rule?_div_trunc`), the flooring division of `|a|` by `|b|` with the sign of `a / b`, computes
`trunc (a / b)` in exact arithmetic, through `Context.copysign_floor_abs_div_real`.

## References

- Dan Hendrycks and Kevin Gimpel, 2016, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415,
  Section 2
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Context

/-- Over the reals, the exact GELU `Context.geluErf Erf.erf` is `x (1 + erf (x / √2)) / 2`,
PyTorch's formula for `x Φ(x)`. It is the function `Tensor.gelu` maps, and so the forward of
`Op.gelu .none` (`Op.rule?_gelu_none`). -/
theorem geluErf_real (x : ℝ) :
    (Context.geluErf (m := Id) Erf.erf x).run = x * (1 + erf (x / √2)) / 2 := by
  have hHalf : ((1 / 2 : Rat) : ℝ) = 1 / 2 := by norm_num
  have hScale : x * √(1 / 2) = x / √2 := by
    rw [one_div, Real.sqrt_inv, div_eq_mul_inv]
  simp only [Context.geluErf, Context.isInf_real, Bool.false_eq_true, ↓reduceIte, hHalf]
  change x * (1 / 2) * (1 + erf (x * √(1 / 2))) = _
  rw [hScale]
  ring

end TorchLean.Context

namespace TorchLean.Activation

/-- Over the reals, the tanh GELU `geluTanh`, the function that the forward of `Op.gelu .tanh`
maps (`Op.rule?_gelu`), is `x (1 + tanh (√(2 / π) (x + 0.044715 x³))) / 2`, the
approximation of Hendrycks and Gimpel with its constants. -/
theorem geluTanh_real (x : ℝ) :
    geluTanh x = x * (1 + Real.tanh (√(2 / Real.pi) * (x + 0.044715 * x ^ 3))) / 2 := by
  simp only [geluTanh, Context.isInf_real, Bool.false_eq_true, ↓reduceIte, geluBeta, geluKappa,
    MathFunctions.sqrt, MathFunctions.pi, MathFunctions.tanh]
  push_cast
  ring_nf

end TorchLean.Activation

namespace TorchLean

/-- Truncation differs from the floor by one exactly when it overshoots the argument. -/
theorem Context.floor_eq_truncRound (q : ℝ) :
    (q - FloatLib.Floats.Formats.Flocq.truncRound q < 0 →
        ⌊q⌋ = FloatLib.Floats.Formats.Flocq.truncRound q - 1) ∧
      (0 ≤ q - FloatLib.Floats.Formats.Flocq.truncRound q →
        ⌊q⌋ = FloatLib.Floats.Formats.Flocq.truncRound q) := by
  unfold FloatLib.Floats.Formats.Flocq.truncRound
  by_cases hq : 0 ≤ q
  · simp only [not_lt.mpr hq, ↓reduceIte]
    exact ⟨fun h => absurd (Int.floor_le q) (by linarith), fun _ => trivial⟩
  · simp only [not_le.mp hq, ↓reduceIte]
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

/-- The flooring-division schedule computes `⌊a / b⌋` in exact real arithmetic, with C's
`fmod` as `a - b * trunc (a / b)`. At `b = 0` both sides are `0`, since Mathlib sets `a / 0 = 0`. -/
theorem divFloorSchedule_real (a b : ℝ) :
    divFloorSchedule (fun x y => x - y * (FloatLib.Floats.Formats.Flocq.truncRound (x / y) : ℝ))
      (fun q => ((⌊q⌋ : ℤ) : ℝ)) (fun _ => 0) (1 / 2) a b = ⌊a / b⌋ := by
  obtain rfl | hb := eq_or_ne b 0
  · simp [divFloorSchedule]
  set t : ℤ := FloatLib.Floats.Formats.Flocq.truncRound (a / b) with ht
  have hFloor := Context.floor_eq_truncRound (a / b)
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
  simp only [beq_iff_eq, hb, ↓reduceIte, beq_self_eq_true, Bool.not_true, Bool.false_and,
    Bool.false_eq_true]
  rw [hCorrected]
  by_cases hk : ((⌊a / b⌋ : ℤ) : ℝ) = 0
  · simp [hk]
  · simp [hk, Int.floor_intCast]

/-- Python's remainder schedule computes `a - b ⌊a / b⌋` in exact real arithmetic, with C's `fmod`
as `a - b * trunc (a / b)`. At `b = 0` both sides are `a`. -/
theorem remainderSchedule_real (a b : ℝ) :
    remainderSchedule
        (fun x y => x - y * (FloatLib.Floats.Formats.Flocq.truncRound (x / y) : ℝ)) a b =
      a - b * ⌊a / b⌋ := by
  obtain rfl | hb := eq_or_ne b 0
  · simp [remainderSchedule]
  have hFloor := Context.floor_eq_truncRound (a / b)
  obtain ⟨t, ht⟩ : ∃ t : ℤ, FloatLib.Floats.Formats.Flocq.truncRound (a / b) = t := ⟨_, rfl⟩
  rw [ht] at hFloor
  unfold remainderSchedule
  simp only [beq_iff_eq, ht]
  have hRemainder : a - b * (t : ℝ) = b * (a / b - t) := by
    field_simp
  have hShift : a - b * ((t : ℝ) - 1) = b * (a / b - t) + b := by
    field_simp
    ring
  simp only [hRemainder]
  rcases lt_trichotomy (a / b - t) 0 with hNeg | hZero | hPos
  · rw [hFloor.1 hNeg, Int.cast_sub, Int.cast_one, hShift]
    rcases lt_or_gt_of_ne hb with hbNeg | hbPos
    · have hm : 0 < b * (a / b - t) := mul_pos_of_neg_of_neg hbNeg hNeg
      simp [hbNeg, hm.ne', not_lt.mpr hm.le]
    · have hm : b * (a / b - t) < 0 := mul_neg_of_pos_of_neg hbPos hNeg
      simp [not_lt.mpr hbPos.le, hm.ne, hm]
  · rw [hFloor.2 hZero.ge, hRemainder]
    simp [hZero]
  · rw [hFloor.2 hPos.le, hRemainder]
    rcases lt_or_gt_of_ne hb with hbNeg | hbPos
    · have hm : b * (a / b - t) < 0 := mul_neg_of_neg_of_pos hbNeg hPos
      simp [hbNeg, hm, hm.ne]
    · have hm : 0 < b * (a / b - t) := mul_pos hbPos hPos
      simp [not_lt.mpr hbPos.le, not_lt.mpr hm.le, hm.ne']

/-- Over the reals, `remainder` is `x - y ⌊x / y⌋`, which is `x` at `y = 0`. -/
theorem Context.remainder_real (x y : ℝ) :
    Context.remainder x y = x - y * ⌊x / y⌋ :=
  remainderSchedule_real x y

/-- Over the reals, the floor of `|a| / |b|` with the sign of `a / b` is the truncated quotient
`FloatLib.Floats.Formats.Flocq.truncRound (a / b)`. -/
theorem Context.copysign_floor_abs_div_real (a b : ℝ) :
    Context.copysign (((⌊|a| / |b|⌋ : ℤ) : ℝ)) (a / b) =
      (FloatLib.Floats.Formats.Flocq.truncRound (a / b) : ℝ) := by
  show (if 0 > a / b then -|((⌊|a| / |b|⌋ : ℤ) : ℝ)| else |((⌊|a| / |b|⌋ : ℤ) : ℝ)|) = _
  rw [← abs_div]
  have hNonneg : (0 : ℝ) ≤ ((⌊|a / b|⌋ : ℤ) : ℝ) := by
    exact_mod_cast Int.floor_nonneg.mpr (abs_nonneg _)
  rw [abs_of_nonneg hNonneg]
  by_cases hq : 0 ≤ a / b
  · rw [ite_eq_right_of_eq_false _ _ (eq_false (not_lt.mpr hq)),
      FloatLib.Floats.Formats.Flocq.truncRound_eq_floor hq, abs_of_nonneg hq]
    rfl
  · have hNeg : a / b < 0 := lt_of_not_ge hq
    rw [ite_eq_left_of_eq_true _ _ (eq_true hNeg),
      FloatLib.Floats.Formats.Flocq.truncRound_eq_ceil hNeg.le, abs_of_neg hNeg, Int.floor_neg,
      Int.cast_neg, neg_neg]
    rfl

end TorchLean

namespace TorchLean.Tensor

/-- Over the reals, the flooring division of `|a|` by `|b|` with the sign of `a / b`, the forward
of `Op.div .trunc` (`Op.rule?_div_trunc`), is the truncated quotient. -/
theorem copysign_floorDivide_abs_real {s : Shape} (a b : Tensor ℝ s) :
    Tensor.copysign (Tensor.floorDivide (Tensor.abs a) (Tensor.abs b)) (a / b) =
      Tensor.trunc (a / b) := by
  have hZero : Context.copysign (0 : ℝ) = fun _ => 0 := by
    funext y
    show (if 0 > y then -|(0 : ℝ)| else |(0 : ℝ)|) = 0
    simp
  have hHalf : ((1 / 2 : Rat) : ℝ) = 1 / 2 := by norm_num
  apply Rep.ext
  intro c
  simp only [copysign_apply, floorDivide, zipWith_apply, trunc_apply, Tensor.div_apply, hZero,
    hHalf, Tensor.abs, Tensor.map_apply]
  exact (congrArg (fun t => Context.copysign t (a c / b c))
    (divFloorSchedule_real (MathFunctions.abs (a c)) (MathFunctions.abs (b c)))).trans
    (Context.copysign_floor_abs_div_real (a c) (b c))

end TorchLean.Tensor
