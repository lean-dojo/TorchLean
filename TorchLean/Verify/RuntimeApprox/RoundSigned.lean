/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConfiguredBinary.RoundSigned
public import FloatLib.Floats.Formats.BinaryInterchange.Transcendentals.CorrectlyRounded.Proof
public import FloatLib.Numerics.Exact.Elementary.Proof
public import FloatLib.Floats.Formats.BinaryInterchange.Algebraic.RootProof
public import FloatLib.Floats.Formats.Posit.Algebraic.Root.Proof

/-!
# Rounding of `roundSigned`

Let `c` compare a real number `y` with rationals, `c q = cmp y q` for every rational `q`. In
every format, `Elementary.roundSigned fmt c` is comparison rounding of `|y|` with the sign bit
set when `y < 0` (`roundSigned_eq_round`). In an IEEE format, below the overflow threshold `Θ` in
magnitude it is finite with the value `roundAt fmt y` (`roundSigned_eq_roundAt`), and at or above
`Θ` in magnitude it is the infinity of the sign of `y` (`once_above`). Both hold for every IEEE
format and every comparator of a real.

At finite inputs `once c` is `roundSigned` of `c` at the decoded inputs (`once_eq`), which are the
real values of the inputs (`rats_cast`). At a finite positive value FloatLib's `rsqrt` is
`roundSigned` of the comparator of `1 / √x` by exact squares (`rsqrt_eq_roundSigned`).

If `E` compares a real `e` with rationals, then `linCompare E c c₀` compares `c * e + c₀` with
zero (`linCompare_eq`) and `ratioCompare E α β γ δ` compares `(α * e + β) / (γ * e + δ)` with
rationals (`ratioCompare_eq`), for every real `e` and all rational coefficients. So
`expRatioCompare α β γ δ r` compares `(α * exp r + β) / (γ * exp r + δ)` with rationals for all
rational coefficients and exponents (`expRatioCompare_eq`).
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model

namespace Elementary

variable {fmt : FloatFormat}

/-- Attaching the sign of `y` to `|y|` gives `y`. -/
private theorem signed_abs (y : ℝ) : (if decide (y < 0) then -|y| else |y|) = y := by
  by_cases h : y < 0
  · simp [h, abs_of_neg h]
  · simp [h, abs_of_nonneg (not_lt.mp h)]

/-- In every format, the signed rounding of a comparator of `y` is comparison rounding of `|y|`
with the sign bit set when `y < 0`. -/
theorem roundSigned_eq_round {c : ℚ → Ordering} {y : ℝ} (hc : ∀ q : ℚ, c q = cmp y q) :
    roundSigned fmt c =
      AlgebraicRounding.round fmt (decide (y < 0)) fun b : ℚ => cmp |y| (b : ℝ) := by
  unfold roundSigned
  by_cases hneg : y < 0
  · have h0 : c 0 = .lt := by rw [hc, Rat.cast_zero, cmp_eq_lt_iff]; exact hneg
    rw [ite_eq_left h0, decide_eq_true hneg]
    congr 1
    funext b
    rw [hc, cmp_swap, Rat.cast_neg, abs_of_neg hneg]
    simp only [cmp, cmpUsing, neg_lt, lt_neg]
  · have h0 : ¬c 0 = .lt := by rw [hc, Rat.cast_zero, cmp_eq_lt_iff]; exact hneg
    rw [ite_eq_right h0, decide_eq_false hneg]
    congr 1
    funext b
    rw [hc, abs_of_nonneg (not_lt.mp hneg)]

/-- In an IEEE format and below the overflow threshold in magnitude, the signed rounding of a
comparator of `y` is finite and is the nearest-even rounding of `y`. -/
theorem roundSigned_eq_roundAt (hfmt : fmt.isIEEE = true) {c : ℚ → Ordering} {y : ℝ}
    (hc : ∀ q : ℚ, c q = cmp y q)
    (hy : |y| < Transcendentals.Certified.overflowThreshold fmt) :
    isFinite (roundSigned fmt c) = true ∧ toReal (roundSigned fmt c) = roundAt fmt y := by
  rw [Transcendentals.CorrectlyRounded.cast_overflowThreshold] at hy
  rw [roundSigned_eq_round hc]
  have hfin := AlgebraicRounding.isFinite_round_of_lt_overflow fmt hfmt (decide (y < 0)) _ |y|
    (fun _ => rfl) hy
  refine ⟨hfin, ?_⟩
  rw [AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ |y| (abs_nonneg y) (fun _ => rfl)
    hfin, signed_abs]

/-- In an IEEE format and at or above the overflow threshold in magnitude, the signed rounding of
a comparator of `y` is the infinity of the sign of `y`. -/
theorem once_above (hfmt : fmt.isIEEE = true) {c : ℚ → Ordering} {y : ℝ}
    (hc : ∀ q : ℚ, c q = cmp y q)
    (hy : (Transcendentals.Certified.overflowThreshold fmt : ℝ) ≤ |y|) :
    roundSigned fmt c = nativeOverflow fmt (decide (y < 0)) := by
  rw [roundSigned_eq_round hc]
  refine AlgebraicRounding.round_eq_nativeOverflow_of_isFinite_eq_false fmt hfmt _ _ ?_
  by_contra hfin
  rw [Bool.not_eq_false] at hfin
  have hval := AlgebraicRounding.toReal_round_eq_roundAt fmt hfmt _ _ |y| (abs_nonneg y)
    (fun _ => rfl) hfin
  rw [signed_abs] at hval
  exact absurd (Transcendentals.CorrectlyRounded.abs_lt_overflowThreshold_of_toReal_eq_roundAt
    hfmt hfin hval) (not_lt.mpr hy)

/-- A finite input decodes to the rational of its real value. -/
theorem rats_cast {n : ℕ} {x : Fin n → Model fmt} (hx : ∀ i, isFinite (x i) = true)
    (i : Fin n) : (rats x i : ℝ) = toReal (x i) := by
  obtain ⟨q, hq, hr⟩ := Algebraic.exists_toRat?_of_isFinite (hx i)
  rw [rats, hq, Option.getD_some, hr]

/-- At finite inputs, `once c` is `roundSigned` of the comparator at the decoded inputs. -/
theorem once_eq {n : ℕ} (c : (Fin n → ℚ) → ℚ → Ordering) {x : Fin n → Model fmt}
    (hx : ∀ i, isFinite (x i) = true) : once c fmt x = roundSigned fmt (c (rats x)) :=
  ite_eq_left hx

/-- At a finite positive value, `rsqrt` is the signed rounding through the comparator of
`1 / √x` by exact squares. -/
theorem rsqrt_eq_roundSigned {v : Model fmt} (hv : isFinite v = true) (hpos : 0 < toReal v) :
    rsqrt v =
      roundSigned fmt (Formats.Posit.Model.RootRounding.compareRoot ((toRat? v).getD 0)⁻¹ 2) := by
  have hzero : isZero v = false := by
    cases h : isZero v
    · rfl
    · exact absurd (toReal_eq_zero_of_isZero v h) hpos.ne'
  have hsign : signBit v = false :=
    (isZero_or_signBit_eq_false_of_toReal_nonneg v hv hpos.le).resolve_left (by simp [hzero])
  obtain ⟨q, hq, hr⟩ := Algebraic.exists_toRat?_of_isFinite hv
  have hq0 : 0 < q := by
    have : (0 : ℝ) < q := hr ▸ hpos
    exact_mod_cast this
  have heq : rsqrt v = AlgebraicRounding.sqrtRat fmt false q⁻¹ := by
    simp [rsqrt, chooseNaN1_none_of_isFinite v hv, hzero, hsign,
      isInf_eq_false_of_isFinite_eq_true v hv, hq]
  rw [heq, AlgebraicRounding.sqrtRat_eq_root, hq, Option.getD_some, roundSigned, ite_eq_right]
  · rfl
  · rw [Formats.Posit.Model.RootRounding.compareRoot, ite_eq_right (lt_irrefl 0), ite_eq_left rfl,
      cmp_eq_lt_iff, not_lt]
    exact (inv_pos.mpr hq0).le

/-- Two comparisons agree when their strict orders do. -/
private theorem cmp_congr {α β : Type*} [LinearOrder α] [LinearOrder β] {a b : α} {a' b' : β}
    (h₁ : a < b ↔ a' < b') (h₂ : b < a ↔ b' < a') : cmp a b = cmp a' b' := by
  simp only [cmp, cmpUsing, h₁, h₂]

/-- `linCompare E c c₀` compares `c * e + c₀` with zero when `E` compares `e` with rationals. -/
theorem linCompare_eq {E : ℚ → Ordering} {e : ℝ} (hE : ∀ b : ℚ, E b = cmp e b) (c c₀ : ℚ) :
    linCompare E c c₀ = cmp ((c : ℝ) * e + c₀) 0 := by
  unfold linCompare
  by_cases h0 : c = 0
  · rw [ite_eq_left h0, h0, Rat.cast_zero, zero_mul, zero_add]
    exact cmp_congr Rat.cast_lt_zero.symm Rat.cast_pos.symm
  rw [ite_eq_right h0, hE]
  have hc : (c : ℝ) ≠ 0 := Rat.cast_ne_zero.mpr h0
  have hb : (c : ℝ) * ((-c₀ / c : ℚ) : ℝ) = -c₀ := by
    push_cast
    field_simp
  by_cases hp : 0 < c
  · have hc' : (0 : ℝ) < c := Rat.cast_pos.mpr hp
    rw [ite_eq_left hp]
    refine cmp_congr ⟨fun h => ?_, fun h => ?_⟩ ⟨fun h => ?_, fun h => ?_⟩
    · nlinarith
    · by_contra h'
      nlinarith
    · nlinarith
    · by_contra h'
      nlinarith
  · have hc' : (c : ℝ) < 0 := Rat.cast_lt_zero.mpr (lt_of_le_of_ne (not_lt.mp hp) h0)
    rw [ite_eq_right hp, cmp_swap]
    refine cmp_congr ⟨fun h => ?_, fun h => ?_⟩ ⟨fun h => ?_, fun h => ?_⟩
    · nlinarith
    · by_contra h'
      nlinarith
    · nlinarith
    · by_contra h'
      nlinarith

/-- `ratioCompare E α β γ δ` compares `(α * e + β) / (γ * e + δ)` with rationals when `E` compares
`e` with rationals. -/
theorem ratioCompare_eq {E : ℚ → Ordering} {e : ℝ} (hE : ∀ b : ℚ, E b = cmp e b)
    (α β γ δ q : ℚ) :
    ratioCompare E α β γ δ q = cmp (((α : ℝ) * e + β) / ((γ : ℝ) * e + δ)) q := by
  unfold ratioCompare
  rw [linCompare_eq hE, linCompare_eq hE]
  have hN : ((α - q * γ : ℚ) : ℝ) * e + ((β - q * δ : ℚ) : ℝ) =
      ((α : ℝ) * e + β) - q * ((γ : ℝ) * e + δ) := by
    push_cast
    ring
  rw [hN]
  rcases lt_trichotomy ((γ : ℝ) * e + δ) 0 with hD | hD | hD
  · rw [hD.cmp_eq_lt, cmp_swap]
    refine cmp_congr ?_ ?_
    · rw [div_lt_iff_of_neg hD, sub_pos]
    · rw [lt_div_iff_of_neg hD, sub_neg]
  · rw [hD, cmp_self_eq_eq, div_zero]
    exact cmp_congr Rat.cast_pos.symm Rat.cast_lt_zero.symm
  · rw [hD.cmp_eq_gt]
    refine cmp_congr ?_ ?_
    · rw [div_lt_iff₀ hD, sub_neg]
    · rw [lt_div_iff₀ hD, sub_pos]

/-- `expRatioCompare α β γ δ r` compares `(α * exp r + β) / (γ * exp r + δ)` with rationals. -/
theorem expRatioCompare_eq (α β γ δ r q : ℚ) :
    expRatioCompare α β γ δ r q =
      cmp (((α : ℝ) * Real.exp r + β) / ((γ : ℝ) * Real.exp r + δ)) q :=
  ratioCompare_eq (FloatLib.Numerics.ElementaryComparison.prepareExp_eq_real r 16) α β γ δ q

end Elementary

end TorchLean
