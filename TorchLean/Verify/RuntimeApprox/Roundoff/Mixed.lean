/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError.Rounding
public import FloatLib.Numerics.MixedError.Elementary
public import TorchLean.Verify.RuntimeApprox.Roundoff.Recovery
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Layers.Loss

/-!
# Mixed-error steps for TorchLean's scalars

FloatLib's `roundoff` composes one local rounding theorem per configured operation with the
propagation lemmas for exact real operations. The analysis takes the steps that belong to
TorchLean's scalar instances from here as rules: the rounding of a natural number cast into the
format and of a division by a count, the finiteness of the default safeguard and the positivity of
the counts that means divide by.
-/

@[expose] public section

open TorchLean.Shape (rank)

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics

/-- The slice count of a mean reduction is positive as a real number. -/
theorem cast_axisMeanDenom_pos (s : Shape) (dim : Fin s.rank) :
    (0 : ℝ) < Tensor.axisMeanDenom s dim :=
  Nat.cast_pos.mpr (Tensor.axisMeanDenom_pos s dim)

/-- The element count of a mean, taken as one on an empty shape, is positive as a real number. -/
theorem cast_meanDenominator_pos (s : Shape) :
    (0 : ℝ) < Tensor.meanDenominator s :=
  Nat.cast_pos.mpr (Tensor.meanDenominator_pos s)

/-- A count taken as one when it is zero is positive as a real number. -/
theorem cast_ite_eq_zero_one_pos (n : ℕ) : (0 : ℝ) < ((if n = 0 then 1 else n : ℕ) : ℝ) :=
  Nat.cast_pos.mpr (by split <;> omega)

/-- The extent of a nonempty axis is positive as a real number. -/
theorem cast_axisSize_pos {axis : ℕ} {s : Shape} (h : Shape.NonemptyAxis axis s) :
    (0 : ℝ) < s.axisSize axis (h := h.axisInBounds) := by
  refine Nat.cast_pos.mpr ?_
  induction h with
  | zero => exact Nat.succ_pos _
  | succ _ ih => exact ih

section Configured

open ExecFloat.Binary

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]
    {x₀ r a : ℝ}

local notation "Value" =>
  ExecFloat.Binary format.expWidth format.fracWidth format.encoding format.exponentBias
    format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
    format.exponentBias_le_maxFinite plan code

section Casts

/-- A natural number at most the largest finite value converts to a finite configured value. -/
theorem isFinite_natCast_of_le (hformat : format.isIEEE = true) {n : ℕ}
    (hrange : (n : ℝ) ≤ Model.toReal (Model.posMaxFinite format)) :
    isFinite (n : ExecFloat (Configured.Family format code plan)) = true := by
  change Model.isFinite (toModel (ofModel (Model.ofRat format (n : ℚ)))) = true
  rw [toModel_ofModel]
  exact Model.isFinite_ofRat _ hformat (by rwa [Rat.cast_natCast, Nat.abs_cast])

/-- A natural number converted to a finite configured value has relative error `u` and no
absolute error. -/
theorem natCast_rounding (hformat : format.isIEEE = true) {n : ℕ}
    (hfinite : isFinite (n : ExecFloat (Configured.Family format code plan)) = true) :
    MixedError (toReal (n : ExecFloat (Configured.Family format code plan))) n
      (Model.unitRoundoffAt format) 0 :=
  ofRat_natCast_rounding (plan := plan) (code := code) n hfinite hformat

/-- The configured cast of the natural number one is exactly one. -/
theorem toReal_natCast_one (hformat : format.isIEEE = true) :
    toReal ((1 : ℕ) : ExecFloat (Configured.Family format code plan)) = 1 := by
  change Model.toReal (toModel (ofModel (Model.ofRat format ((1 : ℕ) : ℚ)))) = 1
  rw [Nat.cast_one]
  exact toReal_one

/-- The default safeguard is finite in an IEEE format: it is `1e-6` rounded once, or the least
positive subnormal when that rounding gives zero. -/
theorem isFinite_defaultEpsilon (hformat : format.isIEEE = true) :
    isFinite (Context.defaultEpsilon : ExecFloat (Configured.Family format code plan)) = true := by
  change isFinite (if isZero (ofModel (Model.ofRat format (1 / 1000000 : ℚ)) :
      ExecFloat (Configured.Family format code plan)) then ofNatBits 1
    else ofModel (Model.ofRat format (1 / 1000000 : ℚ))) = true
  split
  · exact isFinite_ofNatBits_one
  · change Model.isFinite (toModel (ofModel (Model.ofRat format (1 / 1000000 : ℚ)) :
      ExecFloat (Configured.Family format code plan))) = true
    rw [toModel_ofModel]
    exact Model.isFinite_ofRat _ hformat
      (le_trans (by norm_num) (Model.one_le_toReal_posMaxFinite format))

end Casts

section CountDivision

/-- A finite value `x` with rational value `q`, divided by a nonzero count `n`, is `x` when `q` is
zero and `q / n` rounded once otherwise. -/
theorem divNat_eq_of_toRat? {x : ExecFloat (Configured.Family format code plan)} {n : ℕ} {q : ℚ}
    (hn : n ≠ 0) (hq : toRat? x = some q) :
    Context.divNat x n = if q = 0 then x else ofRat (q / n) := by
  change ConfiguredBinary.divNat x n = _
  unfold ConfiguredBinary.divNat
  simp only [hn, ↓reduceIte, hq]

/-- A value without a rational value, an infinity or a NaN, divided by a nonzero count is that
value divided by one. -/
theorem divNat_eq_of_toRat?_eq_none {x : ExecFloat (Configured.Family format code plan)} {n : ℕ}
    (hn : n ≠ 0) (hq : toRat? x = none) :
    Context.divNat x n = x / ofModel (Model.ofRat format 1) := by
  change ConfiguredBinary.divNat x n = _
  unfold ConfiguredBinary.divNat
  simp only [hn, ↓reduceIte, hq]

/-- A finite value divided by a nonzero count is finite in an IEEE format. The analysis tries this
finiteness rule after `isFinite_divNat_of_mixedError`, for an operand whose exact value it cannot
enclose, such as an opaque function of a value. -/
theorem isFinite_divNat (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)} {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (hx : isFinite x = true) : isFinite (Context.divNat x n) = true := by
  obtain ⟨q, hq, hreal⟩ := Model.Algebraic.exists_toRat?_of_isFinite hx
  have hq' : toRat? x = some q := hq
  have hxq : toReal x = q := hreal
  have hn : 0 < n := Nat.pos_of_ne_zero (Nat.cast_ne_zero.mp hn)
  rw [divNat_eq_of_toRat? hn.ne' hq']
  split
  · exact hx
  · refine isFinite_ofRat (plan := plan) (code := code) (q / n) ?_ hformat
    have hbound := abs_toReal_le_posMaxFinite_of_isFinite hx
    rw [hxq] at hbound
    have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr hn.ne'
    rw [Rat.cast_div, Rat.cast_natCast, abs_div, Nat.abs_cast]
    exact (div_le_self (abs_nonneg _) hn1).trans hbound

/-- The finiteness rule of a quotient by a nonzero count: a finite operand gives a finite
quotient, since `|x / n| ≤ |x|`, so no range condition enters. The proof ignores the enclosure of
the exact quotient. The enclosure is a premise so that the analysis derives it first, analyzing
the operand on the way, and then reads the operand's finiteness off that work, as for the range
theorems of the configured operations. -/
theorem isFinite_divNat_of_mixedError (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)} {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (hx : isFinite x = true) (_bound : MixedError (toReal x / n) x₀ r a) :
    isFinite (Context.divNat x n) = true :=
  isFinite_divNat hformat hn hx

/-- A finite value divided by a nonzero count is the exact quotient rounded once. -/
theorem toReal_divNat (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)} {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (hx : isFinite x = true) :
    toReal (Context.divNat x n) = Model.roundAt format (toReal x / n) := by
  obtain ⟨q, hq, hreal⟩ := Model.Algebraic.exists_toRat?_of_isFinite hx
  have hq' : toRat? x = some q := hq
  have hxq : toReal x = q := hreal
  have hfinite := isFinite_divNat hformat hn hx
  rw [divNat_eq_of_toRat? (Nat.cast_ne_zero.mp hn) hq'] at hfinite ⊢
  by_cases hzero : q = 0
  · subst hzero
    simp only [↓reduceIte]
    rw [hxq]
    simp [Model.roundAt_zero]
  · simp only [hzero, ↓reduceIte] at hfinite ⊢
    rw [toReal_ofRat_eq_roundAt _ hfinite hformat, hxq, Rat.cast_div, Rat.cast_natCast]

/-- The operand of a finite quotient by a nonzero count is finite in an IEEE format: a value
without a rational value is divided by one, which keeps an infinity or a NaN. -/
theorem isFinite_of_isFinite_divNat (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)} {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (h : isFinite (Context.divNat x n) = true) : isFinite x = true := by
  cases hq : (toRat? x) with
  | some q => exact Model.Algebraic.isFinite_of_toRat?_eq_some hq
  | none =>
    rw [divNat_eq_of_toRat?_eq_none (Nat.cast_ne_zero.mp hn) hq] at h
    exact Model.isFinite_left_of_isFinite_div hformat (by simpa [isFinite] using h)

/-- A finite quotient by a nonzero count is within half an ulp of the exact quotient. The premise
is the quotient's finiteness, and the operand's follows from it (`isFinite_of_isFinite_divNat`). -/
theorem abs_toReal_divNat_sub_le (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)} {n : ℕ} (hn : (n : ℝ) ≠ 0)
    (hq : isFinite (Context.divNat x n) = true) :
    |toReal (Context.divNat x n) - toReal x / n| ≤ Model.halfUlpAt format (toReal x / n) := by
  rw [toReal_divNat hformat hn (isFinite_of_isFinite_divNat hformat hn hq)]
  exact Model.abs_roundAt_sub_le format _

/-- A finite quotient by a positive count has relative error `u` and absolute error the
underflow error against the exact quotient. As for the configured division, the premise is the
quotient's finiteness, which the analysis proves first, through `isFinite_divNat_of_mixedError`.
Where no rule proves the count positive, such as for the input cells of a pooling window, the
analysis uses `isFinite_divNat_and_rounding`. -/
theorem divNat_rounding (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)} {n : ℕ} (hn : (0 : ℝ) < n)
    (hq : isFinite (Context.divNat x n) = true) :
    MixedError (toReal (Context.divNat x n)) (toReal x / n)
      (Model.unitRoundoffAt format) (Model.underflowErrorAt format) :=
  mixedError_toReal_of_eq_roundAt
    (toReal_divNat hformat hn.ne' (isFinite_of_isFinite_divNat hformat hn.ne' hq))

/-- A real `y` with `u * |y| < |y|` is nonzero: `y = 0` makes it `0 < 0`. A count with the margin
of the rounding of its quotient is therefore nonzero. -/
theorem ne_zero_of_mul_abs_lt_abs {u y : ℝ} (h : u * |y| < |y|) : y ≠ 0 := by
  rintro rfl
  simp at h

/-- A finite value divided by a count `n` with `u * |n| < |n|` is finite, and its error against
the reference `z` of the operand's exact quotient composes the rounding of `toReal x / n`
(relative `u`, absolute the underflow error) with the error of that quotient against `z`. The
condition is the one the division by the cast count needs to exclude a zero divisor, and it fails
at `n = 0`. Because the quotient's enclosure is an input, the analysis encloses the operand before
it needs the operand's finiteness. The condition is left as a premise where no rule proves the
count positive and `divNat_rounding` does not apply, such as the count of the cells of an
average-pool window that excludes padding. -/
theorem isFinite_divNat_and_rounding (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)} {n : ℕ} {z r a : ℝ}
    (h : MixedError (toReal x / n) z r a)
    (hn : Model.unitRoundoffAt format * |(n : ℝ)| < |(n : ℝ)|) (hx : isFinite x = true) :
    isFinite (Context.divNat x n) = true ∧
      MixedError (toReal (Context.divNat x n)) z
        ((1 + Model.unitRoundoffAt format) * r + Model.unitRoundoffAt format)
        ((1 + Model.unitRoundoffAt format) * a + Model.underflowErrorAt format) := by
  have hcount := ne_zero_of_mul_abs_lt_abs hn
  exact ⟨isFinite_divNat hformat hcount hx,
    (mixedError_toReal_of_eq_roundAt (toReal_divNat hformat hcount hx)).trans h⟩

/-- A finite quotient by the count one is exact. The premise is the quotient's finiteness, which
the analysis proves after enclosing the operand, and the operand's finiteness follows from it. -/
theorem divNat_one_rounding (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)}
    (hq : isFinite (Context.divNat x 1) = true) :
    MixedError (toReal (Context.divNat x 1)) (toReal x) 0 0 := by
  have hx := isFinite_of_isFinite_divNat hformat (by norm_num) hq
  refine MixedError.of_eq ?_
  rw [toReal_divNat hformat (by norm_num) hx, Nat.cast_one, div_one]
  exact Model.roundAt_toReal_eq (toModel x) hx

/-- A finite quotient by the slice count of a vector, which is one, is exact. -/
theorem divNat_axisMeanDenom_rounding (hformat : format.isIEEE = true)
    {x : ExecFloat (Configured.Family format code plan)} {k : ℕ} (h : 0 < Shape.rank [k])
    (hq : isFinite (Context.divNat x (Tensor.axisMeanDenom [k] ⟨0, h⟩)) = true) :
    MixedError (toReal (Context.divNat x (Tensor.axisMeanDenom [k] ⟨0, h⟩))) (toReal x) 0 0 :=
  divNat_one_rounding hformat hq

end CountDivision

end Configured

end TorchLean.Roundoff
