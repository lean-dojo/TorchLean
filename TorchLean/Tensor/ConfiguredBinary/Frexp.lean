/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Numeric.Frexp
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
import FloatLib.Floats.Formats.BinaryInterchange.Configured.Operations.Proof
import FloatLib.Floats.Formats.BinaryInterchange.Operations.Proof.Exponent
import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Exact
import FloatLib.Floats.Formats.BinaryInterchange.Analysis.StandardModel

/-!
# C's `frexp` in a configured binary format

We wrote `ConfiguredBinary.frexp` by hand from C's `frexp` (C99 7.12.6.4, WG14 N1256) and the
`logB` and `scaleB` of IEEE 754-2019 (clause 5.3.3). At a finite nonzero `x` the exponent is
`e = logB x + 1` and the mantissa `scaleB x (-e)`, so a subnormal `x` gets a normalized mantissa,
as in C. Elsewhere the exponent is `0` and the mantissa `scaleB x 0`: `x` itself at a zero or an
infinity, and the quiet NaN of `x` at a NaN. Annex F (F.9.3.4) fixes the zeros and leaves the
exponent unspecified at an infinity or a NaN. Glibc stores `0` there and returns `x + x`, which
quiets a signaling NaN, and `torch.frexp` (PyTorch 2.13) returns those results on the CPU.

The mantissa is exact only where the format holds every normalized mantissa `m`, `1/2 ≤ |m| < 1`,
of its values. Its lowest bit then has exponent `-(fracWidth + 1)`, which is at least the least
subnormal exponent `1 - bias - fracWidth` exactly when the exponent bias is at least two. The
minimal default-bias format, with exponent width two and bias one, has `3 = 0.11₂ · 2²` but not
`0.11₂`, so its rounded mantissa of `3` would be `1`. FloatLib proves that a representable dyadic
rounds to itself only in an IEEE format (`FloatFormat.isIEEE`: the IEEE encoding with the bias
`2 ^ (expWidth - 1) - 1`). A configured format therefore has `Frexp`
(`configuredBinaryFrexpCapability`) only in an IEEE format of bias at least two, that is of
exponent width at least three, and only where the exponents of its finite values fit in the int32
word of `Op.frexp`, so never at exponent width 32 or more. Another encoding or bias has none, even
where its mantissas are exact, as at exponent width two and fraction width one with bias two.

Proved here, in those formats, for every finite `x`: the exponent is the real `frexpExponent` of
`x` and the mantissa is finite with the real value `frexpMantissa x` (`frexp_of_isFinite`), so `x`
is the mantissa times `2 ^ e` (`toReal_frexp_mul`) and a nonzero mantissa lies in `[1/2, 1)` in
absolute value (`abs_toReal_frexp_mem`). In every encoding the exponent of a finite nonzero `x`
lies in `[minSubnormalExponent + 1, maxNormalExponent + 1]` (`frexpExponent_mem_of_isFinite`),
`[-1073, 1024]` at binary64. In an IEEE format of bias at least two where that range lies in
`[-2³¹, 2³¹)`, the int32 word that `Op.frexp` writes, `BitVec.ofInt 32 (frexp x).2`, does not wrap,
and the mantissa times two to its signed value is `x` (`toReal_frexp_mul_toInt`).
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

namespace ConfiguredBinary

/-- C's `frexp` in the format: the exponent `e`, which is `logB x + 1` at a finite nonzero `x` and
`0` elsewhere, and the mantissa `scaleB x (-e)` rounded to the format. It is C's `frexp` where that
rounding is exact, in an IEEE format of exponent bias at least two (`frexp_of_isFinite`). Elsewhere
it need not be: in the IEEE format of exponent width two and fraction width one, of bias one, `3`
gives the mantissa `1` and the exponent `2`. -/
def frexp (x : Value) : Value × ℤ :=
  let e := if ExecFloat.Binary.isFinite x && !ExecFloat.Binary.isZero x then
    (ExecFloat.Binary.logBInt x).value + 1 else 0
  (ExecFloat.Binary.scale x (-e), e)

private theorem toReal_shift (d : FloatLib.Numerics.Dyadic) (n : ℤ) :
    ({ d with exponent := d.exponent + n } : FloatLib.Numerics.Dyadic).toReal =
      d.toReal * 2 ^ n := by
  simp only [FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand,
    zpow_add₀ (two_ne_zero' ℝ)]
  ring

/-- In an IEEE format with an exponent bias of at least two, at a finite `x`, the exponent of
`frexp x` is the real `frexpExponent` of `x`, and its mantissa is finite with the value
`frexpMantissa x`: the scaling is exact. -/
theorem frexp_of_isFinite (hformat : format.isIEEE = true) (hbias : 2 ≤ format.exponentBias)
    {x : Value} (hx : ExecFloat.Binary.isFinite x = true) :
    (frexp x).2 = frexpExponent (ExecFloat.Binary.toReal x) ∧
      ExecFloat.Binary.isFinite (frexp x).1 = true ∧
      ExecFloat.Binary.toReal (frexp x).1 = frexpMantissa (ExecFloat.Binary.toReal x) := by
  obtain ⟨d, hd⟩ := Model.exists_toDyadic?_of_isFinite hx
  have hvalue := Model.exactValue_eq_finite_iff.mpr hd
  have hreal : ExecFloat.Binary.toReal x = d.toReal := by
    simp [ExecFloat.Binary.toReal, Model.toReal, Model.toReal?, hd]
  have hsig := Model.toDyadic?_significand_lt _ hd
  have hmin := Model.minSubnormalExponent_le_toDyadic? _ hd
  have hexp : (frexp x).2 = frexpExponent (ExecFloat.Binary.toReal x) ∧
      format.minSubnormalExponent ≤ d.exponent + -(frexp x).2 := by
    by_cases hs : d.significand = 0
    · have hzero : ExecFloat.Binary.toReal x = 0 := by
        rw [hreal, Model.Dyadic.toReal_eq_zero_iff]; exact hs
      have hz : ExecFloat.Binary.isZero x = true :=
        (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hx).2 hzero
      simp [frexp, hx, hz, hzero, frexpExponent, hmin]
    · have hnz : ExecFloat.Binary.isZero x = false := by
        cases h : ExecFloat.Binary.isZero x
        · rfl
        · have := (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hx).1 h
          rw [hreal, Model.Dyadic.toReal_eq_zero_iff] at this
          exact absurd this hs
      have hbin := Model.logBInt_of_finite_nonzero _ d hvalue hs
      have hbounds := Model.logBInt_bounds _ d hvalue hs
      rw [hbin] at hbounds
      have hcast := And.intro ((Rat.cast_le (K := ℝ)).2 hbounds.1)
        ((Rat.cast_lt (K := ℝ)).2 hbounds.2)
      push_cast at hcast
      have he : (frexp x).2 = (d.significand.log2 : ℤ) + d.exponent + 1 := by
        simp [frexp, hx, hnz, ExecFloat.Binary.logBInt_eq, hbin]
      have habs : |d.toReal| = (d.significand : ℝ) * 2 ^ d.exponent := by
        cases hneg : d.negative <;>
          simp [FloatLib.Numerics.Dyadic.toReal, FloatLib.Numerics.Dyadic.signedSignificand, hneg,
            abs_mul, abs_of_pos (zpow_pos two_pos _ : (0 : ℝ) < 2 ^ d.exponent)]
      have hlow : (2 : ℝ) ^ ((d.significand.log2 : ℤ) + d.exponent) ≤ |d.toReal| := by
        rw [habs]; exact hcast.1
      have hhigh : |d.toReal| < (2 : ℝ) ^ ((d.significand.log2 : ℤ) + d.exponent + 1) := by
        rw [habs]; exact hcast.2
      have hmem := Set.mem_Icc.1 (frexpExponent_mem_Icc hlow hhigh)
      have hlog : d.significand.log2 < format.fracWidth + 1 := (Nat.log2_lt hs).2 hsig
      refine ⟨by rw [he, hreal]; omega, ?_⟩
      rw [he]
      simp only [FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
        Int.ofNat_eq_natCast]
      omega
  refine ⟨hexp.1, ?_⟩
  have hscale : ExecFloat.Binary.toModel (frexp x).1 =
      Model.Internal.Directed.roundDyadic format .nearest
        { d with exponent := d.exponent + -(frexp x).2 } := by
    show ExecFloat.Binary.toModel (ExecFloat.Binary.scale x (-(frexp x).2)) = _
    rw [ExecFloat.Binary.toModel_scale, ← Model.value_scale,
      Model.scale_status_of_finite _ _ _ d hvalue]
  have hmantissa :
      ({ d with exponent := d.exponent + -(frexp x).2 } : FloatLib.Numerics.Dyadic).toReal =
        frexpMantissa (ExecFloat.Binary.toReal x) := by
    rw [toReal_shift, frexpMantissa, ← hexp.1, hreal, zpow_neg, div_eq_mul_inv]
  have hone : |frexpMantissa (ExecFloat.Binary.toReal x)| ≤ 1 := by
    by_cases h : ExecFloat.Binary.toReal x = 0
    · simp [frexpMantissa, h]
    · exact (frexpMantissa_mem h).2.le
  have hexact := Model.Internal.Directed.roundDyadic_of_representable format hformat .nearest
    { d with exponent := d.exponent + -(frexp x).2 } hsig hexp.2
    (by rw [hmantissa]; exact hone.trans (Model.one_le_toReal_posMaxFinite format))
  refine ⟨?_, ?_⟩
  · show Model.isFinite (ExecFloat.Binary.toModel (frexp x).1) = true
    rw [hscale]; exact hexact.1
  · show Model.toReal (ExecFloat.Binary.toModel (frexp x).1) = _
    rw [hscale, hexact.2, hmantissa]

/-- A finite `x` is its mantissa times two to its exponent. -/
theorem toReal_frexp_mul (hformat : format.isIEEE = true) (hbias : 2 ≤ format.exponentBias)
    {x : Value} (hx : ExecFloat.Binary.isFinite x = true) :
    ExecFloat.Binary.toReal (frexp x).1 * 2 ^ (frexp x).2 = ExecFloat.Binary.toReal x := by
  obtain ⟨he, -, hm⟩ := frexp_of_isFinite hformat hbias hx
  rw [hm, he, ← frexp_reconstruct]

/-- The mantissa of a finite nonzero `x` lies in `[1/2, 1)` in absolute value. -/
theorem abs_toReal_frexp_mem (hformat : format.isIEEE = true) (hbias : 2 ≤ format.exponentBias)
    {x : Value} (hx : ExecFloat.Binary.isFinite x = true) (hzero : ExecFloat.Binary.toReal x ≠ 0) :
    1 / 2 ≤ |ExecFloat.Binary.toReal (frexp x).1| ∧ |ExecFloat.Binary.toReal (frexp x).1| < 1 := by
  rw [(frexp_of_isFinite hformat hbias hx).2.2]
  exact frexpMantissa_mem hzero

/-- In every encoding, a finite nonzero `x` lies in `[2 ^ minSubnormalExponent,
2 ^ (maxNormalExponent + 1))` in absolute value, so its exponent lies in
`[minSubnormalExponent + 1, maxNormalExponent + 1]`. -/
theorem frexpExponent_mem_of_isFinite {x : Value} (hx : ExecFloat.Binary.isFinite x = true)
    (hzero : ExecFloat.Binary.toReal x ≠ 0) :
    frexpExponent (ExecFloat.Binary.toReal x) ∈
      Set.Icc (format.minSubnormalExponent + 1) (format.maxNormalExponent + 1) := by
  obtain ⟨d, hd⟩ := Model.exists_toDyadic?_of_isFinite hx
  have hreal : ExecFloat.Binary.toReal x = d.toReal := by
    simp [ExecFloat.Binary.toReal, Model.toReal, Model.toReal?, hd]
  have hlow := Model.bpow_minSubnormalExponent_le_abs_toReal hx hzero
  have hhigh := Model.abs_toReal_lt_bpow_of_toDyadic?_some _ hd
  simp only [Model.bpow, Formats.Flocq.bpow_binaryRadix] at hlow hhigh
  rw [← hreal] at hhigh
  exact frexpExponent_mem_Icc hlow hhigh

/-- In an IEEE format of exponent bias at least two whose exponents of finite nonzero values,
`[minSubnormalExponent + 1, maxNormalExponent + 1]`, lie in `[-2³¹, 2³¹)`, the int32 word of the
exponent, `BitVec.ofInt 32 (frexp x).2` as `Op.frexp` writes it, does not wrap, and a finite `x`
is its mantissa times two to the word's signed value. The range alone is not enough: in the IEEE
format of exponent width two and fraction width one, of bias one, the exponent `2` of `3` fits, but
its mantissa `1` rebuilds `4`. -/
theorem toReal_frexp_mul_toInt (hformat : format.isIEEE = true) (hbias : 2 ≤ format.exponentBias)
    (hlow : -2 ^ 31 ≤ format.minSubnormalExponent + 1)
    (hhigh : format.maxNormalExponent + 1 < 2 ^ 31) {x : Value}
    (hx : ExecFloat.Binary.isFinite x = true) :
    ExecFloat.Binary.toReal (frexp x).1 * 2 ^ (BitVec.ofInt 32 (frexp x).2).toInt =
      ExecFloat.Binary.toReal x := by
  have hrange : -2 ^ 31 ≤ (frexp x).2 ∧ (frexp x).2 < 2 ^ 31 := by
    rw [(frexp_of_isFinite hformat hbias hx).1]
    by_cases hzero : ExecFloat.Binary.toReal x = 0
    · simp [frexpExponent, hzero]
    · have hmem := frexpExponent_mem_of_isFinite hx hzero
      exact ⟨hlow.trans hmem.1, hmem.2.trans_lt hhigh⟩
  rw [BitVec.toInt_ofInt_eq_self (by norm_num) hrange.1 hrange.2]
  exact toReal_frexp_mul hformat hbias hx

end ConfiguredBinary

/-- C's `frexp` (`ConfiguredBinary.frexp`) and `ldexp` (FloatLib's `scaleB`, `x 2ⁿ` rounded once
to nearest even), in an IEEE format with an exponent bias of at least two, where the mantissa is
exact (`ConfiguredBinary.frexp_of_isFinite`), whose exponents of finite nonzero values,
`[minSubnormalExponent + 1, maxNormalExponent + 1]`, lie in `[-2³¹, 2³¹)`, so that the int32 word
`Op.frexp` writes does not wrap (`ConfiguredBinary.toReal_frexp_mul_toInt`), and `none` in every
other format. The IEEE format of exponent width 32 has the exponent `2³¹` at its largest finite
value, so it has none. -/
instance configuredBinaryFrexpCapability : Op.Capability Frexp Value :=
  ⟨if format.isIEEE && decide (2 ≤ format.exponentBias) &&
      decide (-2 ^ 31 ≤ format.minSubnormalExponent + 1) &&
      decide (format.maxNormalExponent + 1 < 2 ^ 31) then
    some ⟨ConfiguredBinary.frexp, fun x n => ExecFloat.Binary.scale x n⟩
  else none⟩

end TorchLean
