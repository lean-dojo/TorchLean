/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Error
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Positive

/-!
# `reduce_precision` rounds once

The JAX import (`python/torchlean/jaxpr.py`, `_Builder.reduce_precision`) rounds an operand `x` of
a format `fmt` to `m` mantissa bits, where `m` is below the format's fraction width, to nearest,
ties to even. Where `exponent_bits` is narrower than the format's exponent field, a range step of
selections follows the rounding. Each conversion is the IR's `Binary.cast`, which is FloatLib's
`Model.cast` (`Binary.cast_value`).

For any formats `wide` and `narrow`, with `wide` in IEEE's encoding and holding `narrow`'s grid
and range, `toReal_cast_cast` proves that the conversion to `narrow` and back gives `narrow`'s
rounding of the value. An operand of binary16, bfloat16, binary32 or binary64, each an IEEE
format, is converted to `withFraction fmt m` (its exponent field, bias and encoding, `m` fraction
bits) and back. `toReal_reduce_ieee` proves that for every IEEE format, every such `m` and every
finite `x` whose first conversion does not overflow, the result is finite and is
`roundAt (withFraction fmt m) (toReal x)`, one rounding.

A float8 operand takes four conversions: to binary32, to `reduced fmt m` (one more exponent bit
than `fmt`, its bias, `m` fraction bits and IEEE's encoding), back to binary32 and back to `fmt`.
`toReal_cast_of_generic` proves that converting a value on a format's grid to that format is
exact. `toReal_reduce` composes the two with an exact first conversion: for every finite `x` of
any format that the premises admit, a finite result of the four conversions is
`roundAt narrow (toReal x)`. The range step selects the value, a signed zero or a signed
infinity, so `toReal_cast_of_generic` also converts its finite results back exactly.
`toReal_reduce_float8` is the import's case: for each of the six float8 settings in `settings`
and every finite word, the premises hold. Its premise that the conversion to `reduced fmt m` is
finite is checked by `decide +kernel` at every word, since FloatLib proves that bound only for
IEEE formats, whose bias is the standard one.
-/

@[expose] public section

namespace TorchLean.IR.ReducePrecision

open FloatLib.Floats.Formats
open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Model
open FloatLib.Numerics

/-- The format of `fmt`'s exponent field and one more bit, `fmt`'s bias, `m` fraction bits and
IEEE's encoding. -/
def reduced (fmt : FloatFormat) (m : Nat) (hm : 0 < m := by decide) : FloatFormat where
  expWidth := fmt.expWidth + 1
  fracWidth := m
  exponentBias := fmt.exponentBias
  encoding := .ieee
  expWidth_ge_two := by have := fmt.expWidth_ge_two; omega
  fracWidth_pos := hm
  exponentBias_pos := fmt.exponentBias_pos
  exponentBias_le_maxFinite := by
    show fmt.exponentBias ≤ 2 ^ (fmt.expWidth + 1) - 2
    have hp : 1 ≤ 2 ^ fmt.expWidth := Nat.one_le_two_pow
    have h := fmt.exponentBias_le_maxFinite
    have hmax : fmt.encoding.maxFiniteExponent fmt.expWidth ≤ 2 ^ fmt.expWidth - 1 := by
      cases fmt.encoding
      all_goals first | exact Nat.sub_le_sub_left (by decide : 1 ≤ 2) _ | exact le_rfl
    rw [Nat.pow_succ]
    omega

/-- `fmt` with `m` fraction bits, its exponent field, bias and encoding kept. -/
def withFraction (fmt : FloatFormat) (m : Nat) (hm : 0 < m := by decide) : FloatFormat :=
  { fmt with fracWidth := m, fracWidth_pos := hm }

/-- A format whose largest normal exponent is below another's has the smaller largest finite
value. -/
theorem toReal_posMaxFinite_le {a b : FloatFormat}
    (h : a.maxNormalExponent < b.maxNormalExponent) :
    toReal (posMaxFinite a) ≤ toReal (posMaxFinite b) := by
  have hsplit : bpow b.maxNormalExponent = bpow (Int.ofNat b.fracWidth) *
      bpow (b.maxNormalExponent - Int.ofNat b.fracWidth) := by
    rw [← bpow_add]
    congr 1
    ring
  refine (toReal_posMaxFinite_lt_bpow a).le.trans
    ((bpow_le_bpow_of_le (right := b.maxNormalExponent) (by omega)).trans ?_)
  rw [toReal_posMaxFinite, hsplit, bpow_ofNat]
  exact mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.le_add_right _ _) (bpow_pos _).le

/-- At most an IEEE format's fraction bits give a largest finite value no larger than its own. -/
theorem toReal_posMaxFinite_withFraction_le {fmt : FloatFormat} (hfmt : fmt.isIEEE = true)
    {m : Nat} (hm : 0 < m) (hmf : m ≤ fmt.fracWidth) :
    toReal (posMaxFinite (withFraction fmt m hm)) ≤ toReal (posMaxFinite fmt) := by
  have hfield : ∀ g : FloatFormat, g.encoding = .ieee →
      (g.maxFiniteFracField : ℝ) = 2 ^ g.fracWidth - 1 := by
    intro g hg
    have h : 1 ≤ 2 ^ g.fracWidth := Nat.one_le_two_pow
    simp only [FloatFormat.maxFiniteFracField, FloatFormat.fracMaskNat, hg]
    rw [Nat.cast_sub h, Nat.cast_pow, Nat.cast_ofNat, Nat.cast_one]
  have he := FloatFormat.encoding_eq_ieee_of_isIEEE fmt hfmt
  have hnarrow : ((withFraction fmt m hm).maxFiniteFracField : ℝ) = 2 ^ m - 1 :=
    hfield (withFraction fmt m hm) he
  have hsplit : bpow (fmt.maxNormalExponent - Int.ofNat m) =
      bpow (Int.ofNat (fmt.fracWidth - m)) *
        bpow (fmt.maxNormalExponent - Int.ofNat fmt.fracWidth) := by
    rw [← bpow_add]
    congr 1
    simp only [Int.ofNat_eq_natCast]
    omega
  have hp : (2 : ℝ) ^ fmt.fracWidth = 2 ^ m * 2 ^ (fmt.fracWidth - m) := by
    rw [← pow_add, Nat.add_sub_cancel' hmf]
  have hq : (1 : ℝ) ≤ 2 ^ (fmt.fracWidth - m) := one_le_pow₀ (by norm_num)
  rw [toReal_posMaxFinite, toReal_posMaxFinite]
  change ((pow2 m + (withFraction fmt m hm).maxFiniteFracField : Nat) : ℝ) *
      bpow (fmt.maxNormalExponent - Int.ofNat m) ≤ _
  rw [hsplit, bpow_ofNat, ← mul_assoc]
  refine mul_le_mul_of_nonneg_right ?_ (bpow_pos _).le
  push_cast
  simp only [pow2_eq_two_pow, Nat.cast_pow, Nat.cast_ofNat]
  rw [hnarrow, hfield fmt he, hp]
  nlinarith

/-- A property of every word of a format holds at every value of it. -/
theorem forall_words {fmt : FloatFormat} {P : Model fmt → Prop}
    (h : ∀ n < 2 ^ fmt.bitWidth, P ⟨BitVec.ofNat _ n⟩) (x : Model fmt) : P x := by
  simpa using h x.bits.toNat x.bits.isLt

/-- Converting a finite value to a format whose grid and range `wide` holds, and back, gives the
format's rounding of the value. -/
theorem toReal_cast_cast {wide narrow : FloatFormat} (y : Model wide)
    (hwide : wide.isIEEE = true) (hencoding : narrow.encoding ≠ .finite)
    (hfrac : narrow.fracWidth ≤ wide.fracWidth)
    (hmin : wide.minSubnormalExponent ≤ narrow.minSubnormalExponent)
    (hmax : toReal (posMaxFinite narrow) ≤ toReal (posMaxFinite wide))
    (hy : isFinite y = true) (hz : isFinite (cast wide narrow y) = true) :
    isFinite (cast narrow wide (cast wide narrow y)) = true ∧
      toReal (cast narrow wide (cast wide narrow y)) = roundAt narrow (toReal y) := by
  have hout := isFinite_cast_of_abs_toReal_le_posMaxFinite _ hwide hz
    ((abs_toReal_le_posMaxFinite_of_isFinite _ hz).trans hmax)
  refine ⟨hout, (cast_exact_of_gridExtension _ hwide hfrac hmin hout).trans ?_⟩
  simp [cast_policy _ hy hz, hencoding]

/-- At an IEEE format and at most its fraction bits, the import's two conversions round once:
where the first is finite, the result is finite and is `withFraction fmt m`'s rounding of the
value. -/
theorem toReal_reduce_ieee {fmt : FloatFormat} (hfmt : fmt.isIEEE = true) {m : Nat} (hm : 0 < m)
    (hmf : m ≤ fmt.fracWidth) (x : Model fmt) (hx : isFinite x = true)
    (hz : isFinite (cast fmt (withFraction fmt m hm) x) = true) :
    isFinite (cast (withFraction fmt m hm) fmt (cast fmt (withFraction fmt m hm) x)) = true ∧
      toReal (cast (withFraction fmt m hm) fmt (cast fmt (withFraction fmt m hm) x)) =
        roundAt (withFraction fmt m hm) (toReal x) := by
  have he : (withFraction fmt m hm).encoding = .ieee :=
    FloatFormat.encoding_eq_ieee_of_isIEEE fmt hfmt
  refine toReal_cast_cast x hfmt (by rw [he]; decide) hmf ?_
    (toReal_posMaxFinite_withFraction_le hfmt hm hmf) hx hz
  show fmt.minNormalExponent - Int.ofNat fmt.fracWidth ≤ fmt.minNormalExponent - Int.ofNat m
  simp only [Int.ofNat_eq_natCast]
  omega

/-- A finite conversion of a value on the target's grid, and in its range where the target
saturates, is exact. -/
theorem toReal_cast_of_generic {wide target : FloatFormat} (y : Model wide)
    (hy : isFinite y = true)
    (hgrid : Flocq.GenericRepresentable binaryRadix (fexpOf target) (toReal y))
    (hrange : target.encoding = .finite → |toReal y| ≤ toReal (posMaxFinite target))
    (hout : isFinite (cast wide target y) = true) :
    toReal (cast wide target y) = toReal y := by
  rw [cast_policy _ hy hout, roundAt_of_genericFormat _ hgrid]
  split_ifs with h
  · exact clampAt_eq_self _ (hrange h)
  · rfl

/-- The four conversions of `reduce_precision` round once: a finite result is `narrow`'s rounding
of the source value. -/
theorem toReal_reduce {source wide narrow : FloatFormat} (x : Model source)
    (hwide : wide.isIEEE = true) (hencoding : narrow.encoding ≠ .finite)
    (hsource : source.fracWidth ≤ wide.fracWidth ∧
      wide.minSubnormalExponent ≤ source.minSubnormalExponent)
    (hnarrow : narrow.fracWidth ≤ source.fracWidth ∧
      source.minSubnormalExponent ≤ narrow.minSubnormalExponent)
    (hmax : toReal (posMaxFinite source) ≤ toReal (posMaxFinite wide))
    (hmaxn : toReal (posMaxFinite narrow) ≤ toReal (posMaxFinite wide))
    (hx : isFinite x = true)
    (hround : isFinite (cast wide narrow (cast source wide x)) = true)
    (hrange : source.encoding = .finite →
      |roundAt narrow (toReal x)| ≤ toReal (posMaxFinite source))
    (hout : isFinite (cast wide source
      (cast narrow wide (cast wide narrow (cast source wide x)))) = true) :
    toReal (cast wide source (cast narrow wide (cast wide narrow (cast source wide x)))) =
      roundAt narrow (toReal x) := by
  have hy := isFinite_cast_of_abs_toReal_le_posMaxFinite x hwide hx
    ((abs_toReal_le_posMaxFinite_of_isFinite x hx).trans hmax)
  have hyv := cast_exact_of_gridExtension x hwide hsource.1 hsource.2 hy
  obtain ⟨hz, hzv⟩ := toReal_cast_cast _ hwide hencoding (hnarrow.1.trans hsource.1)
    (hsource.2.trans hnarrow.2) hmaxn hy hround
  rw [hyv] at hzv
  have hgrid : Flocq.GenericRepresentable binaryRadix (fexpOf source)
      (toReal (cast narrow wide (cast wide narrow (cast source wide x)))) := by
    rw [hzv]
    exact genericFormat_of_gridExtension hnarrow.1 hnarrow.2 (genericFormat_roundAt _ _)
  rw [toReal_cast_of_generic _ hz hgrid (by rw [hzv]; exact hrange) hout, hzv]

/-- Each float8 format with `reduced fmt m` for each `m` below its fraction width, the settings
at which the import rounds. -/
def settings : List (FloatFormat × FloatFormat) :=
  [(.e5m2, reduced .e5m2 1), (.e4m3, reduced .e4m3 1), (.e4m3, reduced .e4m3 2),
    (.e4m3fnuz, reduced .e4m3fnuz 1), (.e4m3fnuz, reduced .e4m3fnuz 2),
    (.e5m2fnuz, reduced .e5m2fnuz 1)]

/-- At each float8 setting, a finite result of the import's four conversions is the rounding of
the operand to the reduced format. -/
theorem toReal_reduce_float8 {fmt narrow : FloatFormat} (hset : (fmt, narrow) ∈ settings)
    (x : Model fmt) (hx : isFinite x = true)
    (hout : isFinite (cast .binary32 fmt (cast narrow .binary32
      (cast .binary32 narrow (cast fmt .binary32 x)))) = true) :
    toReal (cast .binary32 fmt (cast narrow .binary32
      (cast .binary32 narrow (cast fmt .binary32 x)))) = roundAt narrow (toReal x) := by
  simp only [settings, List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at hset
  have hround : ∀ y : Model fmt, isFinite y = true →
      isFinite (cast .binary32 narrow (cast fmt .binary32 y)) = true := by
    refine forall_words ?_
    rcases hset with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
      decide +kernel
  rcases hset with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
    exact toReal_reduce x FloatFormat.isIEEE_binary32 (by decide) (by decide) (by decide)
      (toReal_posMaxFinite_le (by decide)) (toReal_posMaxFinite_le (by decide)) hx
      (hround x hx) (fun h => absurd h (by decide)) hout

end TorchLean.IR.ReducePrecision
