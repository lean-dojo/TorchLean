/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Arithmetic.SignedSemantics.Core
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Algebraic.Proof
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError.Rounding
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Operations.Proof
public import FloatLib.Floats.Formats.BinaryInterchange.Dyadic.Classification
public import FloatLib.Floats.Formats.BinaryInterchange.Model.Packing.Special
public import FloatLib.Floats.Formats.BinaryInterchange.Operations.Proof.RoundToIntegral
public import TorchLean.Verify.RuntimeApprox.Elementary
public import TorchLean.Verify.RuntimeApprox.Roundoff.Recovery
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.ConfiguredBinary
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Elementwise operators for `roundoff`

This module registers as `roundoff` rules the read lemmas of the elementwise operators of
`TorchLean.Tensor.Elementwise` and of `zipWith3`. Each states that a coordinate of the
result is the scalar function at the operands' entries there, so the analysis reads through these
operators at every shape and continues with the scalar function.

It also registers the integral roundings of a configured scalar as exact: in an IEEE format, on a
finite operand of value `a`, `floor`, `ceil`, `trunc` and `round` return `⌊a⌋`, `⌈a⌉`, the
truncation of `a` and the nearest integer with ties to even, and the result is finite. The operand
must be exact, since these functions are not continuous. `floor`, `ceil` and `round` also need the
format's integers to reach its precision, `fracWidth ≤ maxNormalExponent`, as every IEEE interchange
format has, and truncation needs nothing more.

`copysign x y` of a finite `x` is `x` when the sign bits of `x` and `y` agree and `-x` otherwise,
and the result is finite with the error of `x`. A finite value is neither a NaN nor an infinity, so
the predicates `isNaN` and `isInf`, the self-comparison `!(x == x)` of a maximum fold, and the
branches that test them in `maximum`, `fmax` or `nanToNum`, are decided on finite operands.
`quietNaN` changes only a NaN, whose real value is zero like that of the quiet NaN it becomes, so it
carries the bound of its operand unchanged.

The IEEE maximum and minimum of finite values with real values `a` and `b` are finite, with the real
values `max a b` and `min a b`. The maximum of a finite value and zero, the clamp of a
normalization's variance, has the real value `max a 0`. The analysis uses these rules at the steps
of the maxima and minima of `reduce` and `scatter` and of `Softmax.maxVec`, and at the extrema of
computed values. Outside a loop step, the analysis in `TorchLean.Verify.RuntimeApprox.Roundoff`
reads an extremum of exact data through its cases. In an IEEE format the configured value of the
rational `1 / 2`, the weight of a tie in the gradients of `max` and `min`, is finite with the real
value `1 / 2` (`toReal_ratCast_half`), since one half is `2 ^ (-1)`.

`hypot` of exact operands of values `a` and `b` rounds `√(a² + b²)` once when the result is finite,
by FloatLib's `toReal_hypot_eq_roundAt`. For operands within `εx` and `εy` of `a` and `b`, it is
within half an ulp of its exact value plus `εx + εy` of `√(a² + b²)`.

`atanBackward x g`, the derivative of `atan` at `x` times `g`, divides `g` by `x * x + 1`, or by `x`
twice when `x * x + 1` overflows. For exact finite operands of values `a` and `b` both branches give
a finite result within one mixed bound of `b / (a * a + 1)` (`mixedError_toReal_atanBackward`).
`hypotBackward z (Context.hypot x y) g`, the derivative of `hypot` in its operand `z` times `g`,
rounds `g * z` and its quotient by the radius, or rounds the exact quotient once when a checked
step raises a flag. For exact operands of values `a`, `b` and `c`, with `g` and `Context.hypot x y`
finite, every branch gives a finite result within one mixed bound of `c * a / √(a ^ 2 + b ^ 2)` for
`z = x` and of `c * b / √(a ^ 2 + b ^ 2)` for `z = y` (`isFinite_and_mixedError_hypotBackward_left`,
`isFinite_and_mixedError_hypotBackward_right`). The theorems on `atanBackward` and `hypotBackward`
are not roundoff rules: at some operands their bounds are larger than the ones the analysis composes
from the steps, for `atanBackward` at `x = 0` and `g = 11 / 8`, so the routes through these two
functions keep the composed bounds. `angleTangent x y dx dy`, the derivative of the angle of
`(x, y)` in direction `(dx, dy)`, divides the four operands by `max |x| |y|` and rounds eleven
steps, or rounds the exact quotient once when a checked step raises a flag or the difference of
the scaled products cancels to zero. For exact finite operands of values `a`, `b`, `c` and `d` and
`q = (a * d - b * c) / (a ^ 2 + b ^ 2)`, the result is within
`K * (|a * d| + |b * c|) / (a ^ 2 + b ^ 2) + η` of `q`, `K = 1 / ((1 - u) * (1 - 2 * u) ^ 4) - 1`,
when `|q|` is at most the largest finite value (`abs_toReal_angleTangent_sub_le`), and a finite
result is within one mixed bound of `q` (`mixedError_toReal_angleTangent`). `isFinite_angleTangent`
gives that finite result under the same range condition. These three are not roundoff rules: they
need finite operands and the range condition, and the flag conditions of a route through
`angleTangent` allow an infinite operand, where its slope is zero, so these routes keep their flag
conditions.

`tan`, `exp2`, `log2`, `log10`, `asinh`, `acosh` and `atanh` of an exact finite operand in the
domain round the real function once, `asinh` and `acosh` always and the others when the result is
finite (`TorchLean.Verify.RuntimeApprox.Elementary`). The finiteness rules `isFinite_tan`,
`isFinite_exp2`, `isFinite_log2`, `isFinite_log10` and `isFinite_atanh` give that finite result
when the real function is below the overflow threshold in magnitude, and `isFinite_asinh` and
`isFinite_acosh` give it in the domain. The absolute bounds take the domain and this range
condition. The mixed bounds take as premise the finite result, which implies the domain, because
mixed analysis proves an order premise or rejects the rule but keeps a finiteness premise open. The
finiteness rules turn that premise into conditions on the operand.

In an IEEE format, C's `fmod` of finite exact operands with a nonzero divisor is exact and finite:
FloatLib's IEEE remainder is the dividend minus an integer multiple of the divisor, at most half the
divisor in magnitude, and the correction by the divisor's magnitude is an exact addition
(`toReal_fmod`). Python's `remainder` corrects that result by one more addition, which may round.
We keep the definitions of Lean's native `Float32` folded in the analysis (`nativeFloat32`), as
FloatLib's arithmetic is. Evaluating core's bit-level model of a native comparison with a literal
by reduction, such as dropout's guard `Context.geBool p 1`, exceeds the default heartbeat budget.
-/

public section

attribute [roundoff_rule]
  TorchLean.Tensor.zipWith3_apply
  TorchLean.Tensor.select_apply
  TorchLean.Tensor.sign_apply
  TorchLean.Tensor.reciprocal_apply
  TorchLean.Tensor.sqrt_apply
  TorchLean.Tensor.rsqrt_apply
  TorchLean.Tensor.floor_apply
  TorchLean.Tensor.ceil_apply
  TorchLean.Tensor.round_apply
  TorchLean.Tensor.trunc_apply
  TorchLean.Tensor.frac_apply
  TorchLean.Tensor.exp2_apply
  TorchLean.Tensor.expm1_apply
  TorchLean.Tensor.log2_apply
  TorchLean.Tensor.log10_apply
  TorchLean.Tensor.log1p_apply
  TorchLean.Tensor.tan_apply
  TorchLean.Tensor.asinh_apply
  TorchLean.Tensor.acosh_apply
  TorchLean.Tensor.atanh_apply
  TorchLean.Tensor.isNaN_apply
  TorchLean.Tensor.isInf_apply
  TorchLean.Tensor.isFinite_apply
  TorchLean.Tensor.maximum_apply
  TorchLean.Tensor.minimum_apply
  TorchLean.Tensor.fmax_apply
  TorchLean.Tensor.fmin_apply
  TorchLean.Tensor.fmod_apply
  TorchLean.Tensor.remainder_apply
  TorchLean.Tensor.copysign_apply
  TorchLean.Tensor.hypot_apply
  TorchLean.Tensor.eq_apply
  TorchLean.Tensor.ne_apply
  TorchLean.Tensor.lt_apply
  TorchLean.Tensor.le_apply
  TorchLean.Tensor.gt_apply
  TorchLean.Tensor.ge_apply
  TorchLean.Tensor.clamp_apply
  TorchLean.Tensor.lerp_apply
  TorchLean.Tensor.addcmul_apply
  TorchLean.Tensor.addcdiv_apply
  TorchLean.Tensor.nanToNum_apply
  TorchLean.Tensor.atan2_apply
  TorchLean.Tensor.atan_apply
  TorchLean.Tensor.asin_apply
  TorchLean.Tensor.acos_apply
  TorchLean.Tensor.erf_apply

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open ExecFloat.Binary (toModel toReal isFinite)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" =>
  ExecFloat.Binary format.expWidth format.fracWidth format.encoding format.exponentBias
    format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
    format.exponentBias_le_maxFinite plan code

/-- The floor of a finite configured value is finite. -/
theorem isFinite_floor {x : Value} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : isFinite x = true) :
    isFinite (Rounding.floor x) = true := by
  unfold isFinite at *
  exact (ExecFloat.Binary.toModel_roundToIntegral x .down) ▸
    Model.isFinite_roundToIntegral hformat hrange _ _ hx

/-- The ceiling of a finite configured value is finite. -/
theorem isFinite_ceil {x : Value} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : isFinite x = true) :
    isFinite (Rounding.ceil x) = true := by
  unfold isFinite at *
  exact (ExecFloat.Binary.toModel_roundToIntegral x .up) ▸
    Model.isFinite_roundToIntegral hformat hrange _ _ hx

/-- The rounding to nearest of a finite configured value is finite. -/
theorem isFinite_round {x : Value} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : isFinite x = true) :
    isFinite (Rounding.round x) = true := by
  unfold isFinite at *
  exact (ExecFloat.Binary.toModel_roundToIntegral x .nearest) ▸
    Model.isFinite_roundToIntegral hformat hrange _ _ hx

/-- The truncation of a finite configured value is finite. -/
theorem isFinite_trunc {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true) :
    isFinite (Rounding.trunc x) = true := by
  unfold isFinite at *
  exact (ExecFloat.Binary.toModel_roundToIntegral x .towardZero) ▸
    Model.isFinite_roundToIntegral_towardZero hformat _ hx

/-- A finite configured value of value `a` has floor `⌊a⌋`. -/
theorem toReal_floor {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : toReal x = a)
    (hfinite : isFinite x = true) : toReal (Rounding.floor x) = (⌊a⌋ : ℝ) := by
  subst hx
  unfold toReal isFinite at *
  change Model.toReal (toModel (ExecFloat.Binary.roundToIntegral x .down)) = _
  rw [ExecFloat.Binary.toModel_roundToIntegral]
  exact Model.toReal_roundToIntegral_down hformat hrange _ hfinite

/-- A finite configured value of value `a` has ceiling `⌈a⌉`. -/
theorem toReal_ceil {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : toReal x = a)
    (hfinite : isFinite x = true) : toReal (Rounding.ceil x) = (⌈a⌉ : ℝ) := by
  subst hx
  unfold toReal isFinite at *
  change Model.toReal (toModel (ExecFloat.Binary.roundToIntegral x .up)) = _
  rw [ExecFloat.Binary.toModel_roundToIntegral]
  exact Model.toReal_roundToIntegral_up hformat hrange _ hfinite

/-- A finite configured value of value `a` rounds to the nearest integer with ties to even. -/
theorem toReal_round {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : toReal x = a)
    (hfinite : isFinite x = true) :
    toReal (Rounding.round x) = (FloatLib.Floats.Formats.Flocq.nearestEven a : ℝ) := by
  subst hx
  unfold toReal isFinite at *
  change Model.toReal (toModel (ExecFloat.Binary.roundToIntegral x .nearest)) = _
  rw [ExecFloat.Binary.toModel_roundToIntegral]
  exact Model.toReal_roundToIntegral_nearest hformat hrange _ hfinite

/-- A finite configured value of value `a` truncates to
`FloatLib.Floats.Formats.Flocq.truncRound a`. -/
theorem toReal_trunc {x : Value} {a : ℝ} (hformat : format.isIEEE = true) (hx : toReal x = a)
    (hfinite : isFinite x = true) :
    toReal (Rounding.trunc x) = (FloatLib.Floats.Formats.Flocq.truncRound a : ℝ) := by
  subst hx
  unfold toReal isFinite at *
  change Model.toReal (toModel (ExecFloat.Binary.roundToIntegral x .towardZero)) = _
  rw [ExecFloat.Binary.toModel_roundToIntegral,
    Model.toReal_roundToIntegral_towardZero hformat _ hfinite]
  unfold FloatLib.Floats.Formats.Flocq.truncRound
  split_ifs <;> first | rfl | (exfalso; linarith)

/-- `toReal_floor` as an absolute bound for an exact operand. -/
theorem abs_toReal_floor_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : |toReal x - a| ≤ 0)
    (hfinite : isFinite x = true) : |toReal (Rounding.floor x) - (⌊a⌋ : ℝ)| ≤ 0 := by
  rw [toReal_floor hformat hrange (by simpa [sub_eq_zero] using hx) hfinite, sub_self, abs_zero]

/-- `toReal_ceil` as an absolute bound for an exact operand. -/
theorem abs_toReal_ceil_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : |toReal x - a| ≤ 0)
    (hfinite : isFinite x = true) : |toReal (Rounding.ceil x) - (⌈a⌉ : ℝ)| ≤ 0 := by
  rw [toReal_ceil hformat hrange (by simpa [sub_eq_zero] using hx) hfinite, sub_self, abs_zero]

/-- `toReal_round` as an absolute bound for an exact operand. -/
theorem abs_toReal_round_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent) (hx : |toReal x - a| ≤ 0)
    (hfinite : isFinite x = true) :
    |toReal (Rounding.round x) - (FloatLib.Floats.Formats.Flocq.nearestEven a : ℝ)| ≤ 0 := by
  rw [toReal_round hformat hrange (by simpa [sub_eq_zero] using hx) hfinite, sub_self, abs_zero]

/-- `toReal_trunc` as an absolute bound for an exact operand. -/
theorem abs_toReal_trunc_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0) (hfinite : isFinite x = true) :
    |toReal (Rounding.trunc x) - (FloatLib.Floats.Formats.Flocq.truncRound a : ℝ)| ≤ 0 := by
  rw [toReal_trunc hformat (by simpa [sub_eq_zero] using hx) hfinite, sub_self, abs_zero]

/-- `toReal_floor` as a mixed bound for an exact operand. -/
theorem mixedError_toReal_floor {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent)
    (hx : MixedError (toReal x) a 0 0) (hfinite : isFinite x = true) :
    MixedError (toReal (Rounding.floor x)) (⌊a⌋ : ℝ) 0 0 :=
  mixedError_zero_zero_iff.mpr
    (toReal_floor hformat hrange (mixedError_zero_zero_iff.mp hx) hfinite)

/-- `toReal_ceil` as a mixed bound for an exact operand. -/
theorem mixedError_toReal_ceil {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent)
    (hx : MixedError (toReal x) a 0 0) (hfinite : isFinite x = true) :
    MixedError (toReal (Rounding.ceil x)) (⌈a⌉ : ℝ) 0 0 :=
  mixedError_zero_zero_iff.mpr
    (toReal_ceil hformat hrange (mixedError_zero_zero_iff.mp hx) hfinite)

/-- `toReal_round` as a mixed bound for an exact operand. -/
theorem mixedError_toReal_round {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hrange : (format.fracWidth : Int) ≤ format.maxNormalExponent)
    (hx : MixedError (toReal x) a 0 0) (hfinite : isFinite x = true) :
    MixedError (toReal (Rounding.round x)) (FloatLib.Floats.Formats.Flocq.nearestEven a : ℝ) 0 0 :=
  mixedError_zero_zero_iff.mpr
    (toReal_round hformat hrange (mixedError_zero_zero_iff.mp hx) hfinite)

/-- `toReal_trunc` as a mixed bound for an exact operand. -/
theorem mixedError_toReal_trunc {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0) (hfinite : isFinite x = true) :
    MixedError (toReal (Rounding.trunc x)) (FloatLib.Floats.Formats.Flocq.truncRound a : ℝ) 0 0 :=
  mixedError_zero_zero_iff.mpr (toReal_trunc hformat (mixedError_zero_zero_iff.mp hx) hfinite)

/-- Copying a sign keeps a configured value finite. -/
theorem isFinite_copysign {x y : Value} (hx : isFinite x = true) :
    isFinite (Context.copysign x y) = true := by
  unfold isFinite at *
  change Model.isFinite (toModel (ExecFloat.Binary.copySign x y)) = true
  rw [ExecFloat.Binary.toModel_copySign, Model.isFinite_copySign]
  exact hx

/-- In an IEEE format, `copysign x y` of a finite `x` is `x` when the sign bits agree and `-x`
otherwise. -/
theorem toReal_copysign {x y : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true) :
    toReal (Context.copysign x y) =
      if ExecFloat.Binary.signBit x == ExecFloat.Binary.signBit y then toReal x
      else -toReal x := by
  unfold toReal isFinite at *
  change Model.toReal (toModel (ExecFloat.Binary.copySign x y)) = _
  rw [ExecFloat.Binary.toModel_copySign]
  have hsigned := FloatFormat.supportsSignedZero_eq_true_of_isIEEE format hformat
  unfold Model.copySign ExecFloat.Binary.signBit
  simp only [hsigned, Bool.not_true, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  split_ifs
  · rfl
  · rw [← Model.neg_eq_toggleSign_of_supportsSignedZero _ hsigned, Model.toReal_neg _ hx]

/-- `toReal_copysign` as a mixed bound: the result carries the error of `x`, against `x₀` or
`-x₀` as the sign bits decide. -/
theorem mixedError_toReal_copysign {x y : Value} {x₀ r a : ℝ} (hformat : format.isIEEE = true)
    (hx : isFinite x = true) (ex : MixedError (toReal x) x₀ r a) :
    MixedError (toReal (Context.copysign x y))
      (if ExecFloat.Binary.signBit x == ExecFloat.Binary.signBit y then x₀ else -x₀) r a := by
  rw [toReal_copysign hformat hx]
  split_ifs
  · exact ex
  · exact ex.neg

/-- `toReal_copysign` as an absolute bound: the result carries the error of `x`. -/
theorem abs_toReal_copysign_sub_le {x y : Value} {x₀ ε : ℝ} (hformat : format.isIEEE = true)
    (hx : isFinite x = true) (ex : |toReal x - x₀| ≤ ε) :
    |toReal (Context.copysign x y) -
        (if ExecFloat.Binary.signBit x == ExecFloat.Binary.signBit y then x₀ else -x₀)| ≤ ε := by
  rw [toReal_copysign hformat hx]
  split_ifs
  · exact ex
  · rw [show -toReal x - -x₀ = -(toReal x - x₀) by ring, abs_neg]
    exact ex

/-- `FloatLib.Floats.Formats.Flocq.truncRound q` is `m` when `q` is nonnegative and in
`[m, m + 1)`, or negative and in `(m - 1, m]`. -/
theorem _root_.TorchLean.Context.truncRound_eq_of_mem {q : ℝ} {m : ℤ}
    (h : (0 ≤ q ∧ (m : ℝ) ≤ q ∧ q < m + 1) ∨ (q < 0 ∧ (m : ℝ) - 1 < q ∧ q ≤ m)) :
    FloatLib.Floats.Formats.Flocq.truncRound q = m := by
  unfold FloatLib.Floats.Formats.Flocq.truncRound
  rcases h with ⟨hq, h1, h2⟩ | ⟨hq, h1, h2⟩
  · simp only [not_lt.mpr hq, ↓reduceIte, Int.floor_eq_iff]; exact ⟨h1, h2⟩
  · simp only [hq, ↓reduceIte, Int.ceil_eq_iff]; exact ⟨h1, h2⟩

/-- C's `fmod` from a remainder of integer quotient: if `r = a - k b` with `|r| ≤ |b| / 2`, then
`fmodReal a b` is `r` when `r` is zero or has the sign of `a`, and `r` plus `|b|` with the sign of
`a` otherwise. -/
theorem _root_.TorchLean.Context.fmodReal_eq_of_remainder {a b r : ℝ} {k : ℤ} (hb : b ≠ 0)
    (hr : r = a - k * b) (habs : |r| ≤ |b| / 2) :
    ((r = 0 ∨ (r < 0 ↔ a < 0)) → Context.fmodReal a b = r) ∧
      (r ≠ 0 → ¬(r < 0 ↔ a < 0) →
        Context.fmodReal a b = r + if a < 0 then -|b| else |b|) := by
  unfold Context.fmodReal
  set q := a / b with hq
  have ha : a = q * b := by rw [hq, div_mul_cancel₀ a hb]
  set d := q - k with hd
  have hrd : r = d * b := by rw [hr, ha, hd]; ring
  have hbabs : 0 < |b| := abs_pos.mpr hb
  have hdabs : |d| ≤ 1 / 2 := by
    rw [hrd, abs_mul] at habs
    by_contra h
    push Not at h
    nlinarith
  obtain ⟨hdlo, hdhi⟩ := abs_le.mp hdabs
  have hqk : q = k + d := by rw [hd]; ring
  -- A zero quotient has a zero remainder.
  have hq0 : q = 0 → d = 0 := by
    intro h0
    have hk : |(k : ℝ)| ≤ 1 / 2 := by
      rw [show (k : ℝ) = -d by linarith, abs_neg]; exact hdabs
    rw [(Int.abs_lt_one_iff (a := k)).mp (by exact_mod_cast hk.trans_lt one_half_lt_one)] at hqk
    simp only [Int.cast_zero, zero_add] at hqk
    rw [← hqk, h0]
  have hfmod (m : ℤ) (hm : FloatLib.Floats.Formats.Flocq.truncRound q = m) :
      a - b * (m : ℝ) = r + (k - m) * b := by
    rw [hrd, ha, hqk]; ring
  rcases lt_or_gt_of_ne hb with hbneg | hbpos
  · -- A negative divisor reverses the signs of `r` against `d` and of `a` against `q`.
    have hr_neg : r < 0 ↔ 0 < d := ⟨fun h => by nlinarith, fun h => by nlinarith⟩
    have ha_neg : a < 0 ↔ 0 < q := ⟨fun h => by nlinarith, fun h => by nlinarith⟩
    have hr0 : r = 0 ↔ d = 0 := by
      rw [hrd]; exact ⟨fun h => (mul_eq_zero.mp h).resolve_right hb, fun h => by rw [h, zero_mul]⟩
    have habs_b : |b| = -b := abs_of_neg hbneg
    refine ⟨fun h => ?_, fun hr0' hsign => ?_⟩
    · have hm : FloatLib.Floats.Formats.Flocq.truncRound q = k := by
        apply Context.truncRound_eq_of_mem
        rcases h with h | h
        · have := hr0.mp h; rcases le_or_gt 0 q with h' | h'
          · exact Or.inl ⟨h', by linarith, by linarith⟩
          · exact Or.inr ⟨h', by linarith, by linarith⟩
        · rcases le_or_gt 0 q with h' | h'
          · by_cases hz : q = 0
            · have := hq0 hz; exact Or.inl ⟨h', by linarith, by linarith⟩
            · have hqpos : 0 < q := lt_of_le_of_ne h' (Ne.symm hz)
              have : 0 < d ∨ d = 0 := by
                rcases lt_trichotomy d 0 with hd' | hd' | hd'
                · exfalso
                  have : ¬ r < 0 := fun hr' => by linarith [hr_neg.mp hr']
                  exact this (h.mpr (ha_neg.mpr hqpos))
                · exact Or.inr hd'
                · exact Or.inl hd'
              exact Or.inl ⟨h', by rcases this with h'' | h'' <;> linarith, by linarith⟩
          · have : d ≤ 0 := by
              by_contra hd'
              push Not at hd'
              have := h.mp (hr_neg.mpr hd')
              linarith [ha_neg.mp this]
            exact Or.inr ⟨h', by linarith, by linarith⟩
      rw [hm, hfmod k hm]; ring
    · have hd0 : d ≠ 0 := fun h => hr0' (hr0.mpr h)
      have hqne : q ≠ 0 := fun h => hd0 (hq0 h)
      rcases lt_or_gt_of_ne hqne with hqneg | hqpos
      · -- `a > 0`, so `r` is negative and `d` positive: the truncation is `k + 1`.
        have ha' : ¬ a < 0 := fun h => by linarith [ha_neg.mp h]
        have hdpos : 0 < d := by
          by_contra hd'
          push Not at hd'
          exact hsign ⟨fun h => absurd (hr_neg.mp h) (by linarith), fun h => absurd h ha'⟩
        have hm : FloatLib.Floats.Formats.Flocq.truncRound q = k + 1 := by
          apply Context.truncRound_eq_of_mem; push_cast;
            exact Or.inr ⟨hqneg, by linarith, by linarith⟩
        rw [hm, hfmod (k + 1) hm, habs_b]; simp only [ha', ↓reduceIte]; push_cast; ring
      · have ha' : a < 0 := ha_neg.mpr hqpos
        have hdneg : d < 0 := by
          by_contra hd'
          push Not at hd'
          rcases hd'.lt_or_eq with hd'' | hd''
          · exact hsign ⟨fun _ => ha', fun _ => hr_neg.mpr hd''⟩
          · exact hd0 hd''.symm
        have hm : FloatLib.Floats.Formats.Flocq.truncRound q = k - 1 := by
          apply Context.truncRound_eq_of_mem; push_cast;
            exact Or.inl ⟨hqpos.le, by linarith, by linarith⟩
        rw [hm, hfmod (k - 1) hm, habs_b]; simp only [ha', ↓reduceIte]; push_cast; ring
  · have hr_neg : r < 0 ↔ d < 0 := ⟨fun h => by nlinarith, fun h => by nlinarith⟩
    have ha_neg : a < 0 ↔ q < 0 := ⟨fun h => by nlinarith, fun h => by nlinarith⟩
    have hr0 : r = 0 ↔ d = 0 := by
      rw [hrd]; exact ⟨fun h => (mul_eq_zero.mp h).resolve_right hb, fun h => by rw [h, zero_mul]⟩
    have habs_b : |b| = b := abs_of_pos hbpos
    refine ⟨fun h => ?_, fun hr0' hsign => ?_⟩
    · have hm : FloatLib.Floats.Formats.Flocq.truncRound q = k := by
        apply Context.truncRound_eq_of_mem
        rcases h with h | h
        · have := hr0.mp h; rcases le_or_gt 0 q with h' | h'
          · exact Or.inl ⟨h', by linarith, by linarith⟩
          · exact Or.inr ⟨h', by linarith, by linarith⟩
        · rcases le_or_gt 0 q with h' | h'
          · have : 0 ≤ d := by
              by_contra hd'
              push Not at hd'
              have := ha_neg.mp (h.mp (hr_neg.mpr hd'))
              linarith
            exact Or.inl ⟨h', by linarith, by linarith⟩
          · have : d ≤ 0 := by
              by_contra hd'
              push Not at hd'
              have := h.mpr (ha_neg.mpr h')
              linarith [hr_neg.mp this]
            exact Or.inr ⟨h', by linarith, by linarith⟩
      rw [hm, hfmod k hm]; ring
    · have hd0 : d ≠ 0 := fun h => hr0' (hr0.mpr h)
      have hqne : q ≠ 0 := fun h => hd0 (hq0 h)
      rcases lt_or_gt_of_ne hqne with hqneg | hqpos
      · have ha' : a < 0 := ha_neg.mpr hqneg
        have hdpos : 0 < d := by
          by_contra hd'
          push Not at hd'
          rcases hd'.lt_or_eq with hd'' | hd''
          · exact hsign ⟨fun _ => ha', fun _ => hr_neg.mpr hd''⟩
          · exact hd0 hd''
        have hm : FloatLib.Floats.Formats.Flocq.truncRound q = k + 1 := by
          apply Context.truncRound_eq_of_mem; push_cast;
            exact Or.inr ⟨hqneg, by linarith, by linarith⟩
        rw [hm, hfmod (k + 1) hm, habs_b]; simp only [ha', ↓reduceIte]; push_cast; ring
      · have ha' : ¬ a < 0 := fun h => by linarith [ha_neg.mp h]
        have hdneg : d < 0 := by
          by_contra hd'
          push Not at hd'
          exact hsign ⟨fun h => absurd (hr_neg.mp h) (by linarith), fun h => absurd h ha'⟩
        have hm : FloatLib.Floats.Formats.Flocq.truncRound q = k - 1 := by
          apply Context.truncRound_eq_of_mem; push_cast;
            exact Or.inl ⟨hqpos.le, by linarith, by linarith⟩
        rw [hm, hfmod (k - 1) hm, habs_b]; simp only [ha', ↓reduceIte]; push_cast; ring


/-- A finite nonzero configured value has its sign bit set exactly when it is negative. -/
theorem signBit_eq_decide {x : Value} (hx : isFinite x = true)
    (hzero : ExecFloat.Binary.isZero x = false) :
    ExecFloat.Binary.signBit x = decide (toReal x < 0) := by
  unfold isFinite at hx
  unfold ExecFloat.Binary.isZero at hzero
  obtain ⟨d, hd⟩ := Model.exists_toDyadic?_of_isFinite hx
  have hsig := Model.significand_ne_zero_of_toDyadic?_some_of_isZero_eq_false hd hzero
  have hsign := Model.sign_eq_signBit_of_toDyadic?_some hd
  have hval : toReal x = d.toReal := by
    unfold toReal; simp [Model.toReal_eq, hd]
  unfold ExecFloat.Binary.signBit
  rw [← hsign, hval]
  cases hneg : d.negative
  · exact (decide_eq_false (not_lt.mpr
      (Model.Dyadic.toReal_pos_of_negative_eq_false hneg hsig).le)).symm
  · exact (decide_eq_true (Model.Dyadic.toReal_lt_zero_of_negative_eq_true hneg hsig)).symm

/-- In an IEEE format, C's `fmod` of finite configured values with a nonzero divisor is finite
and exact: it decodes to `Context.fmodReal` of the operands' values. -/
theorem toReal_fmod {x y : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hy : isFinite y = true) (hy0 : ExecFloat.Binary.isZero y = false) :
    isFinite (Rounding.fmod x y) = true ∧
      toReal (Rounding.fmod x y) = Context.fmodReal (toReal x) (toReal y) := by
  change isFinite (fmodSchedule (ExecFloat.Binary.remainder · ·) ExecFloat.Binary.isZero
      ExecFloat.Binary.signBit ExecFloat.Binary.copySign x y) = true ∧
    toReal (fmodSchedule (ExecFloat.Binary.remainder · ·) ExecFloat.Binary.isZero
      ExecFloat.Binary.signBit ExecFloat.Binary.copySign x y) =
      Context.fmodReal (toReal x) (toReal y)
  have hrM : toModel (ExecFloat.Binary.remainder x y) =
      Model.remainder (toModel x) (toModel y) := ExecFloat.Binary.toModel_remainder x y
  obtain ⟨k, hk⟩ := Model.exists_toReal_remainder_eq_sub hformat (toModel x) (toModel y) hx hy hy0
  have hk' : toReal (ExecFloat.Binary.remainder x y) = toReal x - k * toReal y := by
    unfold toReal; rw [hrM]; exact hk
  have habs : |toReal (ExecFloat.Binary.remainder x y)| ≤ |toReal y| / 2 := by
    unfold toReal; rw [hrM]
    exact Model.abs_toReal_remainder_le hformat (toModel x) (toModel y) hx hy hy0
  have hrfin : isFinite (ExecFloat.Binary.remainder x y) = true := by
    unfold isFinite; rw [hrM]
    obtain ⟨X, hX⟩ := Model.exists_toDyadic?_of_isFinite hx
    obtain ⟨Y, hY⟩ := Model.exists_toDyadic?_of_isFinite hy
    exact (Model.remainder_exact hformat _ _ X Y (Model.exactValue_eq_finite_iff.mpr hX)
      (Model.exactValue_eq_finite_iff.mpr hY)
      (Model.significand_ne_zero_of_toDyadic?_some_of_isZero_eq_false hY hy0)).1
  have hyreal : toReal y ≠ 0 := fun h => by
    have := (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hy).mpr h
    rw [hy0] at this; exact absurd this (by decide)
  obtain ⟨hsame, hdiff⟩ := Context.fmodReal_eq_of_remainder hyreal hk' habs
  -- A nonzero remainder has a nonzero dividend: a zero dividend leaves a multiple of the
  -- divisor of magnitude at most half the divisor, which is zero.
  have hdividend : toReal (ExecFloat.Binary.remainder x y) ≠ 0 → toReal x ≠ 0 := by
    intro hr hx0
    apply hr
    have hkb : |(k : ℝ)| * |toReal y| ≤ |toReal y| / 2 := by
      rw [← abs_mul, ← abs_neg, show -(↑k * toReal y) = toReal x - k * toReal y by rw [hx0]; ring,
        ← hk']
      exact habs
    have hk0 : |(k : ℝ)| ≤ 1 / 2 := by
      have := abs_pos.mpr hyreal
      by_contra h; push Not at h; nlinarith
    rw [hk', hx0, (Int.abs_lt_one_iff (a := k)).mp
      (by exact_mod_cast hk0.trans_lt one_half_lt_one)]; simp
  unfold fmodSchedule
  simp only
  rcases hz : ExecFloat.Binary.isZero (ExecFloat.Binary.remainder x y) with _ | _
  · have hr0 : toReal (ExecFloat.Binary.remainder x y) ≠ 0 := fun h => by
      have := (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hrfin).mpr h
      rw [hz] at this; exact absurd this (by decide)
    have hx0 := hdividend hr0
    have hxz : ExecFloat.Binary.isZero x = false := by
      rcases h : ExecFloat.Binary.isZero x
      · rfl
      · exact absurd ((ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hx).mp h) hx0
    rw [signBit_eq_decide hrfin hz, signBit_eq_decide hx hxz]
    by_cases hs : (toReal (ExecFloat.Binary.remainder x y) < 0 ↔ toReal x < 0)
    · have hdec : decide (toReal (ExecFloat.Binary.remainder x y) < 0) = decide (toReal x < 0) :=
        Bool.eq_iff_iff.mpr (by simpa using hs)
      simp only [hdec, bne_self_eq_false, Bool.not_false, Bool.and_false, Bool.false_eq_true,
        ↓reduceIte]
      exact ⟨hrfin, (hsame (Or.inr hs)).symm⟩
    · have hdec : (decide (toReal (ExecFloat.Binary.remainder x y) < 0) !=
          decide (toReal x < 0)) = true := by
        rw [bne_iff_ne, ne_eq, Bool.eq_iff_iff]; simpa using hs
      simp only [Bool.not_false, hdec, Bool.and_self, ↓reduceIte]
      have hsign : Model.signBit (Model.remainder (toModel x) (toModel y)) ≠
          Model.signBit (toModel x) := by
        intro h; apply hs
        have h' : ExecFloat.Binary.signBit (ExecFloat.Binary.remainder x y) =
            ExecFloat.Binary.signBit x := by
          unfold ExecFloat.Binary.signBit; rw [hrM]; exact h
        rw [signBit_eq_decide hrfin hz, signBit_eq_decide hx hxz] at h'
        simpa using Bool.eq_iff_iff.mp h'
      have hzM : Model.isZero (Model.remainder (toModel x) (toModel y)) = false := by
        rw [← hrM]; exact hz
      obtain ⟨hfin, hval⟩ := Model.remainder_add_copySign_exact hformat (toModel x) (toModel y)
        hx hy hy0 hzM hsign
      have hadd : toModel (ExecFloat.Binary.remainder x y + ExecFloat.Binary.copySign y x) =
          Model.add (Model.remainder (toModel x) (toModel y))
            (Model.copySign (toModel y) (toModel x)) := by
        rw [ExecFloat.Binary.toModel_hAdd (ExecFloat.Binary.remainder x y)
          (ExecFloat.Binary.copySign y x), hrM, ExecFloat.Binary.toModel_copySign]
      refine ⟨by unfold isFinite; rw [hadd]; exact hfin, ?_⟩
      have hcopy : toReal (ExecFloat.Binary.copySign y x) =
          if toReal x < 0 then -|toReal y| else |toReal y| := by
        have h := toReal_copysign (x := y) (y := x) hformat hy
        change toReal (ExecFloat.Binary.copySign y x) = _ at h
        rw [h, signBit_eq_decide hy hy0, signBit_eq_decide hx hxz]
        rcases lt_or_gt_of_ne hyreal with hyn | hyp
        · by_cases hxn : toReal x < 0 <;> simp [hyn, hxn, abs_of_neg hyn]
        · by_cases hxn : toReal x < 0 <;> simp [hxn, abs_of_pos hyp, not_lt.mpr hyp.le]
      rw [hdiff hr0 hs, ← hcopy]
      unfold toReal at *
      rw [hadd, hval, ← hrM, ExecFloat.Binary.toModel_copySign]
  · simp only [Bool.not_true, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
    exact ⟨hrfin, (hsame (Or.inl
      ((ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hrfin).mp hz))).symm⟩

/-- In an IEEE format, C's `fmod` of finite configured values with a nonzero divisor is finite. -/
theorem isFinite_fmod {x y : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hy : isFinite y = true) (hy0 : ExecFloat.Binary.isZero y = false) :
    isFinite (Rounding.fmod x y) = true :=
  (toReal_fmod hformat hx hy hy0).1

/-- `toReal_fmod` as an absolute bound for exact operands. -/
theorem abs_toReal_fmod_sub_le {x y : Value} {a b : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0) (hy : |toReal y - b| ≤ 0) (hxfinite : isFinite x = true)
    (hyfinite : isFinite y = true) (hy0 : ExecFloat.Binary.isZero y = false) :
    |toReal (Rounding.fmod x y) - Context.fmodReal a b| ≤ 0 := by
  rw [(toReal_fmod hformat hxfinite hyfinite hy0).2,
    (by simpa [sub_eq_zero] using hx : toReal x = a),
    (by simpa [sub_eq_zero] using hy : toReal y = b), sub_self, abs_zero]

/-- `toReal_fmod` as a mixed bound for exact operands. -/
theorem mixedError_toReal_fmod {x y : Value} {a b : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0) (hy : MixedError (toReal y) b 0 0)
    (hxfinite : isFinite x = true) (hyfinite : isFinite y = true)
    (hy0 : ExecFloat.Binary.isZero y = false) :
    MixedError (toReal (Rounding.fmod x y)) (Context.fmodReal a b) 0 0 :=
  mixedError_zero_zero_iff.mpr <| by
    rw [(toReal_fmod hformat hxfinite hyfinite hy0).2, mixedError_zero_zero_iff.mp hx,
      mixedError_zero_zero_iff.mp hy]

/-- A finite configured value is not a NaN. -/
theorem isNaN_of_isFinite {x : Value} (hx : isFinite x = true) : Context.isNaN x = false :=
  Model.isNaN_eq_false_of_isFinite_eq_true _ hx

/-- A finite configured value is not an infinity. -/
theorem isInf_of_isFinite {x : Value} (hx : isFinite x = true) : Context.isInf x = false :=
  Model.isInf_eq_false_of_isFinite_eq_true _ hx

/-- A finite configured value equals itself, so the NaN test `!(x == x)` of a maximum fold such as
`Tensor.maxAbs` is false at it. -/
theorem not_beq_self_of_isFinite {x : Value} (hx : isFinite x = true) : (!(x == x)) = false := by
  have hnan : ExecFloat.Binary.isNaN x = false := Model.isNaN_eq_false_of_isFinite_eq_true _ hx
  have hself : (x == x) = true := by
    show ExecFloat.compareEqual x x = true
    simp only [ExecFloat.compareEqual, ExecFloat.Binary.compare_eq_compare_toModel,
      Model.compare_self_of_isNaN_eq_false _ hnan]
    rfl
  rw [hself]
  rfl

/-- `Context.quietNaN` of a configured value is FloatLib's `Model.quietNaN` of its model. -/
theorem toModel_quietNaN (x : Value) :
    toModel (Context.quietNaN x) = Model.quietNaN (toModel x) := by
  show toModel (ExecFloat.Binary.ofModel (Model.quietNaN (toModel x))) = _
  exact ExecFloat.Binary.toModel_ofModel _

/-- Quieting returns a finite value unchanged. -/
theorem isFinite_quietNaN {x : Value} (hx : isFinite x = true) :
    isFinite (Context.quietNaN x) = true := by
  unfold isFinite at *
  rw [toModel_quietNaN, Model.quietNaN_eq_self_of_isNaN_eq_false _
    (Model.isNaN_eq_false_of_isFinite_eq_true _ hx)]
  exact hx

/-- Quieting keeps the real value: a NaN and its quiet NaN both have real value zero. -/
theorem toReal_quietNaN (x : Value) : toReal (Context.quietNaN x) = toReal x := by
  cases hnan : Model.isNaN (toModel x)
  · unfold toReal
    rw [toModel_quietNaN, Model.quietNaN_eq_self_of_isNaN_eq_false _ hnan]
  · have zero {v : Value} (h : Model.isNaN (toModel v) = true) : toReal v = 0 := by
      have hfinite : isFinite v = false := by
        cases hf : isFinite v
        · rfl
        · have := Model.isNaN_eq_false_of_isFinite_eq_true _ hf
          rw [h] at this
          exact absurd this (by decide)
      have hnone := (ExecFloat.Binary.toReal?_eq_none_iff v).2 hfinite
      unfold ExecFloat.Binary.toReal? at hnone
      unfold toReal Model.toReal
      rw [hnone]
    rw [zero hnan, zero (by rw [toModel_quietNaN, Model.isNaN_quietNaN]; exact hnan)]

/-- `toReal_quietNaN` as an absolute bound: the result carries the error of `x`. -/
theorem abs_toReal_quietNaN_sub_le {x : Value} {x₀ ε : ℝ} (ex : |toReal x - x₀| ≤ ε) :
    |toReal (Context.quietNaN x) - x₀| ≤ ε := by
  rw [toReal_quietNaN]
  exact ex

/-- `toReal_quietNaN` as a mixed bound: the result carries the error of `x`. -/
theorem mixedError_toReal_quietNaN {x : Value} {x₀ r a : ℝ} (ex : MixedError (toReal x) x₀ r a) :
    MixedError (toReal (Context.quietNaN x)) x₀ r a := by
  rw [toReal_quietNaN]
  exact ex

/-- The tie result of an extremum of configured values is the result itself. -/
theorem isFinite_extremaTie {x y r : Value} (hr : isFinite r = true) :
    isFinite (Context.extremaTie x y r) = true := hr

/-- The tie result carries the absolute error of the result. -/
theorem abs_toReal_extremaTie_sub_le {x y r : Value} {r₀ ε : ℝ} (er : |toReal r - r₀| ≤ ε) :
    |toReal (Context.extremaTie x y r) - r₀| ≤ ε := er

/-- The tie result carries the mixed error of the result. -/
theorem mixedError_toReal_extremaTie {x y r : Value} {r₀ e a : ℝ}
    (er : MixedError (toReal r) r₀ e a) :
    MixedError (toReal (Context.extremaTie x y r)) r₀ e a := er

/-- The IEEE maximum of finite configured values has the real value `max (toReal x) (toReal y)`. -/
theorem toReal_maximum {x y : Value} (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal (Context.maximum x y) = max (toReal x) (toReal y) := by
  have hr : ∀ r : Value, Context.extremaTie x y r = r := fun _ => rfl
  have hlt := ExecFloat.Binary.lt_iff_toReal_lt hx hy
  have hgt := ExecFloat.Binary.lt_iff_toReal_lt hy hx
  unfold Context.maximum
  rw [isNaN_of_isFinite hx, isNaN_of_isFinite hy]
  simp only [Bool.false_eq_true, ↓reduceIte, hr]
  by_cases h1 : y > x
  · simp only [h1, ↓reduceIte]
    exact (max_eq_right (hlt.1 h1).le).symm
  · by_cases h2 : x > y
    · simp only [h1, h2, ↓reduceIte]
      exact (max_eq_left (hgt.1 h2).le).symm
    · simp only [h1, h2, ↓reduceIte]
      have he : toReal x = toReal y :=
        le_antisymm (not_lt.1 (mt hgt.2 h2)) (not_lt.1 (mt hlt.2 h1))
      split_ifs <;> simp [he]

/-- The IEEE maximum of finite configured values is finite. -/
theorem isFinite_maximum {x y : Value} (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (Context.maximum x y) = true := by
  have hr : ∀ r : Value, Context.extremaTie x y r = r := fun _ => rfl
  unfold Context.maximum
  rw [isNaN_of_isFinite hx, isNaN_of_isFinite hy]
  simp only [Bool.false_eq_true, ↓reduceIte, hr]
  split_ifs <;> assumption

/-- The IEEE minimum of finite configured values has the real value `min (toReal x) (toReal y)`. -/
theorem toReal_minimum {x y : Value} (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal (Context.minimum x y) = min (toReal x) (toReal y) := by
  have hr : ∀ r : Value, Context.extremaTie x y r = r := fun _ => rfl
  have hlt := ExecFloat.Binary.lt_iff_toReal_lt hx hy
  have hgt := ExecFloat.Binary.lt_iff_toReal_lt hy hx
  unfold Context.minimum
  rw [isNaN_of_isFinite hx, isNaN_of_isFinite hy]
  simp only [Bool.false_eq_true, ↓reduceIte, hr]
  by_cases h1 : x > y
  · simp only [h1, ↓reduceIte]
    exact (min_eq_right (hgt.1 h1).le).symm
  · by_cases h2 : y > x
    · simp only [h1, h2, ↓reduceIte]
      exact (min_eq_left (hlt.1 h2).le).symm
    · simp only [h1, h2, ↓reduceIte]
      have he : toReal x = toReal y :=
        le_antisymm (not_lt.1 (mt hgt.2 h1)) (not_lt.1 (mt hlt.2 h2))
      split_ifs <;> simp [he]

/-- The IEEE minimum of finite configured values is finite. -/
theorem isFinite_minimum {x y : Value} (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (Context.minimum x y) = true := by
  have hr : ∀ r : Value, Context.extremaTie x y r = r := fun _ => rfl
  unfold Context.minimum
  rw [isNaN_of_isFinite hx, isNaN_of_isFinite hy]
  simp only [Bool.false_eq_true, ↓reduceIte, hr]
  split_ifs <;> assumption

/-- On a finite configured value, the IEEE maximum against zero, a clamp at zero such as a
normalization's variance clamp, has the real value `max (toReal x) 0`. -/
theorem toReal_maximum_zero {x : Value} (hx : isFinite x = true) :
    toReal (Context.maximum x 0) = max (toReal x) 0 := by
  rw [toReal_maximum hx ExecFloat.Binary.isFinite_zero, ExecFloat.Binary.toReal_zero]

/-- The IEEE maximum of a finite configured value and zero is finite. -/
theorem isFinite_maximum_zero {x : Value} (hx : isFinite x = true) :
    isFinite (Context.maximum x 0) = true :=
  isFinite_maximum hx ExecFloat.Binary.isFinite_zero

/-- Rounding one half to a format returns one half: it is `2 ^ (-1)`, and the format's least
exponent is at most `-1`. -/
private theorem roundAt_half : Model.roundAt format (1 / 2) = 1 / 2 := by
  have hb : (1 / 2 : ℝ) = Formats.Flocq.bpow binaryRadix (-1) := by
    simp [Formats.Flocq.bpow, binaryRadix, Radix.toReal]
  rw [Model.roundAt_eq_round, hb]
  apply Formats.Flocq.round_preserves_generic
  apply Formats.Flocq.generic_format_bpow
  have := format.exponentBias_pos
  have := format.fracWidth_pos
  simp only [Model.fexpOf, Formats.Flocq.fltExp, FloatFormat.minSubnormalExponent,
    FloatFormat.minNormalExponent, Int.ofNat_eq_natCast, max_le_iff]
  omega

/-- In an IEEE format the configured value of the rational `1 / 2` is finite. -/
theorem isFinite_ratCast_half (hformat : format.isIEEE = true) :
    isFinite ((1 / 2 : ℚ) : Value) = true :=
  ExecFloat.Binary.isFinite_ofRat (1 / 2)
    (le_trans (by norm_num) (Model.one_le_toReal_posMaxFinite format)) hformat

/-- In an IEEE format the configured value of the rational `1 / 2` has the real value `1 / 2`. -/
theorem toReal_ratCast_half (hformat : format.isIEEE = true) :
    toReal ((1 / 2 : ℚ) : Value) = ((1 / 2 : ℚ) : ℝ) := by
  have h := ExecFloat.Binary.toReal_ofRat_eq_roundAt (plan := plan) (code := code) (1 / 2)
    (isFinite_ratCast_half hformat) hformat
  have hq : (((1 / 2 : ℚ) : ℝ)) = 1 / 2 := by norm_num
  rw [hq, roundAt_half] at h
  rw [hq]
  exact h

/-- A left fold of the IEEE maximum over finite values is finite and decodes to the left fold of
`max` over their real values. -/
theorem toReal_foldl_maximum {ι : Type} (f : ι → ExecFloat (Configured.Family format code plan)) :
    ∀ (l : List ι) (start : ExecFloat (Configured.Family format code plan)),
      isFinite start = true → (∀ i ∈ l, isFinite (f i) = true) →
      isFinite (l.foldl (fun acc i => Context.maximum acc (f i)) start) = true ∧
        toReal (l.foldl (fun acc i => Context.maximum acc (f i)) start) =
          l.foldl (fun acc i => max acc (toReal (f i))) (toReal start)
  | [], _, hstart, _ => ⟨hstart, rfl⟩
  | a :: l, start, hstart, hf => by
      have ha := hf a (List.mem_cons_self ..)
      obtain ⟨hfinite, hreal⟩ := toReal_foldl_maximum f l (Context.maximum start (f a))
        (isFinite_maximum hstart ha) (fun i hi => hf i (List.mem_cons_of_mem _ hi))
      exact ⟨hfinite, by
        rw [List.foldl_cons, List.foldl_cons, hreal, toReal_maximum hstart ha]⟩

/-- A finite `hypot` of operands of values `a` and `b` is `√(a² + b²)` rounded once. -/
theorem toReal_hypot {x y : Value} {a b : ℝ} (hformat : format.isIEEE = true)
    (hx : toReal x = a) (hy : toReal y = b) (hfinite : isFinite (Context.hypot x y) = true) :
    toReal (Context.hypot x y) = Model.roundAt format (√(a ^ 2 + b ^ 2)) := by
  subst hx hy
  exact ExecFloat.Binary.toReal_hypot_eq_roundAt x y hfinite

/-- `toReal_hypot` as an absolute bound for exact operands. -/
theorem abs_toReal_hypot_sub_le {x y : Value} {a b : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0) (hy : |toReal y - b| ≤ 0)
    (hfinite : isFinite (Context.hypot x y) = true) :
    |toReal (Context.hypot x y) - √(a ^ 2 + b ^ 2)| ≤
      Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) := by
  rw [toReal_hypot hformat (by simpa [sub_eq_zero] using hx) (by simpa [sub_eq_zero] using hy)
    hfinite]
  exact Model.abs_roundAt_sub_le _ _

/-- `toReal_hypot` as a mixed bound for exact operands. -/
theorem mixedError_toReal_hypot {x y : Value} {a b : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0) (hy : MixedError (toReal y) b 0 0)
    (hfinite : isFinite (Context.hypot x y) = true) :
    MixedError (toReal (Context.hypot x y)) (√(a ^ 2 + b ^ 2)) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) :=
  ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| toReal_hypot hformat
    (mixedError_zero_zero_iff.mp hx) (mixedError_zero_zero_iff.mp hy) hfinite

/-- A finite `hypot` of operands within `εx` and `εy` of `a` and `b` is within half an ulp of its
exact value plus `εx + εy` of `√(a² + b²)`, the norm being 1-Lipschitz in each coordinate. -/
theorem abs_toReal_hypot_sub_le_add {x y : Value} {a b εx εy : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ εx) (hy : |toReal y - b| ≤ εy)
    (hfinite : isFinite (Context.hypot x y) = true) :
    |toReal (Context.hypot x y) - √(a ^ 2 + b ^ 2)| ≤
      Model.halfUlpAt format (√(toReal x ^ 2 + toReal y ^ 2)) + (εx + εy) := by
  rw [toReal_hypot hformat rfl rfl hfinite]
  have h := (abs_norm_sub_norm_le (⟨toReal x, toReal y⟩ : ℂ) ⟨a, b⟩).trans
    (Complex.norm_le_abs_re_add_abs_im _)
  simp only [Complex.norm_def, Complex.normSq_mk, Complex.sub_re, Complex.sub_im] at h
  rw [show toReal x * toReal x + toReal y * toReal y = toReal x ^ 2 + toReal y ^ 2 by ring,
    show a * a + b * b = a ^ 2 + b ^ 2 by ring] at h
  calc _ ≤ |Model.roundAt format (√(toReal x ^ 2 + toReal y ^ 2)) -
          √(toReal x ^ 2 + toReal y ^ 2)| +
        |√(toReal x ^ 2 + toReal y ^ 2) - √(a ^ 2 + b ^ 2)| := abs_sub_le _ _ _
    _ ≤ _ := add_le_add (Model.abs_roundAt_sub_le _ _) (h.trans (add_le_add hx hy))

/-- If `p` is `a * a` rounded, `s` is `p + 1` rounded and `r` is `b / s` rounded, then `r` is within
`(k - 1 + u) / (2 - k) * |b / (a * a + 1)| + η` of `b / (a * a + 1)`, where
`k = (1 + u) * (1 + u + η)`. -/
private theorem abs_sub_div_le_of_rounded_denominator {u η a b p s r : ℝ} (hu : 0 ≤ u)
    (hη : 0 ≤ η) (hk : (1 + u) * (1 + u + η) < 2)
    (hp : MixedError p (a * a) u η) (hp0 : 0 ≤ p)
    (hs : MixedError s (p + 1) u 0) (hs1 : 1 ≤ s)
    (hr : MixedError r (b / s) u η) :
    |r - b / (a * a + 1)| ≤
      ((1 + u) * (1 + u + η) - 1 + u) / (2 - (1 + u) * (1 + u + η)) * |b / (a * a + 1)| + η := by
  obtain ⟨θ, hθ⟩ : ∃ θ, θ = (1 + u) * (1 + u + η) - 1 := ⟨_, rfl⟩
  have hD : 2 - (1 + u) * (1 + u + η) = 1 - θ := by rw [hθ]; ring
  rw [← hθ, hD]
  have ha2 : 0 ≤ a * a := mul_self_nonneg a
  have hc : 0 < a * a + 1 := by linarith
  have hθ0 : 0 ≤ θ := by rw [hθ]; nlinarith
  have hθ1 : θ < 1 := by linarith
  have hpb := hp.bound
  have hsb := hs.bound
  rw [abs_of_nonneg ha2] at hpb
  rw [abs_of_nonneg (by linarith : 0 ≤ p + 1), add_zero] at hsb
  have hp' : p ≤ a * a + (u * (a * a) + η) := by linarith [le_abs_self (p - a * a)]
  have he : |a * a + 1 - s| ≤ θ * (a * a + 1) := by
    have h1 : |a * a + 1 - s| ≤ |a * a - p| + |p + 1 - s| := by
      calc |a * a + 1 - s| = |(a * a - p) + (p + 1 - s)| := by congr 1; ring
        _ ≤ _ := abs_add_le _ _
    rw [abs_sub_comm (a * a) p, abs_sub_comm (p + 1) s] at h1
    calc |a * a + 1 - s| ≤ (u * (a * a) + η) + u * (p + 1) := h1.trans (add_le_add hpb hsb)
      _ ≤ θ * (a * a + 1) := by rw [hθ]; nlinarith [mul_nonneg hu hη, mul_nonneg hu ha2]
  have hbs : |b / s - b / (a * a + 1)| ≤ θ * |b / s| := by
    have hs0 : s ≠ 0 := by positivity
    have heq : b / s - b / (a * a + 1) = b / s * ((a * a + 1 - s) / (a * a + 1)) := by
      field_simp
    rw [heq, abs_mul, mul_comm θ]
    refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
    rw [abs_div, abs_of_pos hc, div_le_iff₀ hc]
    exact he
  have hB : |b / s| ≤ |b / (a * a + 1)| / (1 - θ) := by
    rw [le_div_iff₀ (by linarith)]
    have := abs_sub_abs_le_abs_sub (b / s) (b / (a * a + 1))
    nlinarith
  calc |r - b / (a * a + 1)| ≤ |r - b / s| + |b / s - b / (a * a + 1)| := abs_sub_le _ _ _
    _ ≤ (u * |b / s| + η) + θ * |b / s| := add_le_add hr.bound hbs
    _ = (θ + u) * |b / s| + η := by ring
    _ ≤ (θ + u) * (|b / (a * a + 1)| / (1 - θ)) + η := by gcongr
    _ = (θ + u) / (1 - θ) * |b / (a * a + 1)| + η := by ring

/-- If `(1 + u) * (a * a)` is at least `M - 1 - η` and `r` is `b / a / a` rounded twice, then `r` is
within `(2 * u + u ^ 2 + 4 / M) * |b / (a * a + 1)| + (2 + u) * η` of `b / (a * a + 1)`. -/
private theorem abs_sub_div_le_of_overflow {u η M a b t r : ℝ} (hu : 0 ≤ u) (hη : 0 ≤ η)
    (hM : 0 < M) (ha : M - 1 - η ≤ (1 + u) * (a * a))
    (hMa : (1 + u) ^ 3 * M ≤ 4 * (M - 1 - η)) (ha1 : 1 ≤ a * a)
    (ht : MixedError t (b / a) u η) (hr : MixedError r (t / a) u η) :
    |r - b / (a * a + 1)| ≤ (2 * u + u ^ 2 + 4 / M) * |b / (a * a + 1)| + (2 + u) * η := by
  have habs : 1 ≤ |a| := by
    by_contra h
    nlinarith [abs_mul_abs_self a, abs_nonneg a, not_le.mp h]
  have ha0 : 0 < |a| := by linarith
  have hA : 0 < a * a := by linarith
  have hane : a ≠ 0 := fun h => by simp [h] at hA
  have hc : 0 < a * a + 1 := by linarith
  have hZ : |b / a / a| = |b / (a * a + 1)| + |b / (a * a + 1)| / (a * a) := by
    rw [abs_div, abs_div, abs_div, abs_of_pos hc, div_div, abs_mul_abs_self]
    field_simp
  have h1 : |t / a - b / a / a| ≤ u * |b / a / a| + η := by
    rw [← sub_div, abs_div, div_le_iff₀ ha0, abs_div (b / a) a]
    have hexp : (u * (|b / a| / |a|) + η) * |a| = u * |b / a| + η * |a| := by
      rw [add_mul, mul_assoc, div_mul_cancel₀ _ ha0.ne']
    rw [hexp]
    nlinarith [ht.bound]
  have h2 : |r - b / a / a| ≤ (2 * u + u ^ 2) * |b / a / a| + (2 + u) * η := by
    have h2a : |t / a| ≤ (1 + u) * |b / a / a| + η := by
      have := abs_sub_abs_le_abs_sub (t / a) (b / a / a)
      linarith
    calc |r - b / a / a| ≤ |r - t / a| + |t / a - b / a / a| := abs_sub_le _ _ _
      _ ≤ (u * |t / a| + η) + (u * |b / a / a| + η) := add_le_add hr.bound h1
      _ ≤ (u * ((1 + u) * |b / a / a| + η) + η) + (u * |b / a / a| + η) := by gcongr
      _ = _ := by ring
  have h3 : |b / a / a - b / (a * a + 1)| = |b / (a * a + 1)| / (a * a) := by
    rw [div_div, show b / (a * a) - b / (a * a + 1) = b / (a * a + 1) / (a * a) by
      field_simp; ring]
    rw [abs_div, abs_of_pos hA]
  have h4 : (1 + u) ^ 2 * M ≤ 4 * (a * a) := by
    have h1u : 0 < 1 + u := by linarith
    have : (1 + u) * ((1 + u) ^ 2 * M) ≤ (1 + u) * (4 * (a * a)) := by nlinarith
    exact le_of_mul_le_mul_left this h1u
  have h5 : (1 + u) ^ 2 * (|b / (a * a + 1)| / (a * a)) ≤ 4 / M * |b / (a * a + 1)| := by
    have hY0 := abs_nonneg (b / (a * a + 1))
    have : (1 + u) ^ 2 ≤ 4 / M * (a * a) := by
      rw [div_mul_eq_mul_div, le_div_iff₀ hM]
      linarith
    rw [mul_div_assoc', div_le_iff₀ hA]
    nlinarith
  calc |r - b / (a * a + 1)| ≤ |r - b / a / a| + |b / a / a - b / (a * a + 1)| :=
        abs_sub_le _ _ _
    _ ≤ ((2 * u + u ^ 2) * |b / a / a| + (2 + u) * η) + |b / (a * a + 1)| / (a * a) := by
        rw [h3]; linarith
    _ = (2 * u + u ^ 2) * |b / (a * a + 1)| +
          (1 + u) ^ 2 * (|b / (a * a + 1)| / (a * a)) + (2 + u) * η := by rw [hZ]; ring
    _ ≤ (2 * u + u ^ 2) * |b / (a * a + 1)| + 4 / M * |b / (a * a + 1)| + (2 + u) * η := by
        linarith
    _ = _ := by ring

/-- The largest finite value of an IEEE format is at least three. -/
private theorem three_le_toReal_posMaxFinite (hformat : format.isIEEE = true) :
    3 ≤ Model.toReal (Model.posMaxFinite format) := by
  obtain ⟨hencoding, -⟩ := (FloatFormat.isIEEE_eq_true_iff format).mp hformat
  rw [Model.toReal_posMaxFinite, FloatFormat.maxNormalExponent_eq_ieee format hformat]
  simp only [FloatFormat.maxFiniteFracField, FloatFormat.fracMaskNat, hencoding,
    Model.pow2_eq_two_pow, FloatFormat.ieeeMaxNormalExponent, Model.bpow,
    FloatLib.Floats.Formats.Flocq.bpow_binaryRadix]
  have hX : (2 : ℝ) ≤ 2 ^ format.fracWidth := by
    calc (2 : ℝ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ format.fracWidth := pow_le_pow_right₀ one_le_two format.fracWidth_pos
  have hY : (2 : ℝ) ≤ 2 ^ format.bias := by
    calc (2 : ℝ) = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ format.bias := pow_le_pow_right₀ one_le_two format.bias_pos
  rw [Nat.cast_add, Nat.cast_sub Nat.one_le_two_pow]
  push_cast
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast, zpow_sub₀ two_ne_zero, zpow_natCast,
    zpow_natCast, ← mul_div_assoc, le_div_iff₀ (by positivity)]
  nlinarith

/-- In an IEEE format the unit roundoff and the underflow error are at most `1 / 4` and the largest
finite value is at least three. -/
private theorem atanBackward_constants (hformat : format.isIEEE = true) :
    Model.unitRoundoffAt format ≤ 1 / 4 ∧ Model.underflowErrorAt format ≤ 1 / 4 ∧
      3 ≤ Model.toReal (Model.posMaxFinite format) := by
  refine ⟨?_, ?_, three_le_toReal_posMaxFinite hformat⟩
  · rw [Model.unitRoundoffAt_eq, show (1 / 4 : ℝ) = 2 ^ (-2 : ℤ) by norm_num]
    exact zpow_le_zpow_right₀ one_le_two (by have := format.fracWidth_pos; omega)
  · rw [show (1 / 4 : ℝ) = 2 ^ (-2 : ℤ) by norm_num]
    refine Model.underflowErrorAt_le_zpow format ?_
    have := format.exponentBias_pos
    have := format.fracWidth_pos
    simp only [FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent,
      Int.ofNat_eq_natCast]
    omega

open TorchLean.Numeric (atanBackward hypotBackward angleTangent)

/-- When `x * x + 1` is finite, `atanBackward x g` is `g / (x * x + 1)`. -/
private theorem atanBackward_of_isFinite {x g : Value} (hformat : format.isIEEE = true)
    (hd : isFinite (x * x + 1) = true) : atanBackward x g = g / (x * x + 1) := by
  have hcond : (x * x + 1 - (x * x + 1) == 0) = true :=
    ExecFloat.Binary.beq_zero_iff.mpr ⟨ExecFloat.Binary.isFinite_sub_self hd hformat,
      ExecFloat.Binary.toReal_sub_self hd hformat⟩
  unfold atanBackward
  exact ite_eq_left hcond

/-- When `x * x + 1` is not finite, `atanBackward x g` is `g / x / x`. -/
private theorem atanBackward_of_not_isFinite {x g : Value} (hformat : format.isIEEE = true)
    (hd : isFinite (x * x + 1) = false) : atanBackward x g = g / x / x := by
  have hcond : (x * x + 1 - (x * x + 1) == 0) ≠ true := by
    intro h
    have hsub := (ExecFloat.Binary.beq_zero_iff.mp h).1
    unfold isFinite at hsub hd
    simp only [ExecFloat.Binary.toModel_hSub] at hsub
    have := (Model.isFinite_of_isFinite_sub hformat hsub).1
    rw [hd] at this
    exact absurd this (by decide)
  unfold atanBackward
  exact ite_eq_right hcond

/-- On finite operands `atanBackward x g` is finite and within `R * |c| + (2 + u) * η` of
`c = toReal g / (toReal x * toReal x + 1)`, with `R` of `abs_toReal_atanBackward_sub_le`. -/
private theorem atanBackward_core {x g : Value} (hformat : format.isIEEE = true)
    (hx : isFinite x = true) (hg : isFinite g = true) :
    isFinite (atanBackward x g) = true ∧
      |toReal (atanBackward x g) - toReal g / (toReal x * toReal x + 1)| ≤
        (((1 + Model.unitRoundoffAt format) *
              (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format) - 1 +
            Model.unitRoundoffAt format) /
            (2 - (1 + Model.unitRoundoffAt format) *
              (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format)) +
          4 / Model.toReal (Model.posMaxFinite format)) *
          |toReal g / (toReal x * toReal x + 1)| +
        (2 + Model.unitRoundoffAt format) * Model.underflowErrorAt format := by
  obtain ⟨hu4, hη4, hM3⟩ := atanBackward_constants hformat
  have hu := (Model.unitRoundoffAt_pos format).le
  have hη := (Model.underflowErrorAt_pos format).le
  have hbM := ExecFloat.Binary.abs_toReal_le_posMaxFinite_of_isFinite hg
  have hk1 : 1 ≤ (1 + Model.unitRoundoffAt format) *
      (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format) := by nlinarith
  have hk2 : (1 + Model.unitRoundoffAt format) *
      (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format) ≤ 15 / 8 := by nlinarith
  have h4M : 0 ≤ 4 / Model.toReal (Model.posMaxFinite format) :=
    div_nonneg (by norm_num) (by linarith)
  have hY := abs_nonneg (toReal g / (toReal x * toReal x + 1))
  by_cases hd : isFinite (x * x + 1 : Value) = true
  · rw [atanBackward_of_isFinite hformat hd]
    have hp : isFinite (x * x : Value) = true := by
      have hd' := hd
      unfold isFinite at hd' ⊢
      simp only [ExecFloat.Binary.toModel_hAdd] at hd'
      exact (Model.isFinite_of_isFinite_add hformat hd').1
    have hmul := ExecFloat.Binary.mul_rounding hp hformat
    have hadd := ExecFloat.Binary.add_rounding hd hformat
    rw [ExecFloat.Binary.toReal_one hformat] at hadd
    have hp0 : 0 ≤ toReal (x * x : Value) := by
      rw [ExecFloat.Binary.toReal_mul_eq_roundAt hp hformat]
      exact Model.roundAt_nonneg format (mul_self_nonneg _)
    have hs1 : 1 ≤ toReal (x * x + 1 : Value) := by
      rw [ExecFloat.Binary.toReal_add_eq_roundAt hd hformat, ExecFloat.Binary.toReal_one hformat]
      calc (1 : ℝ) = Model.roundAt format 1 := (Model.roundAt_one format).symm
        _ ≤ _ := Model.roundAt_mono format (by linarith)
    have hz : ExecFloat.Binary.isZero (x * x + 1 : Value) = false := by
      cases h : ExecFloat.Binary.isZero (x * x + 1 : Value)
      · rfl
      · rw [ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hd] at h
        linarith
    have hq : isFinite (g / (x * x + 1) : Value) = true := by
      refine ExecFloat.Binary.isFinite_div_of_mixedError hg hd hz (.refl _) ?_ hformat
      simp only [add_zero, one_mul, abs_div]
      rw [abs_of_pos (by linarith : (0 : ℝ) < toReal (x * x + 1 : Value))]
      exact (div_le_self (abs_nonneg _) hs1).trans hbM
    have hdiv := ExecFloat.Binary.div_rounding hd hq hformat
    have hA := abs_sub_div_le_of_rounded_denominator hu hη (by linarith) hmul hp0 hadd hs1 hdiv
    refine ⟨hq, hA.trans ?_⟩
    nlinarith [mul_nonneg h4M hY, mul_nonneg hu hη]
  · have hd' : isFinite (x * x + 1 : Value) = false := by simpa using hd
    rw [atanBackward_of_not_isFinite hformat hd']
    have hlow : Model.toReal (Model.posMaxFinite format) - 1 - Model.underflowErrorAt format ≤
        (1 + Model.unitRoundoffAt format) * (toReal x * toReal x) := by
      have ha2 := mul_self_nonneg (toReal x)
      by_cases hp : isFinite (x * x : Value) = true
      · have hmul := (ExecFloat.Binary.mul_rounding hp hformat).bound
        have hp0 : 0 ≤ toReal (x * x : Value) := by
          rw [ExecFloat.Binary.toReal_mul_eq_roundAt hp hformat]
          exact Model.roundAt_nonneg format (mul_self_nonneg _)
        have hover : ¬ ((1 + 0) * |toReal (x * x : Value) + toReal (1 : Value)| + 0 ≤
            Model.toReal (Model.posMaxFinite format)) := fun h =>
          hd (ExecFloat.Binary.isFinite_add_of_mixedError hp
            (ExecFloat.Binary.isFinite_one hformat) (.refl _) h hformat)
        rw [ExecFloat.Binary.toReal_one hformat] at hover
        simp only [add_zero, one_mul, not_le] at hover
        rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ toReal (x * x : Value) + 1)] at hover
        rw [abs_of_nonneg ha2] at hmul
        have := le_abs_self (toReal (x * x : Value) - toReal x * toReal x)
        nlinarith
      · have hover : ¬ ((1 + 0) * |toReal x * toReal x| + 0 ≤
            Model.toReal (Model.posMaxFinite format)) := fun h =>
          hp (ExecFloat.Binary.isFinite_mul_of_mixedError hx hx (.refl _) h hformat)
        simp only [add_zero, one_mul, not_le] at hover
        rw [abs_of_nonneg ha2] at hover
        nlinarith [mul_nonneg hu ha2]
    have ha1 : 1 ≤ toReal x * toReal x := by
      nlinarith [mul_le_mul_of_nonneg_right hu4 (mul_self_nonneg (toReal x))]
    have hMa : (1 + Model.unitRoundoffAt format) ^ 3 * Model.toReal (Model.posMaxFinite format) ≤
        4 * (Model.toReal (Model.posMaxFinite format) - 1 - Model.underflowErrorAt format) := by
      have h3 : (1 + Model.unitRoundoffAt format) ^ 3 ≤ (5 / 4) ^ 3 :=
        pow_le_pow_left₀ (by linarith) (by linarith) 3
      nlinarith [mul_le_mul_of_nonneg_right h3 (by linarith : (0 : ℝ) ≤
        Model.toReal (Model.posMaxFinite format))]
    have habs : 1 ≤ |toReal x| := by
      nlinarith [abs_mul_abs_self (toReal x), abs_nonneg (toReal x)]
    have hzx : ExecFloat.Binary.isZero x = false := by
      cases h : ExecFloat.Binary.isZero x
      · rfl
      · rw [ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hx] at h
        rw [h] at ha1
        norm_num at ha1
    have htf : isFinite (g / x : Value) = true := by
      refine ExecFloat.Binary.isFinite_div_of_mixedError hg hx hzx (.refl _) ?_ hformat
      simp only [add_zero, one_mul, abs_div]
      exact (div_le_self (abs_nonneg _) habs).trans hbM
    have ht := ExecFloat.Binary.div_rounding hx htf hformat
    have hrf : isFinite (g / x / x : Value) = true := by
      refine ExecFloat.Binary.isFinite_div_of_mixedError htf hx hzx (.refl _) ?_ hformat
      simp only [add_zero, one_mul, abs_div]
      exact (div_le_self (abs_nonneg _) habs).trans
        (ExecFloat.Binary.abs_toReal_le_posMaxFinite_of_isFinite htf)
    have hr := ExecFloat.Binary.div_rounding hx hrf hformat
    have hB := abs_sub_div_le_of_overflow hu hη (by linarith) hlow hMa ha1 ht hr
    refine ⟨hrf, hB.trans ?_⟩
    gcongr ?_ * _ + _
    gcongr ?_ + _
    rw [le_div_iff₀ (by linarith)]
    nlinarith [mul_nonneg (by positivity : (0 : ℝ) ≤ 2 * Model.unitRoundoffAt format +
      Model.unitRoundoffAt format ^ 2) (by linarith : (0 : ℝ) ≤ (1 + Model.unitRoundoffAt format) *
        (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format) - 1)]

/-- In an IEEE format, `atanBackward x g` of finite configured values is finite. -/
theorem isFinite_atanBackward {x g : Value} (hformat : format.isIEEE = true)
    (hx : isFinite x = true) (hg : isFinite g = true) : isFinite (atanBackward x g) = true :=
  (atanBackward_core hformat hx hg).1

/-- In an IEEE format, `atanBackward x g` of exact finite operands of values `a` and `b` is within
`R * |b / (a * a + 1)| + (2 + u) * η` of `b / (a * a + 1)`. Here `u` is the unit roundoff, `η` the
underflow error, `M` the largest finite value, `k = (1 + u) * (1 + u + η)` and
`R = (k - 1 + u) / (2 - k) + 4 / M`. -/
theorem abs_toReal_atanBackward_sub_le {x g : Value} {a b : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0) (hg : |toReal g - b| ≤ 0)
    (hxf : isFinite x = true) (hgf : isFinite g = true) :
    |toReal (atanBackward x g) - b / (a * a + 1)| ≤
      (((1 + Model.unitRoundoffAt format) *
            (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format) - 1 +
          Model.unitRoundoffAt format) /
          (2 - (1 + Model.unitRoundoffAt format) *
            (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format)) +
        4 / Model.toReal (Model.posMaxFinite format)) * |b / (a * a + 1)| +
      (2 + Model.unitRoundoffAt format) * Model.underflowErrorAt format := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  have hb : toReal g = b := by simpa [sub_eq_zero] using hg
  subst ha hb
  exact (atanBackward_core hformat hxf hgf).2

/-- `abs_toReal_atanBackward_sub_le` as a mixed bound for exact finite operands: relative error `R`,
absolute error `(2 + u) * η`. -/
theorem mixedError_toReal_atanBackward {x g : Value} {a b : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0) (hg : MixedError (toReal g) b 0 0)
    (hxf : isFinite x = true) (hgf : isFinite g = true) :
    MixedError (toReal (atanBackward x g)) (b / (a * a + 1))
      (((1 + Model.unitRoundoffAt format) *
            (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format) - 1 +
          Model.unitRoundoffAt format) /
          (2 - (1 + Model.unitRoundoffAt format) *
            (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format)) +
        4 / Model.toReal (Model.posMaxFinite format))
      ((2 + Model.unitRoundoffAt format) * Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  have hb := mixedError_zero_zero_iff.mp hg
  subst ha hb
  obtain ⟨hu4, hη4, hM3⟩ := atanBackward_constants hformat
  have hu := (Model.unitRoundoffAt_pos format).le
  have hη := (Model.underflowErrorAt_pos format).le
  have hk : (1 + Model.unitRoundoffAt format) *
      (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format) ≤ 15 / 8 := by nlinarith
  have hk1 : 1 ≤ (1 + Model.unitRoundoffAt format) *
      (1 + Model.unitRoundoffAt format + Model.underflowErrorAt format) := by nlinarith
  refine ⟨add_nonneg (div_nonneg (by linarith) (by linarith))
    (div_nonneg (by norm_num) (by linarith)), by positivity, ?_⟩
  exact (atanBackward_core hformat hxf hgf).2

/-- A real fixed by rounding is in magnitude at most the rounded square root of any `s ≥ a ^ 2`. -/
private theorem abs_le_roundAt_sqrt {a s : ℝ} (ha : Model.roundAt format a = a) (hs : a ^ 2 ≤ s) :
    |a| ≤ Model.roundAt format (√s) := by
  have habs : Model.roundAt format |a| = |a| := by
    rcases abs_cases a with ⟨h, _⟩ | ⟨h, _⟩ <;> rw [h]
    · exact ha
    · rw [Model.roundAt_neg, Model.IEEERoundingMode.negate_nearest, ha]
  calc |a| = Model.roundAt format |a| := habs.symm
    _ ≤ Model.roundAt format (√s) :=
      Model.roundAt_mono format (by rw [← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt hs)

/-- A rounding of normal magnitude is within `u / (1 - u)` of its argument relative to the
argument, `u` the unit roundoff. -/
private theorem abs_roundAt_sub_le_of_minNormalAt_le {t : ℝ}
    (ht : Model.minNormalAt format ≤ |Model.roundAt format t|) :
    |Model.roundAt format t - t| ≤
      Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) * |t| := by
  have hu := Model.unitRoundoffAt_pos format
  have hu1 := Model.unitRoundoffAt_lt_one format
  have herr := Model.abs_roundAt_sub_le format t
  have he := abs_nonneg (Model.roundAt format t - t)
  rw [div_mul_eq_mul_div, le_div_iff₀ (by linarith)]
  by_cases hnormal : Model.minNormalAt format ≤ |t|
  · have hulp := Model.ulpAt_le_of_minNormalAt_le format hnormal
    rw [← Model.two_mul_unitRoundoffAt] at hulp
    rw [Model.halfUlpAt] at herr
    nlinarith [mul_nonneg hu.le he]
  · rw [Model.halfUlpAt, Model.ulpAt_eq_of_abs_lt_minNormalAt format (not_le.mp hnormal),
      ← Model.two_mul_underflowErrorAt] at herr
    have hη : Model.underflowErrorAt format =
        Model.unitRoundoffAt format * Model.minNormalAt format := by
      rw [Model.underflowErrorAt_eq, Model.unitRoundoffAt_eq, Model.minNormalAt,
        FloatLib.Floats.Formats.Flocq.bpow_binaryRadix, ← zpow_add₀ two_ne_zero]
      congr 1
      simp only [FloatFormat.minSubnormalExponent, Int.ofNat_eq_natCast]
      ring
    have htri := abs_sub_abs_le_abs_sub (Model.roundAt format t) t
    nlinarith [mul_le_mul_of_nonneg_left ht hu.le, mul_le_mul_of_nonneg_left htri hu.le]

/-- Dividing `t` by a rounded radius `ρ` with `max |a| |b| ≤ ρ` instead of by `R = √(a² + b²)`
changes `t / R` by at most `halfUlpAt R / max |a| |b| * |t / R|`. -/
private theorem abs_div_hypot_sub_le {a b t ρ : ℝ} (hρ0 : ρ ≠ 0) (hmρ : max |a| |b| ≤ ρ)
    (hρ : ρ = Model.roundAt format (√(a ^ 2 + b ^ 2))) :
    |t / ρ - t / √(a ^ 2 + b ^ 2)| ≤
      Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b| * |t / √(a ^ 2 + b ^ 2)| := by
  have hR0 : √(a ^ 2 + b ^ 2) ≠ 0 := fun h => hρ0 (by rw [hρ, h, Model.roundAt_zero])
  have hR : 0 < √(a ^ 2 + b ^ 2) := lt_of_le_of_ne (Real.sqrt_nonneg _) (Ne.symm hR0)
  have hm : 0 < max |a| |b| := by
    refine lt_of_le_of_ne ((abs_nonneg a).trans (le_max_left _ _)) fun hzero => hR0 ?_
    have ha : a = 0 := abs_nonpos_iff.mp ((le_max_left _ _).trans hzero.symm.le)
    have hb : b = 0 := abs_nonpos_iff.mp ((le_max_right _ _).trans hzero.symm.le)
    rw [ha, hb]
    norm_num
  have hρpos : 0 < ρ := hm.trans_le hmρ
  have hh := Model.abs_roundAt_sub_le format (√(a ^ 2 + b ^ 2))
  rw [← hρ] at hh
  have key : t / ρ - t / √(a ^ 2 + b ^ 2) =
      t / √(a ^ 2 + b ^ 2) * ((√(a ^ 2 + b ^ 2) - ρ) / ρ) := by
    field_simp
  rw [key, abs_mul, abs_div (√(a ^ 2 + b ^ 2) - ρ) ρ, abs_of_pos hρpos,
    abs_sub_comm (√(a ^ 2 + b ^ 2)) ρ, mul_comm (Model.halfUlpAt format _ / _)]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  rw [div_le_div_iff₀ hρpos hm]
  nlinarith [mul_le_mul_of_nonneg_right hh hm.le,
    mul_le_mul_of_nonneg_left hmρ ((abs_nonneg _).trans hh)]

/-- Relative perturbations by `v` and `e` after an absolute one of `E * w`, with `|t| ≤ w`,
compose to `((1 + v) * (1 + e) * (1 + E) - 1) * w`. -/
private theorem abs_sub_le_of_three {v e E w q p s t : ℝ} (hv : 0 ≤ v) (he : 0 ≤ e)
    (h1 : |q - p| ≤ v * |p|) (h2 : |p - s| ≤ e * |s|) (h3 : |s - t| ≤ E * w) (ht : |t| ≤ w) :
    |q - t| ≤ ((1 + v) * (1 + e) * (1 + E) - 1) * w := by
  have hs : |s| ≤ (1 + E) * w := by have := abs_sub_abs_le_abs_sub s t; linarith
  have hp : |p| ≤ (1 + e) * |s| := by have := abs_sub_abs_le_abs_sub p s; linarith
  have hqt : |q - t| ≤ |q - p| + |p - s| + |s - t| := by
    have h := abs_sub_le q p t
    have h' := abs_sub_le p s t
    linarith
  have hc : 0 ≤ v * (1 + e) + e := by positivity
  nlinarith [mul_le_mul_of_nonneg_left hp hv, mul_le_mul_of_nonneg_left hs hc]

/-- A rounding within `u * |s| + η` of `s`, after a relative perturbation by `e`, is within
`((1 + e) * (1 + v) ^ 2 - 1) * |t| + η` of `t` once `u ≤ v`. -/
private theorem abs_sub_le_of_two {u v e η q s t : ℝ} (hu : 0 ≤ u) (huv : u ≤ v) (he : 0 ≤ e)
    (h1 : |q - s| ≤ u * |s| + η) (h3 : |s - t| ≤ e * |t|) :
    |q - t| ≤ ((1 + e) * (1 + v) ^ 2 - 1) * |t| + η := by
  have hs : |s| ≤ (1 + e) * |t| := by have := abs_sub_abs_le_abs_sub s t; linarith
  have hqt := abs_sub_le q s t
  have hv : 0 ≤ v := hu.trans huv
  have ht := abs_nonneg t
  nlinarith [mul_le_mul_of_nonneg_left hs hu, mul_nonneg (mul_nonneg (sub_nonneg.mpr huv)
    (by linarith : (0 : ℝ) ≤ 1 + e)) ht, mul_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ 1 + e)
    (add_nonneg hv (sq_nonneg v))) ht]

/-- The unit roundoff `u` is at most `u / (1 - u)`, which is nonnegative. -/
private theorem unitRoundoffAt_le_div :
    Model.unitRoundoffAt format ≤
      Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) := by
  have hu := Model.unitRoundoffAt_pos format
  have hu1 := Model.unitRoundoffAt_lt_one format
  rw [le_div_iff₀ (by linarith)]
  nlinarith

/-- The relative error bound of `hypotBackward` is nonnegative. -/
private theorem hypotBackward_relative_nonneg {a b : ℝ} :
    0 ≤ (1 + Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b|) *
      (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1 := by
  have he : 0 ≤ Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b| :=
    div_nonneg ((abs_nonneg _).trans (Model.abs_roundAt_sub_le _ _))
      ((abs_nonneg a).trans (le_max_left _ _))
  have hv := (Model.unitRoundoffAt_pos format).le.trans (unitRoundoffAt_le_div (format := format))
  have h1 : 1 ≤ (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 :=
    one_le_pow₀ (by linarith)
  nlinarith

/-- At a zero radius the coordinate is zero, so the reference is zero. -/
private theorem hypot_zero_bound {a b c w ρ q : ℝ} (hρ : ρ = 0) (hw : |w| ≤ max |a| |b|)
    (hmρ : max |a| |b| ≤ ρ) (hq : q = 0) :
    |q - c * w / √(a ^ 2 + b ^ 2)| ≤
      ((1 + Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b|) *
          (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1) *
        |c * w / √(a ^ 2 + b ^ 2)| + Model.underflowErrorAt format := by
  have hw0 : w = 0 := abs_nonpos_iff.mp (hw.trans (hρ ▸ hmρ))
  rw [hq, hw0, mul_zero, zero_div, sub_zero, abs_zero, mul_zero, zero_add]
  exact (Model.underflowErrorAt_pos format).le

/-- Two normal roundings, of `c * w` and of its quotient by the rounded radius, are within the
relative bound of `hypotBackward` of `c * w / √(a² + b²)`. -/
private theorem hypot_flagless_bound {a b c w ρ p q : ℝ} (hρ0 : ρ ≠ 0)
    (hmρ : max |a| |b| ≤ ρ) (hρ : ρ = Model.roundAt format (√(a ^ 2 + b ^ 2)))
    (hp : p = Model.roundAt format (c * w)) (hq : q = Model.roundAt format (p / ρ))
    (hn : (c * w = 0 ∧ p = 0) ∨
      (Model.minNormalAt format ≤ |p| ∧ Model.minNormalAt format ≤ |q|)) :
    |q - c * w / √(a ^ 2 + b ^ 2)| ≤
      ((1 + Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b|) *
          (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1) *
        |c * w / √(a ^ 2 + b ^ 2)| + Model.underflowErrorAt format := by
  have hη := (Model.underflowErrorAt_pos format).le
  rcases hn with ⟨hcw, hp0⟩ | ⟨hpn, hqn⟩
  · rw [hq, hp0, zero_div, Model.roundAt_zero, hcw, zero_div, sub_zero, abs_zero, mul_zero,
      zero_add]
    exact hη
  · have hρpos : 0 < |ρ| := abs_pos.mpr hρ0
    have h1 : |q - p / ρ| ≤ Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) *
        |p / ρ| := by
      rw [hq] at hqn ⊢
      exact abs_roundAt_sub_le_of_minNormalAt_le hqn
    have h2' : |p - c * w| ≤ Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) *
        |c * w| := by
      rw [hp] at hpn ⊢
      exact abs_roundAt_sub_le_of_minNormalAt_le hpn
    have h2 : |p / ρ - c * w / ρ| ≤ Model.unitRoundoffAt format /
        (1 - Model.unitRoundoffAt format) * |c * w / ρ| := by
      rw [← sub_div, abs_div, abs_div, ← mul_div_assoc]
      exact div_le_div_of_nonneg_right h2' hρpos.le
    have h3 := abs_div_hypot_sub_le (t := c * w) hρ0 hmρ hρ
    have he : 0 ≤ Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b| :=
      div_nonneg ((abs_nonneg _).trans (Model.abs_roundAt_sub_le _ _))
        ((abs_nonneg a).trans (le_max_left _ _))
    have hv := (Model.unitRoundoffAt_pos format).le.trans (unitRoundoffAt_le_div (format := format))
    linarith [abs_sub_le_of_three hv hv h1 h2 h3 le_rfl]

/-- One rounding of `c * w / ρ`, `ρ` the rounded radius, is within the relative bound of
`hypotBackward` of `c * w / √(a² + b²)`. -/
private theorem hypot_replay_bound {a b c w ρ q : ℝ} (hρ0 : ρ ≠ 0)
    (hmρ : max |a| |b| ≤ ρ) (hρ : ρ = Model.roundAt format (√(a ^ 2 + b ^ 2)))
    (hq : q = Model.roundAt format (c * w / ρ)) :
    |q - c * w / √(a ^ 2 + b ^ 2)| ≤
      ((1 + Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b|) *
          (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1) *
        |c * w / √(a ^ 2 + b ^ 2)| + Model.underflowErrorAt format := by
  have h1 : |q - c * w / ρ| ≤ Model.unitRoundoffAt format * |c * w / ρ| +
      Model.underflowErrorAt format := by
    rw [hq]
    exact Model.abs_roundAt_sub_le_unitRoundoffAt format _
  have he : 0 ≤ Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b| :=
    div_nonneg ((abs_nonneg _).trans (Model.abs_roundAt_sub_le _ _))
      ((abs_nonneg a).trans (le_max_left _ _))
  exact abs_sub_le_of_two (Model.unitRoundoffAt_pos format).le unitRoundoffAt_le_div he h1
    (abs_div_hypot_sub_le hρ0 hmρ hρ)

/-- A finite value that is neither zero nor subnormal has normal magnitude. -/
private theorem minNormalAt_le_abs_toReal {v : Value} (hf : isFinite v = true)
    (hzero : ExecFloat.Binary.isZero v = false) (hsub : ExecFloat.Binary.isSubnormal v = false) :
    Model.minNormalAt format ≤ |toReal v| := by
  have hne : toReal v ≠ 0 := fun h => by
    rw [← ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hf, hzero] at h
    exact Bool.false_ne_true h
  rw [Model.minNormalAt, FloatLib.Floats.Formats.Flocq.bpow_binaryRadix]
  refine not_lt.mp fun hlt => ?_
  have h : Model.isSubnormal (toModel v) = true :=
    (Model.isSubnormal_iff_abs_toReal _).mpr ⟨hf, abs_pos.mpr hne, hlt⟩
  unfold ExecFloat.Binary.isSubnormal at hsub
  rw [hsub] at h
  exact Bool.false_ne_true h

/-- `hypotBackward z (Context.hypot x y) g` for `z` one of `x` and `y`, with `g` and the radius
finite, is finite and within the relative bound of `c * w / √(a² + b²)` plus `η`. -/
private theorem hypotBackward_core {x y z g : Value} (hformat : format.isIEEE = true)
    (hz : z = x ∨ z = y) (hgf : isFinite g = true)
    (hfinite : isFinite (Context.hypot x y) = true) :
    isFinite (hypotBackward z (Context.hypot x y) g) = true ∧
      |toReal (hypotBackward z (Context.hypot x y) g) -
          toReal g * toReal z / √(toReal x ^ 2 + toReal y ^ 2)| ≤
        ((1 + Model.halfUlpAt format (√(toReal x ^ 2 + toReal y ^ 2)) /
              max |toReal x| |toReal y|) *
            (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1) *
          |toReal g * toReal z / √(toReal x ^ 2 + toReal y ^ 2)| +
        Model.underflowErrorAt format := by
  have hH : isFinite (ExecFloat.Binary.hypot x y) = true := hfinite
  unfold isFinite at hH
  rw [ExecFloat.Binary.toModel_hypot] at hH
  obtain ⟨hxf, hyf⟩ := Model.isFinite_of_isFinite_hypot hformat hH
  have hzf : isFinite z = true := by rcases hz with rfl | rfl <;> assumption
  have hzm : |toReal z| ≤ max |toReal x| |toReal y| := by
    rcases hz with rfl | rfl
    · exact le_max_left _ _
    · exact le_max_right _ _
  have hρ := toReal_hypot hformat rfl rfl hfinite
  have hmρ : max |toReal x| |toReal y| ≤ toReal (Context.hypot x y) := by
    rw [hρ]
    exact max_le (abs_le_roundAt_sqrt (Model.roundAt_toReal_eq _ hxf)
        (le_add_of_nonneg_right (sq_nonneg _)))
      (abs_le_roundAt_sqrt (Model.roundAt_toReal_eq _ hyf) (le_add_of_nonneg_left (sq_nonneg _)))
  unfold hypotBackward
  simp only [configured_supported hformat, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
  split_ifs with h0 hflags
  · obtain ⟨-, hH0⟩ := ExecFloat.Binary.beq_zero_iff.mp h0
    exact ⟨ExecFloat.Binary.isFinite_mul_of_isZero_left ExecFloat.Binary.isZero_zero hgf hformat,
      hypot_zero_bound hH0 hzm hmρ
        (ExecFloat.Binary.toReal_mul_of_isZero_left ExecFloat.Binary.isZero_zero hgf hformat)⟩
  · have hρ0 : toReal (Context.hypot x y) ≠ 0 := fun h =>
      h0 (ExecFloat.Binary.beq_zero_iff.mpr ⟨hfinite, h⟩)
    obtain ⟨hf1, hf2⟩ := UInt8.or_eq_zero_iff.mp (beq_iff_eq.mp hflags)
    obtain ⟨hPf, hPsub, hPzero⟩ := isFinite_mul_of_flags_eq_zero hf1
    rw [configured_mulWithFlags_fst g z] at hf2 ⊢
    obtain ⟨hQf, hQsub, hQzero⟩ := isFinite_div_of_flags_eq_zero hf2
    rw [configured_divWithFlags_fst _ (Context.hypot x y)]
    refine ⟨hQf, hypot_flagless_bound hρ0 hmρ hρ
      (ExecFloat.Binary.toReal_mul_eq_roundAt hPf hformat)
      (ExecFloat.Binary.toReal_div_eq_roundAt hfinite hQf hformat) ?_⟩
    cases hP0 : ExecFloat.Binary.isZero (g * z)
    · right
      have hQ0 : ExecFloat.Binary.isZero (g * z / Context.hypot x y) = false := by
        cases h : ExecFloat.Binary.isZero (g * z / Context.hypot x y)
        · rfl
        · exact absurd (hP0.symm.trans (hQzero h)) Bool.false_ne_true
      exact ⟨minNormalAt_le_abs_toReal hPf hP0 hPsub, minNormalAt_le_abs_toReal hQf hQ0 hQsub⟩
    · left
      refine ⟨?_, (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hPf).mp hP0⟩
      rcases hPzero hP0 with h | h
      · rw [(ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hgf).mp h, zero_mul]
      · rw [(ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hzf).mp h, mul_zero]
  · have hρ0 : toReal (Context.hypot x y) ≠ 0 := fun h =>
      h0 (ExecFloat.Binary.beq_zero_iff.mpr ⟨hfinite, h⟩)
    have hρpos : 0 < toReal (Context.hypot x y) := lt_of_le_of_ne
      (by rw [hρ]; exact Model.roundAt_nonneg format (Real.sqrt_nonneg _)) (Ne.symm hρ0)
    have hrange : |(((ExecFloat.Binary.toRat? g).getD 0 * (ExecFloat.Binary.toRat? z).getD 0 /
        (ExecFloat.Binary.toRat? (Context.hypot x y)).getD 0 : ℚ) : ℝ)| ≤
        Model.toReal (Model.posMaxFinite format) := by
      rw [Rat.cast_div, Rat.cast_mul, cast_toRat_eq_toReal,
        cast_toRat_eq_toReal, cast_toRat_eq_toReal, abs_div, abs_mul,
        abs_of_pos hρpos, mul_div_assoc]
      exact (mul_le_of_le_one_right (abs_nonneg _) ((div_le_one hρpos).mpr (hzm.trans hmρ))).trans
        (ExecFloat.Binary.abs_toReal_le_posMaxFinite_of_isFinite hgf)
    obtain ⟨hDf, hD⟩ := configured_decode_rounding (format := format) (plan := plan)
      (code := code) _ hformat hrange (configured_decode _)
    rw [Rat.cast_div, Rat.cast_mul, cast_toRat_eq_toReal,
      cast_toRat_eq_toReal, cast_toRat_eq_toReal] at hD
    rw [configured_mulWithFlags_fst g z, configured_divWithFlags_fst _ (Context.hypot x y),
      configured_encode_of_isFinite hzf, configured_encode_of_isFinite hfinite,
      configured_encode_of_isFinite hgf]
    simp only [TorchLean.Numeric.QuotientCoefficients.mul,
      TorchLean.Numeric.QuotientCoefficients.div, configured_decode]
    split_ifs with hrz
    · obtain ⟨hQ0, hD0⟩ := Bool.and_eq_true_iff.mp hrz
      obtain ⟨hQf, hQ⟩ := ExecFloat.Binary.beq_zero_iff.mp hQ0
      obtain ⟨-, hD0'⟩ := ExecFloat.Binary.beq_zero_iff.mp hD0
      exact ⟨hQf, hypot_replay_bound hρ0 hmρ hρ (hQ.trans (hD0'.symm.trans hD))⟩
    · exact ⟨hDf, hypot_replay_bound hρ0 hmρ hρ hD⟩

/-- In an IEEE format, with `g` and `Context.hypot x y` finite,
`hypotBackward x (Context.hypot x y) g` of exact operands of values `a`, `b` and `c` is within
`K * |c * a / R| + η` of `c * a / R`. Here `R = √(a ^ 2 + b ^ 2)`, `u` is the unit roundoff, `η`
the underflow error and `K = (1 + halfUlpAt R / max |a| |b|) * (1 + u / (1 - u)) ^ 2 - 1`. -/
theorem abs_toReal_hypotBackward_left_sub_le {x y g : Value} {a b c : ℝ}
    (hformat : format.isIEEE = true) (hx : |toReal x - a| ≤ 0) (hy : |toReal y - b| ≤ 0)
    (hg : |toReal g - c| ≤ 0) (hgf : isFinite g = true)
    (hfinite : isFinite (Context.hypot x y) = true) :
    |toReal (hypotBackward x (Context.hypot x y) g) - c * a / √(a ^ 2 + b ^ 2)| ≤
      ((1 + Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b|) *
          (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1) *
        |c * a / √(a ^ 2 + b ^ 2)| + Model.underflowErrorAt format := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  have hb : toReal y = b := by simpa [sub_eq_zero] using hy
  have hc : toReal g = c := by simpa [sub_eq_zero] using hg
  subst ha hb hc
  exact (hypotBackward_core hformat (.inl rfl) hgf hfinite).2

/-- In an IEEE format, with `g` and `Context.hypot x y` finite,
`hypotBackward y (Context.hypot x y) g` of exact operands of values `a`, `b` and `c` is within
`K * |c * b / R| + η` of `c * b / R`, with `R`, `u`, `η` and `K` of
`abs_toReal_hypotBackward_left_sub_le`. -/
theorem abs_toReal_hypotBackward_right_sub_le {x y g : Value} {a b c : ℝ}
    (hformat : format.isIEEE = true) (hx : |toReal x - a| ≤ 0) (hy : |toReal y - b| ≤ 0)
    (hg : |toReal g - c| ≤ 0) (hgf : isFinite g = true)
    (hfinite : isFinite (Context.hypot x y) = true) :
    |toReal (hypotBackward y (Context.hypot x y) g) - c * b / √(a ^ 2 + b ^ 2)| ≤
      ((1 + Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b|) *
          (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1) *
        |c * b / √(a ^ 2 + b ^ 2)| + Model.underflowErrorAt format := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  have hb : toReal y = b := by simpa [sub_eq_zero] using hy
  have hc : toReal g = c := by simpa [sub_eq_zero] using hg
  subst ha hb hc
  exact (hypotBackward_core hformat (.inr rfl) hgf hfinite).2

/-- In an IEEE format, with `g` and `Context.hypot x y` finite,
`hypotBackward x (Context.hypot x y) g` of exact operands of values `a`, `b` and `c` is finite and
within relative error `K` and absolute error `η` of `c * a / R`, with `R`, `u`, `η` and `K` of
`abs_toReal_hypotBackward_left_sub_le`. -/
theorem isFinite_and_mixedError_hypotBackward_left {x y g : Value} {a b c : ℝ}
    (hformat : format.isIEEE = true) (hx : MixedError (toReal x) a 0 0)
    (hy : MixedError (toReal y) b 0 0) (hg : MixedError (toReal g) c 0 0)
    (hgf : isFinite g = true) (hfinite : isFinite (Context.hypot x y) = true) :
    isFinite (hypotBackward x (Context.hypot x y) g) = true ∧
      MixedError (toReal (hypotBackward x (Context.hypot x y) g)) (c * a / √(a ^ 2 + b ^ 2))
        ((1 + Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b|) *
            (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1)
        (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  have hb := mixedError_zero_zero_iff.mp hy
  have hc := mixedError_zero_zero_iff.mp hg
  subst ha hb hc
  have h := hypotBackward_core hformat (.inl rfl) hgf hfinite
  exact ⟨h.1, ⟨hypotBackward_relative_nonneg, (Model.underflowErrorAt_pos format).le, h.2⟩⟩

/-- In an IEEE format, with `g` and `Context.hypot x y` finite,
`hypotBackward y (Context.hypot x y) g` of exact operands of values `a`, `b` and `c` is finite and
within relative error `K` and absolute error `η` of `c * b / R`, with `R`, `u`, `η` and `K` of
`abs_toReal_hypotBackward_left_sub_le`. -/
theorem isFinite_and_mixedError_hypotBackward_right {x y g : Value} {a b c : ℝ}
    (hformat : format.isIEEE = true) (hx : MixedError (toReal x) a 0 0)
    (hy : MixedError (toReal y) b 0 0) (hg : MixedError (toReal g) c 0 0)
    (hgf : isFinite g = true) (hfinite : isFinite (Context.hypot x y) = true) :
    isFinite (hypotBackward y (Context.hypot x y) g) = true ∧
      MixedError (toReal (hypotBackward y (Context.hypot x y) g)) (c * b / √(a ^ 2 + b ^ 2))
        ((1 + Model.halfUlpAt format (√(a ^ 2 + b ^ 2)) / max |a| |b|) *
            (1 + Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format)) ^ 2 - 1)
        (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  have hb := mixedError_zero_zero_iff.mp hy
  have hc := mixedError_zero_zero_iff.mp hg
  subst ha hb hc
  have h := hypotBackward_core hformat (.inr rfl) hgf hfinite
  exact ⟨h.1, ⟨hypotBackward_relative_nonneg, (Model.underflowErrorAt_pos format).le, h.2⟩⟩

/-- A real within relative error `v < 1` of `t` lies between `(1 - v) * t` and `t / (1 - v)`. -/
private theorem ratio_bounds_of_abs_sub_le {v r t : ℝ} (hv : 0 ≤ v) (hv1 : v < 1)
    (h : |r - t| ≤ v * t) : (1 - v) * t ≤ r ∧ (1 - v) * r ≤ t := by
  have h1 := neg_abs_le (r - t)
  have h2 := le_abs_self (r - t)
  have hvt : 0 ≤ v * t := (abs_nonneg _).trans h
  constructor
  · linarith
  · nlinarith [mul_nonneg (sub_nonneg.mpr hv1.le) (by linarith : 0 ≤ t + v * t - r),
      mul_nonneg hv hvt]

/-- A product of two operands within relative error `v`, rounded within relative error `v`, is
within `((1 + v) ^ 3 - 1) * |A * D|` of `A * D`. -/
private theorem abs_rounded_mul_sub_le {v A D A' D' L : ℝ} (hv : 0 ≤ v)
    (hA : |A' - A| ≤ v * |A|) (hD : |D' - D| ≤ v * |D|) (hL : |L - A' * D'| ≤ v * |A' * D'|) :
    |L - A * D| ≤ ((1 + v) ^ 3 - 1) * |A * D| := by
  have h2 : |A' * D' - A * D'| ≤ v * |A * D'| := by
    rw [← sub_mul, abs_mul, abs_mul]
    nlinarith [mul_le_mul_of_nonneg_right hA (abs_nonneg D')]
  have h3 : |A * D' - A * D| ≤ v * |A * D| := by
    rw [← mul_sub, abs_mul, abs_mul]
    nlinarith [mul_le_mul_of_nonneg_left hD (abs_nonneg A)]
  have h := abs_sub_le_of_three hv hv hL h2 h3 le_rfl
  calc |L - A * D| ≤ ((1 + v) * (1 + v) * (1 + v) - 1) * |A * D| := h
    _ = ((1 + v) ^ 3 - 1) * |A * D| := by ring

/-- The rounded square `X` of an operand within relative error `v` of `A` lies between
`(1 - v) ^ 3 * A ^ 2` and `A ^ 2 / (1 - v) ^ 3`. -/
private theorem square_ratio_bounds {v A A' X : ℝ} (hv : 0 ≤ v) (hv1 : v < 1)
    (hA : |A' - A| ≤ v * |A|) (hX : |X - A' * A'| ≤ v * |A' * A'|) :
    (1 - v) ^ 3 * A ^ 2 ≤ X ∧ (1 - v) ^ 3 * X ≤ A ^ 2 := by
  have hw : 0 ≤ 1 - v := by linarith
  obtain ⟨h1, h2⟩ := ratio_bounds_of_abs_sub_le hv hv1
    ((abs_abs_sub_abs_le_abs_sub A' A).trans hA)
  rw [abs_mul_self] at hX
  obtain ⟨h3, h4⟩ := ratio_bounds_of_abs_sub_le hv hv1 hX
  have hsq1 : (1 - v) ^ 2 * A ^ 2 ≤ A' * A' := by
    have := mul_le_mul h1 h1 (by positivity) (abs_nonneg _)
    rw [abs_mul_abs_self] at this
    nlinarith [sq_abs A]
  have hsq2 : (1 - v) ^ 2 * (A' * A') ≤ A ^ 2 := by
    have := mul_le_mul h2 h2 (by positivity) (abs_nonneg _)
    rw [abs_mul_abs_self] at this
    nlinarith [abs_mul_abs_self A']
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left hsq1 hw]
  · nlinarith [mul_le_mul_of_nonneg_left h4 (pow_nonneg hw 2)]

/-- The rounded sum `S` of two values within ratio `(1 - v) ^ 3` of `P ≥ 0` and `Q ≥ 0` lies
between `(1 - v) ^ 4 * (P + Q)` and `(P + Q) / (1 - v) ^ 4`. -/
private theorem sum_ratio_bounds {v P Q X Y S : ℝ} (hv : 0 ≤ v) (hv1 : v < 1) (hP : 0 ≤ P)
    (hQ : 0 ≤ Q) (hX : (1 - v) ^ 3 * P ≤ X ∧ (1 - v) ^ 3 * X ≤ P)
    (hY : (1 - v) ^ 3 * Q ≤ Y ∧ (1 - v) ^ 3 * Y ≤ Q) (hS : |S - (X + Y)| ≤ v * |X + Y|) :
    (1 - v) ^ 4 * (P + Q) ≤ S ∧ (1 - v) ^ 4 * S ≤ P + Q := by
  have hw : 0 ≤ 1 - v := by linarith
  have hw3 : 0 ≤ (1 - v) ^ 3 := pow_nonneg hw 3
  have hXY : 0 ≤ X + Y := by nlinarith [mul_nonneg hw3 hP, mul_nonneg hw3 hQ]
  rw [abs_of_nonneg hXY] at hS
  obtain ⟨h1, h2⟩ := ratio_bounds_of_abs_sub_le hv hv1 hS
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left (add_le_add hX.1 hY.1) hw]
  · nlinarith [mul_le_mul_of_nonneg_left h2 hw3]

/-- The rounded difference `N` of `L` and `R`, within relative error `(1 + v) ^ 3 - 1` of `P1` and
`P2`, is within `((1 + v) ^ 4 - 1) * (|P1| + |P2|)` of `P1 - P2`. -/
private theorem abs_rounded_sub_sub_le {v P1 P2 L R N : ℝ} (hv : 0 ≤ v)
    (hL : |L - P1| ≤ ((1 + v) ^ 3 - 1) * |P1|) (hR : |R - P2| ≤ ((1 + v) ^ 3 - 1) * |P2|)
    (hN : |N - (L - R)| ≤ v * |L - R|) :
    |N - (P1 - P2)| ≤ ((1 + v) ^ 4 - 1) * (|P1| + |P2|) := by
  have h1 : |L - R - (P1 - P2)| ≤ ((1 + v) ^ 3 - 1) * (|P1| + |P2|) := by
    calc |L - R - (P1 - P2)| = |(L - P1) - (R - P2)| := by ring_nf
      _ ≤ |L - P1| + |R - P2| := abs_sub _ _
      _ ≤ _ := by linarith
  have h2 : |L - R| ≤ (1 + v) ^ 3 * (|P1| + |P2|) := by
    have := abs_sub_abs_le_abs_sub (L - R) (P1 - P2)
    have := abs_sub P1 P2
    nlinarith
  have h3 := abs_sub_le N (L - R) (P1 - P2)
  nlinarith [mul_le_mul_of_nonneg_left h2 hv]

/-- A rounded quotient `Q` of `N` by `S`, with `N` within `E * W` of `P`, `|P| ≤ W` and
`m * T ≤ S`, `m * S ≤ T` for `T > 0`, is within `((1 + v) * (1 / m) * (1 + E) - 1) * (W / T)`
of `P / T`. -/
private theorem abs_rounded_div_sub_le {v m E P W T N S Q : ℝ} (hv : 0 ≤ v) (hm : 0 < m)
    (hT : 0 < T) (hPW : |P| ≤ W) (hN : |N - P| ≤ E * W) (hS1 : m * T ≤ S) (hS2 : m * S ≤ T)
    (hQ : |Q - N / S| ≤ v * |N / S|) :
    |Q - P / T| ≤ ((1 + v) * (1 / m) * (1 + E) - 1) * (W / T) := by
  have hS : 0 < S := by nlinarith
  have hm1 : m ≤ 1 := by nlinarith
  have hr1 : T / S ≤ 1 / m := by
    rw [div_le_div_iff₀ hS hm]
    linarith
  have hr2 : m ≤ T / S := by
    rw [le_div_iff₀ hS]
    linarith
  have hinv : 1 - m ≤ 1 / m - 1 := by
    rw [div_sub_one hm.ne', le_div_iff₀ hm]
    nlinarith [sq_nonneg (1 - m)]
  have he : 0 ≤ 1 / m - 1 := by
    rw [le_sub_iff_add_le, zero_add, le_div_iff₀ hm]
    linarith
  have h2 : |N / S - N / T| ≤ (1 / m - 1) * |N / T| := by
    have heq : N / S - N / T = N / T * (T / S - 1) := by
      field_simp
    rw [heq, abs_mul, mul_comm]
    refine mul_le_mul_of_nonneg_right ?_ (abs_nonneg _)
    rw [abs_le]
    constructor <;> linarith
  have h3 : |N / T - P / T| ≤ E * (W / T) := by
    rw [← sub_div, abs_div, abs_of_pos hT, mul_div_assoc']
    exact div_le_div_of_nonneg_right hN hT.le
  have ht : |P / T| ≤ W / T := by
    rw [abs_div, abs_of_pos hT]
    exact div_le_div_of_nonneg_right hPW hT.le
  have h := abs_sub_le_of_three hv he hQ h2 h3 ht
  calc |Q - P / T| ≤ ((1 + v) * (1 + (1 / m - 1)) * (1 + E) - 1) * (W / T) := h
    _ = ((1 + v) * (1 / m) * (1 + E) - 1) * (W / T) := by ring

/-- Eleven roundings within relative error `v < 1` of the flagless `angleTangent` steps give a
result within `((1 + v) ^ 5 / (1 - v) ^ 4 - 1) * W / T` of `P / T`, for `P = A * D - B * C`,
`W = |A * D| + |B * C|` and `T = A ^ 2 + B ^ 2 > 0`. -/
private theorem angleTangent_flagless_bound {v A B C D A' B' C' D' L R N X Y S Q : ℝ}
    (hv : 0 ≤ v) (hv1 : v < 1) (hT : 0 < A ^ 2 + B ^ 2)
    (hA : |A' - A| ≤ v * |A|) (hB : |B' - B| ≤ v * |B|) (hC : |C' - C| ≤ v * |C|)
    (hD : |D' - D| ≤ v * |D|) (hL : |L - A' * D'| ≤ v * |A' * D'|)
    (hR : |R - B' * C'| ≤ v * |B' * C'|) (hN : |N - (L - R)| ≤ v * |L - R|)
    (hX : |X - A' * A'| ≤ v * |A' * A'|) (hY : |Y - B' * B'| ≤ v * |B' * B'|)
    (hS : |S - (X + Y)| ≤ v * |X + Y|) (hQ : |Q - N / S| ≤ v * |N / S|) :
    |Q - (A * D - B * C) / (A ^ 2 + B ^ 2)| ≤
      ((1 + v) ^ 5 / (1 - v) ^ 4 - 1) * ((|A * D| + |B * C|) / (A ^ 2 + B ^ 2)) := by
  have hw : 0 < 1 - v := by linarith
  have hNP := abs_rounded_sub_sub_le hv (abs_rounded_mul_sub_le hv hA hD hL)
    (abs_rounded_mul_sub_le hv hB hC hR) hN
  obtain ⟨hS1, hS2⟩ := sum_ratio_bounds hv hv1 (sq_nonneg A) (sq_nonneg B)
    (square_ratio_bounds hv hv1 hA hX) (square_ratio_bounds hv hv1 hB hY) hS
  have h := abs_rounded_div_sub_le hv (pow_pos hw 4) hT (abs_sub _ _) hNP hS1 hS2 hQ
  have hK : (1 + v) * (1 / (1 - v) ^ 4) * (1 + ((1 + v) ^ 4 - 1)) = (1 + v) ^ 5 / (1 - v) ^ 4 := by
    field_simp
    ring
  rwa [hK] at h

/-- With `v = u / (1 - u)` and `u < 1 / 2`, `(1 + v) ^ 5 / (1 - v) ^ 4` is
`1 / ((1 - u) * (1 - 2 * u) ^ 4)`. -/
private theorem angleTangent_constant_eq {u : ℝ} (hu : u < 1 / 2) :
    (1 + u / (1 - u)) ^ 5 / (1 - u / (1 - u)) ^ 4 = 1 / ((1 - u) * (1 - 2 * u) ^ 4) := by
  have h1 : (1 - u) ≠ 0 := (by linarith : (0 : ℝ) < 1 - u).ne'
  have h2 : (1 - 2 * u) ≠ 0 := (by linarith : (0 : ℝ) < 1 - 2 * u).ne'
  rw [show 1 + u / (1 - u) = 1 / (1 - u) by field_simp; ring,
    show 1 - u / (1 - u) = (1 - 2 * u) / (1 - u) by field_simp; ring]
  field_simp

/-- A finite rounding `r` of `t` that is not subnormal, and zero only when `t` is zero, is within
`u / (1 - u)` of `t` relative to `t`, `u` the unit roundoff. -/
private theorem abs_toReal_sub_le_of_rounding {r : Value} {t : ℝ} (hf : isFinite r = true)
    (hsub : ExecFloat.Binary.isSubnormal r = false) (hr : toReal r = Model.roundAt format t)
    (hzero : ExecFloat.Binary.isZero r = true → t = 0) :
    |toReal r - t| ≤ Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) * |t| := by
  cases hz : ExecFloat.Binary.isZero r
  · rw [hr]
    exact abs_roundAt_sub_le_of_minNormalAt_le
      (by rw [← hr]; exact minNormalAt_le_abs_toReal hf hz hsub)
  · rw [hzero hz, (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hf).mp hz]
    simp

/-- A finite product `l * r` that is not subnormal, and zero only when a factor is zero, is within
`u / (1 - u)` of `toReal l * toReal r` relative to it. -/
private theorem abs_toReal_mul_sub_le_of_flags {l r : Value} (hformat : format.isIEEE = true)
    (hl : isFinite l = true) (hr : isFinite r = true)
    (h : isFinite (l * r) = true ∧ ExecFloat.Binary.isSubnormal (l * r) = false ∧
      (ExecFloat.Binary.isZero (l * r) = true →
        ExecFloat.Binary.isZero l = true ∨ ExecFloat.Binary.isZero r = true)) :
    |toReal (l * r) - toReal l * toReal r| ≤
      Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) * |toReal l * toReal r| :=
  abs_toReal_sub_le_of_rounding h.1 h.2.1 (ExecFloat.Binary.toReal_mul_eq_roundAt h.1 hformat)
    fun hz => by
      rcases h.2.2 hz with h0 | h0
      · rw [(ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hl).mp h0, zero_mul]
      · rw [(ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hr).mp h0, mul_zero]

/-- A finite quotient `l / r` that is not subnormal, and zero only when `l` is zero, is within
`u / (1 - u)` of `toReal l / toReal r` relative to it. -/
private theorem abs_toReal_div_sub_le_of_flags {l r : Value} (hformat : format.isIEEE = true)
    (hl : isFinite l = true) (hr : isFinite r = true)
    (h : isFinite (l / r) = true ∧ ExecFloat.Binary.isSubnormal (l / r) = false ∧
      (ExecFloat.Binary.isZero (l / r) = true → ExecFloat.Binary.isZero l = true)) :
    |toReal (l / r) - toReal l / toReal r| ≤
      Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) * |toReal l / toReal r| :=
  abs_toReal_sub_le_of_rounding h.1 h.2.1 (ExecFloat.Binary.toReal_div_eq_roundAt hr h.1 hformat)
    fun hz => by
      rw [(ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hl).mp (h.2.2 hz), zero_div]

/-- The scale `max |x| |y|` of finite configured values `x` and `y` is finite with the real value
`max |toReal x| |toReal y|`. -/
private theorem isFinite_and_toReal_max_abs {x y : Value} (hformat : format.isIEEE = true)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    isFinite (max (MathFunctions.abs x) (MathFunctions.abs y)) = true ∧
      toReal (max (MathFunctions.abs x) (MathFunctions.abs y)) = max |toReal x| |toReal y| := by
  have hsigned := FloatFormat.supportsSignedZero_eq_true_of_isIEEE format hformat
  have hax : isFinite (MathFunctions.abs x) = true := ExecFloat.Binary.isFinite_abs hx
  have hay : isFinite (MathFunctions.abs y) = true := ExecFloat.Binary.isFinite_abs hy
  refine ⟨ExecFloat.Binary.isFinite_max hax hay, ?_⟩
  have habs : ∀ v : Value, isFinite v = true → toReal (MathFunctions.abs v) = |toReal v| := by
    intro v hv
    have decoded : toModel (MathFunctions.abs v) = Model.abs (toModel v) :=
      ExecFloat.ModelCodec.decode_liftUnary Model.abs v
    unfold toReal
    rw [decoded]
    exact Model.toReal_abs_of_isFinite _ hsigned hv
  unfold toReal
  rw [ExecFloat.Binary.toModel_max, Model.toReal_maximum_eq_max_of_isFinite _ _ hax hay]
  exact congrArg₂ max (habs x hx) (habs y hy)

/-- Dividing the four operands by the same `σ ≠ 0` leaves `(a * d - b * c) / (a ^ 2 + b ^ 2)` and
`(|a * d| + |b * c|) / (a ^ 2 + b ^ 2)` unchanged. -/
private theorem angleTangent_scale_eq {σ a b c d : ℝ} (hσ : σ ≠ 0) :
    (a / σ * (d / σ) - b / σ * (c / σ)) / ((a / σ) ^ 2 + (b / σ) ^ 2) =
        (a * d - b * c) / (a ^ 2 + b ^ 2) ∧
      (|a / σ * (d / σ)| + |b / σ * (c / σ)|) / ((a / σ) ^ 2 + (b / σ) ^ 2) =
        (|a * d| + |b * c|) / (a ^ 2 + b ^ 2) := by
  have hσ2 : (0 : ℝ) < σ ^ 2 := by positivity
  have hmul : ∀ p q : ℝ, p / σ * (q / σ) = p * q / σ ^ 2 := fun p q => by
    rw [div_mul_div_comm, sq]
  rw [hmul, hmul, div_pow, div_pow, abs_div, abs_div, abs_of_pos hσ2, ← sub_div, ← add_div,
    ← add_div, div_div_div_cancel_right₀ hσ2.ne', div_div_div_cancel_right₀ hσ2.ne']
  exact ⟨rfl, rfl⟩

/-- For `0 ≤ u < 1 / 2`, `u` is at most `1 / ((1 - u) * (1 - 2 * u) ^ 4) - 1`. -/
private theorem le_angleTangent_constant {u : ℝ} (hu : 0 ≤ u) (hu2 : u < 1 / 2) :
    u ≤ 1 / ((1 - u) * (1 - 2 * u) ^ 4) - 1 := by
  have h0 : 0 < 1 - u := by linarith
  have h1 : 0 < 1 - 2 * u := by linarith
  have h2 : (1 - 2 * u) ^ 4 ≤ 1 := pow_le_one₀ h1.le (by linarith)
  rw [le_sub_iff_add_le, le_div_iff₀ (by positivity)]
  calc (u + 1) * ((1 - u) * (1 - 2 * u) ^ 4) = (1 - u ^ 2) * (1 - 2 * u) ^ 4 := by ring
    _ ≤ 1 * 1 := mul_le_mul (by nlinarith [sq_nonneg u]) h2 (by positivity) zero_le_one
    _ = 1 := one_mul 1

/-- In an IEEE format, `angleTangent x y dx dy` of finite configured values is finite when
`|(x * dy - y * dx) / (x ^ 2 + y ^ 2)|` is at most the largest finite value. A finite result is
within `K * (|x * dy| + |y * dx|) / (x ^ 2 + y ^ 2) + η` of `(x * dy - y * dx) / (x ^ 2 + y ^ 2)`,
with `K = 1 / ((1 - u) * (1 - 2 * u) ^ 4) - 1`. -/
private theorem angleTangent_core {x y dx dy : Value} (hformat : format.isIEEE = true)
    (hxf : isFinite x = true) (hyf : isFinite y = true) (hdxf : isFinite dx = true)
    (hdyf : isFinite dy = true) :
    (|(toReal x * toReal dy - toReal y * toReal dx) / (toReal x ^ 2 + toReal y ^ 2)| ≤
        Model.toReal (Model.posMaxFinite format) → isFinite (angleTangent x y dx dy) = true) ∧
      (isFinite (angleTangent x y dx dy) = true →
        |toReal (angleTangent x y dx dy) -
            (toReal x * toReal dy - toReal y * toReal dx) / (toReal x ^ 2 + toReal y ^ 2)| ≤
          (1 / ((1 - Model.unitRoundoffAt format) * (1 - 2 * Model.unitRoundoffAt format) ^ 4) -
              1) * ((|toReal x * toReal dy| + |toReal y * toReal dx|) /
                (toReal x ^ 2 + toReal y ^ 2)) + Model.underflowErrorAt format) := by
  obtain ⟨hu4, -, -⟩ := atanBackward_constants hformat
  have hu := Model.unitRoundoffAt_pos format
  have hη := (Model.underflowErrorAt_pos format).le
  have hK := le_angleTangent_constant hu.le (by linarith : Model.unitRoundoffAt format < 1 / 2)
  have hPW : |(toReal x * toReal dy - toReal y * toReal dx) / (toReal x ^ 2 + toReal y ^ 2)| ≤
      (|toReal x * toReal dy| + |toReal y * toReal dx|) / (toReal x ^ 2 + toReal y ^ 2) := by
    have hT : 0 ≤ toReal x ^ 2 + toReal y ^ 2 := by positivity
    rw [abs_div, abs_of_nonneg hT]
    exact div_le_div_of_nonneg_right (abs_sub _ _) hT
  have hKW := add_nonneg (mul_nonneg (hu.le.trans hK) ((abs_nonneg _).trans hPW)) hη
  obtain ⟨hsf, hs⟩ := isFinite_and_toReal_max_abs hformat hxf hyf
  have hss : (max (MathFunctions.abs x) (MathFunctions.abs y) -
      max (MathFunctions.abs x) (MathFunctions.abs y) == 0) = true :=
    ExecFloat.Binary.beq_zero_iff.mpr ⟨ExecFloat.Binary.isFinite_sub_self hsf hformat,
      ExecFloat.Binary.toReal_sub_self hsf hformat⟩
  have hpos : ¬(x == 0 && y == 0) = true →
      0 < toReal x ^ 2 + toReal y ^ 2 ∧ 0 < max |toReal x| |toReal y| := by
    intro h0
    by_cases hX : toReal x = 0
    · have hY : toReal y ≠ 0 := fun hY => h0 (Bool.and_eq_true_iff.mpr
        ⟨ExecFloat.Binary.beq_zero_iff.mpr ⟨hxf, hX⟩, ExecFloat.Binary.beq_zero_iff.mpr ⟨hyf, hY⟩⟩)
      exact ⟨by positivity, lt_max_of_lt_right (abs_pos.mpr hY)⟩
    · have hX' : toReal x ≠ 0 := hX
      exact ⟨by positivity, lt_max_of_lt_left (abs_pos.mpr hX')⟩
  unfold angleTangent
  simp only [hss, configured_supported hformat, Bool.not_true, Bool.and_false, Bool.or_false,
    Bool.false_eq_true, ↓reduceIte, configured_divWithFlags_fst, configured_mulWithFlags_fst,
    configured_subWithFlags_fst, configured_addWithFlags_fst]
  set s := max (MathFunctions.abs x) (MathFunctions.abs y)
  split_ifs with h0 hflags
  · obtain ⟨hx0, hy0⟩ := Bool.and_eq_true_iff.mp h0
    have hzf := ExecFloat.Binary.isFinite_mul_of_isZero_left ExecFloat.Binary.isZero_zero hdxf
      hformat
    have hz := (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hzf).mpr
      (ExecFloat.Binary.toReal_mul_of_isZero_left ExecFloat.Binary.isZero_zero hdxf hformat)
    have hwf := ExecFloat.Binary.isFinite_mul_of_isZero_left ExecFloat.Binary.isZero_zero hdyf
      hformat
    refine ⟨fun _ => ExecFloat.Binary.isFinite_add_of_isZero_left hz hwf hformat, fun _ => ?_⟩
    rw [ExecFloat.Binary.toReal_add_of_isZero_left hz hwf hformat,
      ExecFloat.Binary.toReal_mul_of_isZero_left ExecFloat.Binary.isZero_zero hdyf hformat,
      (ExecFloat.Binary.beq_zero_iff.mp hx0).2, (ExecFloat.Binary.beq_zero_iff.mp hy0).2]
    simpa using hη
  · obtain ⟨-, hσ0⟩ := hpos h0
    have hσ : toReal s ≠ 0 := by rw [hs]; exact hσ0.ne'
    obtain ⟨hfl, hcanc⟩ := Bool.and_eq_true_iff.mp hflags
    have hfl' := beq_iff_eq.mp hfl
    simp only [UInt8.or_eq_zero_iff] at hfl'
    obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨f1, f2⟩, f3⟩, f4⟩, f5⟩, f6⟩, f7⟩, f8⟩, f9⟩, f10⟩, f11⟩ := hfl'
    simp only [Bool.not_and, Bool.not_or, bne, Bool.not_not, Bool.or_eq_true, Bool.and_eq_true,
      Bool.not_eq_true'] at hcanc
    have haf := (isFinite_div_of_flags_eq_zero f1).1
    have hbf := (isFinite_div_of_flags_eq_zero f2).1
    have hcf := (isFinite_div_of_flags_eq_zero f3).1
    have hdf := (isFinite_div_of_flags_eq_zero f4).1
    have hpf := isFinite_mul_of_flags_eq_zero f8
    have hqf := isFinite_mul_of_flags_eq_zero f9
    obtain ⟨hnf, hnsub⟩ := isFinite_sub_of_flags_eq_zero f7
    obtain ⟨hSf, hSsub⟩ := isFinite_add_of_flags_eq_zero f10
    have hQf := isFinite_div_of_flags_eq_zero f11
    have hA := abs_toReal_div_sub_le_of_flags hformat hxf hsf
      (isFinite_div_of_flags_eq_zero f1)
    have hB := abs_toReal_div_sub_le_of_flags hformat hyf hsf
      (isFinite_div_of_flags_eq_zero f2)
    have hC := abs_toReal_div_sub_le_of_flags hformat hdxf hsf
      (isFinite_div_of_flags_eq_zero f3)
    have hD := abs_toReal_div_sub_le_of_flags hformat hdyf hsf
      (isFinite_div_of_flags_eq_zero f4)
    have hL := abs_toReal_mul_sub_le_of_flags hformat haf hdf
      (isFinite_mul_of_flags_eq_zero f5)
    have hR := abs_toReal_mul_sub_le_of_flags hformat hbf hcf
      (isFinite_mul_of_flags_eq_zero f6)
    have hX := abs_toReal_mul_sub_le_of_flags hformat haf haf hpf
    have hY := abs_toReal_mul_sub_le_of_flags hformat hbf hbf hqf
    have hN := abs_toReal_sub_le_of_rounding hnf hnsub
      (ExecFloat.Binary.toReal_sub_eq_roundAt hnf hformat) fun hz => by
        rcases hcanc with h | ⟨hl0, hr0⟩
        · exact absurd (h.symm.trans (ExecFloat.Binary.beq_zero_iff.mpr ⟨hnf,
            (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hnf).mp hz⟩)) Bool.false_ne_true
        · rw [(ExecFloat.Binary.beq_zero_iff.mp hl0).2, (ExecFloat.Binary.beq_zero_iff.mp hr0).2,
            sub_zero]
    have key : ∀ P Q : ℝ, 0 ≤ P → 0 ≤ Q → Model.roundAt format P = P →
        Model.roundAt format Q = Q → Model.roundAt format (P + Q) = 0 → P + Q = 0 := by
      intro P Q hP hQ hPr hQr h
      have h1 := Model.roundAt_mono format (le_add_of_nonneg_right hQ : P ≤ P + Q)
      have h2 := Model.roundAt_mono format (le_add_of_nonneg_left hP : Q ≤ P + Q)
      rw [hPr, h] at h1
      rw [hQr, h] at h2
      linarith
    have hS := abs_toReal_sub_le_of_rounding hSf hSsub
      (ExecFloat.Binary.toReal_add_eq_roundAt hSf hformat) fun hz => by
        have h0 := (ExecFloat.Binary.isZero_eq_true_iff_toReal_eq_zero hSf).mp hz
        rw [ExecFloat.Binary.toReal_add_eq_roundAt hSf hformat] at h0
        exact key _ _
          ((Model.roundAt_nonneg format (mul_self_nonneg _)).trans_eq
            (ExecFloat.Binary.toReal_mul_eq_roundAt hpf.1 hformat).symm)
          ((Model.roundAt_nonneg format (mul_self_nonneg _)).trans_eq
            (ExecFloat.Binary.toReal_mul_eq_roundAt hqf.1 hformat).symm)
          (Model.roundAt_toReal_eq _ hpf.1) (Model.roundAt_toReal_eq _ hqf.1) h0
    have hQ := abs_toReal_div_sub_le_of_flags hformat hnf hSf hQf
    have hT : 0 < (toReal x / toReal s) ^ 2 + (toReal y / toReal s) ^ 2 := by
      rw [div_pow, div_pow, ← add_div]
      exact div_pos (hpos h0).1 (by positivity)
    have hv : 0 ≤ Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) :=
      hu.le.trans unitRoundoffAt_le_div
    have hv1 : Model.unitRoundoffAt format / (1 - Model.unitRoundoffAt format) < 1 :=
      (div_lt_one (by linarith)).mpr (by linarith)
    have hmain := angleTangent_flagless_bound hv hv1 hT hA hB hC hD hL hR hN hX hY hS hQ
    rw [angleTangent_constant_eq (by linarith), (angleTangent_scale_eq hσ).1,
      (angleTangent_scale_eq hσ).2] at hmain
    exact ⟨fun _ => hQf.1, fun _ => by linarith⟩
  · obtain ⟨hT0, -⟩ := hpos h0
    rw [configured_encode_of_isFinite hxf, configured_encode_of_isFinite hyf,
      configured_encode_of_isFinite hdxf, configured_encode_of_isFinite hdyf]
    simp only [TorchLean.Numeric.QuotientCoefficients.sub,
      TorchLean.Numeric.QuotientCoefficients.mul, TorchLean.Numeric.QuotientCoefficients.add,
      TorchLean.Numeric.QuotientCoefficients.div, configured_decode]
    split_ifs with hden hrz
    · exfalso
      have h := congrArg (fun v : ℚ => (v : ℝ)) (beq_iff_eq.mp hden)
      simp only [TorchLean.Numeric.QuotientCoefficients.primal, Rat.cast_add, Rat.cast_mul,
        cast_toRat_eq_toReal, Rat.cast_zero] at h
      nlinarith
    · obtain ⟨hP0, hQ0⟩ := Bool.and_eq_true_iff.mp hrz
      obtain ⟨hQf, hQ⟩ := ExecFloat.Binary.beq_zero_iff.mp hQ0
      have hP := congrArg (fun v : ℚ => (v : ℝ)) (beq_iff_eq.mp hP0)
      simp only [TorchLean.Numeric.QuotientCoefficients.primal, Rat.cast_sub, Rat.cast_mul,
        cast_toRat_eq_toReal, Rat.cast_zero] at hP
      refine ⟨fun _ => hQf, fun _ => (le_of_eq ?_).trans hKW⟩
      rw [abs_eq_zero, sub_eq_zero]
      exact hQ.trans (by rw [hP, zero_div])
    · refine ⟨fun hrange => (configured_decode_rounding (format := format) (plan := plan)
        (code := code) _ hformat ?_ (configured_decode _)).1, fun hfin => ?_⟩
      · simpa only [TorchLean.Numeric.QuotientCoefficients.primal, Rat.cast_div, Rat.cast_sub,
          Rat.cast_add, Rat.cast_mul, cast_toRat_eq_toReal, sq] using hrange
      · have hD : ∀ v : ℚ,
            isFinite (ExecFloat.Binary.ofModel (Model.ofRat format v) : Value) = true →
              toReal (ExecFloat.Binary.ofModel (Model.ofRat format v) : Value) =
                Model.roundAt format (v : ℝ) := fun v h => by
          simp only [ExecFloat.Binary.isFinite, ExecFloat.Binary.toReal,
            ExecFloat.Binary.toModel_ofModel] at h ⊢
          exact Model.toReal_ofRat_eq_roundAt _ hformat h
        rw [hD _ hfin]
        simp only [TorchLean.Numeric.QuotientCoefficients.primal, Rat.cast_div, Rat.cast_sub,
          Rat.cast_add, Rat.cast_mul, Rat.cast_pow, cast_toRat_eq_toReal, ← sq]
        have h1 := Model.abs_roundAt_sub_le_unitRoundoffAt format
          ((toReal x * toReal dy - toReal y * toReal dx) / (toReal x ^ 2 + toReal y ^ 2))
        linarith [mul_le_mul hK hPW (abs_nonneg _) (hu.le.trans hK)]

/-- In an IEEE format, `angleTangent x y dx dy` of finite configured values is finite when
`|(x * dy - y * dx) / (x ^ 2 + y ^ 2)|` of their real values is at most the largest finite value. -/
theorem isFinite_angleTangent {x y dx dy : Value} (hformat : format.isIEEE = true)
    (hx : isFinite x = true) (hy : isFinite y = true) (hdx : isFinite dx = true)
    (hdy : isFinite dy = true)
    (hrange : |(toReal x * toReal dy - toReal y * toReal dx) / (toReal x ^ 2 + toReal y ^ 2)| ≤
      Model.toReal (Model.posMaxFinite format)) :
    isFinite (angleTangent x y dx dy) = true :=
  (angleTangent_core hformat hx hy hdx hdy).1 hrange

/-- In an IEEE format, `angleTangent x y dx dy` of finite operands within errors `εx`, `εy`, `εdx`
and `εdy`, each at most zero, of `a`, `b`, `c` and `d`, with `|(a * d - b * c) / (a ^ 2 + b ^ 2)|`
at most the largest finite value, is within `K * (|a * d| + |b * c|) / (a ^ 2 + b ^ 2) + η` of
`(a * d - b * c) / (a ^ 2 + b ^ 2)`. Here `K = 1 / ((1 - u) * (1 - 2 * u) ^ 4) - 1`, `u` is the unit
roundoff, `η` the underflow error, and an error at most zero allows a literal's budget `0 + 0`. -/
theorem abs_toReal_angleTangent_sub_le {x y dx dy : Value} {a b c d εx εy εdx εdy : ℝ}
    (hformat : format.isIEEE = true) (hx : |toReal x - a| ≤ εx) (hy : |toReal y - b| ≤ εy)
    (hdx : |toReal dx - c| ≤ εdx) (hdy : |toReal dy - d| ≤ εdy) (hεx : εx ≤ 0) (hεy : εy ≤ 0)
    (hεdx : εdx ≤ 0) (hεdy : εdy ≤ 0) (hxf : isFinite x = true) (hyf : isFinite y = true)
    (hdxf : isFinite dx = true) (hdyf : isFinite dy = true)
    (hrange : |(a * d - b * c) / (a ^ 2 + b ^ 2)| ≤ Model.toReal (Model.posMaxFinite format)) :
    |toReal (angleTangent x y dx dy) - (a * d - b * c) / (a ^ 2 + b ^ 2)| ≤
      (1 / ((1 - Model.unitRoundoffAt format) * (1 - 2 * Model.unitRoundoffAt format) ^ 4) - 1) *
        ((|a * d| + |b * c|) / (a ^ 2 + b ^ 2)) + Model.underflowErrorAt format := by
  have ha : toReal x = a := sub_eq_zero.mp (abs_nonpos_iff.mp (hx.trans hεx))
  have hb : toReal y = b := sub_eq_zero.mp (abs_nonpos_iff.mp (hy.trans hεy))
  have hc : toReal dx = c := sub_eq_zero.mp (abs_nonpos_iff.mp (hdx.trans hεdx))
  have hd : toReal dy = d := sub_eq_zero.mp (abs_nonpos_iff.mp (hdy.trans hεdy))
  subst ha hb hc hd
  have h := angleTangent_core hformat hxf hyf hdxf hdyf
  exact h.2 (h.1 hrange)

/-- In an IEEE format, a finite `angleTangent x y dx dy` of exact finite operands of values `a`,
`b`, `c` and `d` is within relative error `K` and absolute error
`K * (|a * d| + |b * c| - |a * d - b * c|) / (a ^ 2 + b ^ 2) + η` of
`(a * d - b * c) / (a ^ 2 + b ^ 2)`, with `K`, `u` and `η` of `abs_toReal_angleTangent_sub_le`. -/
theorem mixedError_toReal_angleTangent {x y dx dy : Value} {a b c d : ℝ}
    (hformat : format.isIEEE = true) (hx : MixedError (toReal x) a 0 0)
    (hy : MixedError (toReal y) b 0 0) (hdx : MixedError (toReal dx) c 0 0)
    (hdy : MixedError (toReal dy) d 0 0) (hxf : isFinite x = true) (hyf : isFinite y = true)
    (hdxf : isFinite dx = true) (hdyf : isFinite dy = true)
    (hfinite : isFinite (angleTangent x y dx dy) = true) :
    MixedError (toReal (angleTangent x y dx dy)) ((a * d - b * c) / (a ^ 2 + b ^ 2))
      (1 / ((1 - Model.unitRoundoffAt format) * (1 - 2 * Model.unitRoundoffAt format) ^ 4) - 1)
      ((1 / ((1 - Model.unitRoundoffAt format) * (1 - 2 * Model.unitRoundoffAt format) ^ 4) -
          1) * ((|a * d| + |b * c| - |a * d - b * c|) / (a ^ 2 + b ^ 2)) +
        Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  have hb := mixedError_zero_zero_iff.mp hy
  have hc := mixedError_zero_zero_iff.mp hdx
  have hd := mixedError_zero_zero_iff.mp hdy
  subst ha hb hc hd
  obtain ⟨hu4, -, -⟩ := atanBackward_constants hformat
  have hu := Model.unitRoundoffAt_pos format
  have hK := hu.le.trans (le_angleTangent_constant hu.le (by linarith))
  have hT : 0 ≤ toReal x ^ 2 + toReal y ^ 2 := by positivity
  have hW : 0 ≤ |toReal x * toReal dy| + |toReal y * toReal dx| -
      |toReal x * toReal dy - toReal y * toReal dx| := sub_nonneg.mpr (abs_sub _ _)
  refine ⟨hK, add_nonneg (mul_nonneg hK (div_nonneg hW hT))
    (Model.underflowErrorAt_pos format).le, ?_⟩
  have h := (angleTangent_core hformat hxf hyf hdxf hdyf).2 hfinite
  have heq : (1 / ((1 - Model.unitRoundoffAt format) * (1 - 2 * Model.unitRoundoffAt format) ^ 4) -
        1) * |(toReal x * toReal dy - toReal y * toReal dx) / (toReal x ^ 2 + toReal y ^ 2)| +
      (1 / ((1 - Model.unitRoundoffAt format) * (1 - 2 * Model.unitRoundoffAt format) ^ 4) -
        1) * ((|toReal x * toReal dy| + |toReal y * toReal dx| -
          |toReal x * toReal dy - toReal y * toReal dx|) / (toReal x ^ 2 + toReal y ^ 2)) =
      (1 / ((1 - Model.unitRoundoffAt format) * (1 - 2 * Model.unitRoundoffAt format) ^ 4) -
        1) * ((|toReal x * toReal dy| + |toReal y * toReal dx|) /
          (toReal x ^ 2 + toReal y ^ 2)) := by
    rw [abs_div, abs_of_nonneg hT]
    ring
  linarith

/-- In an IEEE format, `tan` of a finite configured value is finite when the tangent is below the
overflow threshold in magnitude. -/
theorem isFinite_tan {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hrange : |Real.tan (toReal x)| < Model.Transcendentals.Certified.overflowThreshold format) :
    isFinite (Context.tan x) = true := by
  rw [show Context.tan x = TorchLean.ConfiguredBinary.tan x from rfl]
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [TorchLean.ConfiguredBinary.tan, ExecFloat.Binary.toModel,
    Configured.Family.toModel, ExecFloat.ModelCodec.decode_liftUnary] at *
  exact TorchLean.Elementary.isFinite_tan hformat hx hrange

/-- In an IEEE format, `tan` of an exact finite operand of value `a` with `|tan a|` below the
overflow threshold is within half an ulp of `tan a`. -/
theorem abs_toReal_tan_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0)
    (hxf : isFinite x = true)
    (hrange : |Real.tan a| < Model.Transcendentals.Certified.overflowThreshold format) :
    |toReal (Context.tan x) - Real.tan a| ≤ Model.halfUlpAt format (Real.tan a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  have hfinite := isFinite_tan hformat hxf (ha ▸ hrange)
  rw [show Context.tan x = ConfiguredBinary.tan x from rfl] at hfinite ⊢
  rw [ConfiguredBinary.toReal_tan hxf hfinite hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- A finite `tan` of an exact finite operand is rounded once: a mixed bound. -/
theorem mixedError_toReal_tan {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0)
    (hxf : isFinite x = true)
    (hfinite : isFinite (Context.tan x) = true) :
    MixedError (toReal (Context.tan x)) (Real.tan a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show Context.tan x = ConfiguredBinary.tan x from rfl] at hfinite ⊢
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_tan hxf hfinite hformat, ha]

/-- In an IEEE format, `exp2` of a finite configured value `x` is finite when `2 ^ x` is below
the overflow threshold. -/
theorem isFinite_exp2 {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hrange : (2 : ℝ) ^ toReal x < Model.Transcendentals.Certified.overflowThreshold format) :
    isFinite (Context.exp2 x) = true := by
  rw [show Context.exp2 x = TorchLean.ConfiguredBinary.exp2 x from rfl]
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [TorchLean.ConfiguredBinary.exp2, ExecFloat.Binary.toModel,
    Configured.Family.toModel, ExecFloat.ModelCodec.decode_liftUnary] at *
  exact TorchLean.Elementary.isFinite_exp2 hformat hx hrange

/-- In an IEEE format, `exp2` of an exact finite operand of value `a` with `2 ^ a` below the
overflow threshold is within half an ulp of `2 ^ a`. -/
theorem abs_toReal_exp2_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0)
    (hxf : isFinite x = true)
    (hrange : (2 : ℝ) ^ a < Model.Transcendentals.Certified.overflowThreshold format) :
    |toReal (Context.exp2 x) - (2 : ℝ) ^ a| ≤ Model.halfUlpAt format ((2 : ℝ) ^ a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  have hfinite := isFinite_exp2 hformat hxf (ha ▸ hrange)
  rw [show Context.exp2 x = ConfiguredBinary.exp2 x from rfl] at hfinite ⊢
  rw [ConfiguredBinary.toReal_exp2 hxf hfinite hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- A finite `exp2` of an exact finite operand is rounded once: a mixed bound. -/
theorem mixedError_toReal_exp2 {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0)
    (hxf : isFinite x = true)
    (hfinite : isFinite (Context.exp2 x) = true) :
    MixedError (toReal (Context.exp2 x)) ((2 : ℝ) ^ a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show Context.exp2 x = ConfiguredBinary.exp2 x from rfl] at hfinite ⊢
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_exp2 hxf hfinite hformat, ha]

/-- In an IEEE format, `log2` of a positive finite configured value `x` is finite when
`logb 2 x` is below the overflow threshold in magnitude. -/
theorem isFinite_log2 {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hpositive : 0 < toReal x)
    (hrange :
      |Real.logb 2 (toReal x)| < Model.Transcendentals.Certified.overflowThreshold format) :
    isFinite (Context.log2 x) = true := by
  rw [show Context.log2 x = TorchLean.ConfiguredBinary.logb 2 x from rfl]
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [TorchLean.ConfiguredBinary.logb, ExecFloat.Binary.toModel,
    Configured.Family.toModel, ExecFloat.ModelCodec.decode_liftUnary] at *
  refine TorchLean.Elementary.isFinite_logb hformat (by norm_num) hx hpositive ?_
  exact_mod_cast hrange

/-- In an IEEE format, `log2` of an exact finite operand of positive value `a` with
`|logb 2 a|` below the overflow threshold is within half an ulp of `logb 2 a`. -/
theorem abs_toReal_log2_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0)
    (hxf : isFinite x = true) (hpositive : 0 < a)
    (hrange : |Real.logb 2 a| < Model.Transcendentals.Certified.overflowThreshold format) :
    |toReal (Context.log2 x) - Real.logb 2 a| ≤ Model.halfUlpAt format (Real.logb 2 a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  have hfinite := isFinite_log2 hformat hxf (ha ▸ hpositive) (ha ▸ hrange)
  rw [show Context.log2 x = ConfiguredBinary.logb 2 x from rfl] at hfinite ⊢
  rw [ConfiguredBinary.toReal_logb (by norm_num) hxf (ha ▸ hpositive)
      hfinite hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- A finite `log2` of an exact finite operand is rounded once: a mixed bound. The finite result
makes the operand positive. -/
theorem mixedError_toReal_log2 {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0)
    (hxf : isFinite x = true)
    (hfinite : isFinite (Context.log2 x) = true) :
    MixedError (toReal (Context.log2 x)) (Real.logb 2 a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show Context.log2 x = ConfiguredBinary.logb 2 x from rfl] at hfinite ⊢
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_logb (by norm_num) hxf
      (ConfiguredBinary.toReal_pos_of_isFinite_logb hxf hfinite hformat) hfinite
      hformat, ha]
    norm_num

/-- In an IEEE format, `log10` of a positive finite configured value `x` is finite when
`logb 10 x` is below the overflow threshold in magnitude. -/
theorem isFinite_log10 {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hpositive : 0 < toReal x)
    (hrange :
      |Real.logb 10 (toReal x)| < Model.Transcendentals.Certified.overflowThreshold format) :
    isFinite (Context.log10 x) = true := by
  rw [show Context.log10 x = TorchLean.ConfiguredBinary.logb 10 x from rfl]
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [TorchLean.ConfiguredBinary.logb, ExecFloat.Binary.toModel,
    Configured.Family.toModel, ExecFloat.ModelCodec.decode_liftUnary] at *
  refine TorchLean.Elementary.isFinite_logb hformat (by norm_num) hx hpositive ?_
  exact_mod_cast hrange

/-- In an IEEE format, `log10` of an exact finite operand of positive value `a` with
`|logb 10 a|` below the overflow threshold is within half an ulp of `logb 10 a`. -/
theorem abs_toReal_log10_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0)
    (hxf : isFinite x = true) (hpositive : 0 < a)
    (hrange : |Real.logb 10 a| < Model.Transcendentals.Certified.overflowThreshold format) :
    |toReal (Context.log10 x) - Real.logb 10 a| ≤ Model.halfUlpAt format (Real.logb 10 a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  have hfinite := isFinite_log10 hformat hxf (ha ▸ hpositive) (ha ▸ hrange)
  rw [show Context.log10 x = ConfiguredBinary.logb 10 x from rfl] at hfinite ⊢
  rw [ConfiguredBinary.toReal_logb (by norm_num) hxf (ha ▸ hpositive)
      hfinite hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- A finite `log10` of an exact finite operand is rounded once: a mixed bound. The finite result
makes the operand positive. -/
theorem mixedError_toReal_log10 {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0)
    (hxf : isFinite x = true)
    (hfinite : isFinite (Context.log10 x) = true) :
    MixedError (toReal (Context.log10 x)) (Real.logb 10 a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show Context.log10 x = ConfiguredBinary.logb 10 x from rfl] at hfinite ⊢
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_logb (by norm_num) hxf
      (ConfiguredBinary.toReal_pos_of_isFinite_logb hxf hfinite hformat) hfinite
      hformat, ha]
    norm_num

/-- In an IEEE format, `asinh` of a finite configured value is finite. -/
theorem isFinite_asinh {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true) :
    isFinite (Context.asinh x) = true := by
  rw [show Context.asinh x = TorchLean.ConfiguredBinary.asinh x from rfl]
  unfold ExecFloat.Binary.isFinite at *
  simp only [TorchLean.ConfiguredBinary.asinh, ExecFloat.Binary.toModel,
    Configured.Family.toModel, ExecFloat.ModelCodec.decode_liftUnary] at *
  exact TorchLean.Elementary.isFinite_asinh hformat hx

/-- `asinh` of an exact finite operand is rounded once: an absolute bound. -/
theorem abs_toReal_asinh_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0)
    (hxf : isFinite x = true) :
    |toReal (Context.asinh x) - Real.arsinh a| ≤ Model.halfUlpAt format (Real.arsinh a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  rw [show Context.asinh x = ConfiguredBinary.asinh x from rfl]
  rw [ConfiguredBinary.toReal_asinh hxf hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- `asinh` of an exact finite operand is rounded once: a mixed bound. -/
theorem mixedError_toReal_asinh {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0)
    (hxf : isFinite x = true) :
    MixedError (toReal (Context.asinh x)) (Real.arsinh a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show Context.asinh x = ConfiguredBinary.asinh x from rfl]
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_asinh hxf hformat, ha]

/-- `acosh` of an exact finite operand is rounded once: an absolute bound. -/
theorem abs_toReal_acosh_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0)
    (hxf : isFinite x = true) (hdomain : 1 ≤ a) :
    |toReal (Context.acosh x) - Real.arcosh a| ≤ Model.halfUlpAt format (Real.arcosh a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  rw [show Context.acosh x = ConfiguredBinary.acosh x from rfl]
  rw [ConfiguredBinary.toReal_acosh hxf (ha ▸ hdomain) hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- `acosh` of a finite configured value at least one is finite. -/
theorem isFinite_acosh {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hdomain : 1 ≤ toReal x) : isFinite (Context.acosh x) = true := by
  rw [show Context.acosh x = ConfiguredBinary.acosh x from rfl]
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [ConfiguredBinary.acosh, ExecFloat.Binary.toModel,
    Configured.Family.toModel, ExecFloat.ModelCodec.decode_liftUnary] at *
  exact Elementary.isFinite_acosh hformat hx hdomain

/-- A finite `acosh` of an exact finite operand is rounded once: a mixed bound. The finite result
makes the operand at least one. -/
theorem mixedError_toReal_acosh {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0)
    (hxf : isFinite x = true) (hfinite : isFinite (Context.acosh x) = true) :
    MixedError (toReal (Context.acosh x)) (Real.arcosh a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show Context.acosh x = ConfiguredBinary.acosh x from rfl] at hfinite ⊢
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_acosh hxf
      (ConfiguredBinary.one_le_toReal_of_isFinite_acosh hxf hfinite hformat)
      hformat, ha]

/-- In an IEEE format, `atanh` of a finite configured value `x` in `(-1, 1)` is finite when
`artanh x` is below the overflow threshold in magnitude. -/
theorem isFinite_atanh {x : Value} (hformat : format.isIEEE = true) (hx : isFinite x = true)
    (hdomain : |toReal x| < 1)
    (hrange : |Real.artanh (toReal x)| < Model.Transcendentals.Certified.overflowThreshold format) :
    isFinite (Context.atanh x) = true := by
  rw [show Context.atanh x = TorchLean.ConfiguredBinary.atanh x from rfl]
  unfold ExecFloat.Binary.isFinite ExecFloat.Binary.toReal at *
  simp only [TorchLean.ConfiguredBinary.atanh, ExecFloat.Binary.toModel,
    Configured.Family.toModel, ExecFloat.ModelCodec.decode_liftUnary] at *
  exact TorchLean.Elementary.isFinite_atanh hformat hx hdomain hrange

/-- In an IEEE format, `atanh` of an exact finite operand of value `a` in `(-1, 1)` with
`|artanh a|` below the overflow threshold is within half an ulp of `artanh a`. -/
theorem abs_toReal_atanh_sub_le {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : |toReal x - a| ≤ 0)
    (hxf : isFinite x = true) (hdomain : |a| < 1)
    (hrange : |Real.artanh a| < Model.Transcendentals.Certified.overflowThreshold format) :
    |toReal (Context.atanh x) - Real.artanh a| ≤ Model.halfUlpAt format (Real.artanh a) := by
  have ha : toReal x = a := by simpa [sub_eq_zero] using hx
  have hfinite := isFinite_atanh hformat hxf (ha ▸ hdomain) (ha ▸ hrange)
  rw [show Context.atanh x = ConfiguredBinary.atanh x from rfl] at hfinite ⊢
  rw [ConfiguredBinary.toReal_atanh hxf (ha ▸ hdomain) hfinite hformat, ha]
  exact Model.abs_roundAt_sub_le _ _

/-- A finite `atanh` of an exact finite operand is rounded once: a mixed bound. The finite result
puts the operand in `(-1, 1)`. -/
theorem mixedError_toReal_atanh {x : Value} {a : ℝ} (hformat : format.isIEEE = true)
    (hx : MixedError (toReal x) a 0 0)
    (hxf : isFinite x = true)
    (hfinite : isFinite (Context.atanh x) = true) :
    MixedError (toReal (Context.atanh x)) (Real.artanh a) (Model.unitRoundoffAt format)
      (Model.underflowErrorAt format) := by
  have ha := mixedError_zero_zero_iff.mp hx
  rw [show Context.atanh x = ConfiguredBinary.atanh x from rfl] at hfinite ⊢
  exact ExecFloat.Binary.mixedError_toReal_of_eq_roundAt <| by
    rw [ConfiguredBinary.toReal_atanh hxf
      (ConfiguredBinary.abs_toReal_lt_one_of_isFinite_atanh hxf hfinite hformat)
      hfinite hformat, ha]

attribute [roundoff_rule] isFinite_floor isFinite_ceil isFinite_round isFinite_trunc
  abs_toReal_floor_sub_le abs_toReal_ceil_sub_le abs_toReal_round_sub_le abs_toReal_trunc_sub_le
  mixedError_toReal_floor mixedError_toReal_ceil mixedError_toReal_round mixedError_toReal_trunc
  isFinite_copysign mixedError_toReal_copysign abs_toReal_copysign_sub_le
  isNaN_of_isFinite isInf_of_isFinite not_beq_self_of_isFinite abs_toReal_hypot_sub_le
  mixedError_toReal_hypot isFinite_quietNaN abs_toReal_quietNaN_sub_le mixedError_toReal_quietNaN
  isFinite_extremaTie abs_toReal_extremaTie_sub_le mixedError_toReal_extremaTie
  toReal_maximum_zero isFinite_maximum_zero toReal_maximum isFinite_maximum toReal_minimum
  isFinite_minimum isFinite_ratCast_half toReal_ratCast_half
  isFinite_fmod abs_toReal_fmod_sub_le mixedError_toReal_fmod
  isFinite_tan abs_toReal_tan_sub_le mixedError_toReal_tan
  isFinite_exp2 abs_toReal_exp2_sub_le mixedError_toReal_exp2
  isFinite_log2 abs_toReal_log2_sub_le mixedError_toReal_log2
  isFinite_log10 abs_toReal_log10_sub_le mixedError_toReal_log10
  isFinite_asinh abs_toReal_asinh_sub_le mixedError_toReal_asinh
  abs_toReal_acosh_sub_le isFinite_acosh mixedError_toReal_acosh
  isFinite_atanh abs_toReal_atanh_sub_le mixedError_toReal_atanh

-- After `abs_toReal_hypot_sub_le`, which keeps the a priori budget of exact operands.
attribute [roundoff_rule] abs_toReal_hypot_sub_le_add

end TorchLean.Roundoff

public meta section

namespace TorchLean.Roundoff

/-- Lean's native `Float32` as storage: the analysis keeps its definitions folded, as it keeps
FloatLib's arithmetic, and reads a native literal or comparison as written. -/
def nativeFloat32 : FloatLib.Floats.Formats.BinaryInterchange.Roundoff.Extension where
  storage := #[`Float32]

attribute [roundoff_extension] nativeFloat32

end TorchLean.Roundoff
