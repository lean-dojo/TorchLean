/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof
public import NN.Core.Numeric.Enclosure
public import NN.Proofs.RuntimeApprox.Rounding

/-!
# Rounding of recovered coefficients

The existing recovery search returns rounded endpoints of certified coefficient enclosures.
Equal finite endpoint values determine the rounding of every enclosed real value by monotonicity.
The configured result retains its original input and optional seed; earlier floating-point work
can therefore contribute its own error when this numerical rule is composed.

These theorems use the selected configured codec and an IEEE descriptor. They concern successful
recovery; the search may return `none`. Agreement here concerns real values, including signed zeros.
-/

public section

namespace Proofs.RuntimeApprox.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open TorchLean.Numeric TorchLean.Numeric.Internal

/-- Exact rational decoding agrees with the total real observation. Exceptional values decode
as zero here; numerical API entry points must separately establish finiteness. -/
theorem cast_toRat_eq_toReal {format : FloatFormat} (input : Model format) :
    (((Model.toRat? input).getD 0 : Rat) : ℝ) = Model.toReal input := by
  cases decoded : Model.toDyadic? input with
  | none => simp [Model.toRat?, Model.toReal_eq, decoded]
  | some dyadic =>
      simp [Model.toRat?, Model.toReal_eq, decoded, Dyadic.cast_toRat]

/-- Converting an exact rational rounds once, including tensor-size denominators. -/
theorem model_toReal_ofRat {format : FloatFormat} (value : Rat)
    (hformat : format.isIEEE = true) (hfinite : Model.isFinite (Model.ofRat format value) = true) :
    Model.toReal (Model.ofRat format value) = Model.roundAt format (value : ℝ) := by
  have signed : Model.signedScaledRatToReal (decide (value.num < 0))
      value.num.natAbs value.den 0 = (value : ℝ) := by
    rw [Rat.cast_def]
    by_cases h : value.num < 0
    · simp [Model.signedScaledRatToReal, Model.scaledRatToReal, h, abs_of_neg h, neg_div]
    · simp [Model.signedScaledRatToReal, Model.scaledRatToReal, h,
        abs_of_nonneg (le_of_not_gt h)]
  change Model.toReal (Model.roundRat format (decide (value.num < 0))
    value.num.natAbs value.den) = _
  rw [Model.roundRat, Model.toReal_roundRatScaled_eq_roundAt_of_isFinite
    format _ value.num.natAbs value.den 0 hformat value.den_nz hfinite, signed]

/-- Monotonicity traps the target's rounding between equal rounded endpoint values. -/
theorem model_endpoint_rounding {format : FloatFormat} {lower upper : Rat} {target : ℝ}
    (hformat : format.isIEEE = true)
    (hlower : (lower : ℝ) ≤ target) (hupper : target ≤ (upper : ℝ))
    (hfiniteLower : Model.isFinite (Model.ofRat format lower) = true)
    (hfiniteUpper : Model.isFinite (Model.ofRat format upper) = true)
    (hsame : Model.toReal (Model.ofRat format lower) =
      Model.toReal (Model.ofRat format upper)) :
    Model.toReal (Model.ofRat format lower) = Model.roundAt format target := by
  have lowerRounding := model_toReal_ofRat lower hformat hfiniteLower
  have upperRounding := model_toReal_ofRat upper hformat hfiniteUpper
  have roundedLower := Model.roundAt_mono format hlower
  have roundedUpper := Model.roundAt_mono format hupper
  rw [← lowerRounding] at roundedLower
  rw [← upperRounding, ← hsame] at roundedUpper
  exact le_antisymm roundedLower roundedUpper

section Configured

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The configured decoder exposes the rational coefficient before conversion. Finiteness is
a separate property of the converted value. -/
theorem configured_decode (coefficients : QuotientCoefficients 0) :
    @QuotientArithmetic.decode Value
        (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
        coefficients =
      some (ExecFloat.Binary.ofModel
        (Model.ofRat format (QuotientCoefficients.primal coefficients)) : Value) := by
  cases coefficients with
  | scalar value => rfl

/-- A successful configured coefficient conversion is an exact rational observation. -/
theorem configured_encode (value : Value) (q : Rat)
    (encoded : @QuotientArithmetic.encode Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      value = some (.scalar q)) : ExecFloat.Binary.toRat? value = some q := by
  change (ExecFloat.Binary.toRat? value).map Coefficients.scalar = some (.scalar q) at encoded
  obtain ⟨actual, decoded, agrees⟩ := Option.map_eq_some_iff.mp encoded
  exact Coefficients.scalar.inj agrees ▸ decoded

/-- Successful exact encodings of the two packed endpoint roundings give their finiteness and
their common real value. The storage plan and code carrier remain arbitrary. -/
theorem configured_endpoint_rounding {lower upper shared : Rat} {target : ℝ}
    (hformat : format.isIEEE = true)
    (hlower : (lower : ℝ) ≤ target) (hupper : target ≤ (upper : ℝ))
    (lowerEncoding : ExecFloat.Binary.toRat?
      (ExecFloat.Binary.ofModel (Model.ofRat format lower) : Value) = some shared)
    (upperEncoding : ExecFloat.Binary.toRat?
      (ExecFloat.Binary.ofModel (Model.ofRat format upper) : Value) = some shared) :
    ExecFloat.Binary.toReal
      (ExecFloat.Binary.ofModel (Model.ofRat format lower) : Value) =
      Model.roundAt format target := by
  have lowerDecode : Model.toRat? (Model.ofRat format lower) = some shared := by
    simpa only [ExecFloat.Binary.toRat?, ExecFloat.Binary.toModel_ofModel] using lowerEncoding
  have upperDecode : Model.toRat? (Model.ofRat format upper) = some shared := by
    simpa only [ExecFloat.Binary.toRat?, ExecFloat.Binary.toModel_ofModel] using upperEncoding
  have finiteLower := Model.Algebraic.isFinite_of_toRat?_eq_some lowerDecode
  have finiteUpper := Model.Algebraic.isFinite_of_toRat?_eq_some upperDecode
  have realLower := Model.Algebraic.toReal_of_toRat?_eq_some lowerDecode
  have realUpper := Model.Algebraic.toReal_of_toRat?_eq_some upperDecode
  have same := realLower.trans realUpper.symm
  simpa only [ExecFloat.Binary.toReal, ExecFloat.Binary.toModel_ofModel] using
    model_endpoint_rounding hformat hlower hupper finiteLower finiteUpper same

/-- An accepted logarithm enclosure rounds its real target once through the configured codec.
The coefficient may exceed the format's finite input range. -/
theorem configured_logarithm_rounding (input : Rat) {result : Value}
    (hformat : format.isIEEE = true)
    (accepted : @enclosedLogarithm? Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      (.scalar input) = some result) :
    ExecFloat.Binary.toReal result = Model.roundAt format (Real.log (input : ℝ)) := by
  let ops : QuotientArithmetic Value := QuotientArithmetic.configured
  obtain ⟨bounds, contained, lowerDecoded, upper, shared, upperDecoded,
    resultEncoded, upperEncoded⟩ := @enclosedLogarithm?_eq_some Value ops (.scalar input)
      result accepted
  cases bounds with | scalar interval =>
    cases shared with | scalar shared =>
      change interval.ContainsReal some (Real.log (input : ℝ)) at contained
      obtain ⟨lowerBound, upperBound⟩ :=
        (Interval.containsReal_some_iff _ _).mp contained
      change some (ExecFloat.Binary.ofModel (Model.ofRat format interval.lo) : Value) =
        some result at lowerDecoded
      change some (ExecFloat.Binary.ofModel (Model.ofRat format interval.hi) : Value) =
        some upper at upperDecoded
      have lowerEqual := Option.some.inj lowerDecoded
      have upperEqual := Option.some.inj upperDecoded
      have rounded := configured_endpoint_rounding hformat lowerBound upperBound
        ((congrArg ExecFloat.Binary.toRat? lowerEqual).trans
          (configured_encode result shared resultEncoded))
        ((congrArg ExecFloat.Binary.toRat? upperEqual).trans
          (configured_encode upper shared upperEncoded))
      exact (congrArg ExecFloat.Binary.toReal lowerEqual).symm.trans rounded

/-- Successful complete logarithm replay has only the destination's rounding error. -/
theorem enclosedLogarithm_get_error (input : Rat) (hformat : format.isIEEE = true)
    (found : (@enclosedLogarithm? Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      (.scalar input)).isSome) :
    |ExecFloat.Binary.toReal
        ((@enclosedLogarithm? Value
          (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
          (.scalar input)).get found) - Real.log (input : ℝ)| ≤
      Model.epsilonAt format (Real.log (input : ℝ)) := by
  rw [configured_logarithm_rounding input hformat (Option.eq_some_of_isSome found)]
  exact Model.abs_roundAt_sub_le _ _

/-- Actual successful recovery gives nearest-even rounding of its enclosed real coefficient.
The input and optional seed retain the original configured encoding operations. -/
theorem configured_recovery_rounding (input : Value) (seed : Option Value) (isTanh : Bool)
    {result : Value} (hformat : format.isIEEE = true)
    (accepted : @enclosedSlope? Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      input seed isTanh = some result) :
    ExecFloat.Binary.toReal result = Model.roundAt format
      (idealSlope (.scalar ((ExecFloat.Binary.toRat? input).getD 0))
        (.scalar (match seed with
          | none => 1
          | some upstream => (ExecFloat.Binary.toRat? upstream).getD 0)) isTanh).primal := by
  let ops : QuotientArithmetic Value := QuotientArithmetic.configured
  have recovered := @enclosedSlope?_eq_some Value ops input seed isTanh result accepted
  obtain ⟨encodedInput, encodedSeed, bounds, inputEncoded, seedEncoded, contained,
    lowerDecoded, upper, shared, upperDecoded, resultEncoded, upperEncoded⟩ := recovered
  cases encodedInput with | scalar inputQ =>
    cases encodedSeed with | scalar seedQ =>
      cases bounds with | scalar interval =>
        cases shared with | scalar sharedQ =>
          have inputDecoded := configured_encode input inputQ inputEncoded
          have inputValue : inputQ = (ExecFloat.Binary.toRat? input).getD 0 := by
            simp only [inputDecoded, Option.getD_some]
          have seedValue : seedQ = (match seed with
              | none => 1
              | some upstream => (ExecFloat.Binary.toRat? upstream).getD 0) := by
            cases seed with
            | none =>
                change some (Coefficients.scalar (1 : Rat)) =
                  some (Coefficients.scalar seedQ) at seedEncoded
                change seedQ = (1 : Rat)
                exact (Coefficients.scalar.inj (Option.some.inj seedEncoded)).symm
            | some upstream =>
                have decoded := configured_encode upstream seedQ seedEncoded
                change seedQ = (ExecFloat.Binary.toRat? upstream).getD 0
                simp only [decoded, Option.getD_some]
          let target : ℝ := (idealSlope (.scalar inputQ) (.scalar seedQ) isTanh).primal
          have targetContained : interval.ContainsReal some target := by
            cases meaning : idealSlope (.scalar inputQ) (.scalar seedQ) isTanh with
            | scalar reference =>
                simpa only [target, meaning, Related, QuotientCoefficients.primal] using contained
          obtain ⟨lowerBound, upperBound⟩ :=
            (Interval.containsReal_some_iff interval target).mp targetContained
          change some (ExecFloat.Binary.ofModel (Model.ofRat format interval.lo) : Value) =
            some result at lowerDecoded
          change some (ExecFloat.Binary.ofModel (Model.ofRat format interval.hi) : Value) =
            some upper at upperDecoded
          have resultDecoded := configured_encode result sharedQ resultEncoded
          have upperResultDecoded := configured_encode upper sharedQ upperEncoded
          have packedLower := Option.some.inj lowerDecoded
          have packedUpper := Option.some.inj upperDecoded
          have rounded := configured_endpoint_rounding
            hformat lowerBound upperBound
            ((congrArg ExecFloat.Binary.toRat? packedLower).trans resultDecoded)
            ((congrArg ExecFloat.Binary.toRat? packedUpper).trans upperResultDecoded)
          have actualRounding : ExecFloat.Binary.toReal result = Model.roundAt format target :=
            (congrArg ExecFloat.Binary.toReal packedLower).symm.trans rounded
          cases seed <;> simpa only [target, inputValue, seedValue] using actualRounding

/-- The accepted algorithm's value is within one rounding budget of its original-input formula. -/
theorem enclosedSlope_get_error (input : Value) (seed : Option Value) (isTanh : Bool)
    (hformat : format.isIEEE = true)
    (found : (@enclosedSlope? Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      input seed isTanh).isSome) :
    let tail := Real.exp (-(if isTanh then (2 : ℝ) else 1) * |ExecFloat.Binary.toReal input|)
    let numerator := (match seed with
      | none => 1
      | some upstream => ExecFloat.Binary.toReal upstream) * tail
    let reference :=
      (if isTanh then (numerator + numerator) + (numerator + numerator) else numerator) /
        ((1 + tail) * (1 + tail))
    |ExecFloat.Binary.toReal
        ((@enclosedSlope? Value
          (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
          input seed isTanh).get found) - reference| ≤ Model.epsilonAt format reference := by
  intro tail numerator reference
  have accepted := Option.eq_some_of_isSome found
  have rounded := configured_recovery_rounding input seed isTanh hformat accepted
  rw [idealSlope_scalar] at rounded
  have observed : ExecFloat.Binary.toReal
      ((@enclosedSlope? Value
        (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
        input seed isTanh).get found) = Model.roundAt format reference := by
    cases seed <;>
      simpa only [tail, numerator, reference, ExecFloat.Binary.toRat?,
        cast_toRat_eq_toReal, ExecFloat.Binary.toReal, Rat.cast_one] using rounded
  rw [observed]
  exact Model.abs_roundAt_sub_le _ _

end Configured

end Proofs.RuntimeApprox.Roundoff
