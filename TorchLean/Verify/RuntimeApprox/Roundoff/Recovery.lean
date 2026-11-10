/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Rational
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.MixedError
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof
public import TorchLean.Tensor.Numeric.Enclosure
public import TorchLean.Verify.RuntimeApprox.Rounding
import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Rules

/-!
# Rounding of recovered coefficients

`enclosedSlope?` encloses the sigmoid or tanh slope at a configured input and seed in a certified
rational interval and rounds both endpoints. `enclosedPowerSlope?` does the same for the slope
`e * x ^ (e - 1) * g` of a power, and `enclosedPowerTangent?` for the power tangent
`x ^ y * (dy * log x + y * (dx / x))`. When the endpoints round to the same value, monotonicity of
rounding makes it the rounding of the exact value. The theorems are stated at the configured
inputs, so the error of the computation that produced them composes with this rule.

These theorems use the selected configured codec and an IEEE descriptor. They describe a
successful search (a search may also return `none`) and compare real values, which do not
distinguish signed zeros.

Quotient replay encodes every finite value exactly. Decoding always returns a value, the
coefficient rounded once, which is finite when the coefficient lies within the largest finite
magnitude. A checked configured operation computes the ordinary operation, and a zero range flag
certifies a finite result that is not subnormal.
-/

public section

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open TorchLean.Numeric QuotientCoefficients

/-- If two rationals enclosing `target` round to finite values with the same real value, that value
is the rounding of `target`. -/
theorem model_endpoint_rounding {format : FloatFormat} {lower upper : Rat} {target : ℝ}
    (hformat : format.isIEEE = true)
    (hlower : (lower : ℝ) ≤ target) (hupper : target ≤ (upper : ℝ))
    (hfiniteLower : Model.isFinite (Model.ofRat format lower) = true)
    (hfiniteUpper : Model.isFinite (Model.ofRat format upper) = true)
    (hsame : Model.toReal (Model.ofRat format lower) =
      Model.toReal (Model.ofRat format upper)) :
    Model.toReal (Model.ofRat format lower) = Model.roundAt format target := by
  have lowerRounding := Model.toReal_ofRat_eq_roundAt lower hformat hfiniteLower
  have upperRounding := Model.toReal_ofRat_eq_roundAt upper hformat hfiniteUpper
  have roundedLower := Model.roundAt_mono format hlower
  have roundedUpper := Model.roundAt_mono format hupper
  rw [← lowerRounding] at roundedLower
  rw [← upperRounding, ← hsame] at roundedUpper
  exact le_antisymm roundedLower roundedUpper

section Configured

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The configured decoder rounds the rational value of a coefficient into the format. The result
need not be finite (`configured_decode_rounding`). -/
theorem configured_decode (coefficients : QuotientCoefficients 0) :
    @QuotientArithmetic.decode Value
        (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
        coefficients =
      some (ExecFloat.Binary.ofModel
        (Model.ofRat format (QuotientCoefficients.primal coefficients)) : Value) := by
  cases coefficients with
  | scalar value => rfl

/-- If the configured encoder returns the coefficient `q`, the value decodes to the rational `q`. -/
theorem configured_encode (value : Value) (q : Rat)
    (encoded : @QuotientArithmetic.encode Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      value = some (.scalar q)) : ExecFloat.Binary.toRat? value = some q := by
  change (ExecFloat.Binary.toRat? value).map Coefficients.scalar = some (.scalar q) at encoded
  obtain ⟨actual, decoded, agrees⟩ := Option.map_eq_some_iff.mp encoded
  exact Coefficients.scalar.inj agrees ▸ decoded

/-- The rational coefficient of a configured value denotes its real value. Exceptional values
decode as zero on both sides. -/
@[simp] theorem cast_toRat_eq_toReal (value : Value) :
    (((ExecFloat.Binary.toRat? value).getD 0 : Rat) : ℝ) = ExecFloat.Binary.toReal value :=
  Model.cast_toRat_eq_toReal _

/-- Every finite configured value encodes as the rational it denotes. -/
theorem configured_encode_of_isFinite {value : Value}
    (hvalue : ExecFloat.Binary.isFinite value = true) :
    @QuotientArithmetic.encode Value
        (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) value =
      some (.scalar ((ExecFloat.Binary.toRat? value).getD 0)) := by
  obtain ⟨dyadic, decoded⟩ := Model.exists_toDyadic?_of_isFinite hvalue
  change (Model.toRat? (ExecFloat.Binary.toModel value)).map Coefficients.scalar = _
  simp [ExecFloat.Binary.toRat?, Model.toRat?, decoded]

/-- A coefficient within the largest finite magnitude decodes to a finite value, the coefficient
rounded once. -/
theorem configured_decode_rounding (value : Rat) {result : Value}
    (hformat : format.isIEEE = true)
    (hrange : |(value : ℝ)| ≤ Model.toReal (Model.posMaxFinite format))
    (decoded : @QuotientArithmetic.decode Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      (.scalar value) = some result) :
    ExecFloat.Binary.isFinite result = true ∧
      ExecFloat.Binary.toReal result = Model.roundAt format value := by
  obtain rfl := Option.some.inj decoded
  have finite := Model.isFinite_ofRat _ hformat hrange
  simp only [ExecFloat.Binary.isFinite, ExecFloat.Binary.toReal, ExecFloat.Binary.toModel_ofModel]
  exact ⟨finite, Model.toReal_ofRat_eq_roundAt _ hformat finite⟩

/-- Configured IEEE values support quotient replay. -/
theorem configured_supported (hformat : format.isIEEE = true) :
    (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)).supported =
      true := by
  change (format.encoding == .ieee) = true
  simp [FloatFormat.encoding_eq_ieee_of_isIEEE format hformat]

/-- A zero range flag certifies a finite outcome that is not subnormal. -/
theorem configuredFlags_eq_zero
    {outcome : ExecFloat.Binary.IEEEOutcome (format := format) (plan := plan) (code := code)}
    (hflags : QuotientArithmetic.configuredFlags outcome = 0) :
    ExecFloat.Binary.isFinite outcome.value = true ∧
      ExecFloat.Binary.isSubnormal outcome.value = false := by
  unfold QuotientArithmetic.configuredFlags at hflags
  simp_all [Bool.toUInt8]

section Checked

variable [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]

section Add

variable [cap : ExecFloat.Add (Configured.Family format code plan)]
  [ExecFloat.Capability.HasSpec cap Configured.Spec.add]

/-- A checked configured sum computes the ordinary sum. -/
theorem configured_addWithFlags_fst (x y : Value) :
    (@QuotientArithmetic.addWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x y).1 =
      x + y := by
  apply ExecFloat.Binary.toModel_inj.mp
  rw [ExecFloat.Binary.toModel_hAdd x y, ← Model.value_add]
  exact Roundoff.decode_outcome _

/-- A checked configured sum without range flags is finite and not subnormal. -/
theorem isFinite_add_of_flags_eq_zero {x y : Value}
    (hflags : (@QuotientArithmetic.addWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x y).2 = 0) :
    ExecFloat.Binary.isFinite (x + y) = true ∧ ExecFloat.Binary.isSubnormal (x + y) = false := by
  rw [← configured_addWithFlags_fst]
  exact configuredFlags_eq_zero hflags

end Add

section Sub

variable [cap : ExecFloat.Sub (Configured.Family format code plan)]
  [ExecFloat.Capability.HasSpec cap Configured.Spec.sub]

/-- A checked configured difference computes the ordinary difference. -/
theorem configured_subWithFlags_fst (x y : Value) :
    (@QuotientArithmetic.subWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x y).1 =
      x - y := by
  apply ExecFloat.Binary.toModel_inj.mp
  rw [ExecFloat.Binary.toModel_hSub x y, ← Model.value_sub]
  exact Roundoff.decode_outcome _

/-- A checked configured difference without range flags is finite and not subnormal. -/
theorem isFinite_sub_of_flags_eq_zero {x y : Value}
    (hflags : (@QuotientArithmetic.subWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x y).2 = 0) :
    ExecFloat.Binary.isFinite (x - y) = true ∧ ExecFloat.Binary.isSubnormal (x - y) = false := by
  rw [← configured_subWithFlags_fst]
  exact configuredFlags_eq_zero hflags

end Sub

section Mul

variable [cap : ExecFloat.Mul (Configured.Family format code plan)]
  [ExecFloat.Capability.HasSpec cap Configured.Spec.mul]

/-- A checked configured product computes the ordinary product. -/
theorem configured_mulWithFlags_fst (x y : Value) :
    (@QuotientArithmetic.mulWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x y).1 =
      x * y := by
  apply ExecFloat.Binary.toModel_inj.mp
  rw [ExecFloat.Binary.toModel_hMul x y, ← Model.value_mul]
  exact Roundoff.decode_outcome _

/-- A checked configured product without range flags is finite and not subnormal, and if it is
zero then so is a factor. -/
theorem isFinite_mul_of_flags_eq_zero {x y : Value}
    (hflags : (@QuotientArithmetic.mulWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x y).2 = 0) :
    ExecFloat.Binary.isFinite (x * y) = true ∧ ExecFloat.Binary.isSubnormal (x * y) = false ∧
      (ExecFloat.Binary.isZero (x * y) = true →
        ExecFloat.Binary.isZero x = true ∨ ExecFloat.Binary.isZero y = true) := by
  obtain ⟨flags, lost⟩ := UInt8.or_eq_zero_iff.mp hflags
  have range : ExecFloat.Binary.isFinite (x * y) = true ∧
      ExecFloat.Binary.isSubnormal (x * y) = false := by
    rw [← configured_mulWithFlags_fst]
    exact configuredFlags_eq_zero flags
  have zeroFree : (ExecFloat.Binary.isZero (x * y) && !ExecFloat.Binary.isZero x &&
      !ExecFloat.Binary.isZero y).toUInt8 = 0 := by
    rw [← configured_mulWithFlags_fst]
    exact lost
  refine ⟨range.1, range.2, fun zero => ?_⟩
  by_contra nonzero
  simp only [not_or, Bool.not_eq_true] at nonzero
  simp [zero, nonzero.1, nonzero.2, Bool.toUInt8] at zeroFree

end Mul

section Div

variable [cap : ExecFloat.Div (Configured.Family format code plan)]
  [ExecFloat.Capability.HasSpec cap Configured.Spec.div]

/-- A checked configured quotient computes the ordinary quotient. -/
theorem configured_divWithFlags_fst (x y : Value) :
    (@QuotientArithmetic.divWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x y).1 =
      x / y := by
  apply ExecFloat.Binary.toModel_inj.mp
  rw [ExecFloat.Binary.toModel_hDiv x y, ← Model.value_div]
  exact Roundoff.decode_outcome _

/-- A checked configured quotient without range flags is finite and not subnormal, and if it is
zero then so is the dividend. -/
theorem isFinite_div_of_flags_eq_zero {x y : Value}
    (hflags : (@QuotientArithmetic.divWithFlags Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code)) x y).2 = 0) :
    ExecFloat.Binary.isFinite (x / y) = true ∧ ExecFloat.Binary.isSubnormal (x / y) = false ∧
      (ExecFloat.Binary.isZero (x / y) = true → ExecFloat.Binary.isZero x = true) := by
  obtain ⟨flags, lost⟩ := UInt8.or_eq_zero_iff.mp hflags
  have range : ExecFloat.Binary.isFinite (x / y) = true ∧
      ExecFloat.Binary.isSubnormal (x / y) = false := by
    rw [← configured_divWithFlags_fst]
    exact configuredFlags_eq_zero flags
  have zeroFree : (ExecFloat.Binary.isZero (x / y) && !ExecFloat.Binary.isZero x).toUInt8 = 0 := by
    rw [← configured_divWithFlags_fst]
    exact lost
  refine ⟨range.1, range.2, fun zero => ?_⟩
  simpa [zero, Bool.toUInt8] using zeroFree

end Div

end Checked

/-- If the roundings of two rationals enclosing `target` both decode to the rational `shared`, the
rounding of the lower one is the rounding of `target`, for any storage plan and code. -/
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

/-- A successful `enclosedSlope?` returns the rounding of `idealSlope` at the rationals that the
input and the seed encode, with seed `1` when there is none. -/
theorem enclosedSlope_rounding (input : Value) (seed : Option Value) (isTanh : Bool)
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

/-- A successful `enclosedPowerSlope?` returns the rounding of the power slope
`e * (x ^ (e - 1) * g)` at the values of the configured base `x`, exponent `e` and seed `g`. -/
theorem enclosedPowerSlope_rounding (base exponent seed : Value) {result : Value}
    (hformat : format.isIEEE = true)
    (accepted : @enclosedPowerSlope? Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      base exponent seed = some result) :
    ExecFloat.Binary.toReal result = Model.roundAt format (ExecFloat.Binary.toReal exponent *
      (ExecFloat.Binary.toReal base ^ (ExecFloat.Binary.toReal exponent - 1) *
        ExecFloat.Binary.toReal seed)) := by
  let ops : QuotientArithmetic Value := QuotientArithmetic.configured
  have recovered := @enclosedPowerSlope?_eq_some Value ops base exponent seed result accepted
  obtain ⟨encodedBase, encodedExponent, encodedSeed, bounds, baseEncoded, exponentEncoded,
    seedEncoded, branch, lowerDecoded, upper, shared, upperDecoded, resultEncoded,
    upperEncoded⟩ := recovered
  obtain ⟨_, contained⟩ := branch.resolve_right fun ⟨varying, _⟩ =>
    varying (by cases encodedExponent with | scalar value => rfl)
  cases encodedBase with | scalar baseQ =>
    cases encodedExponent with | scalar exponentQ =>
      cases encodedSeed with | scalar seedQ =>
        cases bounds with | scalar interval =>
          cases shared with | scalar sharedQ =>
            let target : ℝ := (exponentQ : ℝ) * ((baseQ : ℝ) ^ ((exponentQ : ℝ) - 1) * (seedQ : ℝ))
            have targetContained : interval.ContainsReal some target := by
              have meaning := idealPower_scalar baseQ seedQ exponentQ
              change Related _ _ (idealPower _ _ exponentQ) at contained
              cases ideal : idealPower (.scalar baseQ) (.scalar seedQ) exponentQ with
              | scalar reference =>
                  rw [ideal] at contained meaning
                  simpa only [target, ← meaning, Related, QuotientCoefficients.primal] using
                    contained
            obtain ⟨lowerBound, upperBound⟩ :=
              (Interval.containsReal_some_iff interval target).mp targetContained
            change some (ExecFloat.Binary.ofModel (Model.ofRat format interval.lo) : Value) =
              some result at lowerDecoded
            change some (ExecFloat.Binary.ofModel (Model.ofRat format interval.hi) : Value) =
              some upper at upperDecoded
            have packedLower := Option.some.inj lowerDecoded
            have packedUpper := Option.some.inj upperDecoded
            have rounded := configured_endpoint_rounding hformat lowerBound upperBound
              ((congrArg ExecFloat.Binary.toRat? packedLower).trans
                (configured_encode result sharedQ resultEncoded))
              ((congrArg ExecFloat.Binary.toRat? packedUpper).trans
                (configured_encode upper sharedQ upperEncoded))
            have value (v : Value) (q : Rat) (encoded : @QuotientArithmetic.encode Value ops v =
                some (.scalar q)) : (q : ℝ) = ExecFloat.Binary.toReal v := by
              rw [← cast_toRat_eq_toReal, configured_encode v q encoded]
              rfl
            rw [← value base baseQ baseEncoded, ← value exponent exponentQ exponentEncoded,
              ← value seed seedQ seedEncoded]
            exact (congrArg ExecFloat.Binary.toReal packedLower).symm.trans rounded

/-- A successful `enclosedPowerTangent?` returns the rounding of the power tangent
`x ^ y * (dy * log x + y * (dx / x))` at the values of the configured base `x`, exponent `y` and
directions `dx` and `dy`. -/
theorem enclosedPowerTangent_rounding
    (base exponent baseDirection exponentDirection : Value) {result : Value}
    (hformat : format.isIEEE = true)
    (accepted : @enclosedPowerTangent? Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      base exponent baseDirection exponentDirection = some result) :
    ExecFloat.Binary.toReal result = Model.roundAt format
      (ExecFloat.Binary.toReal base ^ ExecFloat.Binary.toReal exponent *
        (ExecFloat.Binary.toReal exponentDirection * Real.log (ExecFloat.Binary.toReal base) +
          ExecFloat.Binary.toReal exponent *
            (ExecFloat.Binary.toReal baseDirection / ExecFloat.Binary.toReal base))) := by
  let ops : QuotientArithmetic Value := QuotientArithmetic.configured
  have recovered := @enclosedPowerTangent?_eq_some Value ops base exponent baseDirection
    exponentDirection result accepted
  obtain ⟨encodedBase, encodedExponent, encodedBaseDirection, encodedExponentDirection, bounds,
    baseEncoded, exponentEncoded, baseDirectionEncoded, exponentDirectionEncoded, positive,
    contained, lowerDecoded, upper, shared, upperDecoded, resultEncoded, upperEncoded⟩ := recovered
  cases encodedBase with | scalar baseQ =>
    cases encodedExponent with | scalar exponentQ =>
      cases encodedBaseDirection with | scalar baseDirectionQ =>
        cases encodedExponentDirection with | scalar exponentDirectionQ =>
          cases bounds with | scalar interval =>
            cases shared with | scalar sharedQ =>
              let target : ℝ := (baseQ : ℝ) ^ (exponentQ : ℝ) *
                ((exponentDirectionQ : ℝ) * Real.log baseQ +
                  (exponentQ : ℝ) * ((baseDirectionQ : ℝ) / baseQ))
              have targetContained : interval.ContainsReal some target := by
                have meaning := idealPowerTangent_scalar baseQ exponentQ baseDirectionQ
                  exponentDirectionQ positive
                cases ideal : idealPowerTangent (.scalar baseQ) (.scalar exponentQ)
                    (.scalar baseDirectionQ) (.scalar exponentDirectionQ) with
                | scalar reference =>
                    rw [ideal] at contained meaning
                    simpa only [target, ← meaning, Related, QuotientCoefficients.primal] using
                      contained
              obtain ⟨lowerBound, upperBound⟩ :=
                (Interval.containsReal_some_iff interval target).mp targetContained
              change some (ExecFloat.Binary.ofModel (Model.ofRat format interval.lo) : Value) =
                some result at lowerDecoded
              change some (ExecFloat.Binary.ofModel (Model.ofRat format interval.hi) : Value) =
                some upper at upperDecoded
              have packedLower := Option.some.inj lowerDecoded
              have packedUpper := Option.some.inj upperDecoded
              have rounded := configured_endpoint_rounding hformat lowerBound upperBound
                ((congrArg ExecFloat.Binary.toRat? packedLower).trans
                  (configured_encode result sharedQ resultEncoded))
                ((congrArg ExecFloat.Binary.toRat? packedUpper).trans
                  (configured_encode upper sharedQ upperEncoded))
              have value (v : Value) (q : Rat) (encoded : @QuotientArithmetic.encode Value ops v =
                  some (.scalar q)) : (q : ℝ) = ExecFloat.Binary.toReal v := by
                rw [← cast_toRat_eq_toReal, configured_encode v q encoded]
                rfl
              rw [← value base baseQ baseEncoded, ← value exponent exponentQ exponentEncoded,
                ← value baseDirection baseDirectionQ baseDirectionEncoded,
                ← value exponentDirection exponentDirectionQ exponentDirectionEncoded]
              exact (congrArg ExecFloat.Binary.toReal packedLower).symm.trans rounded

/-- A recovered power slope is within `Model.halfUlpAt` of the slope `e * x ^ (e - 1) * g` at the
values of the configured base `x`, exponent `e` and seed `g`. -/
theorem abs_enclosedPowerSlope_sub_le (base exponent seed : Value)
    (hformat : format.isIEEE = true)
    (found : (@enclosedPowerSlope? Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      base exponent seed).isSome) :
    |ExecFloat.Binary.toReal ((@enclosedPowerSlope? Value
        (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
        base exponent seed).get found) -
      ExecFloat.Binary.toReal exponent *
        ExecFloat.Binary.toReal base ^ (ExecFloat.Binary.toReal exponent - 1) *
        ExecFloat.Binary.toReal seed| ≤
      Model.halfUlpAt format (ExecFloat.Binary.toReal exponent *
        ExecFloat.Binary.toReal base ^ (ExecFloat.Binary.toReal exponent - 1) *
        ExecFloat.Binary.toReal seed) := by
  rw [enclosedPowerSlope_rounding base exponent seed hformat
    (Option.eq_some_of_isSome found), ← mul_assoc]
  exact Model.abs_roundAt_sub_le _ _

/-- The mixed form of `abs_enclosedPowerSlope_sub_le`: one rounding. -/
theorem mixedError_enclosedPowerSlope (base exponent seed : Value)
    (hformat : format.isIEEE = true)
    (found : (@enclosedPowerSlope? Value
      (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
      base exponent seed).isSome) :
    MixedError (ExecFloat.Binary.toReal ((@enclosedPowerSlope? Value
        (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
        base exponent seed).get found))
      (ExecFloat.Binary.toReal exponent *
        ExecFloat.Binary.toReal base ^ (ExecFloat.Binary.toReal exponent - 1) *
        ExecFloat.Binary.toReal seed)
      (Model.unitRoundoffAt format) (Model.underflowErrorAt format) := by
  refine ExecFloat.Binary.mixedError_toReal_of_eq_roundAt ?_
  rw [enclosedPowerSlope_rounding base exponent seed hformat
    (Option.eq_some_of_isSome found), ← mul_assoc]

/-- A successful `enclosedSlope?` is within `Model.halfUlpAt` of the exact slope at the real input
`x` and seed `g` (`1` without a seed): `g * e / (1 + e) ^ 2` with `e = exp (-|x|)`, or for tanh
`4 * g * e / (1 + e) ^ 2` with `e = exp (-2 * |x|)`. -/
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
          input seed isTanh).get found) - reference| ≤ Model.halfUlpAt format reference := by
  intro tail numerator reference
  have accepted := Option.eq_some_of_isSome found
  have rounded := enclosedSlope_rounding input seed isTanh hformat accepted
  rw [idealSlope_scalar] at rounded
  have observed : ExecFloat.Binary.toReal
      ((@enclosedSlope? Value
        (QuotientArithmetic.configured (format := format) (plan := plan) (code := code))
        input seed isTanh).get found) = Model.roundAt format reference := by
    cases seed <;>
      simpa only [tail, numerator, reference, ExecFloat.Binary.toRat?,
        Model.cast_toRat_eq_toReal, ExecFloat.Binary.toReal, Rat.cast_one] using rounded
  rw [observed]
  exact Model.abs_roundAt_sub_le _ _

end Configured

end TorchLean.Roundoff
