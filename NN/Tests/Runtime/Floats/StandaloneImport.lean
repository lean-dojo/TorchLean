/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats

/-!
# Standalone Floating-Point Import

This regression module deliberately imports only `NN.Floats`. It exercises the public numerical
surface without tensors, models, autograd, CUDA, certificate checkers, or external processes. The
repository linter separately enforces that the import closure cannot acquire those dependencies.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32 toBits32 ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace Tests.Floats.StandaloneImport

open TorchLean.Floats
open TorchLean.Floats.IEEE754
open FloatLib.Numerics.Quantization
open FloatLib.Numerics
open FloatLib.Floats.Formats.Flocq
open FloatLib.Floats.ExecFloat.Binary (tanh sin cos)
open FloatLib.Floats.Formats.BinaryInterchange

/-- Scalar affine quantization is available without TorchLean's tensor layer. -/
noncomputable def int8Quantizer : RealAffineQuantizer where
  scale := 1 / 10
  zeroPoint := 0
  qmin := -128
  qmax := 127
  scale_pos := by norm_num
  codeRange := by norm_num

/-- The standalone scalar quantizer retains its code-range theorem. -/
theorem int8Quantizer_codeRange (rnd : ℝ → ℤ) (x : ℝ) :
    int8Quantizer.qmin ≤ int8Quantizer.quantize rnd x ∧
      int8Quantizer.quantize rnd x ≤ int8Quantizer.qmax :=
  int8Quantizer.quantize_mem rnd x

/-- The standalone executable kernel retains exact bit-pattern round trips. -/
theorem executable_bits_roundTrip (bits : UInt32) :
    toBits32 (ofBits32 bits) = bits :=
  Binary.toBits32_ofBits32 bits

/-- Integer configuration boundaries reject zero and negative format precision. -/
theorem checkedFormatPrecision_rejects_nonpositive :
    FormatPrecision.ofInt? 0 = none ∧
      FormatPrecision.ofInt? (-24) = none := by
  norm_num [FormatPrecision.ofInt?]

/-- A positive checked precision supplies valid FLX, FLT, and FTZ exponent selectors. -/
theorem checkedFormatPrecision_provides_valid_exponents
    (precision : FormatPrecision) :
    ValidExp precision.flxExp ∧
      ValidExp (precision.fltExp (-149)) ∧
      ValidExp (precision.ftzExp (-126)) := by
  exact ⟨inferInstance, inferInstance, inferInstance⟩

/-- Negative precision cannot describe an explicit generic float format. -/
theorem negativePrecision_is_not_a_format (x : ℝ) :
    ¬FLXFormat (β := binaryRadix) (-24) x ∧
      ¬FLTFormat (β := binaryRadix) (-149) (-24) x ∧
      ¬FTZFormat (β := binaryRadix) (-126) (-24) x := by
  exact
    ⟨not_flxFormat_of_nonpos (-24) (by norm_num) x,
      not_fltFormat_of_nonpos (-149) (-24) (by norm_num) x,
      not_ftzFormat_of_nonpos (-126) (-24) (by norm_num) x⟩

/-- Exercise bit-level IEEE edge cases through the standalone numerical import. -/
def run : IO Unit := do
  let one := ofBits32 0x3F800000
  let two := one + one
  unless toBits32 two == 0x40000000 do
    throw <| IO.userError s!"standalone IEEE32 addition failed: bits={toBits32 two}"

  let signalingNaN := ofBits32 0x7F800001
  unless toBits32 (Neg.neg signalingNaN) == 0xFF800001 do
    throw <| IO.userError "IEEE32 negation changed a signaling-NaN payload"
  unless toBits32 (Binary.abs (Neg.neg signalingNaN)) == toBits32 signalingNaN do
    throw <| IO.userError "IEEE32 absolute value changed a signaling-NaN payload"
  let (subValue, subStatus) := Binary.subWithStatus one signalingNaN .nearestEven
  unless subStatus.invalid do
    throw <| IO.userError "IEEE32 subtraction failed to signal invalid for a signaling NaN"
  unless toBits32 subValue == 0xFFC00001 do
    throw <| IO.userError "IEEE32 subtraction did not propagate the right signaling NaN"

  let largeNatural : Nat := 2 ^ 53 + 2 ^ 29 + 1
  let castNatural := ofModel <| Model.roundRatQ .binary32 (largeNatural : Rat)
  unless toBits32 castNatural == 0x5A000001 do
    throw <| IO.userError "IEEE32 natural conversion was double-rounded"
  unless (largeNatural : Float32).toBits == 0x5A000001 do
    throw <| IO.userError "native binary32 natural conversion was double-rounded"
  for exponent in [0, 1, 23, 24, 53, 63, 64, 100, 127, 128, 256] do
    let base : Nat := 2 ^ exponent
    let halfUlp : Nat := if exponent < 24 then 0 else 2 ^ (exponent - 24)
    for n in [base - 1, base, base + 1, base + halfUlp - 1,
        base + halfUlp, base + halfUlp + 1, base + 3 * halfUlp] do
      let castN := ofModel <| Model.roundRatQ .binary32 (n : Rat)
      unless (n : Float32).toBits == toBits32 castN do
        throw <| IO.userError s!"native/model natural conversion disagrees at {n}"

  let negativeOne : Binary 8 23 := -1
  let negativeZero : Binary 8 23 := Binary.zero true
  unless toBits32 (Binary.add one negativeOne .towardNegativeInfinity) ==
      toBits32 negativeZero do
    throw <| IO.userError "IEEE32 downward exact cancellation did not return -0"
  unless toBits32 (Binary.sub one one .towardNegativeInfinity) == toBits32 negativeZero do
    throw <| IO.userError "IEEE32 downward exact subtraction did not return -0"
  unless toBits32 (Binary.fma one one negativeOne .towardNegativeInfinity) ==
      toBits32 negativeZero do
    throw <| IO.userError "IEEE32 downward exact FMA cancellation did not return -0"
  unless (Model.mkBits .binary32 false 256 0).toNat == 0 do
    throw <| IO.userError "IEEE32 field constructor leaked exponent bits into the sign field"

  let tiny : Binary 8 23 :=
    ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat 1e-8)))
  unless toBits32 (tanh tiny) == toBits32 tiny do
    throw <| IO.userError "IEEE32 tanh lost a small nonzero input"
  unless toBits32 (tanh (Neg.neg tiny)) == toBits32 (Neg.neg tiny) do
    throw <| IO.userError "IEEE32 tanh lost a small negative input"
  let posMinSubnormal := ofBits32 1
  let negMinSubnormal := ofBits32 0x80000001
  unless toBits32 (tanh posMinSubnormal) == toBits32 posMinSubnormal &&
      toBits32 (tanh negMinSubnormal) == toBits32 negMinSubnormal do
    throw <| IO.userError "IEEE32 tanh did not preserve the minimum signed subnormals"

  let belowQuarter := ofBits32 0x3E7FFFFF
  let quarter := ofBits32 0x3E800000
  let aboveQuarter := ofBits32 0x3E800001
  unless ExecFloat.compare (tanh belowQuarter) (tanh quarter) != some .gt &&
      ExecFloat.compare (tanh quarter) (tanh aboveQuarter) != some .gt do
    throw <| IO.userError "IEEE32 tanh is not monotone across its positive branch boundary"
  unless ExecFloat.compare (tanh (Neg.neg aboveQuarter)) (tanh (Neg.neg quarter)) != some .gt &&
      ExecFloat.compare (tanh (Neg.neg quarter)) (tanh (Neg.neg belowQuarter)) != some .gt do
    throw <| IO.userError "IEEE32 tanh is not monotone across its negative branch boundary"

  let quietNaN := (Binary.canonicalNaN : Binary 8 23)
  let positiveInfinity : Binary 8 23 := Binary.infinity false
  let negativeInfinity : Binary 8 23 := Binary.infinity true
  let positiveZero : Binary 8 23 := Binary.zero false
  let negativeTwo : Binary 8 23 :=
    ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat (-2.0))))
  let negativeHalf : Binary 8 23 :=
    ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat (-0.5))))
  let positiveHalf : Binary 8 23 :=
    ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat 0.5)))
  -- The standalone scalar import exposes FloatLib's model power directly; a TorchLean Context
  -- supplies the configured `^` dictionary when using the tensor/model API.
  let pow (x y : Binary 8 23) := ofModel (Model.pow (toModel x) (toModel y))
  unless toBits32 (pow one quietNaN) == toBits32 one do
    throw <| IO.userError "IEEE32 pow did not retain one for a quiet-NaN exponent"
  unless toBits32 (pow negativeOne positiveInfinity) == toBits32 one do
    throw <| IO.userError "IEEE32 pow mishandled -1 raised to positive infinity"
  unless toBits32 (pow negativeTwo positiveInfinity) == toBits32 positiveInfinity do
    throw <| IO.userError
      "IEEE32 pow mishandled a negative magnitude above one at positive infinity"
  unless toBits32 (pow negativeHalf positiveInfinity) == toBits32 positiveZero do
    throw <| IO.userError
      "IEEE32 pow mishandled a negative magnitude below one at positive infinity"
  unless toBits32 (pow negativeInfinity positiveHalf) == toBits32 positiveInfinity do
    throw <| IO.userError
      "IEEE32 pow mishandled negative infinity at a positive noninteger exponent"
  unless toBits32 (pow negativeInfinity negativeHalf) == toBits32 positiveZero do
    throw <| IO.userError
      "IEEE32 pow mishandled negative infinity at a negative noninteger exponent"

  -- Exercise argument reduction from small values through both signs of the largest finite input.
  let trigTolerance : Float := 1e-5
  let trigSamples : Array UInt32 :=
    #[0x3F800000, 0x42C80000, 0x501502F9, 0x60AD78EC, 0x7F7FFFFF,
      0xBF800000, 0xC2C80000, 0xD01502F9, 0xE0AD78EC, 0xFF7FFFFF]
  for bits in trigSamples do
    let input := ofBits32 bits
    let resultSin := sin input
    let resultCos := cos input
    unless Binary.isFinite resultSin && Binary.isFinite resultCos do
      throw <| IO.userError s!"IEEE32 trigonometric reduction was non-finite for {bits}"
    unless ExecFloat.compare (Binary.abs resultSin) (1 : Binary 8 23) != some .gt &&
        ExecFloat.compare (Binary.abs resultCos) (1 : Binary 8 23) != some .gt do
      throw <| IO.userError s!"IEEE32 trigonometric result escaped [-1, 1] for {bits}"
    let hostInput := Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel input)))
    let hostSin := Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel resultSin)))
    let hostCos := Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel resultCos)))
    unless Float.abs (hostSin - Float.sin hostInput) < trigTolerance &&
        Float.abs (hostCos - Float.cos hostInput) < trigTolerance do
      throw <| IO.userError s!"IEEE32 trigonometric reduction disagreed with reference for {bits}"

end Tests.Floats.StandaloneImport
