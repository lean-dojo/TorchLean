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

namespace Tests.Floats.StandaloneImport

open TorchLean.Floats
open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.IEEE32Exec
open TorchLean.Floats.Quantization

/-- Scalar affine quantization is available without TorchLean's tensor layer. -/
noncomputable def int8Quantizer : AffineQuantizer where
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
    toBits (ofBits bits) = bits :=
  toBits_ofBits bits

/-- Integer configuration boundaries reject zero and negative format precision. -/
theorem checkedFormatPrecision_rejects_nonpositive :
    NeuralFormatPrecision.ofInt? 0 = none ∧
      NeuralFormatPrecision.ofInt? (-24) = none := by
  norm_num [NeuralFormatPrecision.ofInt?]

/-- A positive checked precision supplies valid FLX, FLT, and FTZ exponent selectors. -/
theorem checkedFormatPrecision_provides_valid_exponents
    (precision : NeuralFormatPrecision) :
    NeuralValidExp precision.flxExp ∧
      NeuralValidExp (precision.fltExp (-149)) ∧
      NeuralValidExp (precision.ftzExp (-126)) := by
  exact ⟨inferInstance, inferInstance, inferInstance⟩

/-- Negative precision cannot describe an explicit generic float format. -/
theorem negativePrecision_is_not_a_format (x : ℝ) :
    ¬FLXFormat (β := binaryRadix) (-24) x ∧
      ¬FLTFormat (β := binaryRadix) (-149) (-24) x ∧
      ¬FTZFormat (β := binaryRadix) (-126) (-24) x := by
  exact
    ⟨not_FLXFormat_of_nonpos (-24) (by norm_num) x,
      not_FLTFormat_of_nonpos (-149) (-24) (by norm_num) x,
      not_FTZFormat_of_nonpos (-126) (-24) (by norm_num) x⟩

/-- Exercise bit-level IEEE edge cases through the standalone numerical import. -/
def run : IO Unit := do
  let one := ofBits 0x3F800000
  let two := one + one
  unless two.bits == 0x40000000 do
    throw <| IO.userError s!"standalone IEEE32 addition failed: bits={two.bits}"

  let signalingNaN := ofBits 0x7F800001
  unless (neg signalingNaN).bits == 0xFF800001 do
    throw <| IO.userError "IEEE32 negation changed a signaling-NaN payload"
  unless (abs (neg signalingNaN)).bits == signalingNaN.bits do
    throw <| IO.userError "IEEE32 absolute value changed a signaling-NaN payload"
  let subOutcome := subWithStatus one signalingNaN
  unless subOutcome.status.invalid do
    throw <| IO.userError "IEEE32 subtraction failed to signal invalid for a signaling NaN"
  unless subOutcome.value.bits == 0xFFC00001 do
    throw <| IO.userError "IEEE32 subtraction did not propagate the right signaling NaN"

  let largeNatural : Nat := 2 ^ 53 + 2 ^ 29 + 1
  unless (largeNatural : TorchLean.Floats.IEEE754.IEEE32Exec).bits == 0x5A000001 do
    throw <| IO.userError "IEEE32 natural conversion was double-rounded"
  unless (largeNatural : Float32).toBits == 0x5A000001 do
    throw <| IO.userError "native binary32 natural conversion was double-rounded"
  for exponent in [0, 1, 23, 24, 53, 63, 64, 100, 127, 128, 256] do
    let base : Nat := 2 ^ exponent
    let halfUlp : Nat := if exponent < 24 then 0 else 2 ^ (exponent - 24)
    for n in [base - 1, base, base + 1, base + halfUlp - 1,
        base + halfUlp, base + halfUlp + 1, base + 3 * halfUlp] do
      unless (n : Float32).toBits == (n : TorchLean.Floats.IEEE754.IEEE32Exec).bits do
        throw <| IO.userError s!"native/model natural conversion disagrees at {n}"

  unless (addDown one negOne).bits == negZero.bits do
    throw <| IO.userError "IEEE32 downward exact cancellation did not return -0"
  unless (subDown one one).bits == negZero.bits do
    throw <| IO.userError "IEEE32 downward exact subtraction did not return -0"
  unless (fmaDown one one negOne).bits == negZero.bits do
    throw <| IO.userError "IEEE32 downward exact FMA cancellation did not return -0"
  unless mkBits false 256 0 == 0 do
    throw <| IO.userError "IEEE32 field constructor leaked exponent bits into the sign field"

  let tiny := ofFloat 1e-8
  unless (tanh tiny).bits == tiny.bits do
    throw <| IO.userError "IEEE32 tanh lost a small nonzero input"
  unless (tanh (neg tiny)).bits == (neg tiny).bits do
    throw <| IO.userError "IEEE32 tanh lost a small negative input"
  unless (tanh posMinSubnormal).bits == posMinSubnormal.bits &&
      (tanh negMinSubnormal).bits == negMinSubnormal.bits do
    throw <| IO.userError "IEEE32 tanh did not preserve the minimum signed subnormals"

  let belowQuarter := ofBits 0x3E7FFFFF
  let quarter := ofBits 0x3E800000
  let aboveQuarter := ofBits 0x3E800001
  unless compare (tanh belowQuarter) (tanh quarter) != some .gt &&
      compare (tanh quarter) (tanh aboveQuarter) != some .gt do
    throw <| IO.userError "IEEE32 tanh is not monotone across its positive branch boundary"
  unless compare (tanh (neg aboveQuarter)) (tanh (neg quarter)) != some .gt &&
      compare (tanh (neg quarter)) (tanh (neg belowQuarter)) != some .gt do
    throw <| IO.userError "IEEE32 tanh is not monotone across its negative branch boundary"

  let quietNaN := canonicalNaN
  unless (posOne ^ quietNaN).bits == posOne.bits do
    throw <| IO.userError "IEEE32 pow did not retain one for a quiet-NaN exponent"
  unless (negOne ^ posInf).bits == posOne.bits do
    throw <| IO.userError "IEEE32 pow mishandled -1 raised to positive infinity"
  unless (ofFloat (-2.0) ^ posInf).bits == posInf.bits do
    throw <| IO.userError
      "IEEE32 pow mishandled a negative magnitude above one at positive infinity"
  unless (ofFloat (-0.5) ^ posInf).bits == posZero.bits do
    throw <| IO.userError
      "IEEE32 pow mishandled a negative magnitude below one at positive infinity"
  unless (negInf ^ ofFloat 0.5).bits == posInf.bits do
    throw <| IO.userError
      "IEEE32 pow mishandled negative infinity at a positive noninteger exponent"
  unless (negInf ^ ofFloat (-0.5)).bits == posZero.bits do
    throw <| IO.userError
      "IEEE32 pow mishandled negative infinity at a negative noninteger exponent"

  -- Exercise argument reduction from small values through both signs of the largest finite input.
  let trigTolerance : Float := 1e-5
  let trigSamples : Array UInt32 :=
    #[0x3F800000, 0x42C80000, 0x501502F9, 0x60AD78EC, 0x7F7FFFFF,
      0xBF800000, 0xC2C80000, 0xD01502F9, 0xE0AD78EC, 0xFF7FFFFF]
  for bits in trigSamples do
    let input := ofBits bits
    let resultSin := sin input
    let resultCos := cos input
    unless isFinite resultSin && isFinite resultCos do
      throw <| IO.userError s!"IEEE32 trigonometric reduction was non-finite for {bits}"
    unless compare (abs resultSin) posOne != some .gt &&
        compare (abs resultCos) posOne != some .gt do
      throw <| IO.userError s!"IEEE32 trigonometric result escaped [-1, 1] for {bits}"
    unless Float.abs (toFloat resultSin - Float.sin (toFloat input)) < trigTolerance &&
        Float.abs (toFloat resultCos - Float.cos (toFloat input)) < trigTolerance do
      throw <| IO.userError s!"IEEE32 trigonometric reduction disagreed with reference for {bits}"

end Tests.Floats.StandaloneImport
