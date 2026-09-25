/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Sterbenz
public import NN.Proofs.RuntimeApprox.Reductions.IEEE32
public import NN.Spec.Quantization
public import NN.Spec.Core.FloatInstances -- shake: keep

/-!
# Effective Rounding in Tensor Semantics

This example uses the same shape-indexed tensor operation twice.  The `FP32` tensor gives the
proof-oriented rounded-real semantics.  The `ExecFloat.Binary 8 23` tensor executes binary32
arithmetic from
bits.  On a finite result, the IEEE bridge and the effective rounding calculation identify the
same canonical mantissa and exponent.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32 toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open Spec TorchLean
open TorchLean.Floats
open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.IEEE32Exec
open FloatLib.Numerics.Quantization
open FloatLib.Numerics FloatLib.Floats.Formats.Flocq

open FloatLib.Floats.Formats.BinaryInterchange

namespace NN.Examples.DeepDives.Floats.EffectiveRounding

/--
Vector length; four entries is enough to show the pointwise behaviour without a wall of output.
-/
def width : Nat := 4
/-- The shape shared by every tensor in this example. -/
abbrev vectorShape : Spec.Shape := [width]

/-- Executable binary32 tensors used by the example. -/
def runtimeOnes : Tensor (Binary 8 23) vectorShape :=
  Tensor.full vectorShape (1 : Binary 8 23)

/-- The constant `2`, given by its bit pattern `0x40000000` so no decimal parsing is involved. -/
def runtimeTwos : Tensor (Binary 8 23) vectorShape :=
  Tensor.full vectorShape (ofBits32 0x40000000)

/-- This addition executes pointwise in the bit-level IEEE32 model. -/
def runtimeSum : Tensor (Binary 8 23) vectorShape :=
  Tensor.addSpec runtimeOnes runtimeTwos

/-- Proof-oriented tensors use the same tensor API with rounded-real FP32 scalars. -/
noncomputable def specOne : FP32 :=
  NF.ofReal (β := binaryRadix) (fexp := fexp32) (rnd := rnd32) 1

/-- The rounded-real counterpart of `2`. Exactly representable, so rounding is the identity here. -/
noncomputable def specTwo : FP32 :=
  NF.ofReal (β := binaryRadix) (fexp := fexp32) (rnd := rnd32) 2

/-- A vector of ones in the proof-oriented model. -/
noncomputable def specOnes : Tensor FP32 vectorShape :=
  Tensor.full vectorShape specOne

/-- A vector of twos in the proof-oriented model. -/
noncomputable def specTwos : Tensor FP32 vectorShape :=
  Tensor.full vectorShape specTwo

/--
Their sum, computed by the same `addSpec` the executable tensors use. The two models share the
tensor
API and differ only in the scalar type, which is what makes the comparison below meaningful.
-/
noncomputable def specSum : Tensor FP32 vectorShape :=
  Tensor.addSpec specOnes specTwos

/-- Every proof-oriented tensor entry exposes the effective nearest-even representation. -/
theorem specSum_entry_computed (i : Fin width) :
    FP32.toReal specSum[i] =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32 (specOne.val + specTwo.val))
        exponent := cexp binaryRadix fexp32 (specOne.val + specTwo.val) } := by
  have hitem : specSum[i] = specOne + specTwo := by
    change specSum.getScalar i = specOne + specTwo
    simp [specSum, specOnes, specTwos, Tensor.addSpec,
      Tensor.map2Spec, Tensor.getScalar_eq_apply]
  rw [hitem]
  exact FP32.add_toReal_eq_computed specOne specTwo

/--
Every executable tensor entry reaches the same effective representation once finiteness is checked.
The finiteness premise is discharged by computation for the concrete $1+2$ example.
-/
theorem runtimeSum_entry_computed (i : Fin width) :
    (toModel runtimeSum[i]).toReal =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32
            ((toModel (1 : Binary 8 23)).toReal + (toModel (ofBits32 0x40000000)).toReal))
        exponent := cexp binaryRadix fexp32
          ((toModel (1 : Binary 8 23)).toReal + (toModel (ofBits32 0x40000000)).toReal) } := by
  have hadd (a b : Binary 8 23) : a + b = ExecFloat.add a b := by
    rfl
  have hitem : @Eq (Binary 8 23) (runtimeSum.getScalar i)
      (ExecFloat.add (1 : Binary 8 23) (ofBits32 0x40000000)) := by
    simp only [runtimeSum, runtimeOnes, runtimeTwos, Tensor.addSpec,
      Tensor.getScalar_map2Spec, Tensor.getScalar_full]
    exact hadd _ _
  have hfinite :
      Binary.isFinite (ExecFloat.add (1 : Binary 8 23) (ofBits32 0x40000000)) = true := by
    change Model.isFinite (toModel (ExecFloat.add (1 : Binary 8 23) (ofBits32 0x40000000))) = true
    rw [IEEE32Exec.toModel_add, Model.Proof.add_eq_spec]
    decide
  have hlookup : @Eq (Binary 8 23) runtimeSum[i] (runtimeSum.getScalar i) := by
    exact (Tensor.getScalar_eq_apply runtimeSum i).symm
  rw [hlookup, hitem]
  rw [toReal_add_eq_fp32Round_of_isFinite hfinite, fp32Round_eq_computed]

/-! ## Exact subtraction, local spacing, and absorption -/

/-- Sterbenz's lemma certifies the concrete binary32 subtraction $2-1$ as exact. -/
theorem two_sub_one_exact :
    round32 ((2 : ℝ) - 1) = (2 : ℝ) - 1 := by
  have hOne : genericFormat binaryRadix fexp32 (1 : ℝ) := by
    simpa [bpow, binaryRadix, Radix.toReal] using
      (generic_format_bpow (β := binaryRadix) (fexp := fexp32) 0
        (by norm_num [fexp32, fltExp]))
  have hTwo : genericFormat binaryRadix fexp32 (2 : ℝ) := by
    simpa [bpow, binaryRadix, Radix.toReal] using
      (generic_format_bpow (β := binaryRadix) (fexp := fexp32) 1
        (by norm_num [fexp32, fltExp]))
  exact round32_sub_exact_of_sterbenz hTwo hOne (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

private theorem toReal_of_decoded {x : (Binary 8 23)} {d : FloatLib.Numerics.Dyadic}
    (h : (toModel x).toDyadic? = some d) : (toModel x).toReal = d.toReal := by
  change Model.toDyadic? (toModel x) = some d at h
  rw [Model.toReal_eq, h]

private theorem toReal_one : (toModel (1 : Binary 8 23)).toReal = (1 : ℝ) := by
  rw [toReal_of_decoded (show (toModel (1 : Binary 8 23)).toDyadic? = some ⟨false, 8388608, -23⟩ by
    decide)]
  norm_num

private theorem toReal_two : (toModel (ofBits32 0x40000000)).toReal = (2 : ℝ) := by
  rw [toReal_of_decoded
    (show (toModel (ofBits32 0x40000000)).toDyadic? = some ⟨false, 8388608, -22⟩ by decide)]
  norm_num

/-- The corresponding executable binary32 operation denotes the exact real subtraction. -/
theorem runtime_two_sub_one_exact :
    (toModel (ExecFloat.sub (ofBits32 0x40000000) (1 : Binary 8 23))).toReal =
      (toModel (ofBits32 0x40000000)).toReal - (toModel (1 : Binary 8 23)).toReal := by
  have hbits : ExecFloat.sub (ofBits32 0x40000000) (1 : Binary 8 23) = (1 : Binary 8 23) := by
    rw [FloatLib.Floats.ExecFloat.Proof.sub_eq_spec]
    decide
  rw [hbits, toReal_one, toReal_two]
  norm_num

/-- The decoded significand at `1.0` has scale `2^-23`, its binary32 spacing. -/
theorem posOne_ulpExp : ((toModel (1 : Binary 8 23)).toDyadic?).map (·.exponent) = some (-23) := by
  decide

/-- The computed exponent therefore denotes the mathematical ULP at `1.0`. -/
theorem posOne_ulp :
    bpow binaryRadix (-23) = ulp32 ((toModel (1 : Binary 8 23)).toReal) := by
  rw [toReal_one]
  simpa [bpow, binaryRadix, Radix.toReal, fexp32, fltExp] using
    (ulp_bpow (β := binaryRadix) (fexp := fexp32) 0).symm

/-- Infinity has no finite dyadic significand or scale. -/
theorem posInf_ulpExp : ((toModel (Binary.infinity false : Binary 8 23)).toDyadic?).map (·.exponent)
  = none := by
  decide

/-- Adding the smallest positive subnormal does not change executable binary32 `1.0`. -/
theorem posOne_absorbs_posMinSubnormal : ExecFloat.add (1 : Binary 8 23) (ofBits32 1) = (1 : Binary
  8 23) := by
  decide

/--
The executable absorption result transports to the rounded-real binary32 specification.
-/
theorem posOne_add_posMinSubnormal_rounds_to_posOne :
    round32 ((toModel (1 : Binary 8 23)).toReal + (toModel (ofBits32 1)).toReal) = (toModel (1 :
      Binary 8 23)).toReal := by
  have h := toReal_add_eq_fp32Round_of_isFinite
    (x := (1 : Binary 8 23)) (y := ofBits32 1) (by decide)
  rw [posOne_absorbs_posMinSubnormal] at h
  exact h.symm

/-! ## Named rounding modes and fused enclosures -/

/-- The public mode API avoids passing a raw integer-rounding function at each call site. -/
noncomputable def oneThirdDown : ℝ :=
  RoundingMode.towardNegative.round
    (β := binaryRadix) (fexp := fexp32) (1 / 3)

/-- Directed rounding gives a certified lower endpoint, not merely a differently named value. -/
theorem oneThirdDown_le : oneThirdDown ≤ 1 / 3 := by
  simpa [oneThirdDown, RoundingMode.round, RoundingMode.roundingFunction] using
    (round_floor_le (β := binaryRadix) (fexp := fexp32) (1 / 3))

/-- A concrete fused multiply-add lower endpoint, computed directly from binary32 inputs. -/
def fusedLower : Binary 8 23 :=
  Binary.fmaWithRounding (1 : Binary 8 23) (ofBits32 0x40000000) (ofBits32 0x3e800000)
    .towardNegativeInfinity

/-- The corresponding upper endpoint. -/
def fusedUpper : Binary 8 23 :=
  Binary.fmaWithRounding (1 : Binary 8 23) (ofBits32 0x40000000) (ofBits32 0x3e800000)
    .towardPositiveInfinity

/-- The executable directed FMA endpoints enclose the exact single-rounding expression. -/
theorem fused_enclosure :
    (toModel fusedLower).toEReal ≤
        (((toModel (1 : Binary 8 23)).toReal * (toModel (ofBits32 0x40000000)).toReal + (toModel
          (ofBits32 0x3e800000)).toReal : ℝ) : EReal) ∧
      (((toModel (1 : Binary 8 23)).toReal * (toModel (ofBits32 0x40000000)).toReal + (toModel
        (ofBits32 0x3e800000)).toReal : ℝ) : EReal) ≤
        (toModel fusedUpper).toEReal := by
  have hlo : fusedLower = ofBits32 0x40100000 := by decide
  have hhi : fusedUpper = ofBits32 0x40100000 := by decide
  have hquarter : (toModel (ofBits32 0x3e800000)).toReal = (1 / 4 : ℝ) := by
    rw [toReal_of_decoded
      (show (toModel (ofBits32 0x3e800000)).toDyadic? = some ⟨false, 8388608, -25⟩ by decide)]
    norm_num
  have hresult : (toModel (ofBits32 0x40100000)).toReal = (9 / 4 : ℝ) := by
    rw [toReal_of_decoded
      (show (toModel (ofBits32 0x40100000)).toDyadic? = some ⟨false, 9437184, -22⟩ by decide)]
    norm_num
  have hext : (toModel (ofBits32 0x40100000)).toEReal = ((9 / 4 : ℝ) : EReal) := by
    rw [Model.toEReal_eq_coe_toReal_of_isFinite _ (by decide)]
    exact congrArg (fun r : ℝ => (r : EReal)) hresult
  rw [hlo, hhi, hext, toReal_one, toReal_two, hquarter]
  norm_num

/-- Directed binary32 square-root endpoints for the exact input `2`. -/
def sqrtLower : Binary 8 23 :=
  (Binary.sqrtWithRounding (rounding := .towardNegativeInfinity)) (ofBits32 0x40000000)

/-- The upper endpoint, rounded away from zero, so the pair brackets the exact `sqrt 2`. -/
def sqrtUpper : Binary 8 23 :=
  (Binary.sqrtWithRounding (rounding := .towardPositiveInfinity)) (ofBits32 0x40000000)

/-- The executable endpoints enclose the exact real value `sqrt 2`. -/
theorem sqrt_enclosure :
    (toModel sqrtLower).toEReal ≤ (Real.sqrt ((toModel (ofBits32 0x40000000)).toReal) : EReal) ∧
      (Real.sqrt ((toModel (ofBits32 0x40000000)).toReal) : EReal) ≤
        (toModel sqrtUpper).toEReal := by
  have hlo : toModel sqrtLower = Model.sqrtDown (toModel (ofBits32 0x40000000)) :=
    Binary.toModel_sqrtWithRounding (ofBits32 0x40000000) .towardNegativeInfinity
  have hhi : toModel sqrtUpper = Model.sqrtUp (toModel (ofBits32 0x40000000)) :=
    Binary.toModel_sqrtWithRounding (ofBits32 0x40000000) .towardPositiveInfinity
  constructor
  · rw [hlo]
    exact Model.toEReal_sqrtDown_le (toModel (ofBits32 0x40000000))
      (by decide) (by decide) (by decide)
  · rw [hhi]
    exact Model.le_toEReal_sqrtUp (toModel (ofBits32 0x40000000))
      (by decide) (by decide) (by decide)

/-! ## Fixed grids, quantization, and double rounding -/

/-- A signed affine code set with quarter-unit spacing. The construction is not tied to a tensor
layout or storage width; those choices only determine the integer code bounds. -/
noncomputable def signedQuarterQuantizer : RealAffineQuantizer where
  scale := 1 / 4
  zeroPoint := 0
  qmin := -128
  qmax := 127
  scale_pos := by norm_num
  codeRange := by norm_num

/-- Every in-range code is recovered exactly after dequantization and requantization. -/
theorem signedQuarterQuantizer_roundtrip {code : ℤ}
    (hlo : -128 ≤ code) (hhi : code ≤ 127) :
    signedQuarterQuantizer.quantize nearestEven
        (signedQuarterQuantizer.dequantize code) = code := by
  exact affine_quantize_dequantize signedQuarterQuantizer nearestEven hlo hhi

/-- When saturation is inactive, nearest-even reconstruction is within half a quantization step. -/
theorem signedQuarterQuantizer_error (x : ℝ)
    (hlo : signedQuarterQuantizer.qmin ≤
      signedQuarterQuantizer.rawCode nearestEven x)
    (hhi : signedQuarterQuantizer.rawCode nearestEven x ≤
      signedQuarterQuantizer.qmax) :
    abs (signedQuarterQuantizer.dequantize
      (signedQuarterQuantizer.quantize nearestEven x) - x) ≤ 1 / 8 := by
  have h :=
    affine_dequantize_quantize_error_le_half signedQuarterQuantizer nearestEven x hlo hhi
  simpa only [RealAffineQuantizer.roundedValue, signedQuarterQuantizer,
    show (1 / 4 : ℝ) / 2 = 1 / 8 by norm_num] using h

/-- Four valid codes, represented with the same shape-indexed tensor used by TorchLean models. -/
def quarterCodes : Tensor ℤ vectorShape :=
  Tensor.ofFn (fun i => i.val)

/-- Pointwise tensor Q/DQ is exact on an in-range code tensor. -/
theorem quarterCodes_roundtrip :
    signedQuarterQuantizer.quantizeTensor nearestEven
      (signedQuarterQuantizer.dequantizeTensor quarterCodes) = quarterCodes := by
  apply signedQuarterQuantizer.quantizeTensor_dequantizeTensor
  intro i
  change signedQuarterQuantizer.qmin ≤ quarterCodes.getScalar i ∧
    quarterCodes.getScalar i ≤ signedQuarterQuantizer.qmax
  simp only [quarterCodes, Tensor.getScalar_ofFn]
  have hi : i.val < 4 := i.isLt
  norm_num [signedQuarterQuantizer]
  grind

/-- Round-to-odd on a sufficiently fine binary grid prevents double rounding on the quarter grid. -/
theorem quarterGrid_doubleRounding_safe (extra : ℕ) (x : ℝ) :
    roundAtScale nearestEven (1 / 4) (by norm_num)
        (roundAtScale oddRound
          ((1 / 4) / (2 : ℝ) ^ (extra + 2)) (by positivity) x) =
      roundAtScale nearestEven (1 / 4) (by norm_num) x := by
  exact roundAtScale_nearestEven_after_odd_binary_extra extra (1 / 4) x (by norm_num)

/-! ## IEEE exception status -/

/-- The status-bearing API records division by zero separately from invalid operation. -/
theorem one_div_zero_status :
    (Binary.divWithStatus (1 : Binary 8 23) (Binary.zero false : Binary 8 23)
      .nearestEven).2.divideByZero = true ∧
      (Binary.divWithStatus (1 : Binary 8 23) (Binary.zero false : Binary 8 23)
        .nearestEven).2.invalid = false := by
  decide

/-! ## A training-shaped reduction -/

/-- A fixed two-term dot-product tree. Its shape records the accumulation order. -/
def runtimeDotTree : SumTree ((Binary 8 23) × (Binary 8 23)) :=
  .node
    (.leaf ((1 : Binary 8 23), ofBits32 0x40000000))
    (.leaf (ofBits32 0x40400000, ofBits32 0x40800000))

/-- Every intermediate product and addition in the example remains finite. -/
theorem runtimeDotTree_finite : FiniteEvalDot runtimeDotTree := by
  have hleft : ExecFloat.mul (1 : Binary 8 23) (ofBits32 0x40000000) = ofBits32 0x40000000 := by
    rw [FloatLib.Floats.ExecFloat.Proof.mul_eq_spec]
    decide
  have hright :
      ExecFloat.mul (ofBits32 0x40400000) (ofBits32 0x40800000) = ofBits32 0x41400000 := by
    rw [FloatLib.Floats.ExecFloat.Proof.mul_eq_spec]
    decide
  simp only [runtimeDotTree, FiniteEvalDot, evalDotIEEE, hleft, hright]
  refine ⟨by decide, by decide, ?_⟩
  change Model.isFinite (toModel (ExecFloat.add (ofBits32 0x40000000) (ofBits32 0x41400000))) = true
  rw [IEEE32Exec.toModel_add, Model.Proof.add_eq_spec]
  decide

/--
The executable dot product decodes to the computed mantissa/exponent representation of its final
rounded accumulation. The two leaf multiplications are rounded before this final addition.
-/
theorem runtimeDotTree_computed :
    (toModel (evalDotIEEE runtimeDotTree)).toReal =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix fexp32
            (evalRealDotIEEE (.leaf ((1 : Binary 8 23), ofBits32 0x40000000)) +
              evalRealDotIEEE (.leaf (ofBits32 0x40400000, ofBits32 0x40800000))))
        exponent := cexp binaryRadix fexp32
          (evalRealDotIEEE (.leaf ((1 : Binary 8 23), ofBits32 0x40000000)) +
            evalRealDotIEEE (.leaf (ofBits32 0x40400000, ofBits32 0x40800000))) } := by
  rw [toReal_evalDotIEEE_eq_evalRealDotIEEE_of_FiniteEvalDot runtimeDotTree runtimeDotTree_finite]
  exact evalRealDotIEEE_node_eq_computed _ _

end NN.Examples.DeepDives.Floats.EffectiveRounding
