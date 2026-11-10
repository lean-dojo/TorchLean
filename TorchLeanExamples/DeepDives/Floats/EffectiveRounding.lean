/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Floats.FP32
public import FloatLib.Floats.Formats.BinaryInterchange.Analysis.Sterbenz
public import TorchLean.Verify.Floats.IEEE754.Reductions
public import TorchLean.Tensor.Arithmetic
public import TorchLean.Tensor.ConfiguredBinary -- shake: keep

/-!
# Effective Rounding in Tensor Semantics

The same shape-indexed tensor addition runs at two scalar types. With `FP32` it is the rounded-real
arithmetic the proofs reason about, and with `Binary32` it executes binary32 arithmetic
on bit patterns. When the result is finite, the IEEE bridge and the effective rounding calculation
give the same canonical mantissa and exponent. Smaller examples cover exact subtraction, ULP
spacing, absorption, directed rounding, quantization grids, exception flags and a dot product.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat Binary32)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32 toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

open TorchLean.IEEE754
open FloatLib.Numerics.Quantization
open FloatLib.Numerics FloatLib.Floats.Formats.Flocq

open FloatLib.Floats.Formats.BinaryInterchange

namespace TorchLean.Examples.EffectiveRounding

/--
The vector length. Four entries show the pointwise behaviour without a wall of output.
-/
def width : Nat := 4
/-- The shape shared by every tensor in this example. -/
abbrev vectorShape : Shape := [width]

/-- A vector of executable binary32 ones. -/
def binaryOnes : Tensor Binary32 vectorShape :=
  Tensor.full vectorShape (1 : Binary32)

/-- A vector of twos from the bit pattern `0x40000000`, so no decimal parsing is involved. -/
def binaryTwos : Tensor Binary32 vectorShape :=
  Tensor.full vectorShape (ofBits32 0x40000000)

/-- `binaryOnes + binaryTwos`, executed pointwise in the bit-level binary32 model. -/
def binarySum : Tensor Binary32 vectorShape :=
  binaryOnes + binaryTwos

/-- The real `1` rounded to nearest even in the rounded-real `FP32` model. -/
noncomputable def fp32One : FP32 :=
  NF.ofReal (β := binaryRadix) (fexp := (Model.fexpOf FloatFormat.binary32)) (rnd := nearestEven) 1

/-- The real `2` rounded to nearest even in `FP32`. Two is representable in binary32, so the
rounding leaves it unchanged. -/
noncomputable def fp32Two : FP32 :=
  NF.ofReal (β := binaryRadix) (fexp := (Model.fexpOf FloatFormat.binary32)) (rnd := nearestEven) 2

/-- A vector of ones in the proof-oriented model. -/
noncomputable def fp32Ones : Tensor FP32 vectorShape :=
  Tensor.full vectorShape fp32One

/-- A vector of twos in the proof-oriented model. -/
noncomputable def fp32Twos : Tensor FP32 vectorShape :=
  Tensor.full vectorShape fp32Two

/--
The sum `fp32Ones + fp32Twos`, computed by the same tensor `+` as `binarySum`. The two models
share the tensor API and differ only in the scalar type, which is what makes the comparison of
their entries meaningful.
-/
noncomputable def fp32Sum : Tensor FP32 vectorShape :=
  fp32Ones + fp32Twos

/-- Every entry of `fp32Sum` denotes the effective nearest-even rounding of
`fp32One.val + fp32Two.val`: its scaled mantissa rounded to nearest even, at the canonical
exponent `cexp`. -/
theorem fp32Sum_get_computed (i : Fin width) :
    fp32Sum[i].val =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix (Model.fexpOf FloatFormat.binary32)
            (fp32One.val + fp32Two.val))
        exponent := cexp binaryRadix (Model.fexpOf FloatFormat.binary32)
          (fp32One.val + fp32Two.val) } := by
  have hitem : fp32Sum[i] = fp32One + fp32Two := by
    change fp32Sum.getScalar i = fp32One + fp32Two
    simp [fp32Sum, fp32Ones, fp32Twos, Tensor.add_def,
      Tensor.getScalar_eq_apply]
  rw [hitem]
  exact round_nearestEven_computed (fp32One.val + fp32Two.val)

/--
Every entry of `binarySum` denotes the same effective rounding, computed from the decoded
inputs. The bridge lemma needs the binary32 sum to be finite, and `decide` checks that for the
concrete $1+2$.
-/
theorem binarySum_get_computed (i : Fin width) :
    (toModel binarySum[i]).toReal =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix (Model.fexpOf FloatFormat.binary32)
            ((toModel (1 : Binary32)).toReal + (toModel (ofBits32 0x40000000)).toReal))
        exponent := cexp binaryRadix (Model.fexpOf FloatFormat.binary32)
          ((toModel (1 : Binary32)).toReal + (toModel (ofBits32 0x40000000)).toReal) } := by
  have hitem : @Eq Binary32 (binarySum.getScalar i)
      (ExecFloat.add (1 : Binary32) (ofBits32 0x40000000)) := by
    simp only [binarySum, binaryOnes, binaryTwos, Tensor.add_def,
      Tensor.getScalar_zipWith, Tensor.getScalar_full]
    rfl
  have hfinite :
      Binary.isFinite (ExecFloat.add (1 : Binary32) (ofBits32 0x40000000)) = true := by
    change Model.isFinite (toModel (_ + ofBits32 0x40000000)) = true
    rw [Binary.toModel_hAdd _ (ofBits32 0x40000000), Model.Proof.add_eq_spec]
    decide
  have hlookup : @Eq Binary32 binarySum[i] (binarySum.getScalar i) := by
    exact (Tensor.getScalar_eq_apply binarySum i).symm
  rw [hlookup, hitem]
  change Binary.toReal (_ + ofBits32 0x40000000) = _
  rw [Binary.toReal_add_eq_roundAt hfinite]
  exact round_nearestEven_computed _

/-! ## Exact subtraction, local spacing, and absorption -/

/-- Sterbenz's lemma certifies the concrete binary32 subtraction $2-1$ as exact. -/
theorem two_sub_one_exact :
    Model.roundAt FloatFormat.binary32 ((2 : ℝ) - 1) = (2 : ℝ) - 1 := by
  have hOne : genericFormat binaryRadix (Model.fexpOf FloatFormat.binary32) (1 : ℝ) := by
    simpa [bpow, binaryRadix, Radix.toReal] using
      (generic_format_bpow (β := binaryRadix) (fexp := (Model.fexpOf FloatFormat.binary32)) 0
        (by norm_num [Model.fexpOf, FloatFormat.binary32, fltExp,
          FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent]))
  have hTwo : genericFormat binaryRadix (Model.fexpOf FloatFormat.binary32) (2 : ℝ) := by
    simpa [bpow, binaryRadix, Radix.toReal] using
      (generic_format_bpow (β := binaryRadix) (fexp := (Model.fexpOf FloatFormat.binary32)) 1
        (by norm_num [Model.fexpOf, FloatFormat.binary32, fltExp,
          FloatFormat.minSubnormalExponent, FloatFormat.minNormalExponent]))
  exact Model.roundAt_sub_eq_of_sterbenz FloatFormat.binary32 hTwo hOne (by norm_num) (by norm_num)

private theorem toReal_of_decoded {x : Binary32} {d : FloatLib.Numerics.Dyadic}
    (h : (toModel x).toDyadic? = some d) : (toModel x).toReal = d.toReal := by
  change Model.toDyadic? (toModel x) = some d at h
  rw [Model.toReal_eq, h]

private theorem toReal_one : (toModel (1 : Binary32)).toReal = (1 : ℝ) := by
  rw [toReal_of_decoded (show (toModel (1 : Binary32)).toDyadic? = some ⟨false, 8388608, -23⟩ by
    decide)]
  norm_num

private theorem toReal_two : (toModel (ofBits32 0x40000000)).toReal = (2 : ℝ) := by
  rw [toReal_of_decoded
    (show (toModel (ofBits32 0x40000000)).toDyadic? = some ⟨false, 8388608, -22⟩ by decide)]
  norm_num

/-- The executable binary32 subtraction `2 - 1` decodes to the exact real difference. -/
theorem toReal_two_sub_one :
    (toModel (ExecFloat.sub (ofBits32 0x40000000) (1 : Binary32))).toReal =
      (toModel (ofBits32 0x40000000)).toReal - (toModel (1 : Binary32)).toReal := by
  have hbits : ExecFloat.sub (ofBits32 0x40000000) (1 : Binary32) = (1 : Binary32) := by
    rw [FloatLib.Floats.ExecFloat.Proof.sub_eq_spec]
    decide +kernel
  rw [hbits, toReal_one, toReal_two]
  norm_num

/-- The decoded significand at `1.0` has scale `2^-23`, its binary32 spacing. -/
theorem one_dyadic_exponent :
    ((toModel (1 : Binary32)).toDyadic?).map (·.exponent) = some (-23) := by
  decide

/-- The scale `2^-23` of the decoded `1.0` equals the mathematical ULP `Model.ulpAt` at `1.0`. -/
theorem one_ulp :
    bpow binaryRadix (-23) =
      Model.ulpAt FloatFormat.binary32 ((toModel (1 : Binary32)).toReal) := by
  rw [toReal_one]
  simpa [bpow, binaryRadix, Radix.toReal, Model.ulpAt, Model.fexpOf,
    FloatFormat.binary32, fltExp, FloatFormat.minSubnormalExponent,
    FloatFormat.minNormalExponent] using
    (ulp_bpow (β := binaryRadix) (fexp := (Model.fexpOf FloatFormat.binary32)) 0).symm

/-- Infinity has no finite dyadic significand or scale. -/
theorem infinity_dyadic_none :
    ((toModel (Binary.infinity false : Binary32)).toDyadic?).map (·.exponent) = none := by
  decide

/-- Adding the smallest positive subnormal does not change executable binary32 `1.0`. -/
theorem one_add_min_subnormal : ExecFloat.add (1 : Binary32) (ofBits32 1) = (1 : Binary32) := by
  decide

/--
In the rounded-real binary32 model too, $1 + 2^{-149}$ rounds to `1`. The IEEE bridge carries
the executable absorption result over.
-/
theorem roundAt_absorption :
    Model.roundAt FloatFormat.binary32
      ((toModel (1 : Binary32)).toReal + (toModel (ofBits32 1)).toReal) =
        (toModel (1 : Binary32)).toReal := by
  have hfinite : Binary.isFinite (ExecFloat.add (1 : Binary32) (ofBits32 1)) = true := by
    decide
  have h : (toModel (ExecFloat.add _ (ofBits32 1))).toReal = _ :=
    Binary.toReal_add_eq_roundAt hfinite
  rw [one_add_min_subnormal] at h
  exact h.symm

/-! ## Named rounding modes and fused enclosures -/

/-- The real `1 / 3` rounded down to binary32. The named mode `.down` selects the rounding, so no
raw integer-rounding function is passed at the call site. -/
noncomputable def oneThirdDown : ℝ :=
  Model.roundAt FloatFormat.binary32 (1 / 3) (mode := .down)

/-- Rounding down gives a lower endpoint for `1 / 3`. -/
theorem oneThirdDown_le : oneThirdDown ≤ 1 / 3 := by
  simpa [oneThirdDown, Model.roundAt] using
    (round_floor_le (β := binaryRadix) (fexp := (Model.fexpOf FloatFormat.binary32)) (1 / 3))

/-- The fused multiply-add `1 * 2 + 1/4` of binary32 inputs, rounded down to a lower endpoint. -/
def fusedLower : Binary32 :=
  Binary.fma (1 : Binary32) (ofBits32 0x40000000) (ofBits32 0x3e800000)
    .down

/-- The same fused multiply-add rounded up, the upper endpoint. -/
def fusedUpper : Binary32 :=
  Binary.fma (1 : Binary32) (ofBits32 0x40000000) (ofBits32 0x3e800000)
    .up

/-- The two directed FMA results enclose the exact real value $1 \cdot 2 + 1/4$. The sum $9/4$
is representable, so both endpoints equal it. -/
theorem fused_enclosure :
    (toModel fusedLower).toEReal ≤
        (((toModel (1 : Binary32)).toReal * (toModel (ofBits32 0x40000000)).toReal + (toModel
          (ofBits32 0x3e800000)).toReal : ℝ) : EReal) ∧
      (((toModel (1 : Binary32)).toReal * (toModel (ofBits32 0x40000000)).toReal + (toModel
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

/-- The binary32 square root of `2` rounded down, the lower endpoint. -/
def sqrtLower : Binary32 :=
  (Binary.sqrt (mode := .down)) (ofBits32 0x40000000)

/-- The square root of `2` rounded up (toward $+\infty$), so the pair brackets `sqrt 2`. -/
def sqrtUpper : Binary32 :=
  (Binary.sqrt (mode := .up)) (ofBits32 0x40000000)

/-- The executable endpoints enclose the exact real value `sqrt 2`. -/
theorem sqrt_enclosure :
    (toModel sqrtLower).toEReal ≤ (Real.sqrt ((toModel (ofBits32 0x40000000)).toReal) : EReal) ∧
      (Real.sqrt ((toModel (ofBits32 0x40000000)).toReal) : EReal) ≤
        (toModel sqrtUpper).toEReal := by
  have hlo : toModel sqrtLower =
      Model.sqrt (toModel (ofBits32 0x40000000)) (mode := .down) :=
    Binary.toModel_sqrt (ofBits32 0x40000000) .down
  have hhi : toModel sqrtUpper =
      Model.sqrt (toModel (ofBits32 0x40000000)) (mode := .up) :=
    Binary.toModel_sqrt (ofBits32 0x40000000) .up
  constructor
  · rw [hlo]
    exact Model.toEReal_sqrt_down_le (toModel (ofBits32 0x40000000))
      (by decide) (by decide) (by decide)
  · rw [hhi]
    exact Model.le_toEReal_sqrt_up (toModel (ofBits32 0x40000000))
      (by decide) (by decide) (by decide)

/-! ## Fixed grids, quantization, and double rounding -/

/-- A signed affine quantizer with quarter-unit spacing. Tensor layout and storage width enter
the construction only through the integer code bounds. -/
noncomputable def quantizer : RealAffineQuantizer where
  scale := 1 / 4
  zeroPoint := 0
  qmin := -128
  qmax := 127
  scale_pos := by norm_num
  codeRange := by norm_num

/-- Every code from `-128` to `127` comes back unchanged after dequantization and nearest-even
requantization. -/
theorem quantizer_roundtrip {code : ℤ}
    (hlo : -128 ≤ code) (hhi : code ≤ 127) :
    quantizer.quantize nearestEven
        (quantizer.dequantize code) = code := by
  exact affine_quantize_dequantize quantizer nearestEven hlo hhi

/-- If the raw nearest-even code of `x` lies in the code range, so that nothing saturates, then
quantizing and dequantizing moves `x` by at most `1/8`, half a step. -/
theorem quantizer_error (x : ℝ)
    (hlo : quantizer.qmin ≤
      quantizer.rawCode nearestEven x)
    (hhi : quantizer.rawCode nearestEven x ≤
      quantizer.qmax) :
    abs (quantizer.dequantize
      (quantizer.quantize nearestEven x) - x) ≤ 1 / 8 := by
  have h :=
    affine_dequantize_quantize_error_le_half quantizer nearestEven x hlo hhi
  simpa only [RealAffineQuantizer.roundedValue, quantizer,
    show (1 / 4 : ℝ) / 2 = 1 / 8 by norm_num] using h

/-- The codes `0, 1, 2, 3` in the same shape-indexed tensor type that TorchLean models use. -/
def quarterCodes : Tensor ℤ vectorShape :=
  Tensor.ofFn (fun i => i.val)

/-- Dequantizing `quarterCodes` and quantizing back, entry by entry, returns the same tensor. -/
theorem quarterCodes_roundtrip :
    Tensor.map (quantizer.quantize nearestEven)
      (Tensor.map quantizer.dequantize quarterCodes) = quarterCodes := by
  refine Tensor.map_map_eq_self_of_forall (predicate := fun code =>
    quantizer.qmin ≤ code ∧ code ≤ quantizer.qmax) ?_
    fun _ h => quantizer_roundtrip h.1 h.2
  intro i
  change quantizer.qmin ≤ quarterCodes.getScalar i ∧
    quarterCodes.getScalar i ≤ quantizer.qmax
  simp only [quarterCodes, Tensor.getScalar_ofFn]
  have hi : i.val < 4 := i.isLt
  norm_num [quantizer]
  grind

/-- Rounding `x` to odd on a grid at least two bits finer than the quarter grid, then to nearest
even on the quarter grid, gives the same result as a single nearest-even rounding of `x`. -/
theorem double_rounding_safe (extra : ℕ) (x : ℝ) :
    roundAtScale nearestEven (1 / 4) (by norm_num)
        (roundAtScale oddRound
          ((1 / 4) / (2 : ℝ) ^ (extra + 2)) (by positivity) x) =
      roundAtScale nearestEven (1 / 4) (by norm_num) x := by
  exact roundAtScale_nearestEven_after_odd_binary_extra extra (1 / 4) x (by norm_num)

/-! ## IEEE exception status -/

/-- With status tracking on, `1 / +0` raises the divide-by-zero flag and leaves the
invalid-operation flag clear. -/
theorem one_div_zero_status :
    (Binary.div (1 : Binary32) (Binary.zero false : Binary32)
      (status := true)).status.divideByZero = true ∧
      (Binary.div (1 : Binary32) (Binary.zero false : Binary32)
        (status := true)).status.invalid = false := by
  decide

/-! ## A training-shaped reduction -/

/-- The dot product $1 \cdot 2 + 3 \cdot 4$ as a reduction tree with two leaves. The tree's
shape fixes the accumulation order. -/
def dotTree : ReductionTree (Binary32 × Binary32) :=
  .node
    (.leaf ((1 : Binary32), ofBits32 0x40000000))
    (.leaf (ofBits32 0x40400000, ofBits32 0x40800000))

/-- Both products and the final addition are finite in binary32. -/
theorem dotTree_finite : FiniteDot dotTree := by
  have hleft : ExecFloat.mul (1 : Binary32) (ofBits32 0x40000000) = ofBits32 0x40000000 := by
    rw [FloatLib.Floats.ExecFloat.Proof.mul_eq_spec]
    decide
  have hright :
      ExecFloat.mul (ofBits32 0x40400000) (ofBits32 0x40800000) = ofBits32 0x41400000 := by
    rw [FloatLib.Floats.ExecFloat.Proof.mul_eq_spec]
    decide
  simp only [dotTree, FiniteDot, evalDot, ReductionTree.eval, hleft, hright]
  refine ⟨by decide, by decide, ?_⟩
  change Model.isFinite (toModel ((ofBits32 0x40000000 : Binary32) + ofBits32 0x41400000)) = true
  rw [Binary.toModel_hAdd (ofBits32 0x40000000) (ofBits32 0x41400000), Model.Proof.add_eq_spec]
  decide

/--
The executable dot product denotes the effective nearest-even rounding of its final addition.
The operands of that addition are the two leaf products, each already rounded to binary32.
-/
theorem dotTree_computed :
    (toModel (evalDot dotTree)).toReal =
      FloatLib.Floats.Formats.Flocq.toReal (β := binaryRadix) {
        mantissa := nearestEvenMantissa
          (scaledMantissa binaryRadix (Model.fexpOf FloatFormat.binary32)
            (evalDotReal (.leaf ((1 : Binary32), ofBits32 0x40000000)) +
              evalDotReal (.leaf (ofBits32 0x40400000, ofBits32 0x40800000))))
        exponent := cexp binaryRadix (Model.fexpOf FloatFormat.binary32)
          (evalDotReal (.leaf ((1 : Binary32), ofBits32 0x40000000)) +
            evalDotReal (.leaf (ofBits32 0x40400000, ofBits32 0x40800000))) } := by
  rw [toReal_evalDot dotTree dotTree_finite]
  change Model.roundAt FloatFormat.binary32
    (evalDotReal (.leaf ((1 : Binary32), ofBits32 0x40000000)) +
      evalDotReal (.leaf (ofBits32 0x40400000, ofBits32 0x40800000))) = _
  exact round_nearestEven_computed _

end TorchLean.Examples.EffectiveRounding
