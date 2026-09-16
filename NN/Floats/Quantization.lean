/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.Flocq
public import FloatLib.Numerics.Quantization.Affine


/-!
# Affine Quantization

This module gives one affine quantizer for real scalars. For a positive scale
$s$, zero point $z$, and integer code interval $[q_{\min},q_{\max}]$, quantization and
reconstruction are

$$
Q(x)=\operatorname{clamp}\!\left(\operatorname{round}(x/s)+z,q_{\min},q_{\max}\right),
\qquad
D(k)=s(k-z).
$$

Theorems below cover code-range safety, monotonicity, in-range code round trips, and the half-step
reconstruction bound when saturation is inactive. The shape-polymorphic tensor lift belongs to the
TorchLean specification adapter `NN.Spec.Quantization`.

The equations follow the integer-arithmetic quantization scheme used by common neural-network
runtimes. See Jacob et al., "Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference," CVPR 2018, doi:10.1109/CVPR.2018.00286.
-/

@[expose] public section

open FloatLib.Numerics FloatLib.Floats.Formats.Flocq

namespace TorchLean.Floats.Quantization

/-- Parameters of a bounded affine quantizer. -/
structure AffineQuantizer where
  /-- Distance between adjacent reconstructed real values. -/
  scale : ℝ
  /-- Integer code representing real zero when it lies in the code range. -/
  zeroPoint : ℤ
  /-- Smallest stored code. -/
  qmin : ℤ
  /-- Largest stored code. -/
  qmax : ℤ
  /-- A quantization scale is strictly positive. -/
  scale_pos : 0 < scale
  /-- The code interval is nonempty. -/
  codeRange : qmin ≤ qmax

namespace AffineQuantizer

/-- Saturate an integer to the quantizer's code interval. -/
def clampCode (q : AffineQuantizer) (code : ℤ) : ℤ :=
  FloatLib.Numerics.Quantization.Saturating.clamp q.qmin q.qmax code

/-- Integer code before saturation. -/
noncomputable def rawCode (q : AffineQuantizer) (rnd : ℝ → ℤ) (x : ℝ) : ℤ :=
  rnd (x / q.scale) + q.zeroPoint

/-- Quantize a real value using the supplied integer rounder and saturate it to the code set. -/
noncomputable def quantize (q : AffineQuantizer) (rnd : ℝ → ℤ) (x : ℝ) : ℤ :=
  q.clampCode (q.rawCode rnd x)

/-- Reconstruct a real value from an integer code. -/
noncomputable def dequantize (q : AffineQuantizer) (code : ℤ) : ℝ :=
  q.scale * ((code - q.zeroPoint : ℤ) : ℝ)

/-- Saturation always returns a valid code. -/
theorem clampCode_mem (q : AffineQuantizer) (code : ℤ) :
    q.qmin ≤ q.clampCode code ∧ q.clampCode code ≤ q.qmax :=
  ⟨FloatLib.Numerics.Quantization.Saturating.lower_le_clamp _ _ _,
    FloatLib.Numerics.Quantization.Saturating.clamp_le_upper q.codeRange⟩

/-- Saturation fixes a code already inside the representable interval. -/
@[simp] theorem clampCode_eq_self (q : AffineQuantizer) {code : ℤ}
    (hlo : q.qmin ≤ code) (hhi : code ≤ q.qmax) :
    q.clampCode code = code :=
  FloatLib.Numerics.Quantization.Saturating.clamp_eq_self hlo hhi

/-- Every quantized scalar lies in the declared code interval. -/
theorem quantize_mem (q : AffineQuantizer) (rnd : ℝ → ℤ) (x : ℝ) :
    q.qmin ≤ q.quantize rnd x ∧ q.quantize rnd x ≤ q.qmax :=
  q.clampCode_mem _

/-- Dequantizing the zero point gives real zero. -/
@[simp] theorem dequantize_zeroPoint (q : AffineQuantizer) :
    q.dequantize q.zeroPoint = 0 := by
  simp [dequantize]

/-- Quantization is monotone for every valid integer rounding rule. -/
theorem quantize_mono (q : AffineQuantizer) (rnd : ℝ → ℤ) [ValidRnd rnd]
    {x y : ℝ} (hxy : x ≤ y) :
    q.quantize rnd x ≤ q.quantize rnd y := by
  have hscaled : x / q.scale ≤ y / q.scale :=
    div_le_div_of_nonneg_right hxy q.scale_pos.le
  have hround : rnd (x / q.scale) ≤ rnd (y / q.scale) :=
    ValidRnd.monotone _ _ hscaled
  have hcode : rnd (x / q.scale) + q.zeroPoint ≤ rnd (y / q.scale) + q.zeroPoint :=
    by simpa [add_comm] using add_le_add_right hround q.zeroPoint
  have hmin :
      min q.qmax (rnd (x / q.scale) + q.zeroPoint) ≤
        min q.qmax (rnd (y / q.scale) + q.zeroPoint) :=
    min_le_min (le_refl q.qmax) hcode
  unfold quantize clampCode rawCode FloatLib.Numerics.Quantization.Saturating.clamp
  exact max_le_max (le_refl q.qmin) hmin

/-- Every in-range integer code survives a dequantize/quantize round trip. -/
@[simp] theorem quantize_dequantize (q : AffineQuantizer) (rnd : ℝ → ℤ) [ValidRnd rnd]
    {code : ℤ} (hlo : q.qmin ≤ code) (hhi : code ≤ q.qmax) :
    q.quantize rnd (q.dequantize code) = code := by
  have hscale : q.scale ≠ 0 := ne_of_gt q.scale_pos
  have hscaled : q.dequantize code / q.scale = ((code - q.zeroPoint : ℤ) : ℝ) := by
    simp [dequantize, hscale]
  have hrnd : rnd (((code - q.zeroPoint : ℤ) : ℝ)) = code - q.zeroPoint :=
    ValidRnd.id _
  rw [quantize, rawCode, hscaled]
  rw [hrnd]
  simp only [Int.sub_add_cancel]
  exact q.clampCode_eq_self hlo hhi

/-- Without saturation, nearest affine quantization has error at most half a quantization step. -/
theorem dequantize_rawCode_error_le (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [ValidRndToNearest rnd] (x : ℝ) :
    abs (q.dequantize (q.rawCode rnd x) - x) ≤ q.scale / 2 := by
  have hscale : q.scale ≠ 0 := ne_of_gt q.scale_pos
  have hround := ValidRndToNearest.abs_sub_le_half (rnd := rnd) (x / q.scale)
  have hrewrite :
      q.dequantize (q.rawCode rnd x) - x =
        q.scale * ((rnd (x / q.scale) : ℝ) - x / q.scale) := by
    simp only [dequantize, rawCode, Int.cast_sub, Int.cast_add]
    field_simp
    ring
  rw [hrewrite, abs_mul, abs_of_pos q.scale_pos]
  have hhalf : (2⁻¹ : ℝ) = 1 / 2 := by norm_num
  rw [hhalf] at hround
  exact (mul_le_mul_of_nonneg_left hround q.scale_pos.le).trans_eq (by ring)

/-- The half-step bound survives saturation whenever clipping is inactive. -/
theorem dequantize_quantize_error_le (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [ValidRndToNearest rnd] (x : ℝ)
    (hlo : q.qmin ≤ q.rawCode rnd x) (hhi : q.rawCode rnd x ≤ q.qmax) :
    abs (q.dequantize (q.quantize rnd x) - x) ≤ q.scale / 2 := by
  rw [quantize, q.clampCode_eq_self hlo hhi]
  exact q.dequantize_rawCode_error_le rnd x

end AffineQuantizer

end TorchLean.Floats.Quantization
