/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Quantization
public import NN.Spec.Quantization.Rational

/-!
# Tensor Quantization

This module lifts the scalar affine quantizer from `NN.Floats.Quantization` pointwise over
TorchLean's shape-indexed tensors. The real-domain specification retains arbitrary valid rounding
rules. The rational tensor API imports FloatLib's executable affine quantizer directly.
-/

@[expose] public section

namespace TorchLean.Floats.Quantization

open FloatLib.Floats.Formats.Flocq

open Spec TorchLean

namespace AffineQuantizer

/-- Apply the affine quantizer independently at every coordinate of an arbitrary-rank tensor. -/
noncomputable def quantizeTensor (q : AffineQuantizer) (rnd : ℝ → ℤ) {s : Shape}
    (x : Tensor ℝ s) : Tensor ℤ s :=
  Tensor.map (q.quantize rnd) x

/-- Reconstruct every code in an arbitrary-rank tensor on the quantizer's real grid. -/
noncomputable def dequantizeTensor (q : AffineQuantizer) {s : Shape}
    (codes : Tensor ℤ s) : Tensor ℝ s :=
  Tensor.map q.dequantize codes

/-- Pointwise condition saying that quantization does not clip any coordinate of `x`. -/
def SaturationInactive (q : AffineQuantizer) (rnd : ℝ → ℤ) {s : Shape}
    (x : Tensor ℝ s) : Prop :=
  Tensor.Forall (fun a => q.qmin ≤ q.rawCode rnd a ∧ q.rawCode rnd a ≤ q.qmax) x

/-- Pointwise condition saying that every stored code belongs to the quantizer's code set. -/
def CodesInRange (q : AffineQuantizer) {s : Shape} (codes : Tensor ℤ s) : Prop :=
  Tensor.Forall (fun code => q.qmin ≤ code ∧ code ≤ q.qmax) codes

/-- Every coordinate produced by tensor quantization lies in the declared code interval. -/
theorem quantizeTensor_inRange (q : AffineQuantizer) (rnd : ℝ → ℤ)
    {s : Shape} (x : Tensor ℝ s) :
    q.CodesInRange (q.quantizeTensor rnd x) := by
  apply Tensor.forall_map (Tensor.forall_true x)
  intro a _
  exact q.quantize_mem rnd a

/-- Pointwise order is preserved by tensor quantization. -/
theorem quantizeTensor_mono (q : AffineQuantizer) (rnd : ℝ → ℤ) [ValidRnd rnd]
    {s : Shape} {x y : Tensor ℝ s}
    (hxy : Tensor.Forall₂ (· ≤ ·) x y) :
    Tensor.Forall₂ (· ≤ ·) (q.quantizeTensor rnd x) (q.quantizeTensor rnd y) := by
  induction s with
  | scalar =>
      change x.item ≤ y.item at hxy
      change q.quantize rnd x.item ≤ q.quantize rnd y.item
      exact q.quantize_mono rnd hxy
  | dim n inner ih =>
      intro i
      change Tensor.Forall₂ (· ≤ ·)
        (Tensor.unstack (Tensor.map (q.quantize rnd) x) i)
        (Tensor.unstack (Tensor.map (q.quantize rnd) y) i)
      rw [show Tensor.unstack (Tensor.map (q.quantize rnd) x) i =
          Tensor.map (q.quantize rnd) (Tensor.unstack x i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack (q.quantize rnd) x i).symm]
      rw [show Tensor.unstack (Tensor.map (q.quantize rnd) y) i =
          Tensor.map (q.quantize rnd) (Tensor.unstack y i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack (q.quantize rnd) y i).symm]
      exact ih (hxy i)

/-- An in-range code tensor survives pointwise dequantization and requantization exactly. -/
theorem quantizeTensor_dequantizeTensor (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [ValidRnd rnd] {s : Shape} {codes : Tensor ℤ s} (hcodes : q.CodesInRange codes) :
    q.quantizeTensor rnd (q.dequantizeTensor codes) = codes := by
  induction s with
  | scalar =>
      apply Tensor.ext_scalar
      change q.quantize rnd (q.dequantize codes.item) = codes.item
      exact q.quantize_dequantize rnd hcodes.1 hcodes.2
  | dim n inner ih =>
      apply (Tensor.dimEquiv n inner).injective
      funext i
      change Tensor.unstack
        (Tensor.map (q.quantize rnd) (Tensor.map q.dequantize codes)) i =
          Tensor.unstack codes i
      rw [show Tensor.unstack
          (Tensor.map (q.quantize rnd) (Tensor.map q.dequantize codes)) i =
          Tensor.map (q.quantize rnd)
            (Tensor.unstack (Tensor.map q.dequantize codes) i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack (q.quantize rnd)
          (Tensor.map q.dequantize codes) i).symm]
      rw [show Tensor.unstack (Tensor.map q.dequantize codes) i =
          Tensor.map q.dequantize (Tensor.unstack codes i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack q.dequantize codes i).symm]
      exact ih (hcodes i)

/-- If no coordinate clips, every tensor reconstruction error is at most half a step. -/
theorem dequantizeTensor_quantizeTensor_error_le (q : AffineQuantizer) (rnd : ℝ → ℤ)
    [ValidRndToNearest rnd] {s : Shape} {x : Tensor ℝ s}
    (hinactive : q.SaturationInactive rnd x) :
    Tensor.Forall (fun e : ℝ => abs e ≤ q.scale / 2)
      ((q.dequantizeTensor (q.quantizeTensor rnd x)).subSpec x) := by
  induction s with
  | scalar =>
      change q.qmin ≤ q.rawCode rnd x.item ∧ q.rawCode rnd x.item ≤ q.qmax at hinactive
      change abs (q.dequantize (q.quantize rnd x.item) - x.item) ≤ q.scale / 2
      exact q.dequantize_quantize_error_le rnd x.item hinactive.1 hinactive.2
  | dim n inner ih =>
      intro i
      change Tensor.Forall (fun e : ℝ => abs e ≤ q.scale / 2)
        (Tensor.unstack
          (Tensor.map2Spec (· - ·)
            (Tensor.map q.dequantize (Tensor.map (q.quantize rnd) x)) x) i)
      rw [show Tensor.unstack
          (Tensor.map2Spec (· - ·)
            (Tensor.map q.dequantize (Tensor.map (q.quantize rnd) x)) x) i =
          Tensor.map2Spec (· - ·)
            (Tensor.unstack (Tensor.map q.dequantize (Tensor.map (q.quantize rnd) x)) i)
            (Tensor.unstack x i) by
        exact (TorchLean.Tensor.Internal.Rep.zipWith_unstack (· - ·)
          (Tensor.map q.dequantize (Tensor.map (q.quantize rnd) x)) x i).symm]
      rw [show Tensor.unstack
          (Tensor.map q.dequantize (Tensor.map (q.quantize rnd) x)) i =
          Tensor.map q.dequantize
            (Tensor.unstack (Tensor.map (q.quantize rnd) x) i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack q.dequantize
          (Tensor.map (q.quantize rnd) x) i).symm]
      rw [show Tensor.unstack (Tensor.map (q.quantize rnd) x) i =
          Tensor.map (q.quantize rnd) (Tensor.unstack x i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack (q.quantize rnd) x i).symm]
      exact ih (hinactive i)

end AffineQuantizer
end TorchLean.Floats.Quantization
