/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Numerics.Quantization.Affine
public import NN.Spec.Core.TensorOps

/-!
# Executable Rational Tensor Quantization

FloatLib supplies the scalar nearest-even affine quantizer and its range, round-trip, and error
proofs. This adapter maps those same operations over shape-indexed rational tensors.
-/

@[expose] public section

namespace FloatLib.Numerics.Quantization

open _root_.Spec TorchLean

namespace AffineQuantizer

/-- Apply the affine quantizer independently at every coordinate of an arbitrary-rank tensor. -/
def quantizeTensor (q : AffineQuantizer) {s : Shape}
    (x : Tensor ℚ s) : Tensor ℤ s :=
  Tensor.map q.quantize x

/-- Reconstruct every code in an arbitrary-rank tensor on the quantizer's rational grid. -/
def dequantizeTensor (q : AffineQuantizer) {s : Shape}
    (codes : Tensor ℤ s) : Tensor ℚ s :=
  Tensor.map q.dequantize codes

/-- Pointwise condition saying that quantization does not clip any coordinate of `x`. -/
def SaturationInactive (q : AffineQuantizer) {s : Shape}
    (x : Tensor ℚ s) : Prop :=
  Tensor.Forall (fun a => q.qmin ≤ q.rawCode a ∧ q.rawCode a ≤ q.qmax) x

/-- Pointwise condition saying that every stored code belongs to the quantizer's code set. -/
def CodesInRange (q : AffineQuantizer) {s : Shape} (codes : Tensor ℤ s) : Prop :=
  Tensor.Forall (fun code => q.qmin ≤ code ∧ code ≤ q.qmax) codes

/-- Every coordinate produced by tensor quantization lies in the declared code interval. -/
theorem quantizeTensor_inRange (q : AffineQuantizer)
    {s : Shape} (x : Tensor ℚ s) :
    q.CodesInRange (q.quantizeTensor x) := by
  apply Tensor.forall_map (Tensor.forall_true x)
  intro a _
  exact q.quantize_mem a

/-- An in-range code tensor survives pointwise dequantization and requantization exactly. -/
theorem quantizeTensor_dequantizeTensor (q : AffineQuantizer)
    {s : Shape} {codes : Tensor ℤ s} (hcodes : q.CodesInRange codes) :
    q.quantizeTensor (q.dequantizeTensor codes) = codes := by
  induction s with
  | scalar =>
      apply Tensor.ext_scalar
      change q.quantize (q.dequantize codes.item) = codes.item
      exact q.quantize_dequantize hcodes.1 hcodes.2
  | dim n inner ih =>
      apply (Tensor.dimEquiv n inner).injective
      funext i
      change Tensor.unstack
        (Tensor.map q.quantize (Tensor.map q.dequantize codes)) i =
          Tensor.unstack codes i
      rw [show Tensor.unstack
          (Tensor.map q.quantize (Tensor.map q.dequantize codes)) i =
          Tensor.map q.quantize
            (Tensor.unstack (Tensor.map q.dequantize codes) i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack q.quantize
          (Tensor.map q.dequantize codes) i).symm]
      rw [show Tensor.unstack (Tensor.map q.dequantize codes) i =
          Tensor.map q.dequantize (Tensor.unstack codes i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack q.dequantize codes i).symm]
      exact ih (hcodes i)

/-- If no coordinate clips, every tensor reconstruction error is at most half a step. -/
theorem dequantizeTensor_quantizeTensor_error_le (q : AffineQuantizer)
    {s : Shape} {x : Tensor ℚ s}
    (hinactive : q.SaturationInactive x) :
    Tensor.Forall (fun e : ℚ => abs e ≤ q.scale / 2)
      ((q.dequantizeTensor (q.quantizeTensor x)).subSpec x) := by
  induction s with
  | scalar =>
      change q.qmin ≤ q.rawCode x.item ∧ q.rawCode x.item ≤ q.qmax at hinactive
      change abs (q.dequantize (q.quantize x.item) - x.item) ≤ q.scale / 2
      exact q.dequantize_quantize_error_le x.item hinactive.1 hinactive.2
  | dim n inner ih =>
      intro i
      change Tensor.Forall (fun e : ℚ => abs e ≤ q.scale / 2)
        (Tensor.unstack
          (Tensor.map2Spec (· - ·)
            (Tensor.map q.dequantize (Tensor.map q.quantize x)) x) i)
      rw [show Tensor.unstack
          (Tensor.map2Spec (· - ·)
            (Tensor.map q.dequantize (Tensor.map q.quantize x)) x) i =
          Tensor.map2Spec (· - ·)
            (Tensor.unstack (Tensor.map q.dequantize (Tensor.map q.quantize x)) i)
            (Tensor.unstack x i) by
        exact (TorchLean.Tensor.Internal.Rep.zipWith_unstack (· - ·)
          (Tensor.map q.dequantize (Tensor.map q.quantize x)) x i).symm]
      rw [show Tensor.unstack
          (Tensor.map q.dequantize (Tensor.map q.quantize x)) i =
          Tensor.map q.dequantize
            (Tensor.unstack (Tensor.map q.quantize x) i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack q.dequantize
          (Tensor.map q.quantize x) i).symm]
      rw [show Tensor.unstack (Tensor.map q.quantize x) i =
          Tensor.map q.quantize (Tensor.unstack x i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack q.quantize x i).symm]
      exact ih (hinactive i)

end AffineQuantizer
end FloatLib.Numerics.Quantization
