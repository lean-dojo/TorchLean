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
  apply TorchLean.Tensor.Internal.Rep.ext
  intro i
  have hi := Tensor.forall_iff.mp hcodes i
  simpa [quantizeTensor, dequantizeTensor, Tensor.map] using
    q.quantize_dequantize hi.1 hi.2

/-- If no coordinate clips, every tensor reconstruction error is at most half a step. -/
theorem dequantizeTensor_quantizeTensor_error_le (q : AffineQuantizer)
    {s : Shape} {x : Tensor ℚ s}
    (hinactive : q.SaturationInactive x) :
    Tensor.Forall (fun e : ℚ => abs e ≤ q.scale / 2)
      ((q.dequantizeTensor (q.quantizeTensor x)).subSpec x) := by
  rw [Tensor.forall_iff]
  intro i
  have hi := Tensor.forall_iff.mp hinactive i
  simpa [Tensor.subSpec, quantizeTensor, dequantizeTensor, Tensor.map,
    AffineQuantizer.roundedValue] using
    q.dequantize_quantize_error_le (x i) hi.1 hi.2

end AffineQuantizer
end FloatLib.Numerics.Quantization
