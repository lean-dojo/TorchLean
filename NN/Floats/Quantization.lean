/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all

public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Affine

/-!
# Affine quantization

FloatLib's `RealAffineQuantizer` supplies a real-valued scale, bounded integer codes, and a
caller-chosen rounding rule. Its `AffineQuantizer` supplies executable rational nearest-even
quantization; the Flocq adapter proves agreement with the real specification.

TorchLean adds shape-polymorphic tensor operations in `NN.Spec.Quantization`. Scalar definitions
and rounding proofs stay in FloatLib so tensor clients use the same grids and error bounds.

The equations follow Jacob et al., "Quantization and Training of Neural Networks for Efficient
Integer-Arithmetic-Only Inference," CVPR 2018, doi:10.1109/CVPR.2018.00286.
-/
