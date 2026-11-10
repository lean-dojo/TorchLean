/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Layers.Embedding
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Layers.Attention
public import TorchLean.Tensor.Layers.MoE
public import TorchLean.Tensor.Layers.PositionalEncoding

/-!
# Roundoff coverage of the general affine map

`#roundoff_coverage` runs `roundoff` on the value of the affine map at every index of a leading
shape and of its flattened weight and bias gradients and its input gradient, at a coordinate of the
result. These are the four computations of `Op.linear` at a leading shape of positive rank. The
table's own entry for `Op.linear` is its vector rule, read at the empty leading shape. Shapes, axes,
block sizes and every other parameter are variables, so each theorem holds at every axis and rank.
Each route is analyzed for the configured scalar of an arbitrary IEEE format, for native `Float32`
and for the rounded reals `FP32`. The theorems are declared in the namespace
`TorchLean.Coverage.TensorComposites`, which every module under
`TorchLean/Verify/Coverage/TensorComposites/` shares.
-/

public section

namespace TorchLean.Coverage.TensorComposites

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  linear :=
    Linear.forwardLeading, Linear.weightsDerivLeading, Linear.biasDerivLeading,
    Linear.inputDerivLeading

end TorchLean.Coverage.TensorComposites
