/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage

/-!
# Roundoff coverage of the general convolutions and poolings

`#roundoff_coverage` runs `roundoff` on the value of each convolution, transposed convolution, their
gradients and each pooling for each reduction, at a coordinate of the result. Shapes, axes and every
other parameter are variables, so each theorem holds at every axis and rank. Each route is analyzed
for the configured scalar of an arbitrary IEEE format, for native `Float32` and for the rounded
reals `FP32`. The theorems are declared in the namespace
`TorchLean.Coverage.TensorOperations`, which every module under
`TorchLean/Verify/Coverage/TensorOperations/` shares.
-/

public section

namespace TorchLean.Coverage.TensorOperations

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  window :=
    Tensor.conv, Tensor.convTranspose, Tensor.convBackwardInput, Tensor.convBackwardWeight,
    Tensor.convTransposeBackwardInput, Tensor.convTransposeBackwardWeight, Tensor.pool,
    Tensor.poolBackward, Tensor.smoothMaxPool, Tensor.smoothMaxPoolJVP,
    Tensor.smoothMaxPoolBackward, Tensor.adaptivePool

end TorchLean.Coverage.TensorOperations
