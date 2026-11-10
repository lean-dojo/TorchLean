/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage

/-!
# Roundoff coverage of the general data movement

`#roundoff_coverage` runs `roundoff` on the value of each general data-movement operator, at a
coordinate of the result. Shapes, axes and every other parameter are variables, so each theorem
holds at every axis and rank. Each route is analyzed for the configured scalar of an arbitrary IEEE
format, for native `Float32` and for the rounded reals `FP32`. The theorems are declared in the
namespace `TorchLean.Coverage.TensorOperations`, which every module under
`TorchLean/Verify/Coverage/TensorOperations/` shares.
-/

public section

namespace TorchLean.Coverage.TensorOperations

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  movement :=
    Tensor.broadcast, Tensor.reshape, Tensor.permute, Tensor.slice, Tensor.pad, Tensor.cat,
    Tensor.flip, Tensor.indices, Tensor.gather

end TorchLean.Coverage.TensorOperations
