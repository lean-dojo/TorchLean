/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage

/-!
# Roundoff coverage of the indices of extrema, the cumulative reductions and sorting

`#roundoff_coverage` runs `roundoff` on `argmax`, `argmin`, `cumsum`, `cumprod`, `cummax`, `cummin`,
`sort` and `topk` and their vector-Jacobian products, at a coordinate of each result. The registered
operators are recorded with the operator table's coverage. `dim`, `k`, the direction and the shapes
are variables, so each theorem holds at every axis and rank. An index tensor is read through the
cast of its entry into the scalar. Each route is analyzed for the configured scalar of an arbitrary
IEEE format, for native `Float32` and for the rounded reals `FP32`.
-/

public section

namespace TorchLean.Coverage.Ordering

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  ordering :=
    Tensor.argmax, Tensor.argmin, Tensor.cummax, Tensor.cummin, Tensor.sort, Tensor.topk
  cumulative := Tensor.cumsum, Tensor.cumprod
  derivative :=
    Tensor.cumsumBackward, Tensor.cumprodBackward, Tensor.cumprodJVP, Tensor.cummaxBackward,
    Tensor.cumminBackward, Tensor.sortBackward, Tensor.topkBackward

end TorchLean.Coverage.Ordering
