/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise -- shake: keep
public import TorchLean.Tensor.Elementwise

/-!
# Roundoff coverage of the transcendental functions of core ATen and `jax.lax`

`#roundoff_coverage` runs `roundoff` on the value of each transcendental function of
`TorchLean.Tensor.Elementwise` at a coordinate of the result, for the configured scalar of
an arbitrary IEEE format, for native `Float32` and for the rounded reals `FP32`. Shapes are
variables, so each theorem holds at every shape. The theorems are declared in the namespace
`TorchLean.Coverage.Pointwise`, which every module under
`TorchLean/Verify/Coverage/Pointwise/` shares.
-/

public section

namespace TorchLean.Coverage.Pointwise

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  transcendental :=
    Tensor.exp2, Tensor.expm1, Tensor.log2, Tensor.log10, Tensor.log1p, Tensor.tan, Tensor.asinh,
    Tensor.acosh, Tensor.atanh, Tensor.atan, Tensor.asin, Tensor.acos, Tensor.atan2, Tensor.erf

end TorchLean.Coverage.Pointwise
