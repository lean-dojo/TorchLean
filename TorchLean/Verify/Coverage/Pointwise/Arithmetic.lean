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
# Roundoff coverage of the arithmetic of core ATen and `jax.lax`

`#roundoff_coverage` runs `roundoff` on the value of each arithmetic operator of
`TorchLean.Tensor.Elementwise` at a coordinate of the result, for the configured scalar of
an arbitrary IEEE format, for native `Float32` and for the rounded reals `FP32`. Shapes are
variables, so each theorem holds at every shape. `sqrt` has one route for each constructor of its
policy argument `SqrtPolicy`. The theorems are declared in the namespace
`TorchLean.Coverage.Pointwise`, which every module under
`TorchLean/Verify/Coverage/Pointwise/` shares.
-/

public section

namespace TorchLean.Coverage.Pointwise

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  arithmetic :=
    Tensor.reciprocal, Tensor.sqrt, Tensor.rsqrt, Tensor.hypot, Tensor.lerp, Tensor.addcmul,
    Tensor.addcdiv

end TorchLean.Coverage.Pointwise
