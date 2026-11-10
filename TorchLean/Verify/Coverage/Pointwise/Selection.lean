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
# Roundoff coverage of the selections of core ATen and `jax.lax`

`#roundoff_coverage` runs `roundoff` on the value of each extremum, clamp and selection operator of
`TorchLean.Tensor.Elementwise` at a coordinate of the result, for the configured scalar of
an arbitrary IEEE format, for native `Float32` and for the rounded reals `FP32`. Shapes are
variables, so each theorem holds at every shape. `clamp` has one route for each constructor of its
policy argument `ClampPolicy`. The theorems are declared in the namespace
`TorchLean.Coverage.Pointwise`, which every module under
`TorchLean/Verify/Coverage/Pointwise/` shares.
-/

public section

namespace TorchLean.Coverage.Pointwise

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  selection :=
    Tensor.maximum, Tensor.minimum, Tensor.fmax, Tensor.fmin, Tensor.clamp, Tensor.clampDerivative,
    Tensor.select, Tensor.nanToNum

end TorchLean.Coverage.Pointwise
