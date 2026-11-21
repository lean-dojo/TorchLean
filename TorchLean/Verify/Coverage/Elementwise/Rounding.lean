/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.Coverage.Elementwise.Arithmetic -- shake: keep
public import TorchLean.Verify.Coverage.Pointwise.Rounding -- shake: keep
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the rounding, remainder, sign and exponent operators of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace
`TorchLean.Coverage.Elementwise`, which every module under
`TorchLean/Verify/Coverage/Elementwise/` shares.

It imports the coverage of the arithmetic of the operator table (`Arithmetic.lean`) and of the
tensor rounding functions (`Pointwise/Rounding.lean`), so that a route with the statement of a route
there records that theorem instead of declaring another.
-/

public section

namespace TorchLean.Coverage.Elementwise

#roundoff_coverage
  elementwise :=
    Op.sign, Op.floor, Op.ceil, Op.round, Op.trunc, Op.frac, Op.fmod, Op.remainder, Op.copysign,
    Op.nextafter, Op.frexp

end TorchLean.Coverage.Elementwise
