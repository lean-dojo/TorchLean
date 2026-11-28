/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.Coverage.Pointwise.Arithmetic -- shake: keep
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the arithmetic, powers and roots of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace
`TorchLean.Coverage.Elementwise`, which every module under
`TorchLean/Verify/Coverage/Elementwise/` shares.

It imports the coverage of the tensor arithmetic (`Pointwise/Arithmetic.lean`), so that a route with
the statement of a route there records that theorem instead of declaring another.

`Op.cumsum` is also covered in the orders `pairwise`, `blocked ?width` and `ofTree ?tree`
(`Tensor.ReduceOrder`): each prefix sum is bounded at the depth of the order's tree on the prefix
(`Roundoff.Rules.ReduceOrder`).
-/

public section

namespace TorchLean.Coverage.Elementwise

#roundoff_coverage
  elementwise :=
    Op.abs, Op.sqrt, Op.neg, Op.add, Op.mul, Op.square, Op.reciprocal, Op.sub, Op.div, Op.cumsum,
    Op.scale, Op.rsqrt, Op.hypot, Op.lerp, Op.addcmul, Op.addcdiv, Op.safeInv, Op.smoothAbs,
    Op.safeDiv, Op.pow, Op.powValue, Op.detach, Op.cbrt, Op.fma, Op.cumsum (order := .pairwise),
    Op.cumsum (order := .blocked ?width), Op.cumsum (order := .ofTree ?tree)

end TorchLean.Coverage.Elementwise
