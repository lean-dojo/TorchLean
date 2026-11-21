/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.Coverage.Pointwise.Selection -- shake: keep
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the extrema, clamps and selections of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace
`TorchLean.Coverage.Elementwise`, which every module under
`TorchLean/Verify/Coverage/Elementwise/` shares.

It imports the coverage of the tensor extrema and selections (`Pointwise/Selection.lean`), so that a
route with the statement of a route there records that theorem instead of declaring another.
-/

public section

namespace TorchLean.Coverage.Elementwise

#roundoff_coverage
  elementwise :=
    Op.max, Op.min, Op.clamp, Op.fmax, Op.fmin, Op.nanToNum, Op.select

end TorchLean.Coverage.Elementwise
