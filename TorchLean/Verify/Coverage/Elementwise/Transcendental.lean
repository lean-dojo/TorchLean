/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the transcendental functions of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace
`TorchLean.Coverage.Elementwise`, which every module under
`TorchLean/Verify/Coverage/Elementwise/` shares.
-/

public section

namespace TorchLean.Coverage.Elementwise

#roundoff_coverage
  elementwise :=
    Op.exp, Op.sin, Op.cos, Op.log, Op.safeLog, Op.exp2, Op.expm1, Op.log2, Op.log10, Op.log1p,
    Op.tan, Op.asinh, Op.acosh, Op.atanh, Op.atan, Op.asin, Op.acos, Op.erf, Op.atan2, Op.sinh,
    Op.cosh

end TorchLean.Coverage.Elementwise
