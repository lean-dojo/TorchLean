/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the comparisons and logical operators of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace `TorchLean.Coverage.Layers`,
which every module under `TorchLean/Verify/Coverage/Layers/` shares.
-/

public section

namespace TorchLean.Coverage.Layers

#roundoff_coverage
  layers :=
    Op.eq, Op.ne, Op.lt, Op.le, Op.gt, Op.ge, Op.isNaN, Op.isInf, Op.isFinite, Op.logicalNot,
    Op.logicalAnd, Op.logicalOr, Op.logicalXor

end TorchLean.Coverage.Layers
