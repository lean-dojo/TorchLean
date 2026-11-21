/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the poolings of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace `TorchLean.Coverage.Layers`,
which every module under `TorchLean/Verify/Coverage/Layers/` shares.

`Op.pool` is also covered in the orders `pairwise`, `blocked ?width` and `ofTree ?tree`
(`Tensor.ReduceOrder`). A sum or average window in such an order is bounded at the depth of the
order's tree (`Roundoff.Rules.ReduceOrder`).
-/

public section

namespace TorchLean.Coverage.Layers

#roundoff_coverage
  layers := Op.pool, Op.smoothMaxPool, Op.pool (order := .pairwise),
    Op.pool (order := .blocked ?width), Op.pool (order := .ofTree ?tree)

end TorchLean.Coverage.Layers
