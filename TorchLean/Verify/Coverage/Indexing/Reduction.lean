/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the reductions and the contraction of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace
`TorchLean.Coverage.Indexing`, which every module under
`TorchLean/Verify/Coverage/Indexing/` shares.

Each operator is also covered in the orders `pairwise`, `blocked ?width` and `ofTree ?tree`
(`Tensor.ReduceOrder`), the width and the tree variables of their routes. A sum in such an order
is bounded at the depth of the order's tree (`Roundoff.Rules.ReduceOrder`), on the native Float32
backend too, and a product with the left fold's relative error and the absolute error from
underflow that the tree accumulates; a maximum or minimum on finite entries has the real value of
the left fold, whose bound applies.
-/

public section

namespace TorchLean.Coverage.Indexing

#roundoff_coverage
  indexing := Op.reduce, Op.mean, Op.contract,
    Op.reduce (order := .pairwise), Op.reduce (order := .blocked ?width),
    Op.reduce (order := .ofTree ?tree), Op.mean (order := .pairwise),
    Op.mean (order := .blocked ?width), Op.mean (order := .ofTree ?tree),
    Op.contract (order := .pairwise), Op.contract (order := .blocked ?width),
    Op.contract (order := .ofTree ?tree)

end TorchLean.Coverage.Indexing
