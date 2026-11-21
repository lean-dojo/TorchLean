/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the sorting, cumulative and arg-extremum operators of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace `TorchLean.Coverage.Layers`,
which every module under `TorchLean/Verify/Coverage/Layers/` shares.

`Op.cumprod` is also covered in the orders `pairwise`, `blocked ?width` and `ofTree ?tree`
(`Tensor.ReduceOrder`), the width and the tree variables of its routes. Each prefix product in such
an order has the left fold's relative error and the absolute error from underflow that the order's
tree accumulates (`Roundoff.Rules.ReduceOrder`), on the native Float32 backend too. `Op.cummax`
and `Op.cummin` are covered in the same orders: a running maximum or minimum is the input at the
index the order selects, with the input's error there.
-/

public section

namespace TorchLean.Coverage.Layers

#roundoff_coverage
  layers := Op.sort, Op.topk, Op.cumprod, Op.cummax, Op.cummin, Op.argmax, Op.argmin,
    Op.cumprod (order := .pairwise), Op.cumprod (order := .blocked ?width),
    Op.cumprod (order := .ofTree ?tree), Op.cummax (order := .pairwise),
    Op.cummax (order := .blocked ?width), Op.cummax (order := .ofTree ?tree),
    Op.cummin (order := .pairwise), Op.cummin (order := .blocked ?width),
    Op.cummin (order := .ofTree ?tree)

end TorchLean.Coverage.Layers
