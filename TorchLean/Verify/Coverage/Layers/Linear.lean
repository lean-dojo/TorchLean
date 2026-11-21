/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the affine map and the matrix products of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace `TorchLean.Coverage.Layers`,
which every module under `TorchLean/Verify/Coverage/Layers/` shares.

`Op.matmul` and `Op.scaledMatmul` are also covered in the orders `pairwise`, `blocked ?width` and
`ofTree ?tree` (`Tensor.ReduceOrder`). The forward and tangent of `Op.matmul` in those orders read
an entry through `Tensor.matmulCommonBatchIn_apply`, so their bounds carry the depth of the order's
tree.
-/

public section

namespace TorchLean.Coverage.Layers

#roundoff_coverage
  layers := Op.linear (lead := []), Op.matmul, Op.scaledMatmul, Op.matmul (order := .pairwise),
    Op.matmul (order := .blocked ?width), Op.matmul (order := .ofTree ?tree),
    Op.scaledMatmul (order := .pairwise), Op.scaledMatmul (order := .blocked ?width),
    Op.scaledMatmul (order := .ofTree ?tree)

end TorchLean.Coverage.Layers
