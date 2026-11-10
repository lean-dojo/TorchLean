/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the data movement, fill and cast operators of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace `TorchLean.Coverage.Layers`,
which every module under `TorchLean/Verify/Coverage/Layers/` shares.
-/

public section

namespace TorchLean.Coverage.Layers

#roundoff_coverage
  layers :=
    Op.gather, Op.reshape, Op.permute, Op.broadcast, Op.fullLike, Op.embedding, Op.flip, Op.indices,
    Op.cast

end TorchLean.Coverage.Layers
