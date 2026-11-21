/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Verify.Coverage.Elementwise.Arithmetic -- shake: keep
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the masks of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace
`TorchLean.Coverage.Indexing`, which every module under
`TorchLean/Verify/Coverage/Indexing/` shares.

It imports the coverage of the arithmetic of the operator table (`Elementwise/Arithmetic.lean`), so
that a route with the statement of a route there records that theorem instead of declaring another.
-/

public section

namespace TorchLean.Coverage.Indexing

#roundoff_coverage
  indexing := Op.maskedSoftmax, Op.topKMask, Op.bernoulliMask

end TorchLean.Coverage.Indexing
