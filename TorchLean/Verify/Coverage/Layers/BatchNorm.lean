/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the running batch normalization of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. The theorems are declared in the namespace `TorchLean.Coverage.Layers`,
which every module under `TorchLean/Verify/Coverage/Layers/` shares.
-/

public section

namespace TorchLean.Coverage.Layers

-- `Op.batchNorm`, with batch statistics, is analyzed beside `Op.layerNorm` in
-- `TorchLean.Verify.Coverage.Layers.LayerNorm`: the two share the propagation
-- theorems of `Normalization.normalizedRows`.
#roundoff_coverage
  layers := Op.batchNormRunning

end TorchLean.Coverage.Layers
