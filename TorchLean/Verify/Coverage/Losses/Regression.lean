/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Roundoff coverage of the regression losses of the operator table

`#roundoff_coverage` reads the routes of each registered operator from its arm of
`TorchLean.Op.rule?`. `Op.huberLoss` is registered again with `mean := true`, the mean reduction
losses default to, as routes of their own. The theorems are declared in the namespace
`TorchLean.Coverage.Losses`, which every module under
`TorchLean/Verify/Coverage/Losses/` shares.
-/

public section

namespace TorchLean.Coverage.Losses

#roundoff_coverage
  losses :=
    Op.mseLoss, Op.maeLoss, Op.huberLoss, Op.huberLoss (mean := true), Op.logCoshLoss,
    Op.cosineLoss

end TorchLean.Coverage.Losses
