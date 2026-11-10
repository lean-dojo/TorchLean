/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Reduce.Backward
public import TorchLean.Tensor.Scatter
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Reductions and scatters over any axes for `roundoff`

The read lemmas of `reduce` for sums, products, maxima and minima, and of `scatter`, are `roundoff`
rules. Each reads an output entry as one left fold, so the analysis bounds these operators at any
axes as the folds they are. A maximum or minimum steps with `Context.maximum` or `Context.minimum`,
which `TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise` reads on finite values as the real
`max` or `min`. The read of `contract` is registered with the tensors themselves, in
`TorchLean.Verify.RuntimeApprox.Roundoff`.
-/

public section

attribute [roundoff_rule]
  TorchLean.Tensor.reduce_sum_apply
  TorchLean.Tensor.reduce_prod_apply
  TorchLean.Tensor.reduce_max_apply
  TorchLean.Tensor.reduce_min_apply
  TorchLean.Tensor.scatter_apply
