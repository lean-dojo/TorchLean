/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Cumulative
public import TorchLean.Tensor.Sort
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Cumulative reductions and sorting for `roundoff`

The `roundoff` rules here read `cumprod`, the values of `cummax`, `cummin` and `sort`, both outputs
of `topk`, and the vector-Jacobian products of these operators and of `cumsum`. A value of a
running extremum, a sort or a top-k is the input read at a computed index, so it carries the
input's error at that index and adds none. The generated bound compares it with the real input at
the same index, which need not be the index an exact sort would choose. A cumulative product is
read as its left fold over the prefix. The entries of `cumsum` are registered as sums with the
tensors themselves, in `TorchLean.Verify.RuntimeApprox.Roundoff`.
-/

public section

attribute [roundoff_rule]
  TorchLean.Tensor.cumprod_apply
  TorchLean.Tensor.cummax_fst_apply
  TorchLean.Tensor.cummin_fst_apply
  TorchLean.Tensor.sort_fst_apply
  TorchLean.Tensor.topk_fst_apply
  TorchLean.Tensor.topk_snd_apply
  TorchLean.Tensor.cumsumBackward_apply
  TorchLean.Tensor.cumprodBackward_apply
  TorchLean.Tensor.cummaxBackward_apply
  TorchLean.Tensor.cumminBackward_apply
  TorchLean.Tensor.sortBackward_apply
  TorchLean.Tensor.topkBackward_apply
