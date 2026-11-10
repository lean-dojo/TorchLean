/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Movement
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# General data movement for `roundoff`

The read lemmas of `broadcast`, `permute`, `reshape`, `slice`, `pad`, `flip`, `indices` and
`gather` are `roundoff` rules. Each one names the input coordinate that an output coordinate
reads, so the analysis reads through these operators at any axes of any shape without unfolding
them, and they add no error. Outside the input, `pad` reads its padding value, and `roundoff`
analyzes the two branches. The read of `cat` is registered with the tensors, in
`TorchLean.Verify.RuntimeApprox.Roundoff`.
-/

public section

attribute [roundoff_rule]
  TorchLean.Tensor.broadcast_apply
  TorchLean.Tensor.permute_apply
  TorchLean.Tensor.reshape_apply
  TorchLean.Tensor.slice_apply
  TorchLean.Tensor.pad_apply
  TorchLean.Tensor.flip_apply
  TorchLean.Tensor.indices_apply
  TorchLean.Tensor.gather_apply
