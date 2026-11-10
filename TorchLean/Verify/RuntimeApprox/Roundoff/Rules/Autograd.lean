/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Tape.SavedBackward
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Floats.Formats.BinaryInterchange.Automation.Roundoff.Extension

/-!
# Indexed forward and saved reverse execution for `roundoff`

This module registers the two compiler replacements of the autograd proofs as `roundoff` rules:
`eval` is `evalWithArray` and `backpropCtx` is `backpropCtxSaved`. It also registers the lemmas
saying that converting tensor and gradient contexts to tensor packs commutes with their operations.
The analysis then reads indexed forward and saved reverse execution through their source programs,
including the original gradient-accumulation order.
-/

public section

attribute [roundoff_rule]
  TorchLean.Tape.GraphData.eval_eq_array
  TorchLean.Tape.GraphData.backpropCtx_eq_saved
  TorchLean.Tape.TensorContext.toPack_ofPack
  TorchLean.Tape.TensorContext.toPack_cast
  TorchLean.Tape.TensorContext.toPack_push
  TorchLean.Tape.SavedGraph.toPack_backpropStored
  TorchLean.Tape.GradientContext.toPack_ofPack
  TorchLean.Tape.GradientContext.toPack_cast
  TorchLean.Tape.GradientContext.toPack_addDense
  TorchLean.Tape.GradientContext.toPack_addContributions
  TorchLean.Tape.GradientContext.toPack_pop
