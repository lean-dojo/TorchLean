/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Module.SpecModule

/-!
# Activation module wrappers

The activation specs in `NN/Spec/Layers/Activation.lean` define the pure tensor functions
(`relu_spec`, `sigmoid_spec`, ...). We wrap them as `NNModuleSpec`s so we can compose them via
`SpecChain` and attach simple export/pretty-print metadata.

Note: activations are shape-preserving, so the `export_func.dimensions` field uses `(0, 0)` as
metadata meaning “not applicable”.

In PyTorch you'd reach for `nn.ReLU()`, `torch.sigmoid`, `torch.tanh`, and `torch.softmax(dim=-1)`;
these wrappers exist for the same reason, just as pure, typed `NNModuleSpec`s.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec
open Activation

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- ReLU as a shape-preserving `NNModuleSpec`. -/
def ReLUModuleSpec {α : Type} [Zero α] [Max α] (s : Shape) : NNModuleSpec α s s :=
{ forward := fun x => Activation.reluSpec x, kind := "ReLU", export_func := {
  toPyTorch := "nn.ReLU()",
  dimensions := (0, 0)  -- ReLU preserves shape
} }

/-- Sigmoid as a shape-preserving `NNModuleSpec`. -/
def SigmoidModuleSpec (s : Shape) : NNModuleSpec α s s :=
{ forward := fun x => Activation.sigmoidSpec x, kind := "Sigmoid", export_func := {
  toPyTorch := "nn.Sigmoid()",
  dimensions := (0, 0)  -- sigmoid preserves shape
} }

/-- Tanh as a shape-preserving `NNModuleSpec`. -/
def TanhModuleSpec (s : Shape) : NNModuleSpec α s s :=
{ forward := fun x => Activation.tanhSpec x, kind := "Tanh", export_func := {
  toPyTorch := "nn.Tanh()",
  dimensions := (0, 0)  -- tanh preserves shape
} }

/-- Softmax along the last axis (matches `Activation.softmax_spec`).

In PyTorch terms: `torch.softmax(x, dim=-1)`. -/
def SoftmaxModuleSpec (s : Shape) : NNModuleSpec α s s :=
{ forward := fun x => Activation.softmaxSpec x, kind := "Softmax", export_func := {
  toPyTorch := "nn.Softmax(dim=-1)",
  dimensions := (0, 0)  -- softmax preserves shape
} }

end Spec
