/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Linear
public import NN.Spec.Module.Core

/-!
# Linear modules

The mathematical forward map comes from `Spec.linearSpec`. This file packages it as a
shape-indexed `Spec.Module` for composition and source export.
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
/-- A linear layer as a mathematical module. -/
def linear {inDim outDim : Nat}
  (m : Spec.LinearSpec α inDim outDim) :
  Spec.Module α ([inDim]) ([outDim]) :=
{
  forward := Spec.linearSpec (α := α) m
  kind := "Linear",
  pythonExpr := s!"nn.Linear({inDim}, {outDim})"
}

end Spec.Module
