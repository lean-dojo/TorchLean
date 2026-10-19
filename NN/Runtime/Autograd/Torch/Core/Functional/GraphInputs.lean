/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Functional.Curried
public import NN.Runtime.Autograd.TypedGraph.GraphM.Core

/-!
# Typed Graph Input Binding

Bind curried model arguments to graph variables.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

namespace CurriedRef

/-- Apply a curried reference function to graph variables in shape order. -/
def applyVarList {Γ : List Shape} {β : Type} :
    CurriedRef (fun s => TypedGraph.GraphM.Var s) Γ β →
    TypedGraph.GraphM.VarList Γ → β
  | f, .nil => f
  | f, .cons v vs => applyVarList (Γ := _) (β := β) (f v) vs

end CurriedRef

end Runtime.Autograd.Torch
