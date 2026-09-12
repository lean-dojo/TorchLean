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

Bind curried model arguments to graph variables and projections from runtime tensor packs.
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

/--
Apply a curried function to the coordinate projections of a tensor pack.

Each argument reads its value from the complete runtime pack.
-/
def applyPackProjections {α : Type} [Storage α] {β : Type} {full : List Shape} :
    {rest : List Shape} →
    (TensorPack α full → TensorPack α rest) →
    CurriedRef (fun s => TensorPack α full → Tensor α s) rest β → β
  | [], _drop, f => f
  | _s :: ss, drop, f =>
      let head : TensorPack α full → Tensor α _ := fun xs =>
        match drop xs with
        | .cons x _ => x
      let tail : TensorPack α full → TensorPack α ss := fun xs =>
        match drop xs with
        | .cons _ rest => rest
      applyPackProjections (rest := ss) tail (f head)

end CurriedRef

end Runtime.Autograd.Torch
