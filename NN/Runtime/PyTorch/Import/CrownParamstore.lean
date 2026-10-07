/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# CrownParamstore

PyTorch → CROWN `ParamStore` helpers.

This module is *not* about JSON parsing; it is about what we do **after** we have already loaded
weights into typed Lean tensors.

Why this exists:

- PyTorch “weights” are keyed by module names (`state_dict` keys).
- TorchLean’s graph backend stores parameters by **node id** in
  `NN.MLTheory.CROWN.Graph.ParamStore`.

So any real bridge needs a small amount of “wiring code” that:

1. chooses a node-id scheme (model-specific),
2. inserts the corresponding `(W,b)` tensors into the right slots.

These helpers keep `NN.Runtime.PyTorch.Import.Core` focused on JSON,
and so model example loaders can share the same ParamStore-building utilities.
-/

@[expose] public section


namespace Import
namespace CROWNParamStore

open Spec TorchLean

open NN.MLTheory.CROWN.Graph

/-- Build a graph parameter store from a sequence of layer descriptions.

`nodeId` assigns graph slots; `parameter` converts each description to typed weights and bias.
Layers are inserted in array order, so the last occurrence of a node id wins.
-/
def ofArray {α β : Type} [Storage α] [Context α]
    (nodeId : Nat → Nat) (layers : Array β) (parameter : β → LinParams α) :
    ParamStore α :=
  layers.zipIdx.foldl (fun store layer =>
    { store with linearWB := store.linearWB.insert (nodeId layer.2) (parameter layer.1) }) {}

/-- Convert every weight and bias scalar, preserving the linear layer's dimensions. -/
def map {α β : Type} [Storage α] [Context α] [Storage β] [Context β]
    (f : α → β) (parameters : LinParams α) : LinParams β :=
  { m := parameters.m
    n := parameters.n
    w := Tensor.map f parameters.w
    b := Tensor.map f parameters.b }

end CROWNParamStore
end Import
