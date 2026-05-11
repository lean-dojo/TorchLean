/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Spec.Core.Utils

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

open Spec
open Tensor
open Shape

open NN.MLTheory.CROWN.Graph

/-- Insert one linear layer's parameters at a given node id. -/
def insertLinearWB
  (nodeId : Nat)
  (p : LinParams Float) (ps : ParamStore Float) : ParamStore Float :=
  { ps with linearWB := ps.linearWB.insert nodeId p }

/--
Build a `ParamStore Float` from a list of linear-layer parameters.

`nodeIdOfIndex` tells us which graph node id corresponds to the i-th layer in the list.
This is the only model-specific decision; the remaining steps are model-agnostic parameter assembly.
-/
def ofLinearStack (nodeIdOfIndex : Nat → Nat) (layers : List (LinParams Float)) : ParamStore Float
  :=
  let rec go : List (LinParams Float) → Nat → ParamStore Float → ParamStore Float
    | [], _, ps => ps
    | p :: rest, idx, ps =>
        go rest (idx + 1) (insertLinearWB (nodeId := nodeIdOfIndex idx) p ps)
  go layers 0 {}

/-- Cast linear parameters from Float to an arbitrary scalar type. -/
def castLinParams {α : Type} [Context α] (ofFloat : Float → α) (p : LinParams Float) : LinParams α
  :=
  { m := p.m
    n := p.n
    w := Spec.mapTensor ofFloat p.w
    b := Spec.mapTensor ofFloat p.b }

/--
Build a `ParamStore α` from Float parameters by casting each tensor entry with `ofFloat`.
-/
def ofLinearStackWith {α : Type} [Context α]
  (ofFloat : Float → α)
  (nodeIdOfIndex : Nat → Nat)
  (layers : List (LinParams Float)) : ParamStore α :=
  let layers' : List (LinParams α) := layers.map (castLinParams (α := α) ofFloat)
  let rec go : List (LinParams α) → Nat → ParamStore α → ParamStore α
    | [], _, ps => ps
    | p :: rest, idx, ps =>
        go rest (idx + 1) ({ ps with linearWB := ps.linearWB.insert (nodeIdOfIndex idx) p })
  go layers' 0 {}

end CROWNParamStore
end Import
