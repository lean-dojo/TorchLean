/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Basic and Random IR Lowering

Checked lowering for graph inputs, constants, detachment, and random operations.

Each operation has its own small `lower*` definition. `lowerBasic` only dispatches on the operation
kind, and the `lowerBasic_*` equation lemmas let correctness proofs reduce a dispatch to the branch
they care about without unfolding the whole dispatcher.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)
open NN.IR

namespace Internal

/-- A second `.input` node cannot appear after node 0; the lowering loop rejects it. -/
def lowerInput {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx :=
  throw s!"IRExec: internal error (handled above)"

/-- Checked lowering for `.const s`: read the payload tensor and retag it at the declared shape. -/
def lowerConst {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (s : Shape) : NodeLoweringResult ctx := do
  let payload := ctx.payload
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  let t ← NN.IR.Graph.evalConst (α := α) (payload := payload) (id := n.id) (s := s)
  if hOut : s = τ then
    pure <| fwd (fun _ctx => hOut ▸ t)
  else
    throw s!"IRExec: const node {i}: outShape mismatch: kind={repr s}, declared={repr τ}"

/-- Checked lowering for `.detach`, including scalar tangent removal for dual-valued execution. -/
def lowerDetach {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let s := pNode.outShape
      let ip ← parentIdx pId s
      if hOut : s = τ then
        let forward := fun ctx : TorchLean.TensorPack α Γ =>
          hOut ▸ Tensor.detachSpec (getIdx (α := α) (xs := ctx) ip)
        pure <| fwd forward
      else
        throw s!"IRExec: node {i}: detach expects outShape=parent.outShape ({n.summary})"
  | _ => throw s!"IRExec: node {i}: detach expects 1 parent ({n.summary})"

/-- Checked lowering for `.randUniform seed`: a deterministic tensor keyed by seed and node id. -/
def lowerRandUniform {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (seed : Nat) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match n.parents.isEmpty with
  | true =>
      let key := Spec.Random.keyOf seed i
      let t : Tensor α τ := Spec.Random.uniform (α := α) key (s := τ)
      pure <| fwd (fun _ctx => t)
  | _ => throw s!"IRExec: node {i}: rand_uniform expects 0 parents ({n.summary})"

/-- Checked lowering for `.bernoulliMask seed`: a keyed mask with a scalar keep probability. -/
def lowerBernoulliMask {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (seed : Nat) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId Shape.scalar
      let key := Spec.Random.keyOf seed i
      let forward := fun ctx : TorchLean.TensorPack α Γ =>
        let kpT := getIdx (α := α) (xs := ctx) ip
        let kp : α := kpT.item
        Spec.Random.mask (α := α) key kp (s := τ)
      pure <| fwd forward
  | _ => throw s!"IRExec: node {i}: bernoulli_mask expects 1 parent ({n.summary})"

/-- Checked lowering for graph inputs, constants, detachment, and random operations. -/
def lowerBasic {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx :=
  match kind with
  | .input => lowerInput ctx
  | .const s => lowerConst ctx s
  | .detach => lowerDetach ctx
  | .randUniform seed => lowerRandUniform ctx seed
  | .bernoulliMask seed => lowerBernoulliMask ctx seed
  | _ => throw s!"IRExec: internal error: operation routed to lowerBasic"

variable {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape}

/-- Dispatch equation for `.input`. -/
@[simp] theorem lowerBasic_input (ctx : NodeLoweringContext α Γ) :
    lowerBasic ctx .input = lowerInput ctx := rfl

/-- Dispatch equation for `.const s`. -/
@[simp] theorem lowerBasic_const (ctx : NodeLoweringContext α Γ) (s : Shape) :
    lowerBasic ctx (.const s) = lowerConst ctx s := rfl

/-- Dispatch equation for `.detach`. -/
@[simp] theorem lowerBasic_detach (ctx : NodeLoweringContext α Γ) :
    lowerBasic ctx .detach = lowerDetach ctx := rfl

/-- Dispatch equation for `.randUniform seed`. -/
@[simp] theorem lowerBasic_randUniform (ctx : NodeLoweringContext α Γ) (seed : Nat) :
    lowerBasic ctx (.randUniform seed) = lowerRandUniform ctx seed := rfl

/-- Dispatch equation for `.bernoulliMask seed`. -/
@[simp] theorem lowerBasic_bernoulliMask (ctx : NodeLoweringContext α Γ) (seed : Nat) :
    lowerBasic ctx (.bernoulliMask seed) = lowerBernoulliMask ctx seed := rfl

end Internal
end IRExec
end Autograd
end Runtime
