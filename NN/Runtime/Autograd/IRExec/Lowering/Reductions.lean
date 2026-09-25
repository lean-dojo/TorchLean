/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Reduction IR Lowering

Checked lowering for broadcasts, axis reductions, full reductions, and scalar losses.

Each operation has its own small `lower*` definition. `lowerReduction` only dispatches on the
operation kind, and the `lowerReduction_*` equation lemmas let correctness proofs reduce a dispatch
to the branch they care about without unfolding the whole dispatcher.
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

/-- Checked lowering for `.broadcastTo s₁ s₂`. -/
def lowerBroadcastTo {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (s₁ s₂ : Shape) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TensorReader α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId s₁
      if hCan : Spec.Shape.CanBroadcastTo s₁ s₂ then
        if hOut : s₂ = τ then
          let forward := fun ctx : TensorReader α Γ =>
            let x := readTensor (α := α) (xs := ctx) ip
            hOut ▸ Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) hCan x
          pure <| fwd forward
        else
          throw <|
            s!"IRExec: node {i}: broadcastTo outShape mismatch: kind={repr s₂}, " ++
              s!"declared={repr τ}"
      else
        throw s!"IRExec: node {i}: broadcastTo invalid: {repr s₁} → {repr s₂}"
  | _ => throw s!"IRExec: node {i}: broadcastTo expects 1 parent ({n.summary})"

/-- Checked lowering for `.reduceSum axis`. -/
def lowerReduceSum {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis : Nat) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TensorReader α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let s := pNode.outShape
      let ip ← parentIdx pId s
      match Spec.Shape.nonemptyAxis? (axis := axis) s with
      | none =>
          throw s!"IRExec: node {i}: reduce_sum invalid axis={axis} for shape {repr s}"
      | some hAxis =>
          let hRed := hAxis.down
          let expected : Shape := TorchLean.Tensor.shapeAfterSum s axis
          if hOut : expected = τ then
            let forward := fun ctx : TensorReader α Γ =>
              let x := readTensor (α := α) (xs := ctx) ip
              let y : Tensor α expected := Tensor.reduceSum (α := α) (s := s) axis x hRed
              hOut ▸ y
            pure <| fwd forward
          else
            throw <|
              s!"IRExec: node {i}: reduce_sum outShape mismatch: " ++
                s!"expected={repr expected}, declared={repr τ} ({n.summary})"
  | _ => throw s!"IRExec: node {i}: reduce_sum expects 1 parent ({n.summary})"

/-- Checked lowering for `.reduceMean axis`. -/
def lowerReduceMean {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis : Nat) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TensorReader α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let s := pNode.outShape
      let ip ← parentIdx pId s
      match Spec.Shape.nonemptyAxis? (axis := axis) s with
      | none =>
          throw s!"IRExec: node {i}: reduce_mean invalid axis={axis} for shape {repr s}"
      | some hAxis =>
          let hRed := hAxis.down
          let expected : Shape := TorchLean.Tensor.shapeAfterSum s axis
          if hOut : expected = τ then
            let forward := fun ctx : TensorReader α Γ =>
              let x := readTensor (α := α) (xs := ctx) ip
              let y : Tensor α expected := Tensor.reduceMean (α := α) (s := s) axis x hRed
              hOut ▸ y
            pure <| fwd forward
          else
            throw <|
              s!"IRExec: node {i}: reduce_mean outShape mismatch: " ++
                s!"expected={repr expected}, declared={repr τ} ({n.summary})"
  | _ => throw s!"IRExec: node {i}: reduce_mean expects 1 parent ({n.summary})"

/-- Checked lowering for `.sum`. -/
def lowerSum {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TensorReader α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let s := pNode.outShape
      let ip ← parentIdx pId s
      if hOut : Shape.scalar = τ then
        let forward := fun ctx : TensorReader α Γ =>
          let x := readTensor (α := α) (xs := ctx) ip
          hOut ▸ Tensor.scalar (Tensor.sumSpec (α := α) x)
        pure <| fwd forward
      else
        throw s!"IRExec: node {i}: sum expects scalar outShape ({n.summary})"
  | _ => throw s!"IRExec: node {i}: sum expects 1 parent ({n.summary})"

/-- Checked lowering for `.mseLoss`. -/
def lowerMseLoss {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TensorReader α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match binaryParents? n.parents with
  | some (yId, tId) =>
      let yNode ← g.getNode yId
      let tNode ← g.getNode tId
      if _hShape : yNode.outShape = tNode.outShape then
        if hOut : Shape.scalar = τ then
          let s := yNode.outShape
          let iy ← parentIdx yId s
          let it ← parentIdx tId s
          let forward := fun ctx : TensorReader α Γ =>
            let yhat := readTensor (α := α) (xs := ctx) iy
            let target := readTensor (α := α) (xs := ctx) it
            let diff := Tensor.subSpec (α := α) yhat target
            let sq := Tensor.mulSpec (α := α) diff diff
            let total : α := Tensor.sumSpec (α := α) sq
            let y0 : Tensor α .scalar :=
              Tensor.scalar (total / (↑(TorchLean.Tensor.meanDenominator s) : α))
            Tensor.castShape y0 hOut
          pure <| fwd forward
        else
          throw s!"IRExec: node {i}: mse_loss expects scalar outShape ({n.summary})"
      else
        throw <|
          s!"IRExec: node {i}: mse_loss expects equal shapes, got " ++
            s!"{repr yNode.outShape} vs {repr tNode.outShape}"
  | _ => throw s!"IRExec: node {i}: mse_loss expects 2 parents ({n.summary})"

/-- Checked lowering for broadcasts, axis reductions, full reductions, and scalar losses. -/
def lowerReduction {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx :=
  match kind with
  | .broadcastTo s₁ s₂ => lowerBroadcastTo ctx s₁ s₂
  | .reduceSum axis => lowerReduceSum ctx axis
  | .reduceMean axis => lowerReduceMean ctx axis
  | .sum => lowerSum ctx
  | .mseLoss => lowerMseLoss ctx
  | _ => throw s!"IRExec: internal error: operation routed to lowerReduction"

variable {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape}

/-- Dispatch equation for `.broadcastTo s₁ s₂`. -/
@[simp] theorem lowerReduction_broadcastTo (ctx : NodeLoweringContext α Γ) (s₁ s₂ : Shape) :
    lowerReduction ctx (.broadcastTo s₁ s₂) = lowerBroadcastTo ctx s₁ s₂ := rfl

/-- Dispatch equation for `.reduceSum axis`. -/
@[simp] theorem lowerReduction_reduceSum (ctx : NodeLoweringContext α Γ) (axis : Nat) :
    lowerReduction ctx (.reduceSum axis) = lowerReduceSum ctx axis := rfl

/-- Dispatch equation for `.reduceMean axis`. -/
@[simp] theorem lowerReduction_reduceMean (ctx : NodeLoweringContext α Γ) (axis : Nat) :
    lowerReduction ctx (.reduceMean axis) = lowerReduceMean ctx axis := rfl

/-- Dispatch equation for `.sum`. -/
@[simp] theorem lowerReduction_sum (ctx : NodeLoweringContext α Γ) :
    lowerReduction ctx .sum = lowerSum ctx := rfl

/-- Dispatch equation for `.mseLoss`. -/
@[simp] theorem lowerReduction_mseLoss (ctx : NodeLoweringContext α Γ) :
    lowerReduction ctx .mseLoss = lowerMseLoss ctx := rfl

end Internal
end IRExec
end Autograd
end Runtime
