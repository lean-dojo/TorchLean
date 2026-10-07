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

Each operation has a named lowerer, called directly by the exhaustive `lowerNode` dispatch.
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

/-- The two axis reductions with identical shape and parent validation. -/
inductive AxisReductionKind where
  | sum
  | mean

/-- Convert a lowered axis-reduction case to its IR operation kind. -/
def AxisReductionKind.toOpKind (operation : AxisReductionKind) (axis : Nat) : NN.IR.OpKind :=
  match operation with
  | .sum => .reduceSum axis
  | .mean => .reduceMean axis

/-- Typed denotation of a lowered axis-reduction case. -/
def AxisReductionKind.denote
    {β : Type} [TorchLean.Storage β] [Context β] {shape : Shape}
    (operation : AxisReductionKind) (axis : Nat) (tensor : Tensor β shape)
    (axisValid : Shape.NonemptyAxis axis shape) : Tensor β (Tensor.shapeAfterSum shape axis) :=
  match operation with
  | .sum => Tensor.reduceSum (α := β) (s := shape) axis tensor axisValid
  | .mean => Tensor.reduceMean (α := β) (s := shape) axis tensor axisValid

/-- Diagnostic label retained by checked axis lowering. -/
def AxisReductionKind.label (operation : AxisReductionKind) : String :=
  match operation with
  | .sum => "reduce_sum"
  | .mean => "reduce_mean"

namespace Internal

/-- Checked lowering for `.broadcastTo s₁ s₂`. -/
def lowerBroadcastTo {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (s₁ s₂ : Shape) : NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  match unaryParent? n.parents with
  | some pId =>
      let ip ← parentIdx pId s₁
      if hCan : Spec.Shape.CanBroadcastTo s₁ s₂ then
        if hOut : s₂ = τ then
          let forward := fun ctx : TensorLookup α Γ =>
            let x := ctx.read ip
            hOut ▸ Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) hCan x
          pure <| mkForwardNode (τ := τ) forward
        else
          throw <|
            s!"IRExec: node {i}: broadcastTo outShape mismatch: kind={repr s₂}, " ++
              s!"declared={repr τ}"
      else
        throw s!"IRExec: node {i}: broadcastTo invalid: {repr s₁} → {repr s₂}"
  | _ => throw s!"IRExec: node {i}: broadcastTo expects 1 parent ({n.summary})"

/-- Shared axis-reduction guards: parent node, typed index, nonempty axis, then output shape. -/
@[simp, inline] def lowerAxisReduction {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ)
    (operation : AxisReductionKind) (axis : Nat) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let s := pNode.outShape
      let ip ← parentIdx pId s
      match Spec.Shape.nonemptyAxis? (axis := axis) s with
      | none =>
          throw s!"IRExec: node {i}: {operation.label} invalid axis={axis} for shape {repr s}"
      | some hAxis =>
          let hRed := hAxis.down
          let expected : Shape := TorchLean.Tensor.shapeAfterSum s axis
          if hOut : expected = τ then
            let forward := fun ctx : TensorLookup α Γ =>
              let x := ctx.read ip
              let y : Tensor α expected := operation.denote axis x hRed
              hOut ▸ y
            pure <| mkForwardNode (τ := τ) forward
          else
            throw <|
              s!"IRExec: node {i}: {operation.label} outShape mismatch: " ++
                s!"expected={repr expected}, declared={repr τ} ({n.summary})"
  | _ => throw s!"IRExec: node {i}: {operation.label} expects 1 parent ({n.summary})"

/-- Checked lowering for `.reduceSum axis`. -/
def lowerReduceSum {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis : Nat) : NodeLoweringResult ctx :=
  lowerAxisReduction ctx .sum axis

/-- Checked lowering for `.reduceMean axis`. -/
def lowerReduceMean {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis : Nat) : NodeLoweringResult ctx :=
  lowerAxisReduction ctx .mean axis

/-- Checked lowering for `.sum`. -/
def lowerSum {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let s := pNode.outShape
      let ip ← parentIdx pId s
      if hOut : Shape.scalar = τ then
        let forward := fun ctx : TensorLookup α Γ =>
          let x := ctx.read ip
          hOut ▸ Tensor.scalar (Tensor.sumSpec (α := α) x)
        pure <| mkForwardNode (τ := τ) forward
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
  match binaryParents? n.parents with
  | some (yId, tId) =>
      let yNode ← g.getNode yId
      let tNode ← g.getNode tId
      if _hShape : yNode.outShape = tNode.outShape then
        if hOut : Shape.scalar = τ then
          let s := yNode.outShape
          let iy ← parentIdx yId s
          let it ← parentIdx tId s
          let forward := fun ctx : TensorLookup α Γ =>
            let yhat := ctx.read iy
            let target := ctx.read it
            let diff := Tensor.subSpec (α := α) yhat target
            let sq := Tensor.mulSpec (α := α) diff diff
            let total : α := Tensor.sumSpec (α := α) sq
            let y0 : Tensor α .scalar :=
              Tensor.scalar (total / (↑(TorchLean.Tensor.meanDenominator s) : α))
            Tensor.castShape y0 hOut
          pure <| mkForwardNode (τ := τ) forward
        else
          throw s!"IRExec: node {i}: mse_loss expects scalar outShape ({n.summary})"
      else
        throw <|
          s!"IRExec: node {i}: mse_loss expects equal shapes, got " ++
            s!"{repr yNode.outShape} vs {repr tNode.outShape}"
  | _ => throw s!"IRExec: node {i}: mse_loss expects 2 parents ({n.summary})"

end Internal
end IRExec
end Autograd
end Runtime
