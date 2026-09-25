/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Linear Algebra IR Lowering

Checked lowering for matrix multiplication and payload-backed linear layers.

Both operations accept the same shapes as the IR semantics. `lowerMatmul` uses the checked
`OpContracts.matmulDims` layout for vector promotion and batch broadcasting, then applies
`NN.IR.Graph.matmulWithDims`. Linear layers preserve their leading shape and apply
`NN.IR.Graph.linearLeading`.

Each operation has its own small `lower*` definition. `lowerLinearAlgebra` only dispatches on the
operation kind, and the `lowerLinearAlgebra_*` equation lemmas let correctness proofs reduce a
dispatch to the branch they care about without unfolding the whole dispatcher.
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

/--
Checked lowering for `.matmul` with vector promotion and batch broadcasting.

The shape contract supplies both typed operand shapes and the output shape. The closure calls
the same evaluator as reference IR execution, including its equal-batch typed matrix kernel.
-/
def lowerMatmul {α : Type} [TorchLean.Storage α] [Context α]
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
  | some (aId, bId) =>
      let aNode ← g.getNode aId
      let bNode ← g.getNode bId
      let dims ← OpContracts.matmulDims aNode.outShape bNode.outShape
      let ia ← parentIdx aId dims.leftShape
      let ib ← parentIdx bId dims.rightShape
      if hOut : dims.outShape = n.outShape then
        let forward := fun ctx : TensorReader α Γ =>
          Tensor.castShape
            (NN.IR.Graph.matmulWithDims dims
              (readTensor (α := α) (xs := ctx) ia)
              (readTensor (α := α) (xs := ctx) ib)) hOut
        pure <| fwd forward
      else
          throw <|
            s!"IRExec: node {i}: matmul outShape mismatch: " ++
              s!"expected={repr dims.outShape}, declared={repr τ} ({n.summary})"
  | _ => throw s!"IRExec: node {i}: matmul expects 2 parents ({n.summary})"

/--
Checked lowering for `.linear` over any leading shape.

The final input axis must equal the payload's `inDim`; every leading axis is preserved and the
affine map `y = Wx + b` is applied independently at each leading index.
-/
def lowerLinear {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let payload := ctx.payload
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TensorReader α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some xId =>
      match payload.linear? n.id with
      | none => throw s!"IRExec: missing linear payload for node {n.id}"
      | some p =>
          let xNode ← g.getNode xId
          let xShape : Shape := xNode.outShape
          let ix ← parentIdx xId xShape
          let leading : Shape := Shape.ofList xShape.toList.dropLast
          let expectedIn : Shape := leading.concat [p.inDim]
          let expectedOut : Shape := leading.concat [p.outDim]
          -- Both guards compare fully spelled out shapes. A shape bound by a local `let` can
          -- leave instance search for `Decidable (s = t)` stuck, and the correctness proofs case
          -- on the `dite` terms these produce.
          if hIn : xNode.outShape = leading.concat [p.inDim] then
            if hOut : leading.concat [p.outDim] = n.outShape then
              let forward := fun ctx : TensorReader α Γ =>
                let xIn : Tensor α expectedIn :=
                  Tensor.castShape (readTensor (α := α) (xs := ctx) ix) hIn
                let y : Tensor α expectedOut := NN.IR.Graph.linearLeading leading p.W p.b xIn
                Tensor.castShape y hOut
              pure <| fwd forward
            else
              throw <|
                s!"IRExec: linear {n.id}: declared outShape mismatch: {repr τ} vs " ++
                  s!"expected {repr expectedOut}"
          else
            throw <|
              s!"IRExec: linear {n.id}: parent shape {repr xShape} does not end in " ++
                s!"inDim={p.inDim}"
  | _ => throw s!"IRExec: node {i}: linear expects 1 parent ({n.summary})"

/-- Checked lowering for matrix multiplication and payload-backed linear layers. -/
def lowerLinearAlgebra {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx :=
  match kind with
  | .matmul => lowerMatmul ctx
  | .linear => lowerLinear ctx
  | _ => throw s!"IRExec: internal error: operation routed to lowerLinearAlgebra"

variable {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape}

/-- Dispatch equation for `.matmul`. -/
@[simp] theorem lowerLinearAlgebra_matmul (ctx : NodeLoweringContext α Γ) :
    lowerLinearAlgebra ctx .matmul = lowerMatmul ctx := rfl

/-- Dispatch equation for `.linear`. -/
@[simp] theorem lowerLinearAlgebra_linear (ctx : NodeLoweringContext α Γ) :
    lowerLinearAlgebra ctx .linear = lowerLinear ctx := rfl

end Internal
end IRExec
end Autograd
end Runtime
