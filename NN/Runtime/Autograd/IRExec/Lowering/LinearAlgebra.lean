/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Runtime.Autograd.IRExec.Lowering.Common
public import NN.IR.Semantics

/-!
# Linear Algebra IR Lowering

Checked lowering for matrix multiplication and payload-backed linear layers.

Both operations accept the same shapes as the IR semantics: any shared leading shape followed by
the matrix axes. `lowerMatmul` obtains the output shape from `OpContracts.inferMatmulOutShape`, the
contract shared with shape inference and evaluation, and only decomposes the parent shapes itself
to build typed indices. The closures apply `NN.IR.Graph.matmulLeading` and
`NN.IR.Graph.linearLeading`, the same typed operators the IR evaluator uses.

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
Checked lowering for `.matmul` over any shared leading shape.

The final two axes follow the matrix rule `(... × m × n) · (... × n × p) → (... × m × p)`. The
output shape is taken from `OpContracts.inferMatmulOutShape`; the typed decomposition below is
checked against it so the contract remains the single source of truth.
-/
def lowerMatmul {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) : NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match binaryParents? n.parents with
  | some (aId, bId) =>
      let aNode ← g.getNode aId
      let bNode ← g.getNode bId
      let expected ← OpContracts.inferMatmulOutShape aNode.outShape bNode.outShape
      match aNode.outShape.toList.reverse, bNode.outShape.toList.reverse with
      | inner :: rows :: leadingRev, cols :: inner' :: leadingRev' =>
          if _hLeading : leadingRev = leadingRev' then
            if _hInner : inner = inner' then
              let leading : Shape := Shape.ofList leadingRev.reverse
              let ia ← parentIdx aId (leading.concat [rows, inner])
              let ib ← parentIdx bId (leading.concat [inner, cols])
              -- Both guards compare fully spelled out shapes. A shape bound by a local `let` can
              -- leave instance search for `Decidable (s = t)` stuck, and the correctness proofs
              -- case on the `dite` terms these produce.
              if hExpected : leading.concat [rows, cols] = expected then
                if hOut : expected = n.outShape then
                  let forward := fun ctx : TorchLean.TensorPack α Γ =>
                    let aT := getIdx (α := α) (xs := ctx) ia
                    let bT := getIdx (α := α) (xs := ctx) ib
                    Tensor.castShape (NN.IR.Graph.matmulLeading leading aT bT)
                      (hExpected.trans hOut)
                  pure <| fwd forward
                else
                  throw <|
                    s!"IRExec: node {i}: matmul outShape mismatch: " ++
                      s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              else
                throw <|
                  s!"IRExec: node {i}: matmul internal error: contract shape {repr expected} " ++
                    s!"differs from typed shape {repr (leading.concat [rows, cols])} ({n.summary})"
            else
              throw s!"IRExec: node {i}: matmul inner dims mismatch: {inner} vs {inner'}"
          else
            throw <|
              s!"IRExec: node {i}: matmul leading dimensions mismatch: " ++
                s!"{repr aNode.outShape} vs {repr bNode.outShape}"
      | _, _ =>
          throw <|
            s!"IRExec: node {i}: matmul expects rank≥2 inputs, got {repr aNode.outShape} " ++
              s!"and {repr bNode.outShape}"
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
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
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
              let forward := fun ctx : TorchLean.TensorPack α Γ =>
                let xIn : Tensor α expectedIn :=
                  Tensor.castShape (getIdx (α := α) (xs := ctx) ix) hIn
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
