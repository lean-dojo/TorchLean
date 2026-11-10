/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.AdjointOp
public import TorchLean.NN.Tape.Graph

/-!
# Lawful nodes

Constructors for algebraic tape nodes: a unary node for any `AdjointOp` and the binary addition
node `LawfulNode.add`, which pairs the executable `NodeData.add` of
`TorchLean.NN.TypedGraph.Data` with its local law. Both read their parents through
typed context indices (`Idx`) and scatter their VJPs into the context with `TensorPack.single`. A
graph built from such nodes lowers to a runtime tape through
`Tape.GraphData.toTape` of `TorchLean.NN.TypedGraph`.
-/

@[expose] public section

namespace TorchLean
namespace Tape

open TorchLean.Tensor
open Rep (dot)

noncomputable section

namespace LawfulNode

open TensorPack (getIdx)

/-- Build a proof-carrying unary node from an `AdjointOp`. -/
def ofAdjointOp {α : Type} {Δ : Type} [Storage α] [CommSemiring α]
    {Γ : List Shape} {σ τ : Shape}
    (idx : Idx Γ σ) (op : AdjointOp (α := α) σ τ) : LawfulNode (α := α) (Δ := Δ) Γ τ :=
  { toNodeData :=
      { forward := fun ctx _d => op.op.forward (getIdx (xs := ctx) idx)
        jvp := fun ctx dctx _d => op.jvp (getIdx (xs := ctx) idx) (getIdx (xs := dctx) idx)
        vjp := fun ctx _d δ =>
          TensorPack.single (α := α) (Γ := Γ) idx (op.op.backward (getIdx (xs := ctx) idx) δ) }
    jvp_vjp_adjoint := by
      intro ctx dctx d δ
      change Rep.dot (op.jvp (getIdx ctx idx) (getIdx dctx idx)) δ =
        TensorPack.dotList dctx (TensorPack.single idx (op.op.backward (getIdx ctx idx) δ))
      rw [TensorPack.dotList_single]
      exact op.isAdjointPair _ _ δ }

/-- Proof-carrying binary add node (two parents of the same shape). -/
def add {α : Type} {Δ : Type} [Storage α] [CommSemiring α]
    {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    LawfulNode (α := α) (Δ := Δ) Γ s :=
  { toNodeData := NodeData.add (α := α) (Δ := Δ) (Γ := Γ) (s := s) a b
    jvp_vjp_adjoint := by
      intro ctx dctx d δ
      change Rep.dot (getIdx dctx a + getIdx dctx b) δ =
        TensorPack.dotList dctx
          (TensorPack.add (TensorPack.single a δ) (TensorPack.single b δ))
      rw [Tensor.dot_add_left, TensorPack.dotList_add_right,
        TensorPack.dotList_single, TensorPack.dotList_single] }

end LawfulNode

end
end Tape
end TorchLean
