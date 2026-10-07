/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Core.SemiringCorrectness
public import NN.Proofs.Autograd.Tape.Algebra.Soundness

/-!
# Nodes

Convenience constructors for algebraic tape nodes/graphs.

Build an SSA/DAG graph from local nodes, then lower it to a runtime tape through
`NN.Proofs.Autograd.Runtime.Link`.

The nodes here are a unary adapter for any `OpSpecCorrect` and a binary addition node. Both read
their parents through typed context indices (`Idx`) and scatter their VJPs into the context with
`TensorPack.single`.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec TorchLean
open TorchLean TorchLean.Tensor
open TensorAlgebra

noncomputable section

namespace NodeData

/-- Executable binary add node (two parents of the same shape). -/
def add {α : Type} {Δ : Type} [TorchLean.Storage α] [Zero α] [Add α]
    {Γ : List Shape} {s : Shape}
    (a b : Idx Γ s) : NodeData α Δ Γ s :=
  { forward := fun ctx _d => addSpec (getIdx (xs := ctx) a) (getIdx (xs := ctx) b)
    jvp := fun _ctx dctx _d => addSpec (getIdx (xs := dctx) a) (getIdx (xs := dctx) b)
    vjp := fun _ctx _d δ =>
      TorchLean.TensorPack.add (α := α) (ss := Γ)
        (TensorPack.single (α := α) (Γ := Γ) a δ)
        (TensorPack.single (α := α) (Γ := Γ) b δ) }

end NodeData

namespace Node

/-- Build a proof-carrying unary node from an `OpSpecCorrect`. -/
def ofOpSpecCorrect {α : Type} {Δ : Type} [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {σ τ : Shape}
    (idx : Idx Γ σ) (op : OpSpecCorrect (α := α) σ τ) : Node (α := α) (Δ := Δ) Γ τ :=
  { toNodeData :=
      { forward := fun ctx _d => op.op.forward (getIdx (xs := ctx) idx)
        jvp := fun ctx dctx _d => op.jvp (getIdx (xs := ctx) idx) (getIdx (xs := dctx) idx)
        vjp := fun ctx _d δ =>
          TensorPack.single (α := α) (Γ := Γ) idx (op.op.backward (getIdx (xs := ctx) idx) δ) }
    correct := by
      intro ctx dctx d δ
      change dot (op.jvp (getIdx ctx idx) (getIdx dctx idx)) δ =
        TensorPack.dotList dctx (TensorPack.single idx (op.op.backward (getIdx ctx idx) δ))
      rw [TensorPack.dotList_single]
      exact op.correct _ _ δ }

/-- Proof-carrying binary add node (two parents of the same shape). -/
def add {α : Type} {Δ : Type} [TorchLean.Storage α] [CommSemiring α]
    {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    Node (α := α) (Δ := Δ) Γ s :=
  { toNodeData := NodeData.add (α := α) (Δ := Δ) (Γ := Γ) (s := s) a b
    correct := by
      intro ctx dctx d δ
      change dot (addSpec (getIdx dctx a) (getIdx dctx b)) δ =
        TensorPack.dotList dctx
          (TorchLean.TensorPack.add (TensorPack.single a δ) (TensorPack.single b δ))
      rw [TensorAlgebra.dot_add_left, TensorPack.dotList_add_right,
        TensorPack.dotList_single, TensorPack.dotList_single] }

end Node

end
end Algebra
end Autograd
end Proofs
