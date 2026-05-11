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

This is the "approach (a)" authoring layer: you build an SSA/DAG graph out of local nodes,
then compile it to a runtime tape via `NN/Proofs/Autograd/Runtime/Link.lean`.

This file focuses on *unary* nodes that depend on a single context entry (an `Idx`).
That is enough to build many fixed-parameter inference graphs (e.g. MLP forward + input gradients).
Extending to multi-parent nodes (e.g. weight gradients) is intended, but left incremental.
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor
open TensorAlgebra

noncomputable section

namespace NodeData

/-- Build an executable unary node from a spec `OpSpec`, storing the VJP in a sparse `TList`. -/
def ofOpSpec {α : Type} {Δ : Type} [Zero α] {Γ : List Shape} {σ τ : Shape}
    (idx : Idx Γ σ) (op : Spec.OpSpec α σ τ) : NodeData α Δ Γ τ :=
  { forward := fun ctx _d => op.forward (getIdx (xs := ctx) idx)
    -- JVP is unused by the proof-compiled runtime path; the zero tangent keeps this executable
    -- node total while VJP carries the proof-compiled runtime behavior.
    jvp := fun _ctx _dctx _d => Spec.fill (0 : α) τ
    vjp := fun ctx _d δ => TList.single (α := α) (Γ := Γ) idx (op.backward (getIdx (xs := ctx) idx)
      δ) }

/-- Executable binary add node (two parents of the same shape). -/
def add {α : Type} {Δ : Type} [Zero α] [Add α] {Γ : List Shape} {s : Shape}
    (a b : Idx Γ s) : NodeData α Δ Γ s :=
  { forward := fun ctx _d => addSpec (getIdx (xs := ctx) a) (getIdx (xs := ctx) b)
    jvp := fun _ctx dctx _d => addSpec (getIdx (xs := dctx) a) (getIdx (xs := dctx) b)
    vjp := fun _ctx _d δ =>
      TList.add (α := α) (ss := Γ)
        (TList.single (α := α) (Γ := Γ) a δ)
        (TList.single (α := α) (Γ := Γ) b δ) }

end NodeData

namespace Node

/-- Build a proof-carrying unary node from an `OpSpecCorrect`. -/
def ofOpSpecCorrect {α : Type} {Δ : Type} [CommSemiring α] {Γ : List Shape} {σ τ : Shape}
    (idx : Idx Γ σ) (op : OpSpecCorrect (α := α) σ τ) : Node (α := α) (Δ := Δ) Γ τ :=
  { toNodeData :=
      { forward := fun ctx _d => op.op.forward (getIdx (xs := ctx) idx)
        jvp := fun ctx dctx _d => op.jvp (getIdx (xs := ctx) idx) (getIdx (xs := dctx) idx)
        vjp := fun ctx _d δ => TList.single (α := α) (Γ := Γ) idx (op.op.backward (getIdx (xs :=
          ctx) idx) δ) }
    correct := by
      intro ctx dctx d δ
      -- Reduce to the per-op adjointness law and the `TList.single` dot lemma.
      let x := getIdx (xs := ctx) idx
      let dx := getIdx (xs := dctx) idx
      have hop := op.correct x dx δ
      have hsingle :
          dot (α := α) dx (op.op.backward x δ) =
            TList.dotList (α := α) dctx (TList.single (α := α) (Γ := Γ) idx (op.op.backward x δ)) :=
              by
        simpa using (TList.dotList_single (α := α) (Γ := Γ) (dx := dctx) (idx := idx)
          (v := op.op.backward x δ)).symm
      -- `hop` gives `dot (jvp ...) δ = dot dx (backward ...)`.
      -- Rewrite the RHS into the `TList.dotList` form.
      simpa [x, dx] using hop.trans hsingle }

/-- Proof-carrying binary add node (two parents of the same shape). -/
def add {α : Type} {Δ : Type} [CommSemiring α] {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    Node (α := α) (Δ := Δ) Γ s :=
  { toNodeData := NodeData.add (α := α) (Δ := Δ) (Γ := Γ) (s := s) a b
    correct := by
      intro ctx dctx d δ
      -- Reduce to dot distribution and the fact that `TList.single` is the adjoint of `getIdx`.
      let da := getIdx (xs := dctx) a
      let db := getIdx (xs := dctx) b
      have hsplit :
          dot (α := α) (addSpec da db) δ = dot (α := α) da δ + dot (α := α) db δ := by
        simpa [da, db] using TensorAlgebra.dot_add_left (α := α) (a := da) (b := db) (c := δ)
      have hsingleA :
          TList.dotList (α := α) dctx (TList.single (α := α) (Γ := Γ) a δ) = dot (α := α) da δ := by
        simpa [da] using (TList.dotList_single (α := α) (Γ := Γ) (dx := dctx) (idx := a) (v := δ))
      have hsingleB :
          TList.dotList (α := α) dctx (TList.single (α := α) (Γ := Γ) b δ) = dot (α := α) db δ := by
        simpa [db] using (TList.dotList_single (α := α) (Γ := Γ) (dx := dctx) (idx := b) (v := δ))
      have hadd :
          TList.dotList (α := α) dctx
              (TList.add (α := α) (ss := Γ)
                (TList.single (α := α) (Γ := Γ) a δ)
                (TList.single (α := α) (Γ := Γ) b δ))
            =
          TList.dotList (α := α) dctx (TList.single (α := α) (Γ := Γ) a δ) +
            TList.dotList (α := α) dctx (TList.single (α := α) (Γ := Γ) b δ) := by
        simpa using
          (TList.dotList_add_right (α := α) (ss := Γ) (x := dctx)
            (y := TList.single (α := α) (Γ := Γ) a δ)
            (z := TList.single (α := α) (Γ := Γ) b δ))
      -- Finish.
      calc
        dot (α := α) (NodeData.add (α := α) (Δ := Δ) (Γ := Γ) (s := s) a b |>.jvp ctx dctx d) δ
            = dot (α := α) (addSpec da db) δ := by
                simp [NodeData.add, da, db]
        _ = dot (α := α) da δ + dot (α := α) db δ := hsplit
        _ = TList.dotList (α := α) dctx (TList.single (α := α) (Γ := Γ) a δ) +
              TList.dotList (α := α) dctx (TList.single (α := α) (Γ := Γ) b δ) := by
                simp [hsingleA, hsingleB]
        _ = TList.dotList (α := α) dctx
              (NodeData.add (α := α) (Δ := Δ) (Γ := Γ) (s := s) a b |>.vjp ctx d δ) := by
                simp [NodeData.add, hadd] }

end Node

namespace GraphData

variable {α : Type}
variable {Δ : Type}
variable {Γ : List Shape}

/-- Append a unary node built from an `OpSpec` (executable-only). -/
def snocOpSpec [Zero α] {ss : List Shape} {σ τ : Shape}
    (g : GraphData α Δ Γ ss) (idx : Idx (Γ ++ ss) σ) (op : Spec.OpSpec α σ τ) :
    GraphData α Δ Γ (ss ++ [τ]) :=
  GraphData.snoc g (NodeData.ofOpSpec (α := α) (Δ := Δ) (Γ := Γ ++ ss) idx op)

end GraphData

namespace Graph

variable {α : Type} [CommSemiring α]
variable {Δ : Type}
variable {Γ : List Shape}

/-- Append a unary node built from an `OpSpecCorrect` (proof-carrying). -/
def snocOpSpecCorrect {ss : List Shape} {σ τ : Shape}
    (g : Graph (α := α) Δ Γ ss) (idx : Idx (Γ ++ ss) σ) (op : OpSpecCorrect (α := α) σ τ) :
    Graph (α := α) Δ Γ (ss ++ [τ]) :=
  Graph.snoc g (Node.ofOpSpecCorrect (α := α) (Δ := Δ) (Γ := Γ ++ ss) idx op)

end Graph

end
end Algebra
end Autograd
end Proofs
