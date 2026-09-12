/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.ToDAG.Core

/-!
# Primitive Embedding

GraphSpec’s DAG language (`NN.GraphSpec.DAG`) is the “general graph” IR. To avoid duplicating the
operator semantics, unary sequential primitives

`p : Primitive ps σ τ`

are embedded into DAG primitive ops

`LowerToDAG.Primitive.toDAGPrimOp p : DAG.PrimOp (ps ++ [σ]) τ`.

This file proves the bookkeeping theorem that keeps the “no duplicated semantics” contract explicit:
if you take a sequential primitive, embed it into DAG form, and feed it the obvious argument list
`params ++ [x]`, you get exactly the same pure forward computation.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Primitive

open Spec TorchLean
open TorchLean.Tensor

/--
`splitAppend` undoes `append` in the one-input case.

This typed-list fact is used internally by `LowerToDAG.Primitive.toDAGPrimOp`.
-/
theorem splitAppend_appendSingleton
    {α : Type} [TorchLean.Storage α] [Context α] :
    {ps : List Shape} → {σ : Shape} →
      (params : TorchLean.TensorPack α ps) → (x : TorchLean.Tensor α σ) →
        TorchLean.TensorPack.split (α := α)
            (ss₁ := ps) (ss₂ := [σ])
            (TorchLean.TensorPack.append (α := α)
              (ss₁ := ps) (ss₂ := [σ]) params (.cons x .nil))
          =
        (params, .cons x .nil)
  | ps, σ, params, x => by
      simp

/--
Embedding a sequential primitive into DAG form preserves its pure `specFwd` semantics.

This theorem states that “the DAG primitive is the sequential primitive with
its parameters made explicit as ordinary inputs”.
-/
theorem toDAGPrimOp_specFwd_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {ps : List Shape} {σ τ : Shape}
    (p : Primitive ps σ τ)
    (params : TorchLean.TensorPack α ps) (x : TorchLean.Tensor α σ) :
    (LowerToDAG.Primitive.toDAGPrimOp (ps := ps) (σ := σ) (τ := τ) p).specFwd (α := α)
        (TorchLean.TensorPack.append (α := α)
          (ss₁ := ps) (ss₂ := [σ]) params (.cons x .nil))
    =
    p.specFwd (α := α) params x := by
  -- Unfold the embedding and use `splitAppend_appendSingleton` to simplify the `splitAppend`.
  simp
    [ LowerToDAG.Primitive.toDAGPrimOp
    , splitAppend_appendSingleton (α := α) (ps := ps) (σ := σ) params x
    ]

end Primitive
end GraphSpec
end NN
