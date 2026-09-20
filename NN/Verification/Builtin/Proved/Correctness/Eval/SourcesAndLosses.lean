/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Source Nodes, Detach, and Loss Evaluation

Local semantics for source nodes, detachment, and scalar losses that appear in imported or lowered
IR graphs.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- A one-node graph containing only an input node. -/
def inputGraph (s : Shape) : Graph :=
  { nodes := #[{ id := 0, parents := #[], kind := .input, outShape := s }] }

/-- Local IR semantics for an input node. -/
theorem evalAt_input_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := inputGraph s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x) (vals := #[]) (i := 0)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s x) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, inputGraph, Graph.getNode,
    Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/--
Local IR semantics for `detach`, including removal of scalar differentiation metadata.

For ordinary numeric tensors, `detachSpec` leaves the values unchanged. For dual-valued tensors,
it preserves the primal values and clears their tangents, as the spec interpreter does.
-/
theorem evalAt_detach_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .detach s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.detachSpec x)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraph, unaryNode,
    Graph.getNode, Graph.getNode?, Graph.expectShape,
    Graph.unaryParentId, unaryParent?, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- A graph containing a zero-parent `rand_uniform` node. -/
def randUniformGraph (seed : Nat) (s : Shape) : Graph :=
  { nodes := #[{ id := 0, parents := #[], kind := .randUniform seed, outShape := s }] }

/-- Local IR semantics for deterministic seeded uniform sampling. -/
theorem evalAt_randUniform_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    (seed : Nat) {s : Shape} :
    Graph.evalAt (α := α) (g := randUniformGraph seed s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s (Tensor.default (α := α) (shape := s)))
        (vals := #[]) (i := 0)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) s
          (Spec.Random.uniform
            (α := α) (Spec.Random.keyOf seed 0) (s := s))) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, randUniformGraph, Graph.getNode,
    Graph.getNode?, Bind.bind, Except.bind,
    Pure.pure, Except.pure]

/-- Local IR semantics for deterministic seeded Bernoulli masks. -/
theorem evalAt_bernoulliMask_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    (seed : Nat) {s : Shape} (keepProb : α) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.bernoulliMask seed) .scalar s)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) .scalar (Tensor.scalar keepProb))
        (vals := #[Spec.SomeTensor.mk (α := α) .scalar (Tensor.scalar keepProb)]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) s
          (Spec.Random.mask
            (α := α) (Spec.Random.keyOf seed 1) keepProb (s := s))) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut,
    Graph.getNode, Graph.getNode?,
    Graph.unaryParentId, unaryParent?, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for scalar mean-squared error. -/
theorem evalAt_mseLoss_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (y target : Tensor α s) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut .mseLoss s s .scalar)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s y)
        (vals := #[Spec.SomeTensor.mk (α := α) s y, Spec.SomeTensor.mk (α := α) s target]) (i := 2)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) .scalar
          (Tensor.scalar
            (((Tensor.subSpec (α := α) y target).mulSpec
                (Tensor.subSpec (α := α) y target)).sumSpec /
              (↑(TorchLean.Tensor.meanDenominator s) : α)))) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, binaryGraphOut, binaryNodeOut,
    Graph.getNode, Graph.getNode?,
    Graph.binaryParentIds, binaryParents?, Graph.mseLossSomeTensor, Bind.bind, Except.bind,
    Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
