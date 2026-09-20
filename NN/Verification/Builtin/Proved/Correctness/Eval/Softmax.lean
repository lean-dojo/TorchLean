/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Softmax IR Evaluation

The IR evaluator gives every in-bounds softmax axis the canonical axis-indexed tensor semantics.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for stable innermost-axis softmax with a hard Boolean mask. -/
theorem evalAt_hardMaskedSoftmax_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (scores : Tensor α s) (allowed : Tensor Bool s) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.hardMaskedSoftmax (NN.IR.HardMask.ofTensor allowed)) s s)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s scores)
        (vals := #[Spec.SomeTensor.mk (α := α) s scores]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) s (Spec.hardMaskedSoftmaxSpec scores allowed)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut,
    Graph.getNode, Graph.getNode?,
    Graph.unaryParentId, NN.IR.unaryParent?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for softmax along any in-bounds tensor dimension. -/
theorem evalAt_softmax_axis_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.softmax axis) s s)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Activation.softmaxSpec (α := α) axis x)) := by
  cases hAxis : Shape.axisInBounds? axis s with
  | none =>
      have hSome := Shape.axisInBounds?_isSome (axis := axis) (s := s)
      simp [hAxis] at hSome
  | some h =>
      simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut,
        unaryNodeOut, Graph.getNode, Graph.getNode?, Graph.unaryParentId,
        NN.IR.unaryParent?, Graph.expectShape, hAxis,
        Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
