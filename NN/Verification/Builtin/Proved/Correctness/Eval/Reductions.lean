/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Reduction IR Evaluation

Local semantics for reduction nodes accepted by the shared IR importer.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Axis reductions whose evaluator branches share the same validity and output-shape contract. -/
inductive AxisReductionOperation where
  /-- Sum along an axis. -/
  | sum
  /-- Mean along an axis. -/
  | mean

/-- IR opcode for an axis reduction. -/
def AxisReductionOperation.toOpKind (op : AxisReductionOperation) (axis : Nat) : OpKind :=
  match op with
  | .sum => .reduceSum axis
  | .mean => .reduceMean axis

/-- Typed denotation of an axis reduction. -/
def AxisReductionOperation.denote
    {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (op : AxisReductionOperation) (axis : Nat) (x : Tensor α s)
    (hAxis : Shape.NonemptyAxis axis s) : Tensor α (Tensor.shapeAfterSum s axis) :=
  match op with
  | .sum => Tensor.reduceSum (α := α) (s := s) axis x (hAxis)
  | .mean => Tensor.reduceMean (α := α) (s := s) axis x (hAxis)

/-- Evaluate either supported axis reduction in its canonical two-node graph. -/
theorem evalAt_axisReduction_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (op : AxisReductionOperation) (axis : Nat) (x : Tensor α s)
    (hAxis : PLift (Shape.NonemptyAxis axis s))
    (hAxisLookup : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxis) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (op.toOpKind axis) s (Tensor.shapeAfterSum s axis))
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) (Tensor.shapeAfterSum s axis)
          (op.denote axis x hAxis.down)) := by
  cases op <;>
    simp [AxisReductionOperation.toOpKind, AxisReductionOperation.denote, Graph.evalAt,
      Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut, Graph.getNode,
      Graph.getNode?, hAxisLookup, Bind.bind,
      Graph.unaryParentId, unaryParent?, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for `reduce_sum` along a valid axis. -/
theorem evalAt_reduceSum_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (axis : Nat) (x : Tensor α s)
    (hAxis : PLift (Shape.NonemptyAxis axis s))
    (hAxisLookup : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxis) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.reduceSum axis) s (Tensor.shapeAfterSum s axis))
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) (Tensor.shapeAfterSum s axis)
          (Tensor.reduceSum (α := α) (s := s) axis x
            (hAxis.down))) := by
  exact evalAt_axisReduction_eq .sum axis x hAxis hAxisLookup

/-- Local IR semantics for `reduce_mean` along a valid axis. -/
theorem evalAt_reduceMean_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (axis : Nat) (x : Tensor α s)
    (hAxis : PLift (Shape.NonemptyAxis axis s))
    (hAxisLookup : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxis) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.reduceMean axis) s (Tensor.shapeAfterSum s axis))
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) (Tensor.shapeAfterSum s axis)
          (Tensor.reduceMean (α := α) (s := s) axis x
            (hAxis.down))) := by
  exact evalAt_axisReduction_eq .mean axis x hAxis hAxisLookup

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
