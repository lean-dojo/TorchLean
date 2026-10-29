/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Shape-Changing IR Evaluation

Local semantics for the shape-oriented IR nodes emitted by PyTorch/ONNX import paths.  These facts
pin the executable IR evaluator to the corresponding typed tensor operations.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Shape-changing unary operations with statically typed output tensors. -/
inductive ShapeOperation : Shape → Shape → Type where
  /-- Reshape between shapes with equal element counts. -/
  | reshape (inShape outShape : Shape)
      (hsize : Spec.Shape.size inShape = Spec.Shape.size outShape) :
      ShapeOperation inShape outShape
  /-- Flatten a tensor to one dimension. -/
  | flatten (s : Shape) : ShapeOperation s (.dim (Spec.Shape.size s) .scalar)
  /-- Broadcast to a compatible target shape. -/
  | broadcastTo (s₁ s₂ : Shape) (witness : Shape.CanBroadcastTo s₁ s₂) : ShapeOperation s₁ s₂
  /-- Sum every coordinate to a scalar. -/
  | sum (s : Shape) : ShapeOperation s .scalar

/-- IR opcode represented by a typed shape operation. -/
def ShapeOperation.toOpKind {inShape outShape : Shape} :
    ShapeOperation inShape outShape → OpKind
  | .reshape _ _ _ => .reshape inShape outShape
  | .flatten _ => .flatten inShape
  | .broadcastTo _ _ _ => .broadcastTo inShape outShape
  | .sum _ => .sum

/-- Typed denotation of a shape operation. -/
def ShapeOperation.denote
    {α : Type} [TorchLean.Storage α] [Context α] {inShape outShape : Shape}
    (op : ShapeOperation inShape outShape) (x : Tensor α inShape) : Tensor α outShape :=
  match op with
  | .reshape _ _ hsize => Tensor.reshapeSpec (α := α) x hsize
  | .flatten _ => Tensor.flattenSpec (α := α) x
  | .broadcastTo _ _ witness => Tensor.broadcastTo (α := α) witness x
  | .sum _ => Tensor.scalar (Tensor.sumSpec (α := α) x)

/-- Evaluate any typed shape operation in its canonical two-node graph. -/
theorem evalAt_shapeOperation_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape outShape : Shape} (op : ShapeOperation inShape outShape)
    (x : Tensor α inShape) :
    Graph.evalAt (α := α) (g := unaryGraphOut op.toOpKind inShape outShape)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) inShape x)
        (vals := #[Spec.SomeTensor.mk (α := α) inShape x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) outShape (op.denote x)) := by
  cases op <;>
    simp_all [ShapeOperation.toOpKind, ShapeOperation.denote, Graph.evalAt, Graph.evalNode,
      unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.expectShape,
      Graph.unaryParentId,
      unaryParent?, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for `reshape` when the element counts match. -/
theorem evalAt_reshape_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape outShape : Shape} (x : Tensor α inShape)
    (hsize : Spec.Shape.size inShape = Spec.Shape.size outShape) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.reshape inShape outShape) inShape outShape)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) inShape x)
        (vals := #[Spec.SomeTensor.mk (α := α) inShape x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) outShape
          (Tensor.reshapeSpec (α := α) (source := inShape) (target := outShape) x hsize)) := by
  exact evalAt_shapeOperation_eq (.reshape inShape outShape hsize) x

/-- Local IR semantics for `flatten`. -/
theorem evalAt_flatten_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.flatten s) s (.dim (Spec.Shape.size s) .scalar))
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) (.dim (Spec.Shape.size s) .scalar)
          (Tensor.flattenSpec (α := α) (shape := s) x)) := by
  exact evalAt_shapeOperation_eq (.flatten s) x

/-- Local IR semantics for `broadcastTo` when the shapes are broadcast compatible. -/
theorem evalAt_broadcastTo_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s₁ s₂ : Shape} (x : Tensor α s₁) (cb : Shape.CanBroadcastTo s₁ s₂) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.broadcastTo s₁ s₂) s₁ s₂)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s₁ x)
        (vals := #[Spec.SomeTensor.mk (α := α) s₁ x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) s₂
          (Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb x)) := by
  exact evalAt_shapeOperation_eq (.broadcastTo s₁ s₂ cb) x

/-- Local IR semantics for reduction to a scalar sum. -/
theorem evalAt_sum_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraphOut .sum s .scalar)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) .scalar (Tensor.scalar (Tensor.sumSpec (α := α) x))) := by
  exact evalAt_shapeOperation_eq (.sum s) x

/-- A checked native slice evaluates to the source coordinate slice for every valid axis. -/
theorem evalAt_slice_eq {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (axis start length : Nat) [hAxis : Shape.AxisInBounds axis s]
    (x : Tensor α s) (hRange : start + length ≤ Shape.axisSize s axis) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.slice axis start length) s (s.replaceAxis axis length))
        (payload := {}) (input := Spec.SomeTensor.mk s x)
        (vals := #[Spec.SomeTensor.mk s x]) (i := 1) =
      .ok (Spec.SomeTensor.mk (s.replaceAxis axis length)
        (Tensor.sliceAxisRangeSpec axis x start length hRange)) := by
  simp [Graph.evalAt, Graph.evalNode, unaryGraphOut, unaryNodeOut, Graph.getNode,
    Graph.expectShape, Graph.unaryParentId, unaryParent?, Shape.axisInBounds?, hAxis.proof,
    hRange, Bind.bind, Except.bind, Pure.pure, Except.pure]

private theorem reshapeSpec_dim {α : Type} [TorchLean.Storage α]
    {n : Nat} {s t : Shape} (h : s.size = t.size) (values : Fin n → Tensor α s) :
    Tensor.reshapeSpec (target := .dim n t) (Tensor.dim values)
        (by simpa only [Shape.size] using congrArg (n * ·) h) =
      Tensor.dim (fun i => Tensor.reshapeSpec (values i) h) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  rintro ⟨i, coordinate⟩
  have hOuter : (Shape.dim n s).size = (Shape.dim n t).size := by
    simpa only [Shape.size] using congrArg (n * ·) h
  have hCoordinate :
      TorchLean.Tensor.Internal.Rep.reshapeCoordEquiv
          (Shape.internalSize_congr hOuter) (i, coordinate) =
        (i, TorchLean.Tensor.Internal.Rep.reshapeCoordEquiv
          (Shape.internalSize_congr h) coordinate) := by
    apply TorchLean.Tensor.Internal.Coord.linearize_injective
    apply Fin.ext
    rw [Tensor.reshapeCoordEquiv_linearize_val,
      TorchLean.Tensor.Internal.Coord.linearize_cons_val,
      TorchLean.Tensor.Internal.Coord.linearize_cons_val,
      Tensor.reshapeCoordEquiv_linearize_val]
    simp only [Shape.internalSize_eq, h]
  simp only [Tensor.reshapeSpec, Tensor.dim,
    TorchLean.Tensor.Internal.Rep.reshape_apply_coordEquiv,
    hCoordinate, TorchLean.Tensor.Internal.Rep.stack_apply]

/-- Dropping any singleton slice axis agrees with selection without scalar arithmetic.

The reshape only removes the length-one axis; all remaining coordinates keep their row-major
order, including shapes with empty leading or trailing axes.
-/
theorem reshape_sliceAxisRangeSpec_one_eq_selectSpec {α : Type} [TorchLean.Storage α] :
    (axis : Nat) → {shape : Shape} → [_hAxis : Shape.AxisInBounds axis shape] →
      (x : Tensor α shape) → (index : Fin (Shape.axisSize shape axis)) →
      Tensor.reshapeSpec
          (Tensor.sliceAxisRangeSpec axis x index.val 1 (Nat.succ_le_of_lt index.isLt))
          (Tensor.size_shapeAfterSumKeepDim shape axis) =
        Tensor.selectSpec axis x index
  | 0, .dim n s, _, x, index => by
      have hRange : index.val + 1 ≤ Shape.axisSize (.dim n s) 0 :=
        Nat.succ_le_of_lt index.isLt
      have hSlice : Tensor.sliceAxisRangeSpec 0 x index.val 1 hRange =
          Tensor.dim (fun _ : Fin 1 => Tensor.selectSpec 0 x index) := by
        apply TorchLean.Tensor.Internal.Rep.ext
        rintro ⟨i, j⟩
        have hCoordinate :
            Tensor.sliceAxisRangeCoordinate 0 (shape := .dim n s) index.val 1
              hRange (i, j) = (index, j) := by
          apply Prod.ext
          · apply Fin.ext
            simp [Tensor.sliceAxisRangeCoordinate]
          · rfl
        calc
          Tensor.sliceAxisRangeSpec 0 x index.val 1 hRange (i, j) =
              x (Tensor.sliceAxisRangeCoordinate 0 index.val 1 hRange (i, j)) :=
            Tensor.sliceAxisRangeSpec_apply 0 x index.val 1 hRange (i, j)
          _ = x (index, j) := congrArg x hCoordinate
          _ = Tensor.dim (fun _ : Fin 1 => Tensor.selectSpec 0 x index) (i, j) := by
            exact (TorchLean.Tensor.Internal.Rep.unstack_apply x index j).symm.trans
              (TorchLean.Tensor.Internal.Rep.stack_apply
                (fun _ : Fin 1 => Tensor.unstack x index) (i, j)).symm
      let hSize : (Shape.dim 1 s).size = s.size := by simp [Shape.size]
      change Tensor.reshapeSpec (Tensor.sliceAxisRangeSpec 0 x index.val 1 hRange) hSize = _
      rw [hSlice]
      dsimp only [Shape.replaceAxis]
      have hStack : Tensor.dim (fun _ : Fin 1 => Tensor.selectSpec 0 x index) =
          Tensor.reshapeSpec (Tensor.selectSpec 0 x index) hSize.symm := by
        exact (TorchLean.Tensor.Internal.Rep.reshape_one_cons _ _).symm
      rw [hStack, Tensor.reshapeSpec_roundtrip]
  | axis + 1, .dim n s, hAxis, x, index => by
      let innerAxis : Shape.AxisInBounds axis s := ⟨by
        have := hAxis.proof
        simp only [Shape.rank] at this
        omega⟩
      let := innerAxis
      have hRange : index.val + 1 ≤ Shape.axisSize s axis :=
        Nat.succ_le_of_lt index.isLt
      have hSlice : Tensor.sliceAxisRangeSpec (axis + 1) x index.val 1
            (Nat.succ_le_of_lt index.isLt) =
          Tensor.dim (fun outer =>
            Tensor.sliceAxisRangeSpec axis (Tensor.unstack x outer) index.val 1 hRange) := by
        dsimp only [Tensor.sliceAxisRangeSpec, Shape.replaceAxis, Tensor.dim, Tensor.unstack]
        apply TorchLean.Tensor.Internal.Rep.ext
        rintro ⟨outer, coordinate⟩
        simp only [TorchLean.Tensor.Internal.Rep.pull_apply, Tensor.sliceAxisRangeCoordinate,
          TorchLean.Tensor.Internal.Rep.stack_apply,
          TorchLean.Tensor.Internal.Rep.unstack_apply]
        exact TorchLean.Tensor.Internal.Rep.pull_apply _ _ _
      rw [hSlice]
      change Tensor.reshapeSpec (target := .dim n (s.eraseAxis axis))
          (Tensor.dim (fun outer =>
            Tensor.sliceAxisRangeSpec axis (Tensor.unstack x outer) index.val 1 hRange))
          (by simpa only [Shape.size, Tensor.shapeAfterSumKeepDim, Tensor.shapeAfterSum] using
            congrArg (n * ·) (Tensor.size_shapeAfterSumKeepDim s axis)) = _
      rw [reshapeSpec_dim]
      congr 1
      funext outer
      exact reshape_sliceAxisRangeSpec_one_eq_selectSpec axis (Tensor.unstack x outer) index

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
