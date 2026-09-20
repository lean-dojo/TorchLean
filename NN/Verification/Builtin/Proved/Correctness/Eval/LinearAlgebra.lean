/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Linear Algebra IR Evaluation

Local semantics for matrix multiplication nodes accepted by the shared IR importer.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

namespace Correctness

namespace IRStep

private theorem evalParsedMatmul_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {leading parsed : Shape} {m n p : Nat} (h : parsed = leading)
    (left : Tensor α (leading.concat [m, n]))
    (right : Tensor α (leading.concat [n, p]))
    (i nodeId leftId rightId : Nat) :
    (do
        let left' ← Graph.expectShape (α := α) (expected := parsed.concat [m, n])
          (Spec.SomeTensor.mk (α := α) (leading.concat [m, n]) left)
        let right' ← Graph.expectShape (α := α) (expected := parsed.concat [n, p])
          (Spec.SomeTensor.mk (α := α) (leading.concat [n, p]) right)
        let value := Spec.SomeTensor.ofTensor (Graph.matmulLeading parsed left' right')
        Graph.normalizeNodeOutput (α := α) i
          { id := nodeId, parents := #[leftId, rightId], kind := .matmul,
            outShape := leading.concat [m, p] } value) =
      .ok (Spec.SomeTensor.mk (α := α) (leading.concat [m, p])
        (Graph.matmulLeading leading left right)) := by
  subst parsed
  simp [Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

private theorem evalNode_matmul_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {leftShape rightShape outShape : Shape}
    (op : MatmulOperation leftShape rightShape outShape)
    (left : Tensor α leftShape) (right : Tensor α rightShape)
    (payload : Payload α) (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α))
    (i nodeId leftId rightId : Nat) (leftValue rightValue : Spec.SomeTensor α)
    (hSomeLeft : vals[leftId]? = some leftValue)
    (hSomeRight : vals[rightId]? = some rightValue)
    (hLeft : Graph.expectShape (α := α) (expected := leftShape) leftValue = .ok left)
    (hRight : Graph.expectShape (α := α) (expected := rightShape) rightValue = .ok right) :
    Graph.evalNode payload input vals i
        { id := nodeId, parents := #[leftId, rightId], kind := .matmul, outShape := outShape } =
      .ok (Spec.SomeTensor.mk (α := α) outShape (op.denote left right)) := by
  have hLeftShape : leftValue.shape = leftShape := shape_eq_of_expectShape_eq_ok hLeft
  have hRightShape : rightValue.shape = rightShape := shape_eq_of_expectShape_eq_ok hRight
  have hLeftValue : leftValue = Spec.SomeTensor.mk (α := α) leftShape left := by
    have hCast : leftValue.cast hLeftShape = left := by
      have hExpected := expectShape_eq_ok (α := α) leftValue hLeftShape
      rw [hLeft] at hExpected
      exact Except.ok.inj hExpected.symm
    rw [← Spec.SomeTensor.ofTensor_cast leftValue hLeftShape, hCast]
    rfl
  have hRightValue : rightValue = Spec.SomeTensor.mk (α := α) rightShape right := by
    have hCast : rightValue.cast hRightShape = right := by
      have hExpected := expectShape_eq_ok (α := α) rightValue hRightShape
      rw [hRight] at hExpected
      exact Except.ok.inj hExpected.symm
    rw [← Spec.SomeTensor.ofTensor_cast rightValue hRightShape, hCast]
    rfl
  subst leftValue
  subst rightValue
  cases op with
  | leading batchAxes m n p =>
      simp [Graph.evalNode, Graph.binaryParentIds, binaryParents?, hSomeLeft, hSomeRight,
        MatmulOperation.denote, Shape.toList, Shape.ofList, Shape.reverse_concat,
        Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Evaluate any supported matrix multiplication in its canonical three-node graph. -/
theorem evalAt_matmul_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {leftShape rightShape outShape : Shape}
    (op : MatmulOperation leftShape rightShape outShape)
    (left : Tensor α leftShape) (right : Tensor α rightShape) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut .matmul leftShape rightShape outShape)
        (payload := {})
        (input := Spec.SomeTensor.mk (α := α) leftShape left)
        (vals := #[Spec.SomeTensor.mk (α := α) leftShape left,
          Spec.SomeTensor.mk (α := α) rightShape right])
        (i := 2)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) outShape (op.denote left right)) := by
  cases op with
  | leading batchAxes m n p =>
      simp [MatmulOperation.denote, Graph.evalAt, Graph.evalNode,
        binaryGraphOut, binaryNodeOut, Graph.getNode,
        Graph.getNode?, Graph.binaryParentIds, binaryParents?,
        Shape.reverse_concat, Shape.toList, Shape.ofList,
        Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Evaluate a typed matrix multiplication node in an arbitrary lowered graph. -/
theorem evalAt_matmul_of_getNode
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape leftShape rightShape outShape : Shape} {ss : List Shape}
    (op : MatmulOperation leftShape rightShape outShape)
    (leftIdx : Idx (Ctx inShape ss) leftShape) (rightIdx : Idx (Ctx inShape ss) rightShape)
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (i : Nat) (node : NN.IR.Node)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : g.getNode i = pure node)
    (hKind : node.kind = .matmul)
    (hParents : node.parents = #[leftIdx.id, rightIdx.id])
    (hOut : node.outShape = outShape) :
    Graph.evalAt (α := α) g payload input vals i = (do
      let left ← getVal (α := α) (inShape := inShape) (ss := ss) vals leftIdx
      let right ← getVal (α := α) (inShape := inShape) (ss := ss) vals rightIdx
      pure (Spec.SomeTensor.mk (α := α) outShape (op.denote left right))) := by
  let leftValue := packedAt vals leftIdx hShapes
  let rightValue := packedAt vals rightIdx hShapes
  let left : Tensor α leftShape := tensorAt vals leftIdx hShapes
  let right : Tensor α rightShape := tensorAt vals rightIdx hShapes
  have hExpectLeft : Graph.expectShape (α := α) (expected := leftShape) leftValue = .ok left := by
    simpa [leftValue, left] using expectShape_packedAt_eq_ok vals leftIdx hShapes
  have hExpectRight :
      Graph.expectShape (α := α) (expected := rightShape) rightValue = .ok right := by
    simpa [rightValue, right] using expectShape_packedAt_eq_ok vals rightIdx hShapes
  have hGetLeft : getVal (α := α) (inShape := inShape) (ss := ss) vals leftIdx = .ok left := by
    simpa [left] using getVal_eq_ok_of_shapesOfVals_eq vals leftIdx hShapes
  have hGetRight : getVal (α := α) (inShape := inShape) (ss := ss) vals rightIdx = .ok right := by
    simpa [right] using getVal_eq_ok_of_shapesOfVals_eq vals rightIdx hShapes
  rcases node with ⟨nodeId, parents, kind, nodeOutShape⟩
  change parents = #[leftIdx.id, rightIdx.id] at hParents
  change nodeOutShape = outShape at hOut
  change kind = .matmul at hKind
  subst parents
  subst nodeOutShape
  subst kind
  calc
    Graph.evalAt (α := α) g payload input vals i =
        Graph.evalNode payload input vals i
          { id := nodeId, parents := #[leftIdx.id, rightIdx.id], kind := .matmul,
            outShape := outShape } := by
      simp [Graph.evalAt, hGetNode]
    _ = .ok (Spec.SomeTensor.mk (α := α) outShape (op.denote left right)) :=
      evalNode_matmul_eq op left right payload input vals i nodeId leftIdx.id rightIdx.id
        leftValue rightValue (getElem?_eq_some_packedAt vals leftIdx hShapes)
        (getElem?_eq_some_packedAt vals rightIdx hShapes) hExpectLeft hExpectRight
    _ = (do
        let left ← getVal (α := α) (inShape := inShape) (ss := ss) vals leftIdx
        let right ← getVal (α := α) (inShape := inShape) (ss := ss) vals rightIdx
        pure (Spec.SomeTensor.mk (α := α) outShape (op.denote left right))) := by
      simp [hGetLeft, hGetRight, Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
