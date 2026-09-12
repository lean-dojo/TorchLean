/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Elementwise
public import NN.Verification.Builtin.Proved.Correctness.Eval.LinearAlgebra

/-!
# Lowered Forward Evaluation: Payload-Free Nodes

One lemma per payload-free operator of the proved forward fragment. Each lemma assumes that the
IR graph holds the node emitted by `lowerNode` at index `id` and that the runtime value array
matches the typed shape context; it concludes that IR evaluation at `id` returns exactly what the
typed evaluator `evalNode` returns for the source node. The IR payload is arbitrary because these
operators never read it.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/-- A lowered `add` node evaluates like the typed `add` node. -/
theorem evalAt_eq_evalNode_add
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (a b : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[a.id, b.id], kind := .add, outShape := s }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.add a b) params vals := by
  simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
    IRStep.evalAt_binaryElementwise_of_getNode (α := α) .add a b G payload input vals id _
      hShapes hGetNode rfl rfl rfl

/-- A lowered `sub` node evaluates like the typed `sub` node. -/
theorem evalAt_eq_evalNode_sub
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (a b : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[a.id, b.id], kind := .sub, outShape := s }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.sub a b) params vals := by
  simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
    IRStep.evalAt_binaryElementwise_of_getNode (α := α) .sub a b G payload input vals id _
      hShapes hGetNode rfl rfl rfl

/-- A lowered `mulElem` node evaluates like the typed `mulElem` node. -/
theorem evalAt_eq_evalNode_mulElem
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (a b : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[a.id, b.id], kind := .mulElem, outShape := s }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.mulElem a b) params vals := by
  simpa [evalNode, IRStep.BinaryElementwiseOp.denote] using
    IRStep.evalAt_binaryElementwise_of_getNode (α := α) .mul a b G payload input vals id _
      hShapes hGetNode rfl rfl rfl

/-- A lowered `relu` node evaluates like the typed `relu` node. -/
theorem evalAt_eq_evalNode_relu
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (xIdx : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .relu, outShape := s }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.relu xIdx) params vals := by
  simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
    IRStep.evalAt_unaryElementwise_of_getNode (α := α) .relu xIdx G payload input vals id _
      hShapes hGetNode rfl rfl rfl

/-- A lowered `exp` node evaluates like the typed `exp` node. -/
theorem evalAt_eq_evalNode_exp
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (xIdx : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .exp, outShape := s }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.exp xIdx) params vals := by
  simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
    IRStep.evalAt_unaryElementwise_of_getNode (α := α) .exp xIdx G payload input vals id _
      hShapes hGetNode rfl rfl rfl

/-- A lowered `inv` node evaluates like the typed `inv` node. -/
theorem evalAt_eq_evalNode_inv
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (xIdx : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .inv, outShape := s }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.inv xIdx) params vals := by
  simpa [evalNode, IRStep.UnaryElementwiseOp.denote] using
    IRStep.evalAt_unaryElementwise_of_getNode (α := α) .inv xIdx G payload input vals id _
      hShapes hGetNode rfl rfl rfl

/--
A lowered `log` node evaluates like the typed `log` node. Both evaluators reject nonpositive
inputs with the same error, so the agreement holds without a positivity side condition.
-/
theorem evalAt_eq_evalNode_log
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (xIdx : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .log, outShape := s }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.log xIdx) params vals := by
  have hExpect := expectShape_packedAt_eq_ok vals xIdx hShapes
  have hGetVal := getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
  cases hPos : Tensor.allSpec (α := α) (s := s) (fun v => decide (0 < v))
      (tensorAt vals xIdx hShapes) <;>
    simp [Graph.evalAt, Graph.evalNode, Graph.unaryParentId, unaryParent?,
      Graph.normalizeNodeOutput, hGetNode, getElem?_eq_some_packedAt vals xIdx hShapes, hExpect,
      evalNode, hGetVal, hPos, throw, throwThe, MonadExceptOf.throw,
      Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- A lowered `matmul` node evaluates like the typed `matmul` node. -/
theorem evalAt_eq_evalNode_matmul
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape}
    {leftShape rightShape outShape : Shape}
    (op : MatmulOperation leftShape rightShape outShape)
    (a : Idx (Ctx inShape ss) leftShape) (b : Idx (Ctx inShape ss) rightShape)
    (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[a.id, b.id], kind := .matmul, outShape := outShape }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := outShape) (Node.matmul op a b) params vals := by
  simpa [evalNode] using
    IRStep.evalAt_matmul_of_getNode (α := α) op a b G payload input vals id _
      hShapes hGetNode rfl rfl rfl

/-- A lowered `reshape` node evaluates like the typed `reshape` node. -/
theorem evalAt_eq_evalNode_reshape
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape}
    (inS outS : Shape) (h : Spec.Shape.size inS = Spec.Shape.size outS)
    (xIdx : Idx (Ctx inShape ss) inS) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .reshape inS outS, outShape := outS }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := outS)
        (Node.reshape inS outS h xIdx) params vals := by
  have hExpect := expectShape_packedAt_eq_ok vals xIdx hShapes
  have hGetVal := getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
  simp [Graph.evalAt, Graph.evalNode, Graph.unaryParentId, unaryParent?,
    Graph.normalizeNodeOutput, hGetNode, getElem?_eq_some_packedAt vals xIdx hShapes, hExpect, h,
    evalNode, hGetVal, Bind.bind, Except.bind, Pure.pure, Except.pure]

/--
A lowered `transpose` node evaluates like the typed `transpose` node. Both evaluators run the same
dynamic permutation, so the proof follows the three fallible steps case by case.
-/
theorem evalAt_eq_evalNode_transpose
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s out : Shape}
    (axis₁ axis₂ : Nat) (hOut : OpContracts.inferTransposeOutShape axis₁ axis₂ s = .ok out)
    (xIdx : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .transpose axis₁ axis₂, outShape := out }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := out)
        (Node.transpose axis₁ axis₂ hOut xIdx) params vals := by
  have hGetVal := getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
  have hPacked := ofTensor_tensorAt_eq_packedAt vals xIdx hShapes
  cases hPerm : OpContracts.transposePerm s.rank axis₁ axis₂ with
  | error e =>
      simp [Graph.evalAt, Graph.evalNode, Graph.unaryParentId, unaryParent?, hGetNode,
        getElem?_eq_some_packedAt vals xIdx hShapes, ← hPacked, evalNode, hGetVal, hPerm,
        Bind.bind, Except.bind, Pure.pure, Except.pure]
  | ok perm =>
      cases hEval : Graph.permuteSomeTensor (α := α)
          (Spec.SomeTensor.mk (α := α) s (tensorAt vals xIdx hShapes)) perm with
      | error e =>
          simp [Graph.evalAt, Graph.evalNode, Graph.unaryParentId, unaryParent?, hGetNode,
            getElem?_eq_some_packedAt vals xIdx hShapes, ← hPacked, evalNode, hGetVal, hPerm,
            hEval, Bind.bind, Except.bind, Pure.pure, Except.pure]
      | ok y =>
          cases hShape : Graph.expectShape (α := α) (expected := out) y with
          | error e =>
              simp [Graph.evalAt, Graph.evalNode, Graph.unaryParentId, unaryParent?, hGetNode,
                getElem?_eq_some_packedAt vals xIdx hShapes, ← hPacked, evalNode, hGetVal, hPerm,
                hEval, hShape, Bind.bind, Except.bind, Pure.pure, Except.pure]
          | ok ty =>
              simp [Graph.evalAt, Graph.evalNode, Graph.unaryParentId, unaryParent?,
                Graph.normalizeNodeOutput, hGetNode, getElem?_eq_some_packedAt vals xIdx hShapes,
                ← hPacked, evalNode, hGetVal, hPerm, hEval, hShape,
                Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- A lowered `softmax` node evaluates like the typed `softmax` node. -/
theorem evalAt_eq_evalNode_softmax
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (axis : Nat) (hAxis : Shape.AxisInBounds axis s)
    (xIdx : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .softmax axis, outShape := s }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.softmax axis hAxis xIdx) params vals := by
  have hExpect := expectShape_packedAt_eq_ok vals xIdx hShapes
  have hGetVal := getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
  cases hDynamic : Shape.axisInBounds? axis s with
  | none =>
      have hSome := Shape.axisInBounds?_isSome (axis := axis) (s := s) (h := hAxis)
      simp [hDynamic] at hSome
  | some h =>
      simp [Graph.evalAt, Graph.evalNode, Graph.unaryParentId, unaryParent?,
        Graph.normalizeNodeOutput, hGetNode, getElem?_eq_some_packedAt vals xIdx hShapes, hDynamic,
        hExpect, evalNode, hGetVal, Bind.bind, Except.bind, Pure.pure, Except.pure]

/--
A lowered `mseLoss` node evaluates like the typed `mseLoss` node. Both evaluators perform the
same dynamic equal-shape check before averaging the squared error.
-/
theorem evalAt_eq_evalNode_mseLoss
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (yhat target : Idx (Ctx inShape ss) s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[yhat.id, target.id], kind := .mseLoss, outShape := .scalar }) :
    Graph.evalAt (α := α) G payload input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := .scalar) (Node.mseLoss yhat target) params vals := by
  have hSomeY := getElem?_eq_some_packedAt vals yhat hShapes
  have hSomeT := getElem?_eq_some_packedAt vals target hShapes
  let yV := packedAt vals yhat hShapes
  let tV := packedAt vals target hShapes
  have hSameShape : yV.shape = tV.shape :=
    (packedAt_shape vals yhat hShapes).trans (packedAt_shape vals target hShapes).symm
  let yT : Tensor α yV.shape := yV.tensor
  let tT : Tensor α yV.shape := hSameShape.symm ▸ tV.tensor
  let diff : Tensor α yV.shape := Tensor.subSpec (α := α) yT tT
  let mean : α :=
    (Tensor.mulSpec (α := α) diff diff).sumSpec / (↑(TorchLean.Tensor.meanDenominator yV.shape) : α)
  have hIREval :
      Graph.evalAt (α := α) G payload input vals id =
        Except.ok (Spec.SomeTensor.mk (α := α) .scalar (Tensor.scalar mean)) := by
    simp [Graph.evalAt, Graph.evalNode, Graph.binaryParentIds, binaryParents?,
      Graph.normalizeNodeOutput, Graph.mseLossSomeTensor, hGetNode, hSomeY, hSomeT, mean, diff,
      yT, tT, yV, tV, Bind.bind, Except.bind, Pure.pure, Except.pure]
  have hTypedEval :
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
          (out := .scalar) (Node.mseLoss yhat target) params vals =
        Except.ok (Spec.SomeTensor.mk (α := α) .scalar (Tensor.scalar mean)) := by
    simp only [evalNode, getValue?, hSomeY, hSomeT, Bind.bind, Except.bind]
    rw [dif_pos hSameShape]
    rfl
  exact hIREval.trans hTypedEval.symm

end Correctness

end NN.Verification.Builtin.Proved
