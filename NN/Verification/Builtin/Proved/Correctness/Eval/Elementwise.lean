/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Elementwise IR Evaluation

These lemmas cover the common elementwise operators emitted by the PyTorch and ONNX bridges.  Each
statement is local to one IR node: if the parent values are already present in the evaluator table,
`Graph.evalAt` returns the corresponding spec tensor operation.
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

/-- Same-shape binary elementwise operations supported by the local evaluator theorem. -/
inductive BinaryElementwiseOp where
  /-- Elementwise addition. -/
  | add
  /-- Elementwise subtraction. -/
  | sub
  /-- Elementwise multiplication. -/
  | mul
  /-- Elementwise maximum. -/
  | max
  /-- Elementwise minimum. -/
  | min

/-- Translate a binary elementwise operation to its IR opcode. -/
def BinaryElementwiseOp.toOpKind : BinaryElementwiseOp → OpKind
  | .add => .add
  | .sub => .sub
  | .mul => .mulElem
  | .max => .maxElem
  | .min => .minElem

/-- Denotation of a same-shape binary elementwise operation. -/
def BinaryElementwiseOp.denote
    {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (op : BinaryElementwiseOp)
    (a b : Tensor α s) : Tensor α s :=
  match op with
  | .add => Tensor.addSpec a b
  | .sub => Tensor.subSpec a b
  | .mul => Tensor.mulSpec a b
  | .max => Tensor.maxSpec a b
  | .min => Tensor.minSpec a b

private theorem evalNode_binaryElementwise_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (op : BinaryElementwiseOp) (a b : Tensor α s)
    (payload : Payload α) (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α))
    (i nodeId aId bId : Nat) (aVal bVal : Spec.SomeTensor α)
    (hSomeA : vals[aId]? = some aVal) (hSomeB : vals[bId]? = some bVal)
    (ha : Graph.expectShape (α := α) (expected := s) aVal = .ok a)
    (hb : Graph.expectShape (α := α) (expected := s) bVal = .ok b) :
    Graph.evalNode payload input vals i
        { id := nodeId, parents := #[aId, bId], kind := op.toOpKind, outShape := s } =
      .ok (Spec.SomeTensor.mk (α := α) s (op.denote a b)) := by
  cases op <;>
    simp [BinaryElementwiseOp.toOpKind, BinaryElementwiseOp.denote, Graph.evalNode,
      Graph.binaryParentIds, binaryParents?, hSomeA, hSomeB, ha, hb,
      Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Evaluate any supported same-shape binary elementwise node. -/
theorem evalAt_binaryElementwise_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (op : BinaryElementwiseOp) (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph op.toOpKind s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s a)
        (vals := #[Spec.SomeTensor.mk (α := α) s a, Spec.SomeTensor.mk (α := α) s b]) (i := 2)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (op.denote a b)) := by
  have ha :
      Graph.expectShape (α := α) (expected := s) (Spec.SomeTensor.mk (α := α) s a) = .ok a := by
    simp [Graph.expectShape, Pure.pure, Except.pure]
  have hb :
      Graph.expectShape (α := α) (expected := s) (Spec.SomeTensor.mk (α := α) s b) = .ok b := by
    simp [Graph.expectShape, Pure.pure, Except.pure]
  simpa [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, binaryGraph, binaryNode,
    Graph.getNode, Graph.getNode?] using
    (evalNode_binaryElementwise_eq op a b ({} : Payload α) (Spec.SomeTensor.mk (α := α) s a)
      #[Spec.SomeTensor.mk (α := α) s a, Spec.SomeTensor.mk (α := α) s b] 2 2 0 1
      (Spec.SomeTensor.mk (α := α) s a) (Spec.SomeTensor.mk (α := α) s b)
      (by simp) (by simp) ha hb)

/-- Same-shape unary elementwise operations without additional runtime side conditions. -/
inductive UnaryElementwiseOp where
  /-- Elementwise absolute value. -/
  | abs
  /-- Elementwise square root. -/
  | sqrt
  /-- Elementwise reciprocal. -/
  | inv
  /-- Rectified linear unit. -/
  | relu
  /-- Hyperbolic tangent. -/
  | tanh
  /-- Logistic sigmoid. -/
  | sigmoid
  /-- Stable softplus. -/
  | softplus
  /-- Elementwise exponential. -/
  | exp
  /-- Elementwise sine. -/
  | sin
  /-- Elementwise cosine. -/
  | cos

/-- Translate a unary elementwise operation to its IR opcode. -/
def UnaryElementwiseOp.toOpKind : UnaryElementwiseOp → OpKind
  | .abs => .abs
  | .sqrt => .sqrt
  | .inv => .inv
  | .relu => .relu
  | .tanh => .tanh
  | .sigmoid => .sigmoid
  | .softplus => .softplus
  | .exp => .exp
  | .sin => .sin
  | .cos => .cos

/-- Denotation of a same-shape unary elementwise operation. -/
def UnaryElementwiseOp.denote
    {α : Type} [TorchLean.Storage α] [Context α] {s : Shape} (op : UnaryElementwiseOp)
    (x : Tensor α s) : Tensor α s :=
  match op with
  | .abs => Tensor.absSpec x
  | .sqrt => Tensor.sqrtSpec x
  | .inv => Tensor.invSpec x
  | .relu => Activation.reluSpec x
  | .tanh => Activation.tanhSpec x
  | .sigmoid => Activation.sigmoidSpec x
  | .softplus => Activation.softplusSpec x
  | .exp => Tensor.expSpec x
  | .sin => Tensor.mapSpec (fun v => MathFunctions.sin v) x
  | .cos => Tensor.mapSpec (fun v => MathFunctions.cos v) x

private theorem evalNode_unaryElementwise_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (op : UnaryElementwiseOp) (x : Tensor α s)
    (payload : Payload α) (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α))
    (i nodeId parentId : Nat) (parent : Spec.SomeTensor α)
    (hSome : vals[parentId]? = some parent)
    (hx : Graph.expectShape (α := α) (expected := s) parent = .ok x) :
    Graph.evalNode payload input vals i
        { id := nodeId, parents := #[parentId], kind := op.toOpKind, outShape := s } =
      .ok (Spec.SomeTensor.mk (α := α) s (op.denote x)) := by
  cases op <;>
    simp [UnaryElementwiseOp.toOpKind, UnaryElementwiseOp.denote, Graph.evalNode,
      Graph.unaryParentId, unaryParent?, hSome, hx,
      Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Evaluate any supported same-shape unary elementwise node. -/
theorem evalAt_unaryElementwise_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (op : UnaryElementwiseOp) (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph op.toOpKind s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (op.denote x)) := by
  have hx :
      Graph.expectShape (α := α) (expected := s) (Spec.SomeTensor.mk (α := α) s x) = .ok x := by
    simp [Graph.expectShape, Pure.pure, Except.pure]
  simpa [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraph, unaryNode,
    Graph.getNode, Graph.getNode?] using
    (evalNode_unaryElementwise_eq op x ({} : Payload α) (Spec.SomeTensor.mk (α := α) s x)
      #[Spec.SomeTensor.mk (α := α) s x] 1 1 0 (Spec.SomeTensor.mk (α := α) s x)
      (by simp) hx)

/-- Evaluate a binary elementwise node in an arbitrary graph from the lowering pass's shape
invariant. -/
theorem evalAt_binaryElementwise_of_getNode
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape s : Shape} {ss : List Shape}
    (op : BinaryElementwiseOp) (a b : Idx (Ctx inShape ss) s)
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α))
    (i : Nat) (n : NN.IR.Node)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : g.getNode i = pure n)
    (hKind : n.kind = op.toOpKind) (hParents : n.parents = #[a.id, b.id])
    (hOut : n.outShape = s) :
    Graph.evalAt (α := α) g payload input vals i = (do
      let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
      let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
      pure (Spec.SomeTensor.mk (α := α) s (op.denote ta tb))) := by
  let aVal := packedAt vals a hShapes
  let bVal := packedAt vals b hShapes
  let ta : Tensor α s := tensorAt vals a hShapes
  let tb : Tensor α s := tensorAt vals b hShapes
  have hExpectA : Graph.expectShape (α := α) (expected := s) aVal = .ok ta := by
    simpa [aVal, ta] using expectShape_packedAt_eq_ok vals a hShapes
  have hExpectB : Graph.expectShape (α := α) (expected := s) bVal = .ok tb := by
    simpa [bVal, tb] using expectShape_packedAt_eq_ok vals b hShapes
  have hGetA : getVal (α := α) (inShape := inShape) (ss := ss) vals a = .ok ta := by
    simpa [ta] using getVal_eq_ok_of_shapesOfVals_eq vals a hShapes
  have hGetB : getVal (α := α) (inShape := inShape) (ss := ss) vals b = .ok tb := by
    simpa [tb] using getVal_eq_ok_of_shapesOfVals_eq vals b hShapes
  rcases n with ⟨nodeId, parents, kind, outShape⟩
  change parents = #[a.id, b.id] at hParents
  change outShape = s at hOut
  change kind = op.toOpKind at hKind
  subst parents
  subst outShape
  subst kind
  calc
    Graph.evalAt (α := α) g payload input vals i =
        Graph.evalNode payload input vals i
          { id := nodeId, parents := #[a.id, b.id], kind := op.toOpKind, outShape := s } := by
      simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hGetNode]
    _ = .ok (Spec.SomeTensor.mk (α := α) s (op.denote ta tb)) :=
      evalNode_binaryElementwise_eq op ta tb payload input vals i nodeId a.id b.id aVal bVal
        (getElem?_eq_some_packedAt vals a hShapes)
        (getElem?_eq_some_packedAt vals b hShapes) hExpectA hExpectB
    _ = (do
        let ta ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals a
        let tb ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals b
        pure (Spec.SomeTensor.mk (α := α) s (op.denote ta tb))) := by
      simp [hGetA, hGetB, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Evaluate a unary elementwise node in an arbitrary graph from the lowering pass's shape
invariant. -/
theorem evalAt_unaryElementwise_of_getNode
    {α : Type} [TorchLean.Storage α] [Context α]
    {inShape s : Shape} {ss : List Shape}
    (op : UnaryElementwiseOp) (x : Idx (Ctx inShape ss) s)
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α))
    (i : Nat) (n : NN.IR.Node)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : g.getNode i = pure n)
    (hKind : n.kind = op.toOpKind) (hParents : n.parents = #[x.id])
    (hOut : n.outShape = s) :
    Graph.evalAt (α := α) g payload input vals i = (do
      let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
      pure (Spec.SomeTensor.mk (α := α) s (op.denote tx))) := by
  let xVal := packedAt vals x hShapes
  let tx : Tensor α s := tensorAt vals x hShapes
  have hExpect : Graph.expectShape (α := α) (expected := s) xVal = .ok tx := by
    simpa [xVal, tx] using expectShape_packedAt_eq_ok vals x hShapes
  have hGet : getVal (α := α) (inShape := inShape) (ss := ss) vals x = .ok tx := by
    simpa [tx] using getVal_eq_ok_of_shapesOfVals_eq vals x hShapes
  rcases n with ⟨nodeId, parents, kind, outShape⟩
  change parents = #[x.id] at hParents
  change outShape = s at hOut
  change kind = op.toOpKind at hKind
  subst parents
  subst outShape
  subst kind
  calc
    Graph.evalAt (α := α) g payload input vals i =
        Graph.evalNode payload input vals i
          { id := nodeId, parents := #[x.id], kind := op.toOpKind, outShape := s } := by
      simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hGetNode]
    _ = .ok (Spec.SomeTensor.mk (α := α) s (op.denote tx)) :=
      evalNode_unaryElementwise_eq op tx payload input vals i nodeId x.id xVal
        (getElem?_eq_some_packedAt vals x hShapes) hExpect
    _ = (do
        let tx ← getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x
        pure (Spec.SomeTensor.mk (α := α) s (op.denote tx))) := by
      simp [hGet, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for elementwise addition. -/
theorem evalAt_add_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .add s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s a)
        (vals := #[Spec.SomeTensor.mk (α := α) s a, Spec.SomeTensor.mk (α := α) s b]) (i := 2)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.addSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .add a b

/-- Local IR semantics for elementwise subtraction. -/
theorem evalAt_sub_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .sub s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s a)
        (vals := #[Spec.SomeTensor.mk (α := α) s a, Spec.SomeTensor.mk (α := α) s b]) (i := 2)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.subSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .sub a b

/-- Local IR semantics for elementwise multiplication. -/
theorem evalAt_mulElem_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .mulElem s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s a)
        (vals := #[Spec.SomeTensor.mk (α := α) s a, Spec.SomeTensor.mk (α := α) s b]) (i := 2)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.mulSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .mul a b

/-- Local IR semantics for elementwise maximum. -/
theorem evalAt_maxElem_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .maxElem s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s a)
        (vals := #[Spec.SomeTensor.mk (α := α) s a, Spec.SomeTensor.mk (α := α) s b]) (i := 2)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.maxSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .max a b

/-- Local IR semantics for elementwise minimum. -/
theorem evalAt_minElem_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .minElem s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s a)
        (vals := #[Spec.SomeTensor.mk (α := α) s a, Spec.SomeTensor.mk (α := α) s b]) (i := 2)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.minSpec (α := α) a b)) := by
  exact evalAt_binaryElementwise_eq .min a b

/-- Local IR semantics for elementwise absolute value. -/
theorem evalAt_abs_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .abs s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.absSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .abs x

/-- Local IR semantics for elementwise square root. -/
theorem evalAt_sqrt_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sqrt s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.sqrtSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .sqrt x

/-- Local IR semantics for elementwise reciprocal. -/
theorem evalAt_inv_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .inv s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.invSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .inv x

/-- Local IR semantics for ReLU. -/
theorem evalAt_relu_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .relu s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Activation.reluSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .relu x

/-- Local IR semantics for tanh. -/
theorem evalAt_tanh_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .tanh s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Activation.tanhSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .tanh x

/-- Local IR semantics for sigmoid. -/
theorem evalAt_sigmoid_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sigmoid s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Activation.sigmoidSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .sigmoid x

/-- Local IR semantics for softplus. -/
theorem evalAt_softplus_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .softplus s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Activation.softplusSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .softplus x

/-- The scalar epsilon parent is read in the same context as the input tensor.

In particular, this equality also preserves an epsilon tangent when the context is a dual number
type. The IR node uses the source operation's total logarithm semantics; checking a positive
logarithm interval belongs to certificate construction. -/
theorem evalNode_safeLog_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) (epsilon : α) :
    Graph.evalNode (α := α) {} (Spec.SomeTensor.mk s x)
        #[Spec.SomeTensor.mk s x, Spec.SomeTensor.mk .scalar (Tensor.scalar epsilon)] 2
        { id := 2, parents := #[0, 1], kind := .safeLog, outShape := s } =
      .ok (Spec.SomeTensor.mk s (Activation.safeLogSpec x epsilon)) := by
  simp [Graph.evalNode, Graph.evalNodeRaw, Graph.normalizeNodeOutput, Graph.binaryParentIds,
    binaryParents?, Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for exp. -/
theorem evalAt_exp_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .exp s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.expSpec (α := α) x)) := by
  exact evalAt_unaryElementwise_eq .exp x

/-- Local IR semantics for sin. -/
theorem evalAt_sin_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sin s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) s (Tensor.mapSpec (fun v => MathFunctions.sin v) x)) := by
  exact evalAt_unaryElementwise_eq .sin x

/-- Local IR semantics for cos. -/
theorem evalAt_cos_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .cos s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) s (Tensor.mapSpec (fun v => MathFunctions.cos v) x)) := by
  exact evalAt_unaryElementwise_eq .cos x

/-- Local IR semantics for log on inputs satisfying the IR positivity side condition. -/
theorem evalAt_log_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (x : Tensor α s)
    (hpos : Tensor.allSpec (α := α) (s := s) (fun v => decide (0 < v)) x = true) :
    Graph.evalAt (α := α) (g := unaryGraph .log s) (payload := {})
        (input := Spec.SomeTensor.mk (α := α) s x)
        (vals := #[Spec.SomeTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.SomeTensor.mk (α := α) s (Tensor.logSpec (α := α) x)) := by
  simp [Graph.evalAt, Graph.evalNode, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?,
    Graph.unaryParentId, unaryParent?, Graph.expectShape, hpos,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
