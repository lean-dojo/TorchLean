/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.LoweringPrefix
public import NN.Verification.Builtin.Proved.Correctness.Eval.LoweredNodeBasic
public import NN.Verification.Builtin.Proved.Correctness.Eval.LoweredNodePayload
public import NN.Verification.Builtin.Proved.Correctness.Eval.NodeShape

/-!
# Lowered Forward Evaluation: SSA Denotation Agreement

The main theorem `denoteAllFrom_lowerForwardLetChain_eq_evalForwardLetChainVals` states that
running the IR evaluator over a lowered let-chain produces the same value vector as the typed
evaluator. The proof is an induction over the chain: `evalAt_eq_evalNode_of_lowerNode` dispatches
the one-node agreement to the per-operator lemmas, and the remaining lemmas here unfold one
evaluation step on each side.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin

/--
Evaluating the IR node emitted by `lowerNode` agrees with `evalNode` on the source node, provided
the graph holds that node at `id` and the parameter store agrees with the lowering at `id`.
-/
theorem evalAt_eq_evalNode_of_lowerNode
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (node : Node α paramShapes inShape ss out) (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat)
    (G : Graph) (P : NN.MLTheory.CROWN.Graph.ParamStore α)
    (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α))
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id = pure (lowerNode (α := α) id node params ps).1)
    (hConst : P.constVals.get? id = (lowerNode (α := α) id node params ps).2.constVals.get? id)
    (hLin : P.linearWB.get? id = (lowerNode (α := α) id node params ps).2.linearWB.get? id)
    (hConv : P.convCfg.get? id = (lowerNode (α := α) id node params ps).2.convCfg.get? id)
    (hLayerNorm :
      P.layerNorm.get? id = (lowerNode (α := α) id node params ps).2.layerNorm.get? id) :
    Graph.evalAt (α := α) G (payloadOfParamStore (α := α) P) input vals id =
      evalNode (α := α) node params vals := by
  cases node with
  | const wf t =>
      exact evalAt_eq_evalNode_const wf t params G P input vals id hGetNode
        (hConst.trans (by simp [lowerNode]))
  | paramConst wf p =>
      exact evalAt_eq_evalNode_paramConst wf p params G P input vals id hGetNode
        (hConst.trans (by simp [lowerNode]))
  | add a b =>
      exact evalAt_eq_evalNode_add a b params G _ input vals id hShapes hGetNode
  | sub a b =>
      exact evalAt_eq_evalNode_sub a b params G _ input vals id hShapes hGetNode
  | mulElem a b =>
      exact evalAt_eq_evalNode_mulElem a b params G _ input vals id hShapes hGetNode
  | relu xIdx =>
      exact evalAt_eq_evalNode_relu xIdx params G _ input vals id hShapes hGetNode
  | exp xIdx =>
      exact evalAt_eq_evalNode_exp xIdx params G _ input vals id hShapes hGetNode
  | log xIdx =>
      exact evalAt_eq_evalNode_log xIdx params G _ input vals id hShapes hGetNode
  | inv xIdx =>
      exact evalAt_eq_evalNode_inv xIdx params G _ input vals id hShapes hGetNode
  | matmul op a b =>
      exact evalAt_eq_evalNode_matmul op a b params G _ input vals id hShapes hGetNode
  | reshape inS _ h xIdx =>
      exact evalAt_eq_evalNode_reshape inS _ h xIdx params G _ input vals id hShapes hGetNode
  | transpose axis₁ axis₂ hOut xIdx =>
      exact evalAt_eq_evalNode_transpose axis₁ axis₂ hOut xIdx params G _ input vals id hShapes
        hGetNode
  | softmax axis hAxis xIdx =>
      exact evalAt_eq_evalNode_softmax axis hAxis xIdx params G _ input vals id hShapes hGetNode
  | layerNorm op xIdx =>
      exact evalAt_eq_evalNode_layerNorm op xIdx params G P input vals id hShapes hGetNode
        (hLayerNorm.trans (by simp [lowerNode]))
  | linear inDim outDim w b xIdx =>
      exact evalAt_eq_evalNode_linear inDim outDim w b xIdx params G P input vals id hShapes
        hGetNode (hLin.trans (by simp [lowerNode]))
  | conv inC outC kernelShape stride padding inSpatial hIn hKernel hStride hInfer kernel bias
      xIdx =>
      exact evalAt_eq_evalNode_conv inC outC kernelShape stride padding inSpatial hIn hKernel
        hStride hInfer kernel bias xIdx params G P input vals id hShapes hGetNode
        (hConv.trans (by simp [lowerNode, loweredConvParams]))
  | mseLoss yhat target =>
      exact evalAt_eq_evalNode_mseLoss yhat target params G _ input vals id hShapes hGetNode

/-- The lowering accumulator after appending the lowering of one node at the fresh id. -/
def lowerStep
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {mid : Shape}
    (node : Node α paramShapes inShape ss mid) (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α) : NN.Verification.Builtin.LoweredIR α :=
  let res := lowerNode (α := α) c.graph.nodes.size node params c.ps
  { c with
      graph := { nodes := c.graph.nodes.push res.1 }
      ps := res.2
      outputId := c.graph.nodes.size }

/-- Lowering a `let1` chain lowers the head node and continues from the extended accumulator. -/
theorem lowerForwardLetChain_let1
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {mid out : Shape}
    (node : Node α paramShapes inShape ss mid)
    (gNext : ForwardLetChain α paramShapes inShape (ss ++ [mid]) out)
    (params : TorchLean.TensorPack α paramShapes) (c : NN.Verification.Builtin.LoweredIR α) :
    lowerForwardLetChain (α := α) (ForwardLetChain.let1 node gNext) params c =
      lowerForwardLetChain (α := α) gNext params (lowerStep node params c) := by
  rfl

/-- The id reserved by `lowerStep` stays in range after lowering the rest of the chain. -/
theorem size_lt_lowerForwardLetChain_lowerStep
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {mid out : Shape}
    (node : Node α paramShapes inShape ss mid)
    (gNext : ForwardLetChain α paramShapes inShape (ss ++ [mid]) out)
    (params : TorchLean.TensorPack α paramShapes) (c : NN.Verification.Builtin.LoweredIR α) :
    c.graph.nodes.size <
      (lowerForwardLetChain (α := α) gNext params (lowerStep node params c)).graph.nodes.size :=
  Nat.lt_of_lt_of_le (by simp [lowerStep])
    (lowerForwardLetChain_nodesSize_le gNext params (lowerStep node params c))

/--
The node lowered by `lowerStep` evaluates like its source node inside the fully lowered graph:
lowering the rest of the chain neither moves the node nor disturbs its payload entries.
-/
theorem evalAt_lowerForwardLetChain_lowerStep_eq_evalNode
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {mid out : Shape}
    (node : Node α paramShapes inShape ss mid)
    (gNext : ForwardLetChain α paramShapes inShape (ss ++ [mid]) out)
    (params : TorchLean.TensorPack α paramShapes) (c : NN.Verification.Builtin.LoweredIR α)
    (input : Spec.SomeTensor α) (vals : Array (Spec.SomeTensor α))
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    Graph.evalAt (α := α)
        (lowerForwardLetChain (α := α) gNext params (lowerStep node params c)).graph
        (payloadOfParamStore (α := α)
          (lowerForwardLetChain (α := α) gNext params (lowerStep node params c)).ps)
        input vals c.graph.nodes.size =
      evalNode (α := α) node params vals := by
  have hLt : c.graph.nodes.size < (lowerStep node params c).graph.nodes.size := by
    simp [lowerStep]
  have hnId : (lowerNode (α := α) c.graph.nodes.size node params c.ps).1.id =
      c.graph.nodes.size := by
    cases node <;> simp [lowerNode]
  refine evalAt_eq_evalNode_of_lowerNode node params c.ps c.graph.nodes.size _ _ input vals
    hShapes ?_ ?_ ?_ ?_ ?_
  · exact (lowerForwardLetChain_getNode_lt gNext params (lowerStep node params c) hLt).trans
      (by simp [lowerStep, Graph.getNode, Graph.getNode?, hnId])
  · exact (lowerForwardLetChain_ps_constVals_get?_lt gNext params (lowerStep node params c)
      hLt).trans rfl
  · exact (lowerForwardLetChain_ps_linearWB_get?_lt gNext params (lowerStep node params c)
      hLt).trans rfl
  · exact (lowerForwardLetChain_ps_convCfg_get?_lt gNext params (lowerStep node params c)
      hLt).trans rfl
  · exact (lowerForwardLetChain_ps_layerNorm_get?_lt gNext params (lowerStep node params c)
      hLt).trans rfl

/-- One in-range step of `denoteAllFrom`: evaluate node `i`, push it, and continue at `i + 1`. -/
theorem denoteAllFrom_eq_bind_of_lt
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (i : Nat) (vals : Array (Spec.SomeTensor α)) (hi : i < g.nodes.size) :
    Graph.denoteAllFrom (α := α) g payload input i vals =
      (do
        let v ← Graph.evalAt (α := α) g payload input vals i
        Graph.denoteAllFrom (α := α) g payload input (i + 1) (vals.push v)) := by
  rw [Graph.denoteAllFrom.eq_1]
  simp [hi]

/-- Pushing a successfully evaluated node value extends the shape context by its output shape. -/
theorem shapesOfVals_push_of_evalNode_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {mid : Shape}
    (node : Node α paramShapes inShape ss mid) (params : TorchLean.TensorPack α paramShapes)
    (vals : Array (Spec.SomeTensor α)) (v : Spec.SomeTensor α)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hEval : evalNode (α := α) node params vals = Except.ok v) :
    shapesOfVals (α := α) (vals.push v) = Ctx inShape (ss ++ [mid]) := by
  have hv : v.1 = mid := evalNode_ok_shape_of_hShapes node params vals hShapes hEval
  simp only [shapesOfVals_push, hShapes, hv, Ctx, List.cons_append]

/--
`denoteAllFrom` for the lowered IR agrees with the forward-fragment evaluator that returns all
intermediate values. Lowering preserves the full SSA value vector up to the current
lowering point, not only the final output.
-/
theorem denoteAllFrom_lowerForwardLetChain_eq_evalForwardLetChainVals
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α)
    (x : Tensor α inShape)
    (vals : Array (Spec.SomeTensor α))
    (hSize : vals.size = c.graph.nodes.size)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    (NN.IR.Graph.denoteAllFrom (α := α)
      (g := (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss) (out := out) g params c).graph)
      (payload := payloadOfParamStore (α := α)
        (lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss) (out := out) g params c).ps)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := c.graph.nodes.size)
      (vals := vals)
        =
    evalForwardLetChainVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params vals) := by
  induction g generalizing c vals with
  | ret y =>
      simp [lowerForwardLetChain, evalForwardLetChainVals, Graph.denoteAllFrom]
  | @let1 ss₀ mid₀ out₀ node gNext ih =>
      rw [lowerForwardLetChain_let1,
        denoteAllFrom_eq_bind_of_lt _ _ _ _ _
          (size_lt_lowerForwardLetChain_lowerStep node gNext params c),
        evalAt_lowerForwardLetChain_lowerStep_eq_evalNode node gNext params c _ vals hShapes]
      simp only [evalForwardLetChainVals]
      cases hEval : evalNode (α := α) node params vals with
      | error e =>
          rfl
      | ok vOut =>
          have hSize' : (vals.push vOut).size = (lowerStep node params c).graph.nodes.size := by
            simp [lowerStep, hSize]
          have hShapes' := shapesOfVals_push_of_evalNode_ok node params vals vOut hShapes hEval
          have hIH := ih (lowerStep node params c) (vals.push vOut) hSize' hShapes'
          simpa [lowerStep, Bind.bind, Except.bind] using hIH

end Correctness

end NN.Verification.Builtin.Proved
