/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.NodeShape
public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadBridge

/-!
# Compiled Forward Evaluation: SSA Denotation Agreement
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

open NN.Verification.TorchLean

set_option maxHeartbeats 2000000

/--
`denoteAllFrom` for the compiled IR agrees with the forward-fragment evaluator that returns all
intermediate values. Compilation preserves the full SSA value vector up to the current
compilation point, not only the final output.
-/
theorem denoteAllFrom_compileFGraph_eq_evalFGraphVals
    {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α)
    (x : Tensor α inShape)
    (vals : Array (DVal α))
    (hSize : vals.size = c.graph.nodes.size)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
    (NN.IR.Graph.denoteAllFrom (α := α)
      (g := (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss) (out := out) g params c).graph)
      (payload := payloadOfParamStore (α := α)
        (compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss) (out := out) g params c).ps)
      (input := DVal.mk (α := α) inShape x)
      (i := c.graph.nodes.size)
      (vals := vals)
        =
    evalFGraphVals (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
      (out := out) g params vals) := by
    classical
    induction g generalizing c vals with
    | ret y =>
        -- No more nodes: the compiled graph doesn't add nodes, so `denoteAllFrom` returns `vals`.
        simp [compileFGraph, evalFGraphVals, NN.IR.Graph.denoteAllFrom]
    | @let1 ss₀ mid₀ out₀ node gNext ih =>
      let id := c.graph.nodes.size
      let res :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
          mid₀)
          id node params c.ps
      let n : NN.IR.Node := res.1
      let ps' : NN.MLTheory.CROWN.Graph.ParamStore α := res.2
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      let cOut :=
        compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀ ++
          [mid₀]) (out := out₀)
          gNext params c'
      have hLt : id < cOut.graph.nodes.size := by
        have hmono :=
          compileFGraph_nodesSize_le (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c')
        have : id + 1 ≤ cOut.graph.nodes.size := by
          simpa [cOut, c', id, Array.size_push] using hmono
        exact Nat.lt_of_lt_of_le (Nat.lt_succ_self id) this
      -- Rewrite the goal to the one-step expansion and apply IH to the suffix.
      -- The only nontrivial work is showing `evalAt` matches `evalNode` at the fresh id.
      have hConst :
          cOut.ps.constVals.get? id = c'.ps.constVals.get? id :=
        compileFGraph_ps_constVals_get?_lt (α := α) (paramShapes := paramShapes) (inShape :=
          inShape)
          (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      have hLin :
          cOut.ps.linearWB.get? id = c'.ps.linearWB.get? id :=
        compileFGraph_ps_linearWB_get?_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
          (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      have hConv :
          cOut.ps.conv2dCfg.get? id = c'.ps.conv2dCfg.get? id :=
        compileFGraph_ps_conv2dCfg_get?_lt (α := α) (paramShapes := paramShapes)
          (inShape := inShape) (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext)
          (params := params) (c := c') (hk := by
            simp [c', id, Array.size_push])
      -- `getNode` at the fresh index is exactly the freshly pushed node.
      have hnId : n.id = id := by
        cases node <;> simp [n, res, compileNode, id]
      have hGetNode : NN.IR.Graph.getNode (g := cOut.graph) id = pure n := by
        have hPres :=
          compileFGraph_getNode_lt (α := α) (paramShapes := paramShapes) (inShape := inShape)
            (ss := ss₀ ++ [mid₀]) (out := out₀) (g := gNext) (params := params) (c := c')
            (i := id) (hi := by simp [c', id, Array.size_push])
        have hAtPush : NN.IR.Graph.getNode (g := c'.graph) id = pure n := by
          simp [NN.IR.Graph.getNode, NN.IR.Graph.getNode?, c', id, hnId]
        simpa [cOut] using Eq.trans hPres hAtPush
      -- One-step correctness: IR `evalAt` at the fresh id matches `evalNode`.
      have hStep :
          NN.IR.Graph.evalAt (α := α)
              (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (vals := vals) (i := id)
            =
          evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out :=
            mid₀)
              node params vals := by
        classical
        -- reduce to the freshly pushed IR node, then finish by cases on the source node
        -- (doing this case-by-case avoids `simp` timeouts on the full IR evaluator).
        cases node with
        | const wf t =>
            letI : Shape.WellFormed mid₀ := wf
            -- Show the const payload is present and evaluates back to `t`.
            let flat : NN.MLTheory.CROWN.Graph.FlatVec α :=
              flatOfTensor (α := α) (s := mid₀) wf t
            have hnKind : n.kind = .const mid₀ := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            have hGet' : c'.ps.constVals.get? id = some flat := by
              have hIns : (c.ps.constVals.insert id flat).get? id = some flat := by
                -- Use the `m[k]?` lemma; it is definitionaly `m.get? k`.
                simp
              -- `c'.ps.constVals = c.ps.constVals.insert id flat`.
              simp [c', res, compileNode, ps', flat]
            have hGet : cOut.ps.constVals.get? id = some flat :=
              hConst.trans hGet'
            have hStoreConst :
                cOut.ps.constVals.get? id =
                  some ({ n := Shape.size mid₀, v := t.flattenSpec } :
                    NN.MLTheory.CROWN.Graph.FlatVec α) := by
              simpa [flat, flatOfTensor] using hGet
            have hUF : unflattenSpec mid₀ t.flattenSpec = t := by
              simpa using (Spec.Tensor.flatten_unflatten_inverse_wf (α := α) (s := mid₀) (t := t))
            have hn :
                n = { id := id, parents := [], kind := .const mid₀, outShape := mid₀ } := by
              simp [n, res, compileNode]
            have hGetNodeConst :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure ({ id := id, parents := [], kind := .const mid₀, outShape := mid₀ } :
                    NN.IR.Node) := by
              simp [hGetNode, hn]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ t) := by
              simpa [hUF] using
                IRStep.evalAt_const_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (s := mid₀) (inputShape := inShape) (input := x) (vals := vals)
                  (v := t.flattenSpec)
                  hGetNodeConst hStoreConst
            have hEvalNode :
                evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out
                  := mid₀)
                    (Node.const wf t) params vals
                  =
                Except.ok (DVal.mk (α := α) mid₀ t) := by
              rfl
            simp [hEvalNode]
            exact hEvalAt
        | paramConst wf p =>
            letI : Shape.WellFormed mid₀ := wf
            -- Same as `const`, but the stored constant comes from `params`.
            let tp : Tensor α mid₀ := getParam (α := α) (paramShapes := paramShapes) params p
            let flat : NN.MLTheory.CROWN.Graph.FlatVec α :=
              flatOfTensor (α := α) (s := mid₀) wf tp
            have hGet' : c'.ps.constVals.get? id = some flat := by
              simp [c', res, compileNode, ps', flat, tp]
            have hGet : cOut.ps.constVals.get? id = some flat :=
              hConst.trans hGet'
            have hStoreConst :
                cOut.ps.constVals.get? id =
                  some ({ n := Shape.size mid₀, v := tp.flattenSpec } :
                    NN.MLTheory.CROWN.Graph.FlatVec α) := by
              simpa [flat, flatOfTensor] using hGet
            have hUF : unflattenSpec mid₀ tp.flattenSpec = tp := by
              simpa using (Spec.Tensor.flatten_unflatten_inverse_wf (α := α) (s := mid₀) (t :=
                tp))
            have hnKind : n.kind = .const mid₀ := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ tp) := by
              have hn :
                  n = { id := id, parents := [], kind := .const mid₀, outShape := mid₀ } := by
                simp [n, res, compileNode]
              have hGetNodeConst :
                  NN.IR.Graph.getNode (g := cOut.graph) id =
                    pure ({ id := id, parents := [], kind := .const mid₀, outShape := mid₀ } :
                      NN.IR.Node) := by
                simp [hGetNode, hn]
              simpa [hUF] using
                IRStep.evalAt_const_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (s := mid₀) (inputShape := inShape) (input := x) (vals := vals)
                  (v := tp.flattenSpec)
                  hGetNodeConst hStoreConst
            have hEvalNode :
                evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out
                  := mid₀)
                    (Node.paramConst wf p) params vals
                  =
                Except.ok (DVal.mk (α := α) mid₀ tp) := by
              rfl
            simpa [hEvalNode] using hEvalAt
        | add a b =>
            have ha : (vals[a.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := a) (s
                  := mid₀)
            have hb : (vals[b.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := b) (s
                  := mid₀)
            have hnKind : n.kind = .add := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let ta : Tensor α mid₀ := ha ▸ (vals[a.id]!).snd
            let tb : Tensor α mid₀ := hb ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[a.id]!) = Except.ok ta :=
                  by
              simpa [ta] using expectShape_eq_ok (expected := mid₀) (v := vals[a.id]!) ha
            have hExpectB :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[b.id]!) = Except.ok tb :=
                  by
              simpa [tb] using expectShape_eq_ok (expected := mid₀) (v := vals[b.id]!) hb
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals a = Except.ok ta
                  := by
              simpa [ta] using getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := a) ha
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals b = Except.ok tb
                  := by
              simpa [tb] using getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := b) hb
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.addSpec (α := α) ta tb)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hnOut]
              rw [hExpectA, hExpectB]
              rfl
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb, Except.bind, Except.pure, bind, pure] using hEvalAt
        | sub a b =>
            have ha : (vals[a.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := a) (s
                  := mid₀)
            have hb : (vals[b.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := b) (s
                  := mid₀)
            have hnKind : n.kind = .sub := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let ta : Tensor α mid₀ := ha ▸ (vals[a.id]!).snd
            let tb : Tensor α mid₀ := hb ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[a.id]!) = Except.ok ta :=
                  by
              simpa [ta] using expectShape_eq_ok (expected := mid₀) (v := vals[a.id]!) ha
            have hExpectB :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[b.id]!) = Except.ok tb :=
                  by
              simpa [tb] using expectShape_eq_ok (expected := mid₀) (v := vals[b.id]!) hb
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals a = Except.ok ta
                  := by
              simpa [ta] using getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := a) ha
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals b = Except.ok tb
                  := by
              simpa [tb] using getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := b) hb
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.subSpec (α := α) ta tb)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hnOut]
              rw [hExpectA, hExpectB]
              rfl
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb, Except.bind, Except.pure, bind, pure] using hEvalAt
        | mulElem a b =>
            have ha : (vals[a.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := a) (s
                  := mid₀)
            have hb : (vals[b.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := b) (s
                  := mid₀)
            have hnKind : n.kind = .mul_elem := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let ta : Tensor α mid₀ := ha ▸ (vals[a.id]!).snd
            let tb : Tensor α mid₀ := hb ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[a.id]!) = Except.ok ta :=
                  by
              simpa [ta] using expectShape_eq_ok (expected := mid₀) (v := vals[a.id]!) ha
            have hExpectB :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[b.id]!) = Except.ok tb :=
                  by
              simpa [tb] using expectShape_eq_ok (expected := mid₀) (v := vals[b.id]!) hb
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals a = Except.ok ta
                  := by
              simpa [ta] using getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := a) ha
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals b = Except.ok tb
                  := by
              simpa [tb] using getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := b) hb
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.mulSpec (α := α) ta tb)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hnOut]
              rw [hExpectA, hExpectB]
              rfl
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb, Except.bind, Except.pure, bind, pure] using hEvalAt
        | relu xIdx =>
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hnKind : n.kind = .relu := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hx ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              simpa [tx] using
                getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := xIdx) hx
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Activation.reluSpec (α := α) tx)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hnOut]
              rw [hExpectX]
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | exp xIdx =>
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hnKind : n.kind = .exp := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hx ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              simpa [tx] using
                getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := xIdx) hx
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.expSpec (α := α) tx)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hnOut]
              rw [hExpectX]
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | log xIdx =>
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hnKind : n.kind = .log := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hx ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              simpa [tx] using
                getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := xIdx) hx
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) mid₀
                    (if Tensor.allSpec (α := α) (s := mid₀) (fun v => decide (0 < v)) tx then
                      Tensor.logSpec (α := α) tx
                    else
                      panic!
                        ("IR eval: log: input contains values <= 0 (or NaN); " ++
                          "use `safe_log` if you want epsilon protection"))) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hnOut]
              rw [hExpectX]
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | inv xIdx =>
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hnKind : n.kind = .inv := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hx ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hx]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              simpa [tx] using
                getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := xIdx) hx
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok (DVal.mk (α := α) mid₀ (Tensor.invSpec (α := α) tx)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hnOut]
              rw [hExpectX]
              rfl
            simpa [evalNode, hGetValX, DVal.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | matmul2d m nDim p a b =>
            have ha : (vals[a.id]!).1 = .dim m (.dim nDim .scalar) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := a) (s := .dim m (.dim nDim .scalar))
            have hb : (vals[b.id]!).1 = .dim nDim (.dim p .scalar) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := b) (s := .dim nDim (.dim p .scalar))
            have haF : (vals[a.id]!).fst = .dim m (.dim nDim .scalar) := by
              simpa using ha
            have hbF : (vals[b.id]!).fst = .dim nDim (.dim p .scalar) := by
              simpa using hb
            have hnKind : n.kind = .matmul := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim m (.dim p .scalar) := by
              simp [compileNode, res, n]
            let ta : Tensor α (.dim m (.dim nDim .scalar)) := haF ▸ (vals[a.id]!).snd
            let tb : Tensor α (.dim nDim (.dim p .scalar)) := hbF ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim m (.dim nDim .scalar)) (vals[a.id]!) =
                  Except.ok ta := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos haF]
              rfl
            have hExpectB :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim nDim (.dim p .scalar)) (vals[b.id]!) =
                  Except.ok tb := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hbF]
              rfl
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim m (.dim nDim .scalar)) vals a =
                  Except.ok ta := by
              simpa [ta] using getVal_eq_ok_of_hShapes (vals := vals)
                (expected := .dim m (.dim nDim .scalar)) (idx := a) haF
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim nDim (.dim p .scalar)) vals b =
                  Except.ok tb := by
              simpa [tb] using getVal_eq_ok_of_hShapes (vals := vals)
                (expected := .dim nDim (.dim p .scalar)) (idx := b) hbF
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim m (.dim p .scalar))
                    (Tensor.matMulSpec (α := α) (m := m) (n := nDim) (p := p) ta tb)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, haF, hbF, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectA, hExpectB]
              split_ifs with hOut
              · cases hOut
                rfl
              · exfalso
                exact hOut (by simp [hnOut])
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb, Except.bind, Except.pure, bind, pure] using hEvalAt
        | bmm batch m nDim p a b =>
            have ha : (vals[a.id]!).1 = .dim batch (.dim m (.dim nDim .scalar)) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := a) (s := .dim batch (.dim m (.dim nDim .scalar)))
            have hb : (vals[b.id]!).1 = .dim batch (.dim nDim (.dim p .scalar)) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := b) (s := .dim batch (.dim nDim (.dim p .scalar)))
            have haF : (vals[a.id]!).fst = .dim batch (.dim m (.dim nDim .scalar)) := by
              simpa using ha
            have hbF : (vals[b.id]!).fst = .dim batch (.dim nDim (.dim p .scalar)) := by
              simpa using hb
            have hnKind : n.kind = .matmul := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [a.id, b.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim batch (.dim m (.dim p .scalar)) := by
              simp [compileNode, res, n]
            let ta : Tensor α (.dim batch (.dim m (.dim nDim .scalar))) := haF ▸ (vals[a.id]!).snd
            let tb : Tensor α (.dim batch (.dim nDim (.dim p .scalar))) := hbF ▸ (vals[b.id]!).snd
            have hExpectA :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim batch (.dim m (.dim nDim .scalar))) (vals[a.id]!) =
                  Except.ok ta := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos haF]
              rfl
            have hExpectB :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim batch (.dim nDim (.dim p .scalar))) (vals[b.id]!) =
                  Except.ok tb := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hbF]
              rfl
            have hGetValA :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim batch (.dim m (.dim nDim .scalar))) vals a =
                  Except.ok ta := by
              simpa [ta] using getVal_eq_ok_of_hShapes (vals := vals)
                (expected := .dim batch (.dim m (.dim nDim .scalar))) (idx := a) haF
            have hGetValB :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim batch (.dim nDim (.dim p .scalar))) vals b =
                  Except.ok tb := by
              simpa [tb] using getVal_eq_ok_of_hShapes (vals := vals)
                (expected := .dim batch (.dim nDim (.dim p .scalar))) (idx := b) hbF
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim batch (.dim m (.dim p .scalar)))
                    (Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := nDim) (p := p) ta tb))
                      := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, haF, hbF, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectA, hExpectB]
              split_ifs with hOut
              · cases hOut
                rfl
              · exfalso
                exact hOut (by simp [hnOut])
            simpa [evalNode, hGetValA, hGetValB, DVal.mk, ta, tb, Except.bind, Except.pure, bind, pure] using hEvalAt
        | reshape inS mid₀ h xIdx =>
            have hx : (vals[xIdx.id]!).1 = inS := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := inS)
            have hxF : (vals[xIdx.id]!).fst = inS := by
              simpa using hx
            have hnKind : n.kind = .reshape inS mid₀ := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α inS := hxF ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := inS) (vals[xIdx.id]!) = Except.ok tx
                  := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := inS) vals xIdx = Except.ok tx
                  := by
              simpa [tx] using
                getVal_eq_ok_of_hShapes (vals := vals) (expected := inS) (idx := xIdx) hxF
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) mid₀
                    (Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := mid₀) tx h)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX]
              simp [h]
              split_ifs with hOut
              · cases hOut
                rfl
              · exfalso
                exact hOut (by simp [hnOut])
            simpa [evalNode, hGetValX, DVal.mk, h, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | swap_first_two m nDim rest xIdx =>
            have hx : (vals[xIdx.id]!).1 = .dim m (.dim nDim rest) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim m (.dim nDim rest))
            have hxF : (vals[xIdx.id]!).fst = .dim m (.dim nDim rest) := by
              simpa using hx
            have hnKind : n.kind = .swap_first_two := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim nDim (.dim m rest) := by
              simp [compileNode, res, n]
            let tx : Tensor α (.dim m (.dim nDim rest)) := hxF ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := .dim m (.dim nDim rest))
                  (vals[xIdx.id]!) =
                  Except.ok tx := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := .dim m (.dim nDim rest)) vals
                  xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok_of_hShapes (vals := vals)
                (expected := .dim m (.dim nDim rest)) (idx := xIdx) hxF
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim nDim (.dim m rest))
                    (Tensor.swapFirstTwoSpec (α := α) (m := m) (n := nDim) (s := rest) tx)) := by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, hnOut, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX]
              simp [hnOut]
            simpa [evalNode, hGetValX, DVal.mk, tx, hnOut, Except.bind, Except.pure, bind, pure] using hEvalAt
        | transpose3dLastTwo a b c xIdx =>
            have hx : (vals[xIdx.id]!).1 = .dim a (.dim b (.dim c .scalar)) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim a (.dim b (.dim c .scalar)))
            have hxF : (vals[xIdx.id]!).fst = .dim a (.dim b (.dim c .scalar)) := by
              simpa using hx
            have hnKind : n.kind = .transpose3dLastTwo := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim a (.dim c (.dim b .scalar)) := by
              simp [compileNode, res, n]
            let tx : Tensor α (.dim a (.dim b (.dim c .scalar))) := hxF ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim a (.dim b (.dim c .scalar))) (vals[xIdx.id]!) =
                  Except.ok tx := by
              unfold NN.IR.Graph.expectShape
              simp [DVal.shape]
              rw [dif_pos hxF]
              rfl
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim a (.dim b (.dim c .scalar))) vals xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok_of_hShapes (vals := vals)
                (expected := .dim a (.dim b (.dim c .scalar))) (idx := xIdx) hxF
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim a (.dim c (.dim b .scalar)))
                    (Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) tx)) :=
                      by
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, hnOut, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hExpectX]
              simp [hnOut]
            simpa [evalNode, hGetValX, DVal.mk, tx, hnOut, Except.bind, Except.pure, bind, pure] using hEvalAt
        | softmaxLast hRank xIdx =>
            have hAxis : (Shape.rank mid₀ - 1) + 1 = Shape.rank mid₀ := by
              exact Nat.sub_add_cancel (Nat.succ_le_of_lt hRank)
            have hAxisValid : OpContracts.checkAxisValid (Shape.rank mid₀ - 1) mid₀ = .ok () := by
              unfold OpContracts.checkAxisValid
              have hLt : Shape.rank mid₀ - 1 < Shape.rank mid₀ := by
                cases hR : Shape.rank mid₀ with
                | zero =>
                    simp [hR] at hRank
                | succ r =>
                    simp
              simp [hLt]
              rfl
            have hx : (vals[xIdx.id]!).1 = mid₀ := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := xIdx)
                  (s := mid₀)
            have hxF : (vals[xIdx.id]!).fst = mid₀ := by
              simpa using hx
            have hnKind : n.kind = .softmax (Shape.rank mid₀ - 1) := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = mid₀ := by
              simp [compileNode, res, n]
            let tx : Tensor α mid₀ := hxF ▸ (vals[xIdx.id]!).snd
            have hExpectX :
                NN.IR.Graph.expectShape (α := α) (expected := mid₀) (vals[xIdx.id]!) = Except.ok tx
                  := by
              simpa [tx] using expectShape_eq_ok (expected := mid₀) (v := vals[xIdx.id]!) hxF
            have hGetValX :
                getVal (α := α) (inShape := inShape) (ss := ss₀) (s := mid₀) vals xIdx = Except.ok
                  tx := by
              simpa [tx] using getVal_eq_ok_of_hShapes (vals := vals) (expected := mid₀) (idx := xIdx) hxF
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) mid₀
                    (Activation.softmaxSpec (α := α) tx)) := by
              have hAxisValid' :
                  OpContracts.checkAxisValid (Shape.rank mid₀ - 1) n.outShape = Except.ok () := by
                simpa [hnOut] using hAxisValid
              have hAxis' : (Shape.rank mid₀ - 1) + 1 = Shape.rank n.outShape := by
                simpa [hnOut] using hAxis
              unfold NN.IR.Graph.evalAt
              simp [hGetNode, hnKind, hnParents, DVal.shape, DVal.tensor, DVal.mk,
                throw, throwThe, MonadExceptOf.throw]
              rw [hnOut]
              rw [hAxisValid, hExpectX]
              simp [hAxis]
            simpa [evalNode, hGetValX, DVal.mk, tx, Except.bind, Except.pure, bind, pure] using hEvalAt
        | layernorm2d seqLen embedDim hSeq hEmb xIdx =>
            have hParams :
                OpContracts.layerNorm2DParams 1
                    (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) =
                  .ok (seqLen, embedDim) := by
              have hSeqNe : seqLen ≠ 0 := Nat.ne_of_gt hSeq
              have hEmbNe : embedDim ≠ 0 := Nat.ne_of_gt hEmb
              -- For a `(seqLen × embedDim)` tensor, `axis=1` normalizes the last dimension.
              simp [OpContracts.layerNorm2DParams, OpContracts.checkAxisValid,
                OpContracts.checkPositive, Shape.toList, hSeqNe, hEmbNe]
              rfl
            have hx : (vals[xIdx.id]!).1 = .dim seqLen (.dim embedDim .scalar) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim seqLen (.dim embedDim .scalar))
            have hxF : (vals[xIdx.id]!).fst = .dim seqLen (.dim embedDim .scalar) := by
              simpa using hx
            have hnKind : n.kind = .layernorm 1 := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [xIdx.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .dim seqLen (.dim embedDim .scalar) := by
              simp [compileNode, res, n]
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .layernorm 1
                     outShape := .dim seqLen (.dim embedDim .scalar) } : NN.IR.Node) := by
              simp [n, res, compileNode]
            have hGetNodeLN :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := [xIdx.id]
                       kind := .layernorm 1
                       outShape := .dim seqLen (.dim embedDim .scalar) } : NN.IR.Node) := by
              simp [hGetNode, hn]
            cases hnOut
            let tx : Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
              hxF ▸ (vals[xIdx.id]!).snd
            have hExpect :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim seqLen (.dim embedDim .scalar)) (vals[xIdx.id]!) =
                  Except.ok tx := by
              simpa [tx] using
                expectShape_eq_ok (expected := .dim seqLen (.dim embedDim .scalar)) (v :=
                  vals[xIdx.id]!) hxF
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim seqLen (.dim embedDim .scalar)) vals xIdx =
                  Except.ok tx := by
              simpa [tx] using getVal_eq_ok_of_hShapes (vals := vals)
                (expected := .dim seqLen (.dim embedDim .scalar)) (idx
                := xIdx) hxF
            have hLN :
                NN.IR.Graph.layernormPure (α := α) (seqLen := seqLen) (embedDim := embedDim)
                    (Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      tx rfl)
                  =
                Except.ok
                  (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                    (x := Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      tx rfl)
                    (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
                    (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
                    (h_seq_pos := hSeq) (h_embed_pos := hEmb)) := by
              simp [NN.IR.Graph.layernormPure, hSeq, hEmb]
              rfl
            have hNumel :
                Shape.size (.dim seqLen (.dim embedDim .scalar)) =
                  Shape.size (.dim seqLen (.dim embedDim .scalar)) := rfl
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim seqLen (.dim embedDim .scalar))
                    (Tensor.reshapeSpec (α := α)
                      (s₁ := .dim seqLen (.dim embedDim .scalar))
                      (s₂ := .dim seqLen (.dim embedDim .scalar))
                      (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                        (x := Tensor.reshapeSpec (α := α)
                          (s₁ := .dim seqLen (.dim embedDim .scalar))
                          (s₂ := .dim seqLen (.dim embedDim .scalar))
                          tx rfl)
                        (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
                        (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
                        (h_seq_pos := hSeq) (h_embed_pos := hEmb))
                      rfl)) := by
                simp [NN.IR.Graph.evalAt, hGetNodeLN, hExpect, hParams,
                  DVal.shape, DVal.tensor, DVal.mk,
                  throw, throwThe, MonadExceptOf.throw]
                simpa [hExpect, hParams, hNumel, DVal.mk, Except.bind, Except.pure, bind, pure] using
                  congrArg
                    (fun e =>
                      (fun a : Tensor α (.dim seqLen (.dim embedDim .scalar)) =>
                        DVal.mk (α := α) (.dim seqLen (.dim embedDim .scalar))
                          (Tensor.reshapeSpec (α := α)
                            (s₁ := .dim seqLen (.dim embedDim .scalar))
                            (s₂ := .dim seqLen (.dim embedDim .scalar))
                            a rfl)) <$> e)
                    hLN
            simpa [evalNode, hGetVal, DVal.shape, DVal.tensor, hxF, DVal.mk,
              tx, Tensor.reshapeSpec, Tensor.flatten_unflatten_inverse] using hEvalAt
        | linear inDim outDim w b xIdx =>
            let wT : Tensor α (.dim outDim (.dim inDim .scalar)) :=
              getParam (α := α) (paramShapes := paramShapes) params w
            let bT : Tensor α (.dim outDim .scalar) :=
              getParam (α := α) (paramShapes := paramShapes) params b
            have hx : (vals[xIdx.id]!).1 = .dim inDim .scalar := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim inDim .scalar)
            have hxF : (vals[xIdx.id]!).fst = .dim inDim .scalar := by
              simpa using hx
            have hLin' : cOut.ps.linearWB[n.id]? = c'.ps.linearWB[n.id]? := by
              simpa [hnId] using hLin
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .linear
                     outShape := .dim outDim .scalar } : NN.IR.Node) := by
              simp [n, res, compileNode]
            have hGetNodeLinear :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := [xIdx.id]
                       kind := .linear
                       outShape := .dim outDim .scalar } : NN.IR.Node) := by
              simp [hGetNode, hn]
            let xT : Tensor α (.dim inDim .scalar) :=
              hxF ▸ (vals[xIdx.id]!).snd
            have hExpectIn :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim inDim .scalar) (vals[xIdx.id]!) =
                  Except.ok xT := by
              simpa [xT] using expectShape_eq_ok (expected := .dim inDim .scalar) (v :=
                vals[xIdx.id]!) hxF
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim inDim .scalar) vals xIdx =
                  Except.ok xT := by
              simpa [xT] using getVal_eq_ok_of_hShapes (vals := vals) (expected := .dim inDim .scalar)
                (idx := xIdx) hxF
            have hLinearStore :
                cOut.ps.linearWB.get? id =
                  some ({ m := outDim, n := inDim, w := wT, b := bT } :
                    NN.MLTheory.CROWN.Graph.LinParams α) := by
              rw [hLin]
              simp [c', res, compileNode, ps', wT, bT]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) (.dim outDim .scalar)
                      (Tensor.addSpec (α := α)
                        (Tensor.matVecMulSpec (α := α) (m := outDim) (n := inDim) wT xT) bT)) := by
              simpa [Spec.matVecMulSpec] using
                IRStep.evalAt_linear_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (pId := xIdx.id) (outDim := outDim) (inDim := inDim)
                  (inputShape := inShape) (input := x) (vals := vals)
                  (W := wT) (b := bT) (x := xT)
                  hGetNodeLinear hExpectIn hLinearStore
            simpa [evalNode, hGetVal, DVal.shape, DVal.tensor, DVal.mk, hxF, xT, wT, bT] using
              hEvalAt
        | conv2d inC outC kH kW stride padding inH inW hIn hKH hKW hStride hHeight hWidth kernel bias xIdx =>
            let kT : Tensor α (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) :=
              getParam (α := α) (paramShapes := paramShapes) params kernel
            let bT : Tensor α (.dim outC .scalar) :=
              getParam (α := α) (paramShapes := paramShapes) params bias
            let outShape : Shape :=
              .dim outC
                (.dim ((inH + 2 * padding - kH) / stride + 1)
                  (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))
            have hx : (vals[xIdx.id]!).1 = .dim inC (.dim inH (.dim inW .scalar)) := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes)
                  (idx := xIdx) (s := .dim inC (.dim inH (.dim inW .scalar)))
            have hxF : (vals[xIdx.id]!).fst = .dim inC (.dim inH (.dim inW .scalar)) := by
              simpa using hx
            have hn :
                n =
                  ({ id := id
                     parents := [xIdx.id]
                     kind := .conv2d inC outC kH kW stride padding
                     outShape := outShape } : NN.IR.Node) := by
              simp [n, res, compileNode, outShape]
            have hGetNodeConv :
                NN.IR.Graph.getNode (g := cOut.graph) id =
                  pure
                    ({ id := id
                       parents := [xIdx.id]
                       kind := .conv2d inC outC kH kW stride padding
                       outShape := outShape } : NN.IR.Node) := by
              simp [hGetNode, hn]
            let xT : Tensor α (.dim inC (.dim inH (.dim inW .scalar))) :=
              hxF ▸ (vals[xIdx.id]!).snd
            have hExpectIn :
                NN.IR.Graph.expectShape (α := α)
                    (expected := .dim inC (.dim inH (.dim inW .scalar))) (vals[xIdx.id]!) =
                  Except.ok xT := by
              simpa [xT] using
                expectShape_eq_ok (expected := .dim inC (.dim inH (.dim inW .scalar)))
                  (v := vals[xIdx.id]!) hxF
            have hGetVal :
                getVal (α := α) (inShape := inShape) (ss := ss₀)
                    (s := .dim inC (.dim inH (.dim inW .scalar))) vals xIdx =
                  Except.ok xT := by
              simpa [xT] using getVal_eq_ok_of_hShapes (vals := vals)
                (expected := .dim inC (.dim inH (.dim inW .scalar))) (idx := xIdx) hxF
            let spec : Spec.Conv2DSpec inC outC kH kW stride padding α hIn hKH hKW :=
              { kernel := kT, bias := bT }
            let cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α :=
              { inC := inC, outC := outC, kH := kH, kW := kW
                stride := stride, padding := padding
                inH := inH, inW := inW
                hIn := hIn, hKH := hKH, hKW := hKW, hStride := hStride,
                spec := spec }
            have hConvStore :
                cOut.ps.conv2dCfg.get? id = some cfg := by
              rw [hConv]
              simp [c', res, compileNode, ps', cfg, spec, kT, bT]
            have hEvalAt :
                NN.IR.Graph.evalAt (α := α)
                    (g := cOut.graph)
                    (payload := payloadOfParamStore (α := α) cOut.ps)
                    (input := DVal.mk (α := α) inShape x)
                    (vals := vals) (i := id)
                  =
                Except.ok
                  (DVal.mk (α := α) outShape
                    (Spec.conv2dSpec (α := α) (layer := spec) (input := xT))) := by
              simpa [cfg, spec, outShape] using
                IRStep.evalAt_conv2d_from_paramStore_of_getNode
                  (α := α) (g := cOut.graph) (ps := cOut.ps) (i := id) (id := id)
                  (pId := xIdx.id) (cfg := cfg) (inputShape := inShape) (input := x)
                  (vals := vals) (x := xT)
                  hGetNodeConv hExpectIn hConvStore hHeight hWidth
            simpa [evalNode, hGetVal, DVal.shape, DVal.tensor, DVal.mk, hxF, xT, kT, bT,
              spec, outShape] using hEvalAt
        | mseLoss yhat target =>
            rename_i s
            let yV : DVal α := vals[yhat.id]!
            let tV : DVal α := vals[target.id]!
            have hSomeY : vals[yhat.id]? = some yV := by
              simpa [yV] using
                val_get?_eq_some_of_hShapes (α := α) (vals := vals) (idx := yhat)
                  (hShapes := hShapes)
            have hSomeT : vals[target.id]? = some tV := by
              simpa [tV] using
                val_get?_eq_some_of_hShapes (α := α) (vals := vals) (idx := target)
                  (hShapes := hShapes)
            have hy : (vals[yhat.id]!).1 = s := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx := yhat)
                  (s := s)
            have ht : (vals[target.id]!).1 = s := by
              simpa [DVal.shape] using
                shape_of_vals_of_hShapes (α := α) (vals := vals) (hShapes := hShapes) (idx :=
                  target) (s := s)
            have hnKind : n.kind = .mseLoss := by
              simp [compileNode, res, n]
            have hnParents : n.parents = [yhat.id, target.id] := by
              simp [compileNode, res, n]
            have hnOut : n.outShape = .scalar := by
              simp [compileNode, res, n]
            -- `evalAt` and `evalNode` do the same dynamic check (`yhat.shape = target.shape`) and
            -- then compute
            -- the same scalar MSE.
            simp (config := { zeta := true })
              [NN.IR.Graph.evalAt, NN.IR.Graph.mseLossDVal, hGetNode, hnKind, hnParents, hnOut,
              evalNode, getDVal?, yV, tV, hSomeY, hSomeT, hy, ht, DVal.shape, DVal.tensor,
              DVal.mk, Bind.bind, Except.bind, Except.pure, bind, pure, Except.pure, Pure.pure]
      -- Unfold both evaluators one step, then dispatch by cases on the shared `evalNode`.
      have hStart :
          NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (i := id) (vals := vals)
            =
          (do
            let v ← NN.IR.Graph.evalAt (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (vals := vals) (i := id)
            NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (i := id + 1) (vals := vals.push v)) := by
        -- Unfold `denoteAllFrom` once at the top-level; don't simp-recursively unfold the recursive
        -- call.
        rw [NN.IR.Graph.denoteAllFrom.eq_1]
        simp [hLt]
      -- Rewrite the goal using `hStart` and the one-step lemma `hStep`.
      have hStart' :
          NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (i := id) (vals := vals)
            =
          (do
            let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss :=
              ss₀) (out := mid₀)
              node params vals
            NN.IR.Graph.denoteAllFrom (α := α) (g := cOut.graph)
              (payload := payloadOfParamStore (α := α) cOut.ps)
              (input := DVal.mk (α := α) inShape x)
              (i := id + 1) (vals := vals.push vOut)) := by
        -- Rewrite `denoteAllFrom` once, then replace `evalAt` with the already-verified `hStep`.
        -- Doing this explicitly avoids `simp` unfolding `DVal.mk` too early, which can prevent
        -- matching on the `hStart` rewrite.
        rw [hStart]
        have hStep' :
            NN.IR.Graph.evalAt (α := α) (g := cOut.graph)
                (payload := payloadOfParamStore (α := α) cOut.ps)
                (input := ⟨inShape, x⟩)
                (vals := vals) (i := id)
              =
            evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀) (out := mid₀)
              node params vals := by
          simpa [DVal.mk] using hStep
        simp [hStep']
      -- Now split on the result of `evalNode`.
      cases hEval : evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss₀)
        (out := mid₀)
          node params vals with
        | error e =>
            -- If the next DSL node fails, both evaluators stop immediately with the same error.
            -- First unfold compilation/evaluation one step so the goal is stated in terms of
            -- `cOut`.
            simp [compileFGraph, evalFGraphVals]
            -- The simp step above rewrites the compiled graph to `cOut.graph`, but may unfold
            -- `DVal.mk` to `⟨_,_⟩`. Normalize before rewriting with `hStart'`.
            have hStart'' :
                cOut.graph.denoteAllFrom
                    (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) id vals
                  =
                (do
                  let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                    (ss := ss₀) (out := mid₀) node params vals
                  cOut.graph.denoteAllFrom
                    (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) (id + 1) (vals.push vOut)) := by
              simpa [DVal.mk] using hStart'
            rw [hStart'']
            simp [hEval]
            rfl
        | ok vOut =>
          have hSize' : (vals.push vOut).size = c'.graph.nodes.size := by
            simp [c', hSize, Array.size_push]
          have hvOutShape : vOut.1 = mid₀ :=
            evalNode_ok_shape_of_hShapes (α := α) (paramShapes := paramShapes) (inShape := inShape)
              (ss := ss₀) (out := mid₀) node params vals hShapes (v := vOut) (by simp [hEval])
          have hShapes' : shapesOfVals (α := α) (vals.push vOut) = Ctx inShape (ss₀ ++ [mid₀]) := by
            calc
              shapesOfVals (α := α) (vals.push vOut)
                  = shapesOfVals (α := α) vals ++ [vOut.1] := shapesOfVals_push (α := α) (vals :=
                    vals) (v := vOut)
              _ = Ctx inShape ss₀ ++ [vOut.1] := by simp [hShapes]
              _ = (inShape :: ss₀) ++ [mid₀] := by simp [Ctx, hvOutShape]
              _ = Ctx inShape (ss₀ ++ [mid₀]) := by simp [Ctx, List.cons_append]
          have hIH :=
            ih (c := c') (vals := vals.push vOut) (hSize := hSize') (hShapes := hShapes')
          -- Rewrite the overall goal to the suffix goal (start at `id+1`), then discharge with IH.
          simp [compileFGraph, evalFGraphVals]
          have hStart'' :
              cOut.graph.denoteAllFrom
                  (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) id vals
                =
              (do
                let vOut ← evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape)
                  (ss := ss₀) (out := mid₀) node params vals
                cOut.graph.denoteAllFrom
                  (payloadOfParamStore (α := α) cOut.ps) (⟨inShape, x⟩) (id + 1) (vals.push vOut)) := by
            simpa [DVal.mk] using hStart'
          rw [hStart'']
          simp [hEval]
          -- now the goal is exactly the suffix IH (start index is `id+1 = c'.graph.nodes.size`).
          -- `DVal.mk` is definitional `⟨_,_⟩`, but the pretty-printer may choose either form; normalize
          -- before applying the IH.
          simpa [c', id, DVal.mk, Except.bind, Except.pure, bind, pure] using hIH
end Correctness

end NN.Verification.TorchLean.Proved
