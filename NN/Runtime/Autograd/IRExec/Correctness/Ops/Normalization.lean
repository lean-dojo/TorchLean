/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Normalization

Normalization correctness lemmas for the IR-to-forward-executor lowering.

TorchLean’s LayerNorm correctness argument has two layers:

* the spec layer (`Spec.layerNorm`) defines the mathematical normalization over a tensor axis,
  matching the original Layer Normalization formulation from Ba et al. (2016) and the public
  PyTorch `LayerNorm` API;
* the runtime/lowering pass layer (`IRExec.Internal.buildFrom`) lowers `.layernorm axis` IR nodes
  into SSA nodes
  whose `forward` closure computes the same result on the forward-graph execution path.

This file proves the forward-correctness lemmas for LayerNorm and eval-mode BatchNorm lowering:
when `buildFrom` succeeds at IR node position `i`, the IR evaluator and the forward-graph evaluator
append the same output tensor.

The proof is shape-driven: it follows the same dependent matches and checks as the lowering pass, so
failed preconditions discharge as contradictions.

References:
* Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, "Layer Normalization", arXiv:1607.06450.
* PyTorch `LayerNorm` documentation:
  https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

## Main definitions

- `buildFrom_denoteAllFrom_layernorm`: correctness step for `.layernorm axis` lowering.
- `buildFrom_denoteAllFrom_batchNormEval`: correctness for an arbitrary BatchNorm channel axis.

## Implementation notes

- This proof is shape-driven and follows the same checks as lowering, which keeps it
  maintainable as layernorm contracts evolve.
- Branches that fail preconditions are discharged as contradictions close to where they arise,
  keeping the successful path readable.
- The BatchNorm closure applies `Spec.batchNormInference` under `Tensor.mapLeading` directly, so
  the proof compares that term with the one computed by `NN.IR.Graph.evalBatchNorm`.
- LayerNorm carries axis constraints through both the shape discipline and the tensor computation.
  Keep axis-validity and shape-cast facts in small helper lemmas so the semantic theorem stays
  focused on agreement between the lowered node and the evaluator.

## Tags

layernorm, correctness, ir, runtime, semantic equivalence
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/-- Correctness lemma for the `.layernorm` node lowering pass. -/
theorem buildFrom_denoteAllFrom_layernorm
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .layernorm axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  -- Unfold the lowering pass step and specialize to the `.layernorm axis` branch.
  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk, lowerLayernorm] at hBuild

  -- `layernorm` is unary.
  cases hp : unaryParent? n.parents with
  | none =>
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | some pId =>
          -- Compute the matrix-view parameters used by the lowering pass and the IR evaluator.
          cases hParams : OpContracts.layerNormMatrixDims axis n.outShape with
          | error msg =>
              exact False.elim <| throw_bind_ne_ok (by simpa [hp, hParams] using hBuild)
          | ok p =>
              rcases p with ⟨seqLen, embedDim⟩
              let view2d : Shape := .dim seqLen (.dim embedDim .scalar)

              by_cases hNumel : Spec.Shape.size n.outShape = Spec.Shape.size view2d
              · by_cases hSeq : seqLen > 0
                · by_cases hEmb : embedDim > 0
                  · cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
                    | error msg =>
                        have hFalse : False := by
                          simp [hp, hParams, view2d, hNumel, hSeq, hEmb, hIdx] at hBuild
                        cases hFalse
                    | ok ip =>
                        have hAffineOk : ∃ affine,
                            NN.IR.Graph.resolveLayerNormAffine payload i axis n.outShape embedDim =
                              .ok affine := by
                          cases hResult : NN.IR.Graph.resolveLayerNormAffine
                              payload i axis n.outShape embedDim with
                          | error msg =>
                              have hFalse : False := by
                                simp [hp, hParams, view2d, hNumel, hSeq, hEmb, hIdx,
                                  hResult] at hBuild
                              exact False.elim hFalse
                          | ok affine => exact ⟨affine, rfl⟩
                        rcases hAffineOk with ⟨affine, hAffine⟩
                        have hNodeId : n.id = i := NN.IR.Graph.getNode_id_eq hN
                        -- Reduce `hBuild` to the recursive lowering call.
                        simp [hp, hParams, view2d, hNumel, hSeq, hEmb, hIdx, hAffine] at hBuild

                        let gamma : Tensor α [embedDim] := affine.gamma
                        let beta : Tensor α [embedDim] := affine.beta
                        let epsilon : α := affine.epsilon
                        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                          mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape)
                            (fun ctx =>
                            let x : Tensor α n.outShape := getIdx (α := α) (xs := ctx) ip
                            let x2d : Tensor α view2d :=
                              Tensor.reshapeSpec (α := α) (source := n.outShape)
                                (target := view2d) x hNumel
                            let y2d : Tensor α view2d :=
                              Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                                (x := x2d) (gamma := gamma) (beta := beta)
                                (h_seq_pos := hSeq) (h_embed_pos := hEmb) (epsilon := epsilon)
                            Tensor.reshapeSpec (α := α) (source := view2d)
                              (target := n.outShape) y2d hNumel.symm)
                        let st1 : State α inShape :=
                          ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩

                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                                (i := i + 1) st1 = .ok st' := by
                          simpa [st1, nodeData, gamma, beta, epsilon] using hBuild

                        have hGet :
                            vals0[pId]? = some (Spec.SomeTensor.mk (α := α) n.outShape
                                (getIdx (α := α) (xs := ctx) ip)) := by
                          simpa [vals0, ctx] using
                            (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)

                        have hLN :
                            NN.IR.Graph.layerNormMatrix (α := α) seqLen embedDim
                                (Tensor.reshapeSpec (α := α) (source := n.outShape)
                                  (target := view2d) (getIdx (α := α) (xs := ctx) ip) hNumel)
                                gamma beta epsilon =
                              .ok
                                (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                                  (x := Tensor.reshapeSpec (α := α) (source := n.outShape)
                                    (target := view2d)
                                    (getIdx (α := α) (xs := ctx) ip) hNumel)
                                  (gamma := gamma) (beta := beta)
                                  (h_seq_pos := hSeq) (h_embed_pos := hEmb)
                                  (epsilon := epsilon)) := by
                          simp [NN.IR.Graph.layerNormMatrix, hSeq, hEmb]
                          rfl

                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := input) (vals := vals0) (i := i) =
                              .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) :=
                                by
                          -- Focused simplification of the `.layernorm` branch of the evaluator.
                          simp (config := { failIfUnchanged := false })
                            [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                              NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hParams,
                              view2d, hNumel, hNodeId, hAffine,
                              throw_eq_error,
                              nodeData, mkForwardNode, gamma, beta, epsilon]
                          simpa using
                            congrArg
                              (fun e =>
                                (fun a : Tensor α view2d =>
                                  Spec.SomeTensor.mk (α := α) n.outShape
                                    (Tensor.reshapeSpec (α := α) (source := view2d)
                                      (target := n.outShape) a hNumel.symm)) <$> e)
                              hLN

                        have hStep :
                            denoteAllState (α := α) inShape st1 x =
                              vals0.push
                                (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                          simpa [vals0, st1, nodeData, ctx] using
                            (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss)
                              (τ := n.outShape) (gd := gd) (nodeData := nodeData) (x := x))

                        have hTail := ih st1 hRec
                        exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                          (i := i) (x := x) (hi := hi) (τ := n.outShape)
                          (nodeData := nodeData) (st1 := st1) (st' := st')
                          (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                  · exact False.elim <|
                      throw_bind_ne_ok (by simpa [hp, hParams, view2d, hNumel, hSeq, hEmb] using
                        hBuild)
                · exact False.elim <|
                    throw_bind_ne_ok (by simpa [hp, hParams, view2d, hNumel, hSeq] using hBuild)
              · exact False.elim <|
                  throw_bind_ne_ok (by simpa [hp, hParams, view2d, hNumel] using hBuild)

/-- Correctness lemma for fixed-statistics BatchNorm lowering along an arbitrary channel axis. -/
theorem buildFrom_denoteAllFrom_batchNormEval
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (channelAxis channels : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .batchNormEval channelAxis channels)
    (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape
        (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x

  classical
  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk, lowerBatchNormEval] at hBuild
  cases hp : unaryParent? n.parents with
  | none =>
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | some pId =>
          cases hParent : g.getNode pId with
          | error msg =>
              simp [hp, hParent] at hBuild
          | ok parentNode =>
              let expectedIn : Shape := parentNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId expectedIn with
              | error msg =>
                  simp [hp, hParent, expectedIn, hIdx] at hBuild
              | ok ip =>
                  cases hInfer :
                      OpContracts.inferBatchNormEvalOutShape channelAxis channels expectedIn with
                  | error msg =>
                      simp [hp, hParent, expectedIn, hIdx, hInfer] at hBuild
                  | ok inferred =>
                      cases hCfg : payload.batchNormEval? n.id with
                      | none =>
                          simp [hp, hParent, expectedIn, hIdx, hInfer, hCfg] at hBuild
                          exact False.elim <| throw_bind_ne_ok (h := hBuild)
                      | some params =>
                          by_cases hChannels : params.c = channels
                          · let dims := expectedIn.toList
                            let leading : Shape := Shape.ofList (dims.take channelAxis)
                            let spatial : Shape := Shape.ofList (dims.drop (channelAxis + 1))
                            let payloadShape : Shape := leading.concat (.dim params.c spatial)
                            cases hDecision : decEq expectedIn payloadShape with
                            | isTrue hInput =>
                              by_cases hOut : expectedIn = n.outShape
                              · have hOut' : parentNode.outShape = n.outShape := by
                                  simpa [expectedIn] using hOut
                                simp [hp, hParent, expectedIn, hIdx, hCfg, hChannels,
                                  dims, leading, spatial, payloadShape, hDecision, hOut'] at hBuild
                                have hInferOut :
                                    OpContracts.inferBatchNormEvalOutShape channelAxis channels
                                        n.outShape = .ok inferred := by
                                  simpa [hOut] using hInfer
                                simp [hInferOut] at hBuild
                                let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                                  mkForwardNode (α := α) (Γ := [inShape] ++ ss)
                                    (τ := n.outShape) (fun ctx =>
                                      let input : Tensor α payloadShape :=
                                        Tensor.castShape (getIdx (α := α) (xs := ctx) ip) hInput
                                      let output : Tensor α payloadShape :=
                                        Tensor.mapLeading leading
                                          (fun sample => Spec.batchNormInference sample
                                            params.mean params.var params.gamma params.beta
                                            params.eps)
                                          input
                                      Tensor.castShape output (hInput.symm.trans hOut))
                                let st1 : State α inShape :=
                                  ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                                have hRec :
                                    buildFrom (α := α) (g := g) (payload := payload)
                                      (inShape := inShape) (i := i + 1) st1 = .ok st' := by
                                  simpa [st1, nodeData, expectedIn, dims, leading, spatial,
                                    payloadShape] using hBuild
                                have hGet :
                                    vals0[pId]? = some
                                      (Spec.SomeTensor.mk (α := α) expectedIn
                                        (getIdx (α := α) (xs := ctx) ip)) := by
                                  simpa [vals0, ctx, expectedIn] using
                                    (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                                      (gd := gd) (x := x) (pid := pId) (s := expectedIn)
                                      (idx := ip) hIdx)
                                let inputT : Tensor α payloadShape :=
                                  Tensor.castShape (getIdx (α := α) (xs := ctx) ip) hInput
                                let output : Tensor α payloadShape :=
                                  Tensor.mapLeading leading
                                    (fun sample => Spec.batchNormInference sample params.mean
                                      params.var params.gamma params.beta params.eps)
                                    inputT
                                have hBatch :
                                    NN.IR.Graph.evalBatchNorm (α := α) payload n.id channelAxis
                                        channels (Spec.SomeTensor.mk (α := α) expectedIn
                                          (getIdx (α := α) (xs := ctx) ip)) =
                                      .ok (Spec.SomeTensor.mk (α := α) payloadShape output) := by
                                  unfold NN.IR.Graph.evalBatchNorm
                                  simp [hInfer, hCfg, hChannels]
                                  split <;> rename_i hShape
                                  · rename_i _ hInput'
                                    simp only [output, inputT, Pure.pure, Except.pure,
                                      Tensor.eqRec_eq_cast_shape]
                                    rfl
                                  · rename_i _ hInput'
                                    exact False.elim (hInput'
                                      (by simpa [payloadShape, leading, spatial, dims] using
                                        hInput))
                                have hResult : payloadShape = n.outShape :=
                                  hInput.symm.trans hOut
                                have hNorm :=
                                  normalizeNodeOutput_mk_of_eq (α := α) i n output hResult
                                have hEval :
                                    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                        (input := input) (vals := vals0) (i := i) =
                                      .ok (Spec.SomeTensor.mk (α := α) n.outShape
                                        (nodeData.eval ctx)) := by
                                  simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, hN, hk, hp,
                                    hGet, hBatch, hResult, hNorm]
                                  rfl
                                have hStep :
                                    denoteAllState (α := α) inShape st1 x =
                                      vals0.push (Spec.SomeTensor.mk (α := α) n.outShape
                                        (nodeData.eval ctx)) := by
                                  simpa [vals0, st1, nodeData, ctx] using
                                    (denoteAllState_snoc (α := α) (inShape := inShape)
                                      (ss := ss) (τ := n.outShape) (gd := gd)
                                      (nodeData := nodeData) (x := x))
                                have hTail := ih st1 hRec
                                exact buildFrom_denoteAllFrom_finish (α := α) (g := g)
                                  (payload := payload) (i := i) (x := x) (hi := hi)
                                  (τ := n.outShape) (nodeData := nodeData) (st1 := st1)
                                  (st' := st') (ctx := ctx) (vals0 := vals0) (input := input)
                                  hTail hEval hStep
                              · simp [hp, hParent, expectedIn, hIdx, hInfer, hCfg, hChannels,
                                  dims, leading, spatial, payloadShape, hDecision, hOut] at hBuild
                                exact False.elim <| throw_bind_ne_ok (h := hBuild)
                            | isFalse hInput =>
                              simp [hp, hParent, expectedIn, hIdx, hInfer, hCfg, hChannels,
                                dims, leading, spatial, payloadShape, hDecision] at hBuild
                              exact False.elim <| throw_bind_ne_ok (h := hBuild)
                          · simp [hp, hParent, expectedIn, hIdx, hInfer, hCfg, hChannels]
                              at hBuild
                            exact False.elim <| throw_bind_ne_ok (h := hBuild)

end IRExec
end Autograd
end Runtime
