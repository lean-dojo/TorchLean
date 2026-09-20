/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Convolution Correctness

Semantic preservation for arbitrary-rank convolution. `ConvConfig` records the channel axis and
one kernel, stride, and padding value for every spatial axis. Axes before the channel axis are
preserved, so the same theorem covers unbatched tensors and tensors with any number of leading
batch dimensions.

The lowered closure applies `Spec.groupedConvSpec` under `Tensor.mapLeading` directly, transporting
along the payload and output shape equalities checked during lowering; `evalConv_eq_generalSpec`
shows the IR evaluator produces the same term.
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

/-- Successful convolution evaluation agrees with the generalized typed convolution semantics. -/
theorem evalConv_eq_generalSpec
    {α : Type} [TorchLean.Storage α] [Context α]
    (payload : Payload α) (id : Nat) (config : ConvConfig) (params : ConvParams α)
    {parentShape : Shape} (parent : Tensor α parentShape) (leading outShape : Shape)
    (hInfer : OpContracts.inferConvConfigOutShape "conv" config parentShape =
      .ok outShape)
    (hParams : payload.conv? id = some params)
    (hMatches : params.matchesConfig config = true)
    (hLeading : Shape.ofList (parentShape.toList.take config.channelAxis) = leading)
    (hInput : parentShape = params.input leading) :
    NN.IR.Graph.evalConv payload id config (Spec.SomeTensor.ofTensor parent) =
      .ok (Spec.SomeTensor.ofTensor <|
        Tensor.mapLeading leading
          (Spec.groupedConvSpec (α := α) (stride := params.stride)
            (dilation := params.dilation) (paddingBefore := params.padding)
            (paddingAfter := params.paddingAfter) params.groups params.spec.kernel params.spec.bias)
          (hInput ▸ parent)) := by
  unfold NN.IR.Graph.evalConv
  simp only [Spec.SomeTensor.shape_ofTensor, Spec.SomeTensor.tensor_ofTensor]
  rw [hInfer]
  simp only [hParams, hMatches, ite_true]
  rw [hLeading]
  split
  · congr
  · contradiction

/-- Lowering an arbitrary-rank convolution preserves the IR denotation. -/
theorem buildFrom_denoteAllFrom_conv
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (config : ConvConfig)
    (hN : g.getNode i = .ok n) (hk : n.kind = .conv config) (hi : i < g.nodes.size)
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
  let vals0 := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk, lowerConv] at hBuild
  cases hp : unaryParent? n.parents with
  | none => exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | some parentId =>
  cases hParent : g.getNode parentId with
  | error message => simp [hp, hParent] at hBuild
  | ok parentNode =>
  let parentShape := parentNode.outShape
  cases hIdx : mkIdx (inShape := inShape) (ss := ss) parentId parentShape with
  | error message => simp [hp, hParent, parentShape, hIdx] at hBuild
  | ok parentIdx =>
  cases hExpected : OpContracts.inferConvConfigOutShape "conv" config parentShape with
  | error message => simp [hp, hParent, parentShape, hIdx, hExpected] at hBuild
  | ok expected =>
  cases hParams : payload.conv? n.id with
  | none =>
      exact False.elim <| throw_bind_ne_ok <| by
        simpa [hp, hParent, parentShape, hIdx, hExpected, hParams] using hBuild
  | some params =>
  by_cases hMatches : params.matchesConfig config = true
  · simp [hp, hParent, parentShape, hIdx, hExpected, hParams, hMatches] at hBuild
    let leading := Shape.ofList (parentShape.toList.take config.channelAxis)
    let payloadShape := params.input leading
    -- The payload-shape guard is a dependent `if`; splitting keeps `hInput` out of the simp set,
    -- where an equation whose right side mentions its left side would loop.
    split at hBuild
    · rename_i hInput
      by_cases hPayloadOut : params.output leading = expected
      · by_cases hOut : expected = n.outShape
        · simp [parentShape, leading, hPayloadOut, hOut] at hBuild
          let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
            mkForwardNode (fun context =>
              let input : Tensor α payloadShape :=
                Tensor.castShape (getIdx (α := α) (xs := context) parentIdx) hInput
              let output : Tensor α (params.output leading) :=
                Tensor.mapLeading leading
                  (Spec.groupedConvSpec (α := α) (stride := params.stride)
                    (dilation := params.dilation) (paddingBefore := params.padding)
                    (paddingAfter := params.paddingAfter) params.groups
                    params.spec.kernel params.spec.bias)
                  input
              Tensor.castShape output (hPayloadOut.trans hOut))
          let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
          have hRec :
              buildFrom (α := α) (g := g) (payload := payload)
                  (inShape := inShape) (i := i + 1) st1 = .ok st' := by
            simpa [st1, nodeData, parentShape, leading, payloadShape] using hBuild
          have hGet :
              vals0[parentId]? = some
                (Spec.SomeTensor.mk (α := α) parentShape
                  (getIdx (α := α) (xs := ctx) parentIdx)) := by
            simpa [vals0, ctx, parentShape] using
              (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                (gd := gd) (x := x) (pid := parentId) (s := parentShape)
                (idx := parentIdx) hIdx)
          let input : Tensor α payloadShape :=
            Tensor.castShape (getIdx (α := α) (xs := ctx) parentIdx) hInput
          let output : Tensor α (params.output leading) := Tensor.mapLeading leading
            (Spec.groupedConvSpec (α := α) (stride := params.stride)
              (dilation := params.dilation) (paddingBefore := params.padding)
              (paddingAfter := params.paddingAfter) params.groups
              params.spec.kernel params.spec.bias) input
          have hConv :
              NN.IR.Graph.evalConv payload n.id config
                  (Spec.SomeTensor.mk (α := α) parentShape
                    (getIdx (α := α) (xs := ctx) parentIdx)) =
                .ok (Spec.SomeTensor.mk (α := α) (params.output leading) output) := by
            have h := evalConv_eq_generalSpec (payload := payload) (id := n.id)
              (config := config) (params := params)
              (parent := getIdx (α := α) (xs := ctx) parentIdx)
              (leading := leading) (hInfer := hExpected)
              (hParams := hParams) (hMatches := hMatches)
              (hLeading := rfl) (hInput := hInput)
            simpa [Spec.SomeTensor.ofTensor, output, input, Tensor.eqRec_eq_cast_shape,
              ConvParams.output] using h
          have hResultShape : params.output leading = n.outShape := hPayloadOut.trans hOut
          have hNorm := normalizeNodeOutput_mk_of_eq (α := α) i n output hResultShape
          have hEval :
              NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                  (input := Spec.SomeTensor.mk (α := α) inShape x)
                  (vals := vals0) (i := i) =
                .ok (Spec.SomeTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
            simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, hN, hk, hp, hGet, hConv, hResultShape,
              hNorm]
            rfl
          apply buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
            (payload := payload) (gd := gd) (i := i) (st' := st')
            (x := x) (hi := hi) (nodeData := nodeData)
          · exact ih st1 hRec
          · exact hEval
        · exact False.elim <| throw_bind_ne_ok <| by
            simpa [parentShape, leading, hPayloadOut, hOut] using hBuild
      · exact False.elim <| throw_bind_ne_ok <| by
          simpa [parentShape, leading, hPayloadOut] using hBuild
    · exact False.elim <| throw_bind_ne_ok hBuild
  · exact False.elim <| throw_bind_ne_ok <| by
      simpa [hp, hParent, parentShape, hIdx, hExpected, hParams, hMatches] using hBuild

end IRExec
end Autograd
end Runtime
