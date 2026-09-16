/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core
public import NN.Verification.Builtin.Proved.Correctness.Eval.PayloadBridge

/-!
# Lowered Forward Evaluation: Payload-Backed Nodes

One lemma per operator of the proved forward fragment whose lowering writes to the verifier
`ParamStore`: constants, linear layers, LayerNorm (which erases its own entry so the IR falls back
to unit affine parameters), and convolutions. Each lemma assumes the IR graph holds the node
emitted by `lowerNode` at index `id` and that the parameter store agrees with the lowering at that
id; it concludes that IR evaluation returns exactly what the typed evaluator `evalNode` returns.
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

/-- A lowered `const` node evaluates like the typed `const` node. -/
theorem evalAt_eq_evalNode_const
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (wf : Shape.WellFormed s) (t : Tensor α s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (P : NN.MLTheory.CROWN.Graph.ParamStore α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[], kind := .const s, outShape := s })
    (hConst : P.constVals.get? id = some (flatOfTensor (α := α) (s := s) wf t)) :
    Graph.evalAt (α := α) G (payloadOfParamStore (α := α) P) input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.const wf t) params vals := by
  obtain ⟨inputShape, inputT⟩ := input
  have hStore :
      P.constVals.get? id =
        some ({ n := Spec.Shape.size s, v := t.flattenSpec } :
          NN.MLTheory.CROWN.Graph.FlatTensor α) := by
    simpa [flatOfTensor] using hConst
  simpa [evalNode, Pure.pure, Except.pure] using
    IRStep.evalAt_const_from_paramStore_of_getNode (α := α) G P id id s inputShape inputT
      vals t.flattenSpec hGetNode hStore

/-- A lowered `paramConst` node evaluates like the typed `paramConst` node. -/
theorem evalAt_eq_evalNode_paramConst
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (wf : Shape.WellFormed s) (p : Idx paramShapes s) (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (P : NN.MLTheory.CROWN.Graph.ParamStore α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[], kind := .const s, outShape := s })
    (hConst : P.constVals.get? id =
      some (flatOfTensor (α := α) (s := s) wf
        (getParam (α := α) (paramShapes := paramShapes) params p))) :
    Graph.evalAt (α := α) G (payloadOfParamStore (α := α) P) input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.paramConst wf p) params vals := by
  obtain ⟨inputShape, inputT⟩ := input
  have hStore :
      P.constVals.get? id =
        some ({ n := Spec.Shape.size s,
                v := (getParam (α := α) (paramShapes := paramShapes) params p).flattenSpec } :
          NN.MLTheory.CROWN.Graph.FlatTensor α) := by
    simpa [flatOfTensor] using hConst
  simpa [evalNode, Pure.pure, Except.pure] using
    IRStep.evalAt_const_from_paramStore_of_getNode (α := α) G P id id s inputShape inputT
      vals (getParam (α := α) (paramShapes := paramShapes) params p).flattenSpec hGetNode
      hStore

/-- A lowered `linear` node evaluates like the typed `linear` node. -/
theorem evalAt_eq_evalNode_linear
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape}
    (inDim outDim : Nat)
    (w : Idx paramShapes (.dim outDim (.dim inDim .scalar)))
    (b : Idx paramShapes (.dim outDim .scalar))
    (xIdx : Idx (Ctx inShape ss) (.dim inDim .scalar))
    (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (P : NN.MLTheory.CROWN.Graph.ParamStore α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .linear, outShape := .dim outDim .scalar })
    (hLin : P.linearWB.get? id =
      some ({ m := outDim, n := inDim
              w := getParam (α := α) (paramShapes := paramShapes) params w
              b := getParam (α := α) (paramShapes := paramShapes) params b } :
        NN.MLTheory.CROWN.Graph.LinParams α)) :
    Graph.evalAt (α := α) G (payloadOfParamStore (α := α) P) input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := .dim outDim .scalar) (Node.linear inDim outDim w b xIdx) params vals := by
  obtain ⟨inputShape, inputT⟩ := input
  have hGetVal := getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
  have hEvalAt :=
    IRStep.evalAt_linear_from_paramStore_of_getNode (α := α) G P id id xIdx.id outDim inDim
      inputShape inputT vals (getParam (α := α) (paramShapes := paramShapes) params w)
      (getParam (α := α) (paramShapes := paramShapes) params b)
      (tensorAt vals xIdx hShapes) (packedAt vals xIdx hShapes) hGetNode
      (getElem?_eq_some_packedAt vals xIdx hShapes)
      (expectShape_packedAt_eq_ok vals xIdx hShapes) hLin
  simpa [evalNode, hGetVal, Spec.matVecMulSpec, Bind.bind, Except.bind,
    Pure.pure, Except.pure] using hEvalAt

/--
A lowered `layerNorm` node evaluates like the typed `layerNorm` node. The lowering erases any
LayerNorm payload at the fresh id, so the IR evaluator applies unit affine parameters, matching the
typed evaluator's `gamma = 1`, `beta = 0`.
-/
theorem evalAt_eq_evalNode_layerNorm
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {s : Shape}
    (op : LayerNormOperation s) (xIdx : Idx (Ctx inShape ss) s)
    (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (P : NN.MLTheory.CROWN.Graph.ParamStore α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id, parents := #[xIdx.id], kind := .layernorm op.axis, outShape := s })
    (hLayerNorm : P.layerNorm.get? id = none) :
    Graph.evalAt (α := α) G (payloadOfParamStore (α := α) P) input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := s)
        (Node.layerNorm op xIdx) params vals := by
  let matrixShape : Shape := .dim op.rows (.dim op.width .scalar)
  have hExpect := expectShape_packedAt_eq_ok vals xIdx hShapes
  have hGetVal := getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
  let xMatrix : Tensor α matrixShape :=
    Tensor.reshapeSpec (α := α) (source := s) (target := matrixShape)
      (tensorAt vals xIdx hShapes) op.size_eq
  let yMatrix : Tensor α matrixShape :=
    Spec.layerNorm (α := α) (seqLen := op.rows) (embedDim := op.width) (x := xMatrix)
      (gamma := Tensor.full (α := α) (.dim op.width .scalar) 1)
      (beta := Tensor.full (α := α) (.dim op.width .scalar) 0)
      (h_seq_pos := op.rows_pos) (h_embed_pos := op.width_pos)
  have hLN :
      Graph.layerNormWithoutAffine (α := α) op.rows op.width xMatrix = Except.ok yMatrix := by
    unfold Graph.layerNormWithoutAffine Graph.layerNormMatrix
    rw [dite_eq_left op.rows_pos, dite_eq_left op.width_pos]
    rfl
  have hNoLayerNorm : (payloadOfParamStore (α := α) P).layerNorm? id = none := by
    rw [IRStep.payloadOfParamStore_layerNorm?_eq, hLayerNorm]
  have hLayerNormMatrix :
      Graph.layerNormMatrix op.rows op.width xMatrix
          (Tensor.full (α := α) [op.width] 1) (Tensor.full (α := α) [op.width] 0)
          TorchLean.normalizationEpsilon = .ok yMatrix := by
    simpa [Graph.layerNormWithoutAffine, Graph.layerNormMatrix] using hLN
  have hEvalAt :
      Graph.evalAt (α := α) G (payloadOfParamStore (α := α) P) input vals id =
        Except.ok
          (Spec.SomeTensor.mk (α := α) s
            (Tensor.reshapeSpec (α := α) (source := matrixShape) (target := s)
              yMatrix op.size_eq.symm)) := by
    simp [Graph.evalAt, Graph.evalNode, Graph.unaryParentId, unaryParent?,
      Graph.normalizeNodeOutput, hGetNode, getElem?_eq_some_packedAt vals xIdx hShapes, hExpect,
      op.matrixDims, op.size_eq, matrixShape, xMatrix, hNoLayerNorm,
      Graph.resolveLayerNormAffine, hLayerNormMatrix,
      Bind.bind, Except.bind, Pure.pure, Except.pure]
  simpa [evalNode, hGetVal, matrixShape, xMatrix, yMatrix,
    Bind.bind, Except.bind, Pure.pure, Except.pure] using hEvalAt

/-- The IR convolution configuration emitted by lowering a `conv` node. -/
abbrev loweredConvConfig {d : Nat} (inC outC : Nat)
    (kernelShape stride padding : TorchLean.Tensor Nat [d]) : ConvConfig :=
  { spatialRank := d
    kernel := kernelShape
    stride := stride
    padding := padding
    dilation := Tensor.full [d] 1
    paddingAfter := padding
    groups := 1
    channelAxis := 0
    inChannels := inC
    outChannels := outC }

/-- The convolution payload written to the parameter store by lowering a `conv` node. -/
abbrev loweredConvParams
    {α : Type} [TorchLean.Storage α] [Context α] {d : Nat} (inC outC : Nat)
    (kernelShape stride padding inSpatial : TorchLean.Tensor Nat [d])
    (hKernel : ∀ i : Fin d, kernelShape.getScalar i ≠ 0)
    (hStride : ∀ i : Fin d, stride.getScalar i ≠ 0)
    (kT : Tensor α (Shape.ofList (outC :: inC :: Tensor.to kernelShape (List Nat))))
    (bT : Tensor α [outC]) : ConvParams α :=
  { spatialRank := d
    inChannels := inC
    outChannels := outC
    kernel := kernelShape
    stride := stride
    padding := padding
    dilation := Tensor.full [d] 1
    paddingAfter := padding
    groups := 1
    inputSpatial := inSpatial
    kernelNonzero := hKernel
    strideNonzero := hStride
    spec := { kernel := kT, bias := bT } }

/--
A lowered `conv` node evaluates like the typed `conv` node. The IR evaluates a grouped, dilated
convolution with `groups = 1`, unit dilation, and symmetric padding, which agrees with the dense
`Spec.convSpec` used by the typed evaluator up to a shape cast.
-/
theorem evalAt_eq_evalNode_conv
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {d : Nat}
    (inC outC : Nat) (kernelShape stride padding inSpatial : TorchLean.Tensor Nat [d])
    (hIn : inC ≠ 0)
    (hKernel : ∀ i : Fin d, kernelShape.getScalar i ≠ 0)
    (hStride : ∀ i : Fin d, stride.getScalar i ≠ 0)
    (hInfer : OpContracts.inferConvOutShape "conv" 0 inC outC
      kernelShape stride padding (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))) =
        .ok (Shape.ofList
          (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernelShape stride padding)
            (List Nat))))
    (kernel : Idx paramShapes (Shape.ofList (outC :: inC :: (Tensor.to kernelShape (List Nat)))))
    (bias : Idx paramShapes (.dim outC .scalar))
    (xIdx : Idx (Ctx inShape ss) (Shape.ofList (inC :: (Tensor.to inSpatial (List Nat)))))
    (params : TorchLean.TensorPack α paramShapes)
    (G : Graph) (P : NN.MLTheory.CROWN.Graph.ParamStore α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α)) (id : Nat)
    (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss)
    (hGetNode : G.getNode id =
      pure { id := id
             parents := #[xIdx.id]
             kind := .conv (loweredConvConfig inC outC kernelShape stride padding)
             outShape := Shape.ofList
               (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernelShape stride padding)
                 (List Nat)) })
    (hConv : P.convCfg.get? id =
      some (loweredConvParams (α := α) inC outC kernelShape stride padding inSpatial hKernel
        hStride (getParam (α := α) (paramShapes := paramShapes) params kernel)
        (getParam (α := α) (paramShapes := paramShapes) params bias))) :
    Graph.evalAt (α := α) G (payloadOfParamStore (α := α) P) input vals id =
      evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (Node.conv inC outC kernelShape stride padding inSpatial hIn hKernel hStride hInfer
          kernel bias xIdx) params vals := by
  obtain ⟨inputShape, inputT⟩ := input
  let inputShape' : Shape := Shape.ofList (inC :: Tensor.to inSpatial (List Nat))
  let outputShape' : Shape := Shape.ofList
    (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernelShape stride padding) (List Nat))
  let config : ConvConfig := loweredConvConfig inC outC kernelShape stride padding
  let kT : Tensor α (Shape.ofList (outC :: inC :: Tensor.to kernelShape (List Nat))) :=
    getParam (α := α) (paramShapes := paramShapes) params kernel
  let bT : Tensor α [outC] := getParam (α := α) (paramShapes := paramShapes) params bias
  let spec : Spec.ConvSpec d inC outC kernelShape stride padding α := { kernel := kT, bias := bT }
  let cfg : ConvParams α :=
    loweredConvParams (α := α) inC outC kernelShape stride padding inSpatial hKernel hStride kT bT
  let xT : Tensor α inputShape' := tensorAt vals xIdx hShapes
  have hxF : (packedAt vals xIdx hShapes).shape = inputShape' := by
    simp [inputShape']
  have hOutput : cfg.output .scalar = outputShape' := by
    simp only [cfg, loweredConvParams, ConvParams.output, Shape.concat, Shape.ofList, outputShape']
    rw [Spec.convOutSpatialDilated_one_symmetric]
  have hGetNodeConv :
      G.getNode id =
        pure ({ id := id
                parents := #[xIdx.id]
                kind := .conv config
                outShape := cfg.output .scalar } : NN.IR.Node) := by
    rw [hOutput]
    exact hGetNode
  have hExpectIn :
      Graph.expectShape (α := α) (expected := cfg.input .scalar) (packedAt vals xIdx hShapes) =
        Except.ok xT := by
    simpa [cfg, loweredConvParams, ConvParams.input, Shape.concat, inputShape', xT] using
      expectShape_packedAt_eq_ok vals xIdx hShapes
  have hGetVal :
      getVal (α := α) (inShape := inShape) (ss := ss) (s := inputShape') vals xIdx =
        Except.ok xT :=
    getVal_eq_ok_of_shapesOfVals_eq vals xIdx hShapes
  have hConfig : cfg.matchesConfig config = true := by
    simp only [cfg, config]
    unfold loweredConvParams loweredConvConfig
    simp [ConvParams.matchesConfig]
  have hInfer' :
      OpContracts.inferConvConfigOutShape "conv" config (packedAt vals xIdx hShapes).shape =
        .ok (cfg.output .scalar) := by
    rw [hOutput]
    simpa [OpContracts.inferConvOutShape, config, loweredConvConfig, hxF, outputShape',
      Tensor.to_list_eq_data] using hInfer
  have hLeading :
      Shape.ofList ((packedAt vals xIdx hShapes).shape.toList.take config.channelAxis) =
        .scalar := by
    rfl
  have hEvalAt :=
    IRStep.evalAt_conv_from_paramStore_of_getNode (α := α) G P id id xIdx.id cfg config .scalar
      inputShape inputT vals xT (packedAt vals xIdx hShapes) hGetNodeConv
      (getElem?_eq_some_packedAt vals xIdx hShapes) hExpectIn hConv hConfig hInfer' hLeading
  have hPacked :
      Spec.SomeTensor.ofTensor
          (Tensor.mapLeading .scalar
            (Spec.groupedConvSpec (α := α) (stride := cfg.stride)
              (dilation := cfg.dilation) (paddingBefore := cfg.padding)
              (paddingAfter := cfg.paddingAfter) cfg.groups cfg.spec.kernel cfg.spec.bias)
            xT) =
        Spec.SomeTensor.mk (α := α) outputShape'
          (Spec.convSpec (α := α) (layer := spec) (input := xT)) := by
    let generalConv :=
      Spec.groupedConvSpec (α := α) (stride := stride) (dilation := Tensor.full [d] 1)
        (paddingBefore := padding) (paddingAfter := padding) 1 kT bT xT
    let denseConv := Spec.convSpec (α := α) (layer := spec) (input := xT)
    let shapeEq :=
      congrArg (fun spatial => Shape.ofList (outC :: Tensor.to spatial (List Nat)))
        (Spec.convOutSpatialDilated_one_symmetric inSpatial kernelShape stride padding)
    have hCast : Tensor.castShape generalConv shapeEq = denseConv := by
      simpa [generalConv, denseConv, shapeEq, spec] using
        Spec.castShape_groupedConvSpec_one_symmetric
          (α := α) (weights := kT) (bias := bT) (input := xT)
    have hErased : Spec.SomeTensor.ofTensor generalConv = Spec.SomeTensor.ofTensor denseConv := by
      calc
        Spec.SomeTensor.ofTensor generalConv =
            Spec.SomeTensor.ofTensor (Tensor.castShape generalConv shapeEq) :=
          (Spec.SomeTensor.ofTensor_castShape generalConv shapeEq).symm
        _ = Spec.SomeTensor.ofTensor denseConv := congrArg Spec.SomeTensor.ofTensor hCast
    simpa [generalConv, denseConv, cfg, loweredConvParams, spec, outputShape',
      Tensor.mapLeading, Shape.concat] using hErased
  simpa [evalNode, hGetVal, hxF, xT, kT, bT, spec, inputShape', outputShape',
    Bind.bind, Except.bind, Pure.pure, Except.pure] using
    hEvalAt.trans (congrArg Except.ok hPacked)

end Correctness

end NN.Verification.Builtin.Proved
