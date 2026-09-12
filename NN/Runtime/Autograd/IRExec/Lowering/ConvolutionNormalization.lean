/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Runtime.Autograd.IRExec.Lowering.Common
public import NN.IR.Semantics

/-!
# Convolution and Normalization IR Lowering

Checked lowering for pooling, convolution, batch normalization, and layer normalization.

Pooling, convolution, and batch normalization validate the pooling plan, payload, and shapes once
while lowering. The closures then apply the typed specification operators directly to the typed
parent value, transporting along the shape equalities established by those checks. They do not
call the dynamic IR evaluator and cannot fail at runtime.

Each operation has its own small `lower*` definition. `lowerConvolutionNormalization` only
dispatches on the operation kind, and the `lowerConvolutionNormalization_*` equation lemmas let
correctness proofs reduce a dispatch to the branch they care about without unfolding the whole
dispatcher.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)
open NN.IR

namespace Internal

/-- Checked lowering for `.maxPool config` over the spatial suffix selected by the pooling plan. -/
def lowerMaxPool {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (config : WindowConfig) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let sIn := pNode.outShape
      let ip ← parentIdx pId sIn
      let plan ← OpContracts.planPool "max_pool" config sIn
      if hOut : plan.outShape = τ then
        let forward := fun ctx : TorchLean.TensorPack α Γ =>
          let input : Tensor α
              (plan.leading.concat (Shape.ofList (Tensor.to plan.spatial (List Nat)))) :=
            Tensor.castShape (getIdx (α := α) (xs := ctx) ip) plan.concat_eq.symm
          let layer : Spec.MaxPoolSpec config.spatialRank config.kernel config.stride
              config.padding plan.kernelNonzero plan.strideNonzero := {}
          let output : Tensor α plan.outShape :=
            Tensor.mapLeading plan.leading
              (Spec.maxPoolSpatialSpec (α := α) (inSpatial := plan.spatial) layer) input
          Tensor.castShape output hOut
        pure <| fwd forward
      else
        throw s!"IRExec: node {i}: max_pool outShape mismatch ({n.summary})"
  | _ => throw s!"IRExec: node {i}: max_pool expects 1 parent ({n.summary})"

/-- Checked lowering for `.avgPool config` over the spatial suffix selected by the pooling plan. -/
def lowerAvgPool {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (config : WindowConfig) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId =>
      let pNode ← g.getNode pId
      let sIn := pNode.outShape
      let ip ← parentIdx pId sIn
      let plan ← OpContracts.planPool "avg_pool" config sIn
      if hOut : plan.outShape = τ then
        let forward := fun ctx : TorchLean.TensorPack α Γ =>
          let input : Tensor α
              (plan.leading.concat (Shape.ofList (Tensor.to plan.spatial (List Nat)))) :=
            Tensor.castShape (getIdx (α := α) (xs := ctx) ip) plan.concat_eq.symm
          let layer : Spec.AvgPoolSpec config.spatialRank config.kernel config.stride
              config.padding plan.kernelNonzero plan.strideNonzero := {}
          let output : Tensor α plan.outShape :=
            Tensor.mapLeading plan.leading
              (Spec.avgPoolSpatialSpec (α := α) (inSpatial := plan.spatial) layer) input
          Tensor.castShape output hOut
        pure <| fwd forward
      else
        throw s!"IRExec: node {i}: avg_pool outShape mismatch ({n.summary})"
  | _ => throw s!"IRExec: node {i}: avg_pool expects 1 parent ({n.summary})"

/-- Checked lowering for `.conv config` with a payload-backed kernel over any leading shape. -/
def lowerConv {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (config : ConvConfig) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let payload := ctx.payload
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some xId =>
      let xNode ← g.getNode xId
      let expectedIn : Shape := xNode.outShape
      let ix ← parentIdx xId expectedIn
      let expected ← OpContracts.inferConvConfigOutShape "conv" config expectedIn
      match payload.conv? n.id with
      | none => throw s!"IRExec: missing convolution payload for node {n.id}"
      | some params =>
          if params.matchesConfig config then
            let dims := expectedIn.toList
            let leading : Shape := Shape.ofList (dims.take config.channelAxis)
            let payloadShape := params.input leading
            if hInput : expectedIn = payloadShape then
              if hPayloadOut : params.output leading = expected then
                if hOut : expected = τ then
                  let forward := fun ctx : TorchLean.TensorPack α Γ =>
                    let input : Tensor α payloadShape :=
                      Tensor.castShape (getIdx (α := α) (xs := ctx) ix) hInput
                    let output : Tensor α (params.output leading) :=
                      Tensor.mapLeading leading
                        (Spec.groupedConvSpec (α := α) (stride := params.stride)
                          (dilation := params.dilation) (paddingBefore := params.padding)
                          (paddingAfter := params.paddingAfter) params.groups
                          params.spec.kernel params.spec.bias)
                        input
                    Tensor.castShape output (hPayloadOut.trans hOut)
                  pure <| fwd forward
                else
                  throw s!"IRExec: node {i}: conv outShape mismatch ({n.summary})"
              else
                throw <|
                  s!"IRExec: node {i}: convolution payload output shape " ++
                    s!"{repr (params.output leading)} does not match inferred shape " ++
                    s!"{repr expected} ({n.summary})"
            else
              throw <|
                s!"IRExec: node {i}: convolution payload shape {repr payloadShape} " ++
                  s!"does not match parent shape {repr expectedIn} ({n.summary})"
          else
            throw s!"IRExec: node {i}: convolution payload does not match the node configuration"
  | _ => throw s!"IRExec: node {i}: conv expects 1 parent ({n.summary})"

/-- Checked lowering for `.batchNormEval channelAxis channels` with fixed statistics. -/
def lowerBatchNormEval {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (channelAxis channels : Nat) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let payload := ctx.payload
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some xId =>
      let xNode ← g.getNode xId
      let expectedIn : Shape := xNode.outShape
      let ix ← parentIdx xId expectedIn
      let _ ← OpContracts.inferBatchNormEvalOutShape channelAxis channels expectedIn
      match payload.batchNormEval? n.id with
      | none => throw s!"IRExec: missing batch_norm_eval payload for node {n.id}"
      | some params =>
          if _hChannels : params.c = channels then
            let dims := expectedIn.toList
            let leading : Shape := Shape.ofList (dims.take channelAxis)
            let spatial : Shape := Shape.ofList (dims.drop (channelAxis + 1))
            let payloadShape : Shape := leading.concat (.dim params.c spatial)
            match decEq expectedIn payloadShape with
            | isTrue hInput =>
                if hOut : @Eq Shape expectedIn τ then
                  let forward := fun ctx : TorchLean.TensorPack α Γ =>
                    let input : Tensor α payloadShape :=
                      Tensor.castShape (getIdx (α := α) (xs := ctx) ix) hInput
                    let output : Tensor α payloadShape :=
                      Tensor.mapLeading leading
                        (fun sample => Spec.batchNormInference sample params.mean params.var
                          params.gamma params.beta params.eps)
                        input
                    Tensor.castShape output (hInput.symm.trans hOut)
                  pure <| fwd forward
                else
                  throw s!"IRExec: node {i}: batch_norm_eval outShape mismatch ({n.summary})"
            | isFalse _ =>
                throw <|
                  s!"IRExec: node {i}: batch_norm_eval payload shape {repr payloadShape} " ++
                    s!"does not match parent shape {repr expectedIn} ({n.summary})"
          else
            throw <|
              s!"IRExec: node {i}: batch_norm_eval payload channels {params.c} do not " ++
                s!"match node channels {channels} ({n.summary})"
  | _ => throw s!"IRExec: node {i}: batch_norm_eval expects 1 parent ({n.summary})"

/-- Checked lowering for `.layernorm axis` through the matrix view of the normalized suffix. -/
def lowerLayernorm {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (axis : Nat) : NodeLoweringResult ctx := do
  let payload := ctx.payload
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match unaryParent? n.parents with
  | some pId => do
      let (seqLen, embedDim) ←
        match OpContracts.layerNormMatrixDims axis τ with
        | .ok p => pure p
        | .error msg => throw s!"IRExec: node {i}: layernorm: {msg} ({n.summary})"
      let view2d : Shape := .dim seqLen (.dim embedDim .scalar)
      if hNumel : Spec.Shape.size τ = Spec.Shape.size view2d then
        if hSeq : seqLen > 0 then
          if hEmb : embedDim > 0 then
            let ip ← parentIdx pId τ
            let affine ←
              NN.IR.Graph.resolveLayerNormAffine payload i axis τ embedDim
            let forward := fun ctx : TorchLean.TensorPack α Γ =>
              let x : Tensor α τ := getIdx (α := α) (xs := ctx) ip
              let x2d : Tensor α view2d :=
                Tensor.reshapeSpec (α := α) (source := τ) (target := view2d) x hNumel
              let y2d : Tensor α view2d :=
                Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                  (x := x2d) (gamma := affine.gamma) (beta := affine.beta)
                  (h_seq_pos := hSeq) (h_embed_pos := hEmb) (epsilon := affine.epsilon)
              Tensor.reshapeSpec (α := α) (source := view2d) (target := τ) y2d hNumel.symm
            pure <| fwd forward
          else
            throw s!"IRExec: node {i}: layernorm embedDim must be > 0 (got {embedDim})"
        else
          throw s!"IRExec: node {i}: layernorm seqLen must be > 0 (got {seqLen})"
      else
        throw <|
          s!"IRExec: node {i}: layernorm internal error: bad reshape sizes " ++
            s!"({Spec.Shape.size τ} vs {Spec.Shape.size view2d}) ({n.summary})"
  | _ =>
      throw s!"IRExec: node {i}: layernorm expects 1 parent ({n.summary})"

/-- Checked lowering for pooling, convolution, batch normalization, and layer normalization. -/
def lowerConvolutionNormalization {α : Type} [TorchLean.Storage α] [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx :=
  match kind with
  | .maxPool config => lowerMaxPool ctx config
  | .avgPool config => lowerAvgPool ctx config
  | .conv config => lowerConv ctx config
  | .batchNormEval channelAxis channels => lowerBatchNormEval ctx channelAxis channels
  | .layernorm axis => lowerLayernorm ctx axis
  | _ => throw s!"IRExec: internal error: operation routed to lowerConvolutionNormalization"

variable {α : Type} [TorchLean.Storage α] [Context α] {Γ : List Shape}

/-- Dispatch equation for `.maxPool config`. -/
@[simp] theorem lowerConvolutionNormalization_maxPool (ctx : NodeLoweringContext α Γ)
    (config : WindowConfig) :
    lowerConvolutionNormalization ctx (.maxPool config) = lowerMaxPool ctx config := rfl

/-- Dispatch equation for `.avgPool config`. -/
@[simp] theorem lowerConvolutionNormalization_avgPool (ctx : NodeLoweringContext α Γ)
    (config : WindowConfig) :
    lowerConvolutionNormalization ctx (.avgPool config) = lowerAvgPool ctx config := rfl

/-- Dispatch equation for `.conv config`. -/
@[simp] theorem lowerConvolutionNormalization_conv (ctx : NodeLoweringContext α Γ)
    (config : ConvConfig) :
    lowerConvolutionNormalization ctx (.conv config) = lowerConv ctx config := rfl

/-- Dispatch equation for `.batchNormEval channelAxis channels`. -/
@[simp] theorem lowerConvolutionNormalization_batchNormEval (ctx : NodeLoweringContext α Γ)
    (channelAxis channels : Nat) :
    lowerConvolutionNormalization ctx (.batchNormEval channelAxis channels) =
      lowerBatchNormEval ctx channelAxis channels := rfl

/-- Dispatch equation for `.layernorm axis`. -/
@[simp] theorem lowerConvolutionNormalization_layernorm (ctx : NodeLoweringContext α Γ)
    (axis : Nat) :
    lowerConvolutionNormalization ctx (.layernorm axis) = lowerLayernorm ctx axis := rfl

end Internal
end IRExec
end Autograd
end Runtime
