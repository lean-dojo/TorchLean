/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Payload-Backed IR Evaluation

`linear` and `conv` nodes read weights from the external IR payload. These lemmas state the
local contract at that boundary: when the expected payload is present and the shape preconditions
are met, the IR evaluator returns the corresponding spec-layer operation.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- A lookup table with one defined entry. -/
def singletonAt {β : Type} (id : Nat) (x : β) : Nat → Option β :=
  fun j => if j = id then some x else none

/-- Looking up the defining index of a singleton table returns its stored value. -/
@[simp]
theorem singletonAt_self {β : Type} (id : Nat) (x : β) :
    singletonAt id x id = some x := by
  simp [singletonAt]

/-- A payload containing one flat constant at `id`. -/
def singletonConstPayload {α : Type} [TorchLean.Storage α] [Context α] (id : Nat)
    (c : ConstFlat α) : Payload α :=
  { const? := singletonAt id c }

/-- A payload containing one linear layer at `id`. -/
def singletonLinearPayload {α : Type} [TorchLean.Storage α] [Context α] (id : Nat)
    (p : LinearWB α) : Payload α :=
  { linear? := singletonAt id p }

/-- A payload containing one convolution layer at `id`. -/
def singletonConvPayload {α : Type} [TorchLean.Storage α] [Context α] (id : Nat)
    (p : ConvParams α) : Payload α :=
  { conv? := singletonAt id p }

/-- A payload containing one eval-mode BatchNorm layer at `id`. -/
def singletonBatchNormEvalPayload {α : Type} [TorchLean.Storage α] [Context α]
    (id : Nat) (p : BatchNormEvalParams α) : Payload α :=
  { batchNormEval? := singletonAt id p }

/-- A graph containing a zero-parent `const` node. -/
def constGraph (s : Shape) : Graph :=
  { nodes := #[{ id := 0, parents := #[], kind := .const s, outShape := s }] }

/-- Local IR semantics for a payload-backed flat `const` node. -/
theorem evalConst_eq_unflatten
    {α : Type} [TorchLean.Storage α] [Context α]
    (id : Nat) (s : Shape)
    (v : Tensor α [Spec.Shape.size s]) :
    let c : ConstFlat α := { n := Spec.Shape.size s, v := v }
    Graph.evalConst (α := α) (payload := singletonConstPayload (α := α) id c) (id := id) (s := s)
      =
      Except.ok (Tensor.unflattenSpec (α := α) s v) := by
  simp [Graph.evalConst, singletonConstPayload, Graph.castDimScalar, Pure.pure, Except.pure]

/-- Local IR semantics for a payload-backed flat `const` node. -/
theorem evalAt_const_eq_unflatten
    {α : Type} [TorchLean.Storage α] [Context α]
    (s : Shape)
    (v : Tensor α [Spec.Shape.size s]) :
    let c : ConstFlat α := { n := Spec.Shape.size s, v := v }
    Graph.evalAt (α := α) (g := constGraph s)
        (payload := singletonConstPayload (α := α) 0 c)
        (input := Spec.SomeTensor.mk (α := α) s (Tensor.default (α := α) (shape := s)))
        (vals := #[]) (i := 0)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) s (Tensor.unflattenSpec (α := α) s v)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, constGraph, Graph.getNode,
    Graph.getNode?, Graph.evalConst,
    singletonConstPayload, Graph.castDimScalar, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Missing constant payloads are rejected before unflattening. -/
theorem evalConst_missing_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    (payload : Payload α) (id : Nat) (s : Shape)
    (hMissing : payload.const? id = none) :
    Graph.evalConst (α := α) (payload := payload) (id := id) (s := s)
      =
      Except.error s!"IR eval: missing const payload for node {id}" := by
  simp [Graph.evalConst, hMissing]
  rfl

/-- Local IR semantics for a payload-backed `linear` node. -/
theorem evalLinear_eq_affine
    {α : Type} [TorchLean.Storage α] [Context α]
    (id outDim inDim : Nat)
    (W : Tensor α [outDim, inDim])
    (b : Tensor α [outDim])
    (x : Tensor α [inDim]) :
    let p : LinearWB α := { outDim := outDim, inDim := inDim, W := W, b := b }
    Graph.evalLinear (α := α) (payload := singletonLinearPayload (α := α) id p) (id := id)
        (x := Spec.SomeTensor.mk (α := α) (.dim inDim .scalar) x)
        (outShape := .dim outDim .scalar)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) (.dim outDim .scalar)
          (Tensor.addSpec (α := α)
            (Spec.matVecMulSpec (α := α) (m := outDim) (n := inDim) W x) b)) := by
  simp [Graph.evalLinear, Graph.linearLeading, singletonLinearPayload, Graph.expectShape,
    Shape.toList, Shape.ofList, Shape.concat, Bind.bind, Except.bind, Pure.pure, Except.pure]
  cases x
  rfl

/-- Local IR semantics for a payload-backed `linear` node. -/
theorem evalAt_linear_eq_affine
    {α : Type} [TorchLean.Storage α] [Context α]
    (outDim inDim : Nat)
    (W : Tensor α [outDim, inDim])
    (b : Tensor α [outDim])
    (x : Tensor α [inDim]) :
    let p : LinearWB α := { outDim := outDim, inDim := inDim, W := W, b := b }
    Graph.evalAt (α := α)
        (g := unaryGraphOut .linear (.dim inDim .scalar) (.dim outDim .scalar))
        (payload := singletonLinearPayload (α := α) 1 p)
        (input := Spec.SomeTensor.mk (α := α) (.dim inDim .scalar) x)
        (vals := #[Spec.SomeTensor.mk (α := α) (.dim inDim .scalar) x])
        (i := 1)
      =
      Except.ok
        (Spec.SomeTensor.mk (α := α) (.dim outDim .scalar)
          (Tensor.addSpec (α := α)
            (Spec.matVecMulSpec (α := α) (m := outDim) (n := inDim) W x) b)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut,
    Graph.getNode, Graph.getNode?,
    Graph.unaryParentId, NN.IR.unaryParent?, Graph.evalLinear, singletonLinearPayload,
    Graph.expectShape, Graph.linearLeading, Shape.toList, Shape.ofList, Shape.concat,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  cases x
  rfl

/-- Missing linear payloads are rejected before the affine operation is evaluated. -/
theorem evalLinear_missing_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    (payload : Payload α) (id : Nat)
    (hMissing : payload.linear? id = none)
    (x : Spec.SomeTensor α) (outShape : Shape) :
    Graph.evalLinear (α := α) (payload := payload) (id := id) (x := x) (outShape := outShape)
      =
      Except.error s!"IR eval: missing linear payload for node {id}" := by
  simp [Graph.evalLinear, hMissing]
  rfl

/-- The IR configuration carried by a convolution payload with no leading batch axes. -/
def convolutionConfig {α : Type} [TorchLean.Storage α] [Context α] (params : ConvParams α) :
    ConvConfig :=
  { spatialRank := params.spatialRank
    kernel := params.kernel
    stride := params.stride
    padding := params.padding
    dilation := params.dilation
    paddingAfter := params.paddingAfter
    groups := params.groups
    channelAxis := 0
    inChannels := params.inChannels
    outChannels := params.outChannels }

/--
Local IR semantics for a payload-backed arbitrary-rank convolution.

The shape-inference equation is exposed because it is exactly the dynamic check performed at the
IR boundary. Once that check succeeds, evaluation is the typed `Spec.groupedConvSpec` operation.
-/
theorem evalConv_eq_spec
    {α : Type} [TorchLean.Storage α] [Context α]
    (id : Nat)
    (params : ConvParams α)
    (x : Tensor α
      (Shape.ofList (params.inChannels :: Tensor.to params.inputSpatial (List Nat))))
    (outShape : Shape)
    (hInfer : OpContracts.inferConvConfigOutShape "conv" (convolutionConfig params)
      (Shape.ofList (params.inChannels :: Tensor.to params.inputSpatial (List Nat))) =
        .ok outShape) :
    Graph.evalConv (α := α)
        (payload := singletonConvPayload (α := α) id params)
        (id := id) (config := convolutionConfig params)
        (x := Spec.SomeTensor.mk (α := α)
          (Shape.ofList (params.inChannels :: Tensor.to params.inputSpatial (List Nat))) x)
      =
      Except.ok (Spec.SomeTensor.ofTensor
        (Tensor.mapLeading .scalar
          (Spec.groupedConvSpec (α := α) (stride := params.stride)
            (dilation := params.dilation) (paddingBefore := params.padding)
            (paddingAfter := params.paddingAfter) params.groups params.spec.kernel params.spec.bias)
          x)) := by
  have hInfer' :
      OpContracts.inferConvConfigOutShape "conv"
          { spatialRank := params.spatialRank
            kernel := params.kernel
            stride := params.stride
            padding := params.padding
            dilation := params.dilation
            paddingAfter := params.paddingAfter
            groups := params.groups
            channelAxis := 0
            inChannels := params.inChannels
            outChannels := params.outChannels }
          (Shape.ofList (params.inChannels :: Tensor.to params.inputSpatial (List Nat))) =
            .ok outShape := by
    simpa only [convolutionConfig] using hInfer
  have hInferData :
      OpContracts.inferConvConfigOutShape "conv"
          { spatialRank := params.spatialRank
            kernel := params.kernel
            stride := params.stride
            padding := params.padding
            dilation := params.dilation
            paddingAfter := params.paddingAfter
            groups := params.groups
            channelAxis := 0
            inChannels := params.inChannels
            outChannels := params.outChannels }
          (Shape.ofList (params.inChannels :: params.inputSpatial.data.toList)) =
            .ok outShape := by
    simpa only [Tensor.to_list_eq_data] using hInfer'
  have hMatches : params.matchesConfig (convolutionConfig params) = true := by
    change
      (params.spatialRank == params.spatialRank &&
        params.inChannels == params.inChannels &&
        params.outChannels == params.outChannels &&
        Tensor.to params.kernel (List Nat) == Tensor.to params.kernel (List Nat) &&
        Tensor.to params.stride (List Nat) == Tensor.to params.stride (List Nat) &&
        Tensor.to params.padding (List Nat) == Tensor.to params.padding (List Nat) &&
        Tensor.to params.paddingAfter (List Nat) == Tensor.to params.paddingAfter (List Nat) &&
        Tensor.to params.dilation (List Nat) == Tensor.to params.dilation (List Nat) &&
        params.groups == params.groups) = true
    simp
  have hMatchesData :
      params.matchesConfig
          { spatialRank := params.spatialRank
            kernel := params.kernel
            stride := params.stride
            padding := params.padding
            dilation := params.dilation
            paddingAfter := params.paddingAfter
            groups := params.groups
            channelAxis := 0
            inChannels := params.inChannels
            outChannels := params.outChannels } = true := by
    simpa only [convolutionConfig] using hMatches
  simp [Graph.evalConv, hInferData, singletonConvPayload, convolutionConfig,
    hMatchesData, Tensor.to_list_eq_data, ConvParams.input, Shape.ofList, Shape.concat,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  split
  · rfl
  · rename_i _ hFalse _
    exact (hFalse rfl).elim

/-- Missing convolution payloads are rejected before convolution is evaluated. -/
theorem evalConv_missing_payload
    {α : Type} [TorchLean.Storage α] [Context α]
    (payload : Payload α) (id : Nat) (config : ConvConfig)
    (x : Spec.SomeTensor α)
    (outShape : Shape)
    (hInfer : OpContracts.inferConvConfigOutShape "conv" config x.shape = .ok outShape)
    (hMissing : payload.conv? id = none) :
    Graph.evalConv (α := α) (payload := payload) (id := id) (config := config) (x := x)
      = Except.error s!"IR eval: missing conv payload for node {id}" := by
  unfold Graph.evalConv
  rw [hInfer]
  simp only [Bind.bind, Except.bind]
  rw [hMissing]
  rfl

end IRStep

end Correctness

end NN.Verification.Builtin.Proved
