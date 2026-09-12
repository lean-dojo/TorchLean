/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core
public import NN.Verification.Builtin.Correctness
public import Std.Data.HashMap.Lemmas

/-!
# Parameter Store To IR Payload Bridge

The verifier stores parameters in `ParamStore`, while executable graph semantics reads `Payload`.
The declarations below state how that conversion affects lookups and the payload-backed operations
used by the proved forward language. Convolution is stated for an arbitrary number of spatial axes;
tensor layout is represented by the node's channel axis rather than by a format-specific operator.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open NN.IR

namespace Correctness.IRStep

open NN.Verification.Builtin

/-- Convert a verifier flat tensor into the IR constant payload format. -/
def irConstOfFlatTensor {α : Type} [TorchLean.Storage α] [Context α]
    (c : NN.MLTheory.CROWN.Graph.FlatTensor α) : ConstFlat α :=
  { n := c.n, v := c.v }

/-- Convert verifier affine parameters into the IR linear payload format. -/
def irLinearOfLinParams {α : Type} [TorchLean.Storage α] [Context α]
    (p : NN.MLTheory.CROWN.Graph.LinParams α) : LinearWB α :=
  { outDim := p.m, inDim := p.n, W := p.w, b := p.b }

/-- Constant lookup after converting a verifier parameter store to an IR payload. -/
theorem payloadOfParamStore_const?_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).const? id =
      (ps.constVals.get? id).map (irConstOfFlatTensor (α := α)) := by
  rfl

/-- Affine-layer lookup after converting a verifier parameter store to an IR payload. -/
theorem payloadOfParamStore_linear?_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).linear? id =
      (ps.linearWB.get? id).map (irLinearOfLinParams (α := α)) := by
  rfl

/-- Convolution lookup after converting a verifier parameter store to an IR payload. -/
theorem payloadOfParamStore_conv?_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).conv? id = ps.convCfg.get? id := by
  rfl

/-- BatchNorm lookup after converting a verifier parameter store to an IR payload. -/
theorem payloadOfParamStore_batchNormEval?_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).batchNormEval? id = ps.batchNormEval.get? id := by
  rfl

/-- LayerNorm lookup after converting a verifier parameter store to an IR payload. -/
theorem payloadOfParamStore_layerNorm?_eq
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) (id : Nat) :
    (payloadOfParamStore (α := α) ps).layerNorm? id = ps.layerNorm.get? id := by
  rfl

/-- A `const` node reads its value from the matching parameter-store entry. -/
theorem evalAt_const_from_paramStore_of_getNode
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id : Nat) (s inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.SomeTensor α))
    (v : Tensor α [Spec.Shape.size s])
    (hNode : Graph.getNode (g := g) i =
      pure (NN.IR.Node.mk id #[] (.const s) s))
    (hStore : ps.constVals.get? id =
      some ({ n := Spec.Shape.size s, v := v } : NN.MLTheory.CROWN.Graph.FlatTensor α)) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.SomeTensor.ofTensor input) (vals := vals) (i := i) =
      .ok (Spec.SomeTensor.ofTensor (Tensor.unflattenSpec (α := α) s v)) := by
  have hPayload : (payloadOfParamStore (α := α) ps).const? id =
      some ({ n := Spec.Shape.size s, v := v } : ConstFlat α) := by
    rw [payloadOfParamStore_const?_eq, hStore]
    rfl
  simp [Graph.evalAt, Graph.evalNode, hNode, Graph.evalConst, hPayload, Graph.castDimScalar,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- A `linear` node reads its weights and bias from the matching parameter-store entry. -/
theorem evalAt_linear_from_paramStore_of_getNode
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id parentId outDim inDim : Nat)
    (inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.SomeTensor α))
    (weight : Tensor α [outDim, inDim])
    (bias : Tensor α [outDim])
    (x : Tensor α [inDim])
    (parent : Spec.SomeTensor α)
    (hNode : Graph.getNode (g := g) i =
      pure (NN.IR.Node.mk id #[parentId] .linear (.dim outDim .scalar)))
    (hParentValue : vals[parentId]? = some parent)
    (hParent : Graph.expectShape (α := α) (expected := .dim inDim .scalar) parent = .ok x)
    (hStore : ps.linearWB.get? id =
      some ({ m := outDim, n := inDim, w := weight, b := bias } :
        NN.MLTheory.CROWN.Graph.LinParams α)) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.SomeTensor.ofTensor input) (vals := vals) (i := i) =
      .ok (Spec.SomeTensor.ofTensor
        (Tensor.addSpec (Spec.matVecMulSpec weight x) bias)) := by
  have hParentShape : parent.shape = [inDim] :=
    shape_eq_of_expectShape_eq_ok hParent
  have hTensor : parent.cast hParentShape = x := by
    have h := (expectShape_eq_ok parent hParentShape).symm.trans hParent
    injection h
  have hParentEq : parent = Spec.SomeTensor.ofTensor x := by
    calc
      parent = Spec.SomeTensor.ofTensor (parent.cast hParentShape) :=
        (Spec.SomeTensor.ofTensor_cast parent hParentShape).symm
      _ = Spec.SomeTensor.ofTensor x := congrArg Spec.SomeTensor.ofTensor hTensor
  subst parent
  have hPayload : (payloadOfParamStore (α := α) ps).linear? id =
      some ({ outDim := outDim, inDim := inDim, W := weight, b := bias } : LinearWB α) := by
    rw [payloadOfParamStore_linear?_eq, hStore]
    rfl
  have hLinear :
      Graph.evalLinear (α := α) (payloadOfParamStore (α := α) ps) id
          (Spec.SomeTensor.mk (α := α) [inDim] x) (.dim outDim .scalar) =
        .ok (Spec.SomeTensor.mk (α := α) [outDim]
          (Tensor.addSpec (Spec.matVecMulSpec weight x) bias)) := by
    simp [Graph.evalLinear, hPayload, Graph.expectShape, Graph.linearLeading,
      Shape.toList, Shape.ofList, Shape.concat,
      Bind.bind, Except.bind, Pure.pure, Except.pure]
    cases x
    rfl
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hNode,
    Graph.unaryParentId, NN.IR.unaryParent?, hParentValue, hLinear,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Evaluation reads arbitrary-rank convolution parameters from the matching store entry. -/
theorem evalConv_from_paramStore
    {α : Type} [TorchLean.Storage α] [Context α]
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (id : Nat) (params : ConvParams α) (config : ConvConfig)
    (parent : Spec.SomeTensor α) (leading : Shape)
    (hStore : ps.convCfg.get? id = some params)
    (hConfig : params.matchesConfig config = true)
    (hInfer : OpContracts.inferConvConfigOutShape "conv" config parent.shape =
      .ok (params.output leading))
    (hLeading : Shape.ofList (parent.shape.toList.take config.channelAxis) = leading)
    (hInput : parent.shape = params.input leading) :
    Graph.evalConv (α := α) (payloadOfParamStore (α := α) ps) id config parent =
      .ok (Spec.SomeTensor.ofTensor
        (Tensor.mapLeading leading
          (Spec.groupedConvSpec (α := α) (stride := params.stride)
            (dilation := params.dilation) (paddingBefore := params.padding)
            (paddingAfter := params.paddingAfter) params.groups params.spec.kernel params.spec.bias)
          (hInput ▸ parent.tensor))) := by
  have hPayload : (payloadOfParamStore (α := α) ps).conv? id = some params := by
    simpa [payloadOfParamStore_conv?_eq] using hStore
  unfold Graph.evalConv
  rw [hInfer]
  simp only [Bind.bind, Except.bind, hPayload]
  rw [if_pos hConfig, hLeading]
  split
  · congr
  · contradiction

/--
A convolution node reads arbitrary-rank convolution parameters from the matching store entry.

The shape-inference equation is the graph's dynamic check. The remaining hypotheses identify the
typed parent and output shapes; none fixes a spatial rank or memory layout.
-/
theorem evalAt_conv_from_paramStore_of_getNode
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (i id parentId : Nat)
    (params : ConvParams α)
    (config : ConvConfig)
    (leading inputShape : Shape)
    (input : Tensor α inputShape)
    (vals : Array (Spec.SomeTensor α))
    (x : Tensor α (params.input leading))
    (parent : Spec.SomeTensor α)
    (hNode : Graph.getNode (g := g) i =
      pure (NN.IR.Node.mk id #[parentId] (.conv config) (params.output leading)))
    (hParentValue : vals[parentId]? = some parent)
    (hParent : Graph.expectShape (α := α) (expected := params.input leading) parent = .ok x)
    (hStore : ps.convCfg.get? id = some params)
    (hConfig : params.matchesConfig config = true)
    (hInfer : OpContracts.inferConvConfigOutShape "conv" config parent.shape =
      .ok (params.output leading))
    (hLeading : Shape.ofList (parent.shape.toList.take config.channelAxis) = leading) :
    Graph.evalAt (α := α) (g := g) (payload := payloadOfParamStore (α := α) ps)
        (input := Spec.SomeTensor.ofTensor input) (vals := vals) (i := i) =
      .ok (Spec.SomeTensor.ofTensor
        (Tensor.mapLeading leading
          (Spec.groupedConvSpec (α := α) (stride := params.stride)
            (dilation := params.dilation) (paddingBefore := params.padding)
            (paddingAfter := params.paddingAfter) params.groups params.spec.kernel params.spec.bias)
          x)) := by
  have hParentShape : parent.shape = params.input leading :=
    shape_eq_of_expectShape_eq_ok hParent
  have hTensor : parent.cast hParentShape = x := by
    have h := (expectShape_eq_ok parent hParentShape).symm.trans hParent
    injection h
  have hTensor' : hParentShape ▸ parent.tensor = x := by
    rw [Tensor.eqRec_eq_cast_shape]
    exact hTensor
  have hEval :
      Graph.evalConv (α := α) (payloadOfParamStore (α := α) ps) id config parent =
        .ok (Spec.SomeTensor.ofTensor
          (Tensor.mapLeading leading
            (Spec.groupedConvSpec (α := α) (stride := params.stride)
              (dilation := params.dilation) (paddingBefore := params.padding)
              (paddingAfter := params.paddingAfter) params.groups params.spec.kernel
                params.spec.bias)
            x)) := by
    simpa [hTensor'] using
      evalConv_from_paramStore (ps := ps) (id := id) (params := params) (config := config)
        (parent := parent) (leading := leading) hStore hConfig hInfer hLeading hParentShape
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, hNode,
    Graph.unaryParentId, NN.IR.unaryParent?, hParentValue, hEval, ConvParams.output,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

end Correctness.IRStep

end NN.Verification.Builtin.Proved
