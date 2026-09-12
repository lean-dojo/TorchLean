/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Syntax
public import NN.Verification.Builtin.Lowering.API

/-!
# Verified Forward Fragment: Lowering

Lowering from the first-order forward fragment into the verifier IR graph and parameter store.
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.IR

/-! ## Lowering to verifier IR -/

/-- Flatten a well-formed tensor into the `FlatTensor` payload format used by CROWN/LiRPA IR
nodes. -/
def flatOfTensor {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (_wf : Shape.WellFormed s)
    (t : Tensor α s) : NN.MLTheory.CROWN.Graph.FlatTensor α :=
  { n := Spec.Shape.size s, v := Tensor.flattenSpec (α := α) (shape := s) t }

/--
Lower a single forward-fragment node into the verifier IR.

Returns the corresponding `NN.IR.Node` together with an updated CROWN `ParamStore` that contains any
payload required by `.const`, `.linear`, and payload-backed convolution nodes.
-/
def lowerNode
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (id : Nat)
    (node : Node α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    NN.IR.Node × NN.MLTheory.CROWN.Graph.ParamStore α :=
  match node with
  | .const (s := s) wf t =>
      let n : NN.IR.Node := { id := id, parents := #[], kind := .const s, outShape := s }
      let ps' := {
        ps with
        constVals := ps.constVals.insert id (flatOfTensor (α := α) (s := s) wf t) }
      (n, ps')
  | .paramConst (s := s) wf p =>
      let t := getParam (α := α) (paramShapes := paramShapes) params p
      let n : NN.IR.Node := { id := id, parents := #[], kind := .const s, outShape := s }
      let ps' := {
        ps with
        constVals := ps.constVals.insert id (flatOfTensor (α := α) (s := s) wf t) }
      (n, ps')
  | .add (s := s) a b =>
      ({ id := id, parents := #[a.id, b.id], kind := .add, outShape := s }, ps)
  | .sub (s := s) a b =>
      ({ id := id, parents := #[a.id, b.id], kind := .sub, outShape := s }, ps)
  | .mulElem (s := s) a b =>
      ({ id := id, parents := #[a.id, b.id], kind := .mulElem, outShape := s }, ps)
  | .relu (s := s) x =>
      ({ id := id, parents := #[x.id], kind := .relu, outShape := s }, ps)
  | .exp (s := s) x =>
      ({ id := id, parents := #[x.id], kind := .exp, outShape := s }, ps)
  | .log (s := s) x =>
      ({ id := id, parents := #[x.id], kind := .log, outShape := s }, ps)
  | .inv (s := s) x =>
      ({ id := id, parents := #[x.id], kind := .inv, outShape := s }, ps)
  | .matmul (outShape := outShape) _op a b =>
      ({ id := id
         parents := #[a.id, b.id]
         kind := .matmul
         outShape := outShape }, ps)
  | .reshape inS outS _h x =>
      ({ id := id, parents := #[x.id], kind := .reshape inS outS, outShape := outS }, ps)
  | .transpose axis₁ axis₂ _hOut x =>
      ({ id := id, parents := #[x.id], kind := .transpose axis₁ axis₂, outShape := out }, ps)
  | .softmax (s := s) axis _hAxis x =>
      ({ id := id, parents := #[x.id], kind := .softmax axis, outShape := s }, ps)
  | .layerNorm (s := s) op x =>
      let ps' := { ps with layerNorm := ps.layerNorm.erase id }
      ({ id := id
         parents := #[x.id]
         kind := .layernorm op.axis
         outShape := s }, ps')
  | .linear inDim outDim w b x =>
      let wT := getParam (α := α) (paramShapes := paramShapes) params w
      let bT := getParam (α := α) (paramShapes := paramShapes) params b
      let n : NN.IR.Node :=
        { id := id, parents := #[x.id], kind := .linear, outShape := .dim outDim .scalar }
      let ps' :=
        { ps with
            linearWB := ps.linearWB.insert id { m := outDim, n := inDim, w := wT, b := bT } }
      (n, ps')
  | .conv (d := d) inC outC kernelShape stride padding inSpatial _hIn hKernel hStride _hInfer
      kernel bias x =>
      let kT := getParam (α := α) (paramShapes := paramShapes) params kernel
      let bT := getParam (α := α) (paramShapes := paramShapes) params bias
      let outShape : Shape := Shape.ofList
        (outC :: Tensor.to (Spec.convOutSpatial inSpatial kernelShape stride padding)
          (List Nat))
      let n : NN.IR.Node :=
        { id := id
          parents := #[x.id]
          kind := .conv
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
          outShape := outShape }
      let spec : Spec.ConvSpec d inC outC kernelShape stride padding α :=
        { kernel := kT, bias := bT }
      let config : NN.IR.ConvParams α :=
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
          spec := spec }
      let ps' := { ps with convCfg := ps.convCfg.insert id config }
      (n, ps')
  | .mseLoss (s := _s) yhat target =>
      ({ id := id, parents := #[yhat.id, target.id], kind := .mseLoss, outShape := .scalar }, ps)

/--
Lower a forward let-chain into a `LoweredIR` graph.

This threads an accumulator `LoweredIR` that contains:
- the growing `NN.IR.Graph`,
- the payload store (`ParamStore`),
- and the current output id.
-/
def lowerForwardLetChain
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : ForwardLetChain α paramShapes inShape ss out)
    (params : TorchLean.TensorPack α paramShapes)
    (c : NN.Verification.Builtin.LoweredIR α) :
    NN.Verification.Builtin.LoweredIR α :=
  match g with
  | .ret y =>
      { c with outputId := y.id }
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext =>
      let id := c.graph.nodes.size
      let (n, ps') :=
        lowerNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
          mid)
          id node params c.ps
      let c' : NN.Verification.Builtin.LoweredIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape)
        (ss := ss ++ [mid]) (out := out)
        gNext params c'

/--
Lower a proved forward-fragment program into the verifier IR.

The resulting `LoweredIR` can be executed by the IR evaluator, and we prove (in this file) that
its denotation agrees with `evalForward`.
-/
def lowerForwardProgramToIR
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : ForwardProgram α paramShapes inShape outShape)
    (params : TorchLean.TensorPack α paramShapes) :
    NN.Verification.Builtin.LoweredIR α :=
  let input : NN.IR.Node := { id := 0, parents := #[], kind := .input, outShape := inShape }
  let c0 : NN.Verification.Builtin.LoweredIR α :=
    { graph := { nodes := #[input] }, ps := {}, inputId := 0, outputId := 0 }
  lowerForwardLetChain (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
    outShape)
    p params c0

end NN.Verification.Builtin.Proved
