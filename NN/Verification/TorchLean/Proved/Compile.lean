/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Syntax

/-!
# Verified Forward Fragment: Compilation

Compilation from the first-order forward fragment into the verifier IR graph and parameter store.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

/-! ## Compilation to verifier IR -/

/-- Flatten a well-formed tensor into the `FlatVec` payload format used by CROWN/LiRPA IR nodes. -/
def flatOfTensor {α : Type} [Context α] {s : Shape}
    (_wf : Shape.WellFormed s)
    (t : Tensor α s) : NN.MLTheory.CROWN.Graph.FlatVec α :=
  { n := Shape.size s, v := Tensor.flattenSpec (α := α) (s := s) t }

/--
Compile a single forward-fragment node into the verifier IR.

Returns the corresponding `NN.IR.Node` together with an updated CROWN `ParamStore` that contains any
payload required by `.const`, `.linear`, and payload-backed convolution nodes.
-/
def compileNode
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (id : Nat)
    (node : Node α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    NN.IR.Node × NN.MLTheory.CROWN.Graph.ParamStore α :=
  match node with
  | .const (s := s) wf t =>
      let n : NN.IR.Node := { id := id, parents := [], kind := .const s, outShape := s }
      let ps' := { ps with constVals := ps.constVals.insert id (flatOfTensor (α := α) (s := s) wf t)
        }
      (n, ps')
  | .paramConst (s := s) wf p =>
      let t := getParam (α := α) (paramShapes := paramShapes) params p
      let n : NN.IR.Node := { id := id, parents := [], kind := .const s, outShape := s }
      let ps' := { ps with constVals := ps.constVals.insert id (flatOfTensor (α := α) (s := s) wf t)
        }
      (n, ps')
  | .add (s := s) a b =>
      ({ id := id, parents := [a.id, b.id], kind := .add, outShape := s }, ps)
  | .sub (s := s) a b =>
      ({ id := id, parents := [a.id, b.id], kind := .sub, outShape := s }, ps)
  | .mulElem (s := s) a b =>
      ({ id := id, parents := [a.id, b.id], kind := .mul_elem, outShape := s }, ps)
  | .relu (s := s) x =>
      ({ id := id, parents := [x.id], kind := .relu, outShape := s }, ps)
  | .exp (s := s) x =>
      ({ id := id, parents := [x.id], kind := .exp, outShape := s }, ps)
  | .log (s := s) x =>
      ({ id := id, parents := [x.id], kind := .log, outShape := s }, ps)
  | .inv (s := s) x =>
      ({ id := id, parents := [x.id], kind := .inv, outShape := s }, ps)
  | .matmul2d m _n p a b =>
      ({ id := id
         parents := [a.id, b.id]
         kind := .matmul
         outShape := .dim m (.dim p .scalar) }, ps)
  | .bmm batch m _n p a b =>
      ({ id := id
         parents := [a.id, b.id]
         kind := .matmul
         outShape := .dim batch (.dim m (.dim p .scalar)) }, ps)
  | .reshape inS outS _h x =>
      ({ id := id, parents := [x.id], kind := .reshape inS outS, outShape := outS }, ps)
  | .swap_first_two m n rest x =>
      ({ id := id, parents := [x.id], kind := .swap_first_two, outShape := .dim n (.dim m rest) },
        ps)
  | .transpose3dLastTwo _a _b _c x =>
      ({ id := id, parents := [x.id], kind := .transpose3dLastTwo, outShape := out }, ps)
  | .softmaxLast (s := s) _hRank x =>
      let axis := (Shape.rank s) - 1
      ({ id := id, parents := [x.id], kind := .softmax axis, outShape := s }, ps)
  | .layernorm2d seqLen embedDim _hSeq _hEmb x =>
      ({ id := id
         parents := [x.id]
         kind := .layernorm (axis := 1)
         outShape := .dim seqLen (.dim embedDim .scalar) }, ps)
  | .linear inDim outDim w b x =>
      let wT := getParam (α := α) (paramShapes := paramShapes) params w
      let bT := getParam (α := α) (paramShapes := paramShapes) params b
      let n : NN.IR.Node :=
        { id := id, parents := [x.id], kind := .linear, outShape := .dim outDim .scalar }
      let ps' :=
        { ps with
            linearWB := ps.linearWB.insert id { m := outDim, n := inDim, w := wT, b := bT } }
      (n, ps')
  | .conv2d inC outC kH kW stride padding inH inW hIn hKH hKW hStride _hHeight _hWidth kernel bias x =>
      let kT := getParam (α := α) (paramShapes := paramShapes) params kernel
      let bT := getParam (α := α) (paramShapes := paramShapes) params bias
      let outShape : Shape :=
        .dim outC
          (.dim ((inH + 2 * padding - kH) / stride + 1)
            (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))
      let n : NN.IR.Node :=
        { id := id
          parents := [x.id]
          kind := .conv2d inC outC kH kW stride padding
          outShape := outShape }
      let spec : Spec.Conv2DSpec inC outC kH kW stride padding α hIn hKH hKW :=
        { kernel := kT, bias := bT }
      let cfg : NN.MLTheory.CROWN.Graph.Conv2DParams α :=
        { inC := inC, outC := outC, kH := kH, kW := kW
          stride := stride, padding := padding
          inH := inH, inW := inW
          hIn := hIn, hKH := hKH, hKW := hKW, hStride := hStride,
          spec := spec }
      let ps' := { ps with conv2dCfg := ps.conv2dCfg.insert id cfg }
      (n, ps')
  | .mseLoss (s := _s) yhat target =>
      ({ id := id, parents := [yhat.id, target.id], kind := .mseLoss, outShape := .scalar }, ps)

/--
Compile a forward let-chain into a `CompiledIR` graph.

This threads an accumulator `CompiledIR` that contains:
- the growing `NN.IR.Graph`,
- the payload store (`ParamStore`),
- and the current output id.
-/
def compileFGraph
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
    (g : FGraph α paramShapes inShape ss out)
    (params : Runtime.Autograd.Torch.TList α paramShapes)
    (c : NN.Verification.TorchLean.CompiledIR α) :
    NN.Verification.TorchLean.CompiledIR α :=
  match g with
  | .ret y =>
      { c with outputId := y.id }
  | .let1 (ss := ss) (mid := mid) (out := out) node gNext =>
      let id := c.graph.nodes.size
      let (n, ps') :=
        compileNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out :=
          mid)
          id node params c.ps
      let c' : NN.Verification.TorchLean.CompiledIR α :=
        { c with graph := { nodes := c.graph.nodes.push n }, ps := ps', outputId := id }
      compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss ++ [mid])
        (out := out)
        gNext params c'

/--
Compile a proved forward fragment program into the verifier IR.

The resulting `CompiledIR` can be executed by the IR evaluator, and we prove (in this file) that
its denotation agrees with `evalForward1`.
-/
def compileVerifiedForward1
    {α : Type} [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (p : Program α paramShapes inShape outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    NN.Verification.TorchLean.CompiledIR α :=
  let input : NN.IR.Node := { id := 0, parents := [], kind := .input, outShape := inShape }
  let c0 : NN.Verification.TorchLean.CompiledIR α :=
    { graph := { nodes := #[input] }, ps := {}, inputId := 0, outputId := 0 }
  compileFGraph (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := []) (out :=
    outShape)
    p params c0

end NN.Verification.TorchLean.Proved
