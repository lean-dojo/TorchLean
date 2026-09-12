/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Operators.Conv
public import NN.Verification.LiRPA.ExampleInputs

/-!
# LiRPA convolutional certificate checker

LiRPA/IBP certificate checker for a convolution followed by a linear head.

This workflow:
- encodes a convolution as an affine form (so the graph stays in the flat-vector LiRPA engine),
- adds a linear head, and
- checks a JSON certificate (produced by Python) using `NN.Verification.Cert.IBPCert`.

References:
- IBP: arXiv:1810.12715 `https://arxiv.org/abs/1810.12715`
- auto_LiRPA (reference implementation / cert exporter inspiration):
  `https://github.com/Verified-Intelligence/auto_LiRPA`

Export (Python):
`python3.12 scripts/verification/lirpa/export_cnn_cert.py`

Run (Lean):
`lake exe verify -- lirpa-cnn [NN/Examples/Verification/LiRPA/cnn_cert.json]`
-/

@[expose] public section


namespace NN.Verification.LiRPA.Cnn

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

/--
Small fixed graph:
`input(flattened) -> linear(convolution) -> ReLU -> linear(head)`.

We keep it flat so the certificate checker works over `FlatBox` inputs.
-/
def buildGraph : Graph :=
  let inC := 1; let inH := 4; let inW := 4
  let inShape : Shape := [inC, inH, inW]
  let nIn := inShape.size
  let outC := 1; let kH := 3; let kW := 3; let stride := 1; let padding := 0
  let outH := Spec.Shape.slidingWindowOutDim inH kH stride padding
  let outW := Spec.Shape.slidingWindowOutDim inW kW stride padding
  let outShape : Shape := [outC, outH, outW]
  let nConv := outShape.size
  let nOut := 2
  let inputNode : Node := { id := 0, parents := #[], kind := .input, outShape := [nIn] }
  let convAffineNode : Node := { id := 1, parents := #[0], kind := .linear, outShape := [nConv] }
  let reluNode : Node := { id := 2, parents := #[1], kind := .relu, outShape := [nConv] }
  let classifierNode : Node := { id := 3, parents := #[2], kind := .linear, outShape := [nOut] }
  { nodes := #[inputNode, convAffineNode, reluNode, classifierNode] }

/--
Seed deterministic parameters and the input box.

The convolution is materialized as its exact flattened matrix and bias. ReLU remains an explicit
graph node, so IBP applies its interval rule instead of disguising one affine relaxation as an exact
linear layer.
-/
def seedParamsFloat : ParamStore Float :=
  let inC := 1; let outC := 1; let kH := 3; let kW := 3; let stride := 1; let padding := 0
  let inH := 4; let inW := 4
  let kernelShape : TorchLean.Tensor Nat [2] :=
    Tensor.from #[kH, kW]
  let strides : TorchLean.Tensor Nat [2] :=
    Tensor.from #[stride, stride]
  let paddings : TorchLean.Tensor Nat [2] :=
    Tensor.from #[padding, padding]
  let inputSpatial : TorchLean.Tensor Nat [2] :=
    Tensor.from #[inH, inW]
  let inShape := Shape.ofList (inC :: Tensor.to inputSpatial (List Nat))
  let outSpatial := Spec.convOutSpatial inputSpatial kernelShape strides paddings
  let outShape := Shape.ofList (outC :: Tensor.to outSpatial (List Nat))
  let nIn := inShape.size
  let nConv := outShape.size
  let kernelValues : Tensor Float [outC, inC, kH, kW] :=
    Tensor.generate [outC, inC, kH, kW] fun
      | [_, _, i, j] => Float.ofNat (1 + i + j)
      | _ => 0.0
  let kernel :
      Tensor Float (Shape.ofList (outC :: inC :: Tensor.to kernelShape (List Nat))) := by
    have hKernelShape : Tensor.to kernelShape (List Nat) = [kH, kW] := by
      change Tensor.to (Tensor.from #[kH, kW]) (List Nat) = [kH, kW]
      exact Tensor.to_list_from_array #[kH, kW]
    simpa [hKernelShape] using kernelValues
  let bias : Tensor Float [outC] := Tensor.generate [outC] fun _ => 0.0
  let conv : Spec.ConvSpec 2 inC outC kernelShape strides paddings Float :=
    { kernel := kernel, bias := bias }
  -- Seed input box (center ones, eps)
  let inputCenter : Tensor Float inShape := Tensor.full inShape 1.0
  let eps : Float := 0.1
  let rad := Tensor.full (α := Float) inShape eps
  let xB : Box Float inShape :=
    { lo := Tensor.subSpec inputCenter rad, hi := Tensor.addSpec inputCenter rad }
  let convWeight : Tensor Float [nConv, nIn] :=
    NN.MLTheory.CROWN.convLinearMatrix (α := Float)
    (inSpatial := inputSpatial) conv
  let convBias : Tensor Float [nConv] :=
    NN.MLTheory.CROWN.convBiasBroadcast (α := Float) (outSpatial := outSpatial) conv.bias
  -- Linear head 4→2
  let headWeight : Tensor Float [2, nConv] :=
    Tensor.generate [2, nConv] fun
      | [i, j] => Float.ofNat (2 + i + j)
      | _ => 0.0
  let headBias : Tensor Float [2] :=
    Tensor.generate [2] fun
      | [i] => Float.ofNat i
      | _ => 0.0
  let emptyStore : ParamStore Float := {}
  -- set input box
  let inFlat : FlatBox Float :=
    { dim := nIn, lo := Tensor.flattenSpec xB.lo, hi := Tensor.flattenSpec xB.hi }
  let withInputBox := emptyStore.seedInputBox 0 inFlat
  -- Store the exact flattened convolution as a linear node.
  let withConvAffine :=
    { withInputBox with
      linearWB :=
        withInputBox.linearWB.insert 1
          { m := nConv
            n := nIn
            w := convWeight
            b := convBias } }
  -- set head linear
  let withClassifier :=
    { withConvAffine with
      linearWB := withConvAffine.linearWB.insert 3
        ({ m := 2, n := nConv, w := headWeight, b := headBias }) }
  withClassifier

/--
Check an IBP certificate JSON against this CNN graph.

This is wired into `lake exe verify -- lirpa-cnn [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedParamsFloat
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 3) path

end NN.Verification.LiRPA.Cnn
