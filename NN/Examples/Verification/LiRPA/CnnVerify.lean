/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Operators.Conv
public import NN.Verification.Cert.IBPCert

/-!
# CnnVerify

LiRPA/IBP certificate checker: CNN conv2d -> linear head.

This is a small workflow that:
- encodes a single conv2d as an affine form (so the graph stays in the "flat vector" LiRPA engine),
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


namespace NN.Examples.Verification.LiRPA.CnnVerify

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor
open NN.Verification.IBPCert

/--
Small fixed graph:
`input(flattened) -> linear(conv2d-as-affine) -> linear(head)`.

We keep it flat so the certificate checker works over `FlatBox` inputs.
-/
def buildGraph : Graph :=
  let inC := 1; let inH := 4; let inW := 4
  let inShape := Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))
  let nIn := inShape.size
  let outC := 1; let kH := 3; let kW := 3; let stride := 1; let padding := 0
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  let outShape := Shape.dim outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  let nConv := outShape.size
  let nOut := 2
  let n0 : Node := { id := 0, parents := [], kind := .input, outShape := .dim nIn .scalar }
  let n1 : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim nConv .scalar }
  let n2 : Node := { id := 2, parents := [1], kind := .linear, outShape := .dim nOut .scalar }
  { nodes := #[n0, n1, n2] }

/--
Seed deterministic parameters and the input box.

Key trick: we compute an affine form for conv2d (`A, c`) on the input box and insert it as a
`linearWB` entry, so the graph contains only `.linear` nodes.
-/
def seedParamsFloat : ParamStore Float :=
  let inC := 1; let outC := 1; let kH := 3; let kW := 3; let stride := 1; let padding := 0
  let inH := 4; let inW := 4
  have h1 : inC ≠ 0 := by decide
  have h2 : kH ≠ 0 := by decide
  have h3 : kW ≠ 0 := by decide
  let kernel : Tensor Float (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) :=
    Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.dim (fun i => Tensor.dim (fun j =>
      Tensor.scalar (Float.ofNat (1 + (i.val + j.val)))))))
  let bias : Tensor Float (.dim outC .scalar) := Tensor.dim (fun _ => Tensor.scalar (0.0))
  let conv : Spec.Conv2DSpec inC outC kH kW stride padding Float h1 h2 h3 :=
    { kernel := kernel, bias := bias }
  -- Seed input box (center ones, eps)
  let x0 := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar (1.0))))
  let eps : Float := 0.1
  let rad := Spec.fill (α:=Float) eps (.dim inC (.dim inH (.dim inW .scalar)))
  let xB : Box Float (.dim inC (.dim inH (.dim inW .scalar))) :=
    { lo := Tensor.subSpec x0 rad, hi := Tensor.addSpec x0 rad }
  let aff := NN.MLTheory.CROWN.crownConv2dAffineForm (α:=Float) (inC:=inC) (outC:=outC) (kH:=kH)
    (kW:=kW) (stride:=stride) (padding:=padding) (inH:=inH) (inW:=inW) conv xB
  let inShape := Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  let outShape := Shape.dim outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  let nIn := inShape.size
  let nConv := outShape.size
  -- Linear head 4→2
  let Whead : Tensor Float (.dim 2 (.dim nConv .scalar)) := Tensor.dim (fun i => Tensor.dim (fun j
    => Tensor.scalar (Float.ofNat (2 + (i.val + j.val)))))
  let bhead : Tensor Float (.dim 2 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat
    (i.val)))
  let ps0 : ParamStore Float := {}
  -- set input box
  let inFlat : FlatBox Float :=
    { dim := nIn, lo := Tensor.flattenSpec xB.lo, hi := Tensor.flattenSpec xB.hi }
  let ps1 := { ps0 with inputBoxes := ps0.inputBoxes.insert 0 inFlat }
  -- set conv as linear (A,c)
  let ps2 :=
    { ps1 with
      linearWB :=
        ps1.linearWB.insert 1
          { m := nConv
            n := nIn
            w := aff.A
            b := aff.c } }
  -- set head linear
  let ps3 :=
    { ps2 with linearWB := ps2.linearWB.insert 2 ({ m := 2, n := nConv, w := Whead, b := bhead }) }
  ps3

/--
Check an IBP certificate JSON against this CNN graph.

This is wired into `lake exe verify -- lirpa-cnn [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedParamsFloat
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 2) path

end NN.Examples.Verification.LiRPA.CnnVerify
