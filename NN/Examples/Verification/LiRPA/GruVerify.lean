/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.IBPCert

/-!
# GruVerify

LiRPA/IBP certificate checker: GRU-style gate fragment.

This example builds a small nonlinear graph with `sigmoid`, `tanh`, and an elementwise multiply:
`x -> linear -> sigmoid`
`x -> linear -> tanh`
`mul_elem(sigmoid(x), tanh(x))`

It is a focused fragment that exercises common RNN nonlinearities in the
LiRPA certificate checker.

References:
- IBP: arXiv:1810.12715 `https://arxiv.org/abs/1810.12715`
- auto_LiRPA (reference implementation / exporter inspiration):
  `https://github.com/Verified-Intelligence/auto_LiRPA`

Export (Python):
`python3.12 scripts/verification/lirpa/export_gru_cert.py`

Run (Lean):
`lake exe verify -- lirpa-gru [NN/Examples/Verification/LiRPA/gru_gate_cert.json]`
-/

@[expose] public section


namespace NN.Examples.Verification.LiRPA.GruVerify

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor
open NN.Verification.IBPCert

/-- Small nonlinear graph exercising `sigmoid`, `tanh`, and `mul_elem`. -/
def buildGraph : Graph :=
  let n := 3
  let n0 : Node := { id := 0, parents := [], kind := .input, outShape := .dim n .scalar }
  let n1 : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim n .scalar }
  let n2 : Node := { id := 2, parents := [1], kind := .sigmoid, outShape := .dim n .scalar }
  let n3 : Node := { id := 3, parents := [0], kind := .linear, outShape := .dim n .scalar }
  let n4 : Node := { id := 4, parents := [3], kind := .tanh, outShape := .dim n .scalar }
  let n5 : Node := { id := 5, parents := [2, 4], kind := .mul_elem, outShape := .dim n .scalar }
  { nodes := #[n0, n1, n2, n3, n4, n5] }

/-- Seed deterministic linear weights for the two `.linear` nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let n := 3
  let W : Tensor Float (.dim n (.dim n .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + (i.val + j.val)))))
  let b : Tensor Float (.dim n .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val)))
  let ps0 : ParamStore Float := {}
  let ps1 := { ps0 with linearWB := ps0.linearWB.insert 1 ({ m := n, n := n, w := W, b := b }) }
  let ps2 := { ps1 with linearWB := ps1.linearWB.insert 3 ({ m := n, n := n, w := W, b := b }) }
  ps2

/-- Insert an `L∞` input box of radius `eps` around a fixed center point. -/
def seedInputFloat (ps : ParamStore Float) (eps : Float) : ParamStore Float :=
  let n := 3
  let x0 : Tensor Float (.dim n .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val +
    1)))
  let rad := Tensor.scaleSpec (Spec.fill (α:=Float) eps (.dim n .scalar)) 1
  let xB : Box Float (.dim n .scalar) :=
    { lo := Tensor.subSpec x0 rad
      hi := Tensor.addSpec x0 rad }
  { ps with inputBoxes := ps.inputBoxes.insert 0 { dim := n, lo := xB.lo, hi := xB.hi } }

/--
Check an IBP certificate JSON against this GRU-fragment graph.

This is wired into `lake exe verify -- lirpa-gru [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedInputFloat seedParamsFloat (eps := 0.5)
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 5) path

end NN.Examples.Verification.LiRPA.GruVerify
