/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.IBPCert

/-!
# TransformerEncoderVerify

LiRPA/IBP certificate checker: transformer-encoder-like graph.

This transformer encoder block includes:
- attention-like `softmax` flow,
- residual additions,
- `layernorm`, and
- a 2-layer feed-forward network with `relu`.

It exists primarily to exercise certificate checking across a wider set of nonlinear ops than the
MLP/CNN workflows.

References:
- IBP: arXiv:1810.12715 `https://arxiv.org/abs/1810.12715`
- CROWN background: `https://arxiv.org/abs/1811.00866`
- auto_LiRPA (reference implementation / exporter inspiration):
  `https://github.com/Verified-Intelligence/auto_LiRPA`

Export (Python):
`python3.12 scripts/verification/lirpa/export_crown_cert.py`

Run (Lean):
`lake exe verify -- lirpa-encoder [NN/Examples/Verification/LiRPA/transformer_encoder_cert.json]`
-/

@[expose] public section


namespace NN.Examples.Verification.LiRPA.TransformerEncoderVerify

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open _root_.Spec
open _root_.Spec.Tensor
open NN.Verification.IBPCert

/-- Small fixed graph with residual + layernorm + FFN (see module doc). -/
def buildGraph : Graph :=
  let nModel := 4
  let scoresDim := 5
  let nHidden := 6
  let n0 : Node := { id := 0, parents := [], kind := .input, outShape := .dim nModel .scalar }
  let n1 : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim scoresDim .scalar }
  let n2 : Node :=
    { id := 2, parents := [1], kind := .softmax (axis := 0), outShape := .dim scoresDim .scalar }
  let n3 : Node := { id := 3, parents := [2], kind := .matmul, outShape := .dim nModel .scalar }
  let n4 : Node := { id := 4, parents := [0, 3], kind := .add, outShape := .dim nModel .scalar }
  let n5 : Node :=
    { id := 5, parents := [4], kind := .layernorm (axis := 0), outShape := .dim nModel .scalar }
  let n6 : Node := { id := 6, parents := [5], kind := .linear, outShape := .dim nHidden .scalar }
  let n7 : Node := { id := 7, parents := [6], kind := .relu, outShape := .dim nHidden .scalar }
  let n8 : Node := { id := 8, parents := [7], kind := .linear, outShape := .dim nModel .scalar }
  let n9 : Node := { id := 9, parents := [5, 8], kind := .add, outShape := .dim nModel .scalar }
  let n10 : Node :=
    { id := 10, parents := [9], kind := .layernorm (axis := 0), outShape := .dim nModel .scalar }
  { nodes := #[n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10] }

/-- Seed deterministic parameters for the `.linear` / `.matmul` nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let nModel := 4; let scoresDim := 5; let nHidden := 6
  let Wq : Tensor Float (.dim scoresDim (.dim nModel .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + (i.val + 2*j.val)))))
  let bq : Tensor Float (.dim scoresDim .scalar) := Tensor.dim (fun i => Tensor.scalar (0.1 *
    Float.ofNat i.val))
  let Wv : Tensor Float (.dim nModel (.dim scoresDim .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (2 + (i.val + j.val)))))
  let W1 : Tensor Float (.dim nHidden (.dim nModel .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + ((i.val + j.val) %
      3)))))
  let b1 : Tensor Float (.dim nHidden .scalar) := Tensor.dim (fun i => Tensor.scalar (0.05 *
    Float.ofNat i.val))
  let W2 : Tensor Float (.dim nModel (.dim nHidden .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (2 + ((i.val + j.val) %
      4)))))
  let b2 : Tensor Float (.dim nModel .scalar) := Tensor.dim (fun i => Tensor.scalar (0.02 *
    Float.ofNat i.val))
  let ps0 : ParamStore Float := {}
  let ps1 :=
    { ps0 with
      linearWB :=
        ps0.linearWB.insert 1
          { m := scoresDim
            n := nModel
            w := Wq
            b := bq } }
  let ps2 := { ps1 with matmulW := ps1.matmulW.insert 3 ({ m := nModel, n := scoresDim, w := Wv }) }
  let ps3 :=
    { ps2 with linearWB := ps2.linearWB.insert 6 ({ m := nHidden, n := nModel, w := W1, b := b1 }) }
  let ps4 :=
    { ps3 with linearWB := ps3.linearWB.insert 8 ({ m := nModel, n := nHidden, w := W2, b := b2 }) }
  ps4

/-- Insert an `L∞` input box of radius `eps` around a fixed center point. -/
def seedInputFloat (ps : ParamStore Float) (eps : Float) : ParamStore Float :=
  let nModel := 4
  let x0 : Tensor Float (.dim nModel .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat
    (i.val + 1)))
  let rad := Spec.fill (α:=Float) eps (.dim nModel .scalar)
  let xB : Box Float (.dim nModel .scalar) :=
    { lo := Tensor.subSpec x0 rad
      hi := Tensor.addSpec x0 rad }
  { ps with inputBoxes := ps.inputBoxes.insert 0 { dim := nModel, lo := xB.lo, hi := xB.hi } }

/--
Check an IBP certificate JSON against this transformer-encoder graph.

This is wired into `lake exe verify -- lirpa-encoder [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedInputFloat (seedParamsFloat) (eps := (0.5))
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 10) path

end NN.Examples.Verification.LiRPA.TransformerEncoderVerify
