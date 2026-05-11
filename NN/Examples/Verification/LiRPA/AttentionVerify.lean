/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.IBPCert

/-!
# AttentionVerify

LiRPA/IBP certificate checker: attention-like softmax graph.

This example builds a compact computation graph:
`input -> matmul -> softmax -> matmul`,
seeds a small input box, and checks a JSON certificate produced by an external IBP/LiRPA tool.

References:
- IBP: "On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models"
  (arXiv:1810.12715): `https://arxiv.org/abs/1810.12715`
- CROWN (background): `https://arxiv.org/abs/1811.00866`
- auto_LiRPA (common reference implementation):
  `https://github.com/Verified-Intelligence/auto_LiRPA`

Export (Python):
`python3.12 scripts/verification/lirpa/export_attention_cert.py`

Run (Lean):
`lake exe verify -- lirpa-attention [NN/Examples/Verification/LiRPA/attention_softmax_cert.json]`
-/

@[expose] public section


namespace NN.Examples.Verification.LiRPA.AttentionVerify

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open _root_.Spec
open _root_.Spec.Tensor
open NN.Verification.IBPCert

/-- Small fixed graph with one `softmax` node, used to exercise certificate checking. -/
def buildGraph : Graph :=
  let n0 : Node := { id := 0, parents := [], kind := .input, outShape := .dim 4 .scalar }
  let n1 : Node := { id := 1, parents := [0], kind := .matmul, outShape := .dim 5 .scalar }
  let n2 : Node :=
    { id := 2, parents := [1], kind := .softmax (axis := 0), outShape := .dim 5 .scalar }
  let n3 : Node := { id := 3, parents := [2], kind := .matmul, outShape := .dim 3 .scalar }
  { nodes := #[n0, n1, n2, n3] }

/-- Seed deterministic weights for the two matmul nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let Wq : Tensor Float (.dim 5 (.dim 4 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + (i.val + 2*j.val)))))
  let Wv : Tensor Float (.dim 3 (.dim 5 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (2 + (i.val + j.val)))))
  let ps0 : ParamStore Float := {}
  let ps1 := { ps0 with matmulW := ps0.matmulW.insert 1 ({ m := 5, n := 4, w := Wq }) }
  let ps2 := { ps1 with matmulW := ps1.matmulW.insert 3 ({ m := 3, n := 5, w := Wv }) }
  ps2

/-- Insert an `L∞` input box of radius `eps` around a fixed center point. -/
def seedInputFloat (ps : ParamStore Float) (eps : Float) : ParamStore Float :=
  let x0 : Tensor Float (.dim 4 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val +
    1)))
  let rad := Spec.fill (α:=Float) eps (.dim 4 .scalar)
  let xB : Box Float (.dim 4 .scalar) :=
    { lo := Tensor.subSpec x0 rad
      hi := Tensor.addSpec x0 rad }
  { ps with inputBoxes := ps.inputBoxes.insert 0 { dim := 4, lo := xB.lo, hi := xB.hi } }

/--
Check an IBP certificate JSON against this attention graph.

This is wired into `lake exe verify -- lirpa-attention [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedInputFloat (seedParamsFloat) (eps := (0.5))
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 3) path

end NN.Examples.Verification.LiRPA.AttentionVerify
