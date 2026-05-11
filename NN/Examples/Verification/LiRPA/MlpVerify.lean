/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.IBPCert

/-!
# LiRPA MLP certificate checker

This module is a compact end-to-end example of *checking* an IBP certificate for a
compact MLP in TorchLean's graph-based verifier.

It does three things:
1. Defines a compact graph (`buildGraph`),
2. Seeds deterministic parameters and an input box,
3. Checks a JSON certificate produced by an external tool (LiRPA-style workflow).

Run via the unified CLI registry:

- `lake exe verify -- lirpa-mlp [path]`

If `path` is omitted, the CLI uses the default example certificate at:
`NN/Examples/Verification/LiRPA/mlp_cert.json`.
-/

@[expose] public section


namespace NN.Examples.Verification.LiRPA.MlpVerify

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open _root_.Spec
open _root_.Spec.Tensor
open NN.Verification.IBPCert

def buildGraph : Graph :=
  let n0 : Node := { id := 0, parents := [], kind := .input, outShape := .dim 3 .scalar }
  let n1 : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim 4 .scalar }
  let n2 : Node := { id := 2, parents := [1], kind := .relu,   outShape := .dim 4 .scalar }
  let n3 : Node := { id := 3, parents := [2], kind := .linear, outShape := .dim 2 .scalar }
  { nodes := #[n0, n1, n2, n3] }

def seedParamsFloat : ParamStore Float :=
  let W1 : Tensor Float (.dim 4 (.dim 3 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + (i.val + j.val)))))
  let b1 : Tensor Float (.dim 4 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val +
    1)))
  let W2 : Tensor Float (.dim 2 (.dim 4 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (2 + (i.val + j.val)))))
  let b2 : Tensor Float (.dim 2 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat
    (i.val)))
  let ps0 : ParamStore Float := {}
  let ps1 := { ps0 with linearWB := ps0.linearWB.insert 1 ({ m := 4, n := 3, w := W1, b := b1 }) }
  let ps2 := { ps1 with linearWB := ps1.linearWB.insert 3 ({ m := 2, n := 4, w := W2, b := b2 }) }
  ps2

def seedInputFloat (ps : ParamStore Float) (eps : Float) : ParamStore Float :=
  let x0 : Tensor Float (.dim 3 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val +
    1)))
  let rad := Spec.fill (α:=Float) eps (.dim 3 .scalar)
  let xB : Box Float (.dim 3 .scalar) :=
    { lo := Tensor.subSpec x0 rad
      hi := Tensor.addSpec x0 rad }
  { ps with inputBoxes := ps.inputBoxes.insert 0 { dim := 3, lo := xB.lo, hi := xB.hi } }

/-- Check an IBP certificate JSON file and throw an error if it does not match recomputed bounds. -/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedInputFloat (seedParamsFloat) (eps := (1.0))
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 3) path

end NN.Examples.Verification.LiRPA.MlpVerify
