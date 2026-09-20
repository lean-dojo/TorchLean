/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.LiRPA.ExampleInputs

/-!
# LiRPA GRU gate certificate checker

LiRPA/IBP certificate checker: GRU-style gate fragment.

This module builds a small nonlinear graph with `sigmoid`, `tanh`, and an elementwise multiply:
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


namespace NN.Verification.LiRPA.Gru

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open Spec TorchLean
open TorchLean.Tensor

/-- Small nonlinear graph exercising `sigmoid`, `tanh`, and `mulElem`. -/
def buildGraph : Graph :=
  let n := 3
  let inputNode : Node := { id := 0, parents := #[], kind := .input, outShape := [n] }
  let gateLinearNode : Node := { id := 1, parents := #[0], kind := .linear, outShape := [n] }
  let sigmoidGateNode : Node := { id := 2, parents := #[1], kind := .sigmoid, outShape := [n] }
  let candidateLinearNode : Node := { id := 3, parents := #[0], kind := .linear, outShape := [n] }
  let candidateTanhNode : Node := { id := 4, parents := #[3], kind := .tanh, outShape := [n] }
  let gatedCandidateNode : Node :=
    { id := 5, parents := #[2, 4], kind := .mulElem, outShape := [n] }
  { nodes := #[inputNode, gateLinearNode, sigmoidGateNode, candidateLinearNode,
      candidateTanhNode, gatedCandidateNode] }

/-- Seed deterministic linear weights for the two `.linear` nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let n := 3
  let weight : Tensor Float [n, n] :=
    Tensor.generate [n, n] fun
      | [i, j] => Float.ofNat (1 + i + j)
      | _ => 0.0
  let bias : Tensor Float [n] :=
    Tensor.generate [n] fun
      | [i] => Float.ofNat i
      | _ => 0.0
  let emptyStore : ParamStore Float := {}
  let withGateLinear :=
    { emptyStore with
      linearWB := emptyStore.linearWB.insert 1 ({ m := n, n := n, w := weight, b := bias }) }
  let withCandidateLinear :=
    { withGateLinear with
      linearWB := withGateLinear.linearWB.insert 3 ({ m := n, n := n, w := weight, b := bias }) }
  withCandidateLinear

/--
Check an IBP certificate JSON against this GRU-fragment graph.

This is wired into `lake exe verify -- lirpa-gru [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  -- Every input coordinate gets the box $[x_i - \varepsilon, x_i + \varepsilon]$; the
  -- graph has 3 inputs, ids `0 .. 2`.
  let ps := ExampleInputs.seedNaturalInputBox 0 3 0.5 seedParamsFloat
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 5) path

end NN.Verification.LiRPA.Gru
