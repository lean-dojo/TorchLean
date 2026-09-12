/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.LiRPA.ExampleInputs

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


namespace NN.Verification.LiRPA.Mlp

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open Spec TorchLean
open TorchLean.Tensor

/-- Four-node `3 → 4 → 2` ReLU MLP graph used by the compact LiRPA certificate example. -/
def buildGraph : Graph :=
  let inputNode : Node := { id := 0, parents := #[], kind := .input, outShape := [3] }
  let hiddenLinearNode : Node := { id := 1, parents := #[0], kind := .linear, outShape := [4] }
  let reluNode : Node := { id := 2, parents := #[1], kind := .relu, outShape := [4] }
  let outputLinearNode : Node := { id := 3, parents := #[2], kind := .linear, outShape := [2] }
  { nodes := #[inputNode, hiddenLinearNode, reluNode, outputLinearNode] }

/-- Deterministic Float weights and biases for both linear nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let hiddenWeight : Tensor Float [4, 3] :=
    Tensor.generate [4, 3] fun
      | [i, j] => Float.ofNat (1 + i + j)
      | _ => 0.0
  let hiddenBias : Tensor Float [4] :=
    Tensor.generate [4] fun
      | [i] => Float.ofNat (i + 1)
      | _ => 0.0
  let outputWeight : Tensor Float [2, 4] :=
    Tensor.generate [2, 4] fun
      | [i, j] => Float.ofNat (2 + i + j)
      | _ => 0.0
  let outputBias : Tensor Float [2] :=
    Tensor.generate [2] fun
      | [i] => Float.ofNat i
      | _ => 0.0
  let emptyStore : ParamStore Float := {}
  let withHiddenLayer :=
    { emptyStore with
      linearWB := emptyStore.linearWB.insert 1
        ({ m := 4, n := 3, w := hiddenWeight, b := hiddenBias }) }
  let withOutputLayer :=
    { withHiddenLayer with
      linearWB := withHiddenLayer.linearWB.insert 3
        ({ m := 2, n := 4, w := outputWeight, b := outputBias }) }
  withOutputLayer

/-- Check an IBP certificate JSON file and throw an error if it does not match recomputed bounds. -/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  -- Every input coordinate gets the box $[x_i - \varepsilon, x_i + \varepsilon]$; the
  -- graph has 3 inputs, ids `0 .. 2`.
  let ps := ExampleInputs.seedNaturalInputBox 0 3 1.0 seedParamsFloat
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 3) path

end NN.Verification.LiRPA.Mlp
