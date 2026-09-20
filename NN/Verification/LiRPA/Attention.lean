/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.LiRPA.ExampleInputs

/-!
# LiRPA attention certificate checker

LiRPA/IBP certificate checker: attention-like softmax graph.

This module builds a compact computation graph:
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


namespace NN.Verification.LiRPA.Attention

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open Spec TorchLean
open TorchLean.Tensor

/-- Small fixed graph with one `softmax` node, used to exercise certificate checking. -/
def buildGraph : Graph :=
  let inputNode : Node := { id := 0, parents := #[], kind := .input, outShape := [4] }
  let scoreNode : Node := { id := 1, parents := #[0], kind := .matmul, outShape := [5] }
  let softmaxNode : Node :=
    { id := 2, parents := #[1], kind := .softmax (axis := 0), outShape := [5] }
  let valueProjectionNode : Node := { id := 3, parents := #[2], kind := .matmul, outShape := [3] }
  { nodes := #[inputNode, scoreNode, softmaxNode, valueProjectionNode] }

/-- Seed deterministic weights for the two matmul nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let scoreWeight : Tensor Float [5, 4] :=
    Tensor.generate [5, 4] fun
      | [i, j] => Float.ofNat (1 + i + 2 * j)
      | _ => 0.0
  let valueWeight : Tensor Float [3, 5] :=
    Tensor.generate [3, 5] fun
      | [i, j] => Float.ofNat (2 + i + j)
      | _ => 0.0
  let emptyStore : ParamStore Float := {}
  let withScoreWeight :=
    { emptyStore with
      matmulW := emptyStore.matmulW.insert 1 ({ m := 5, n := 4, w := scoreWeight }) }
  let withValueWeight :=
    { withScoreWeight with
      matmulW := withScoreWeight.matmulW.insert 3 ({ m := 3, n := 5, w := valueWeight }) }
  withValueWeight

/--
Check an IBP certificate JSON against this attention graph.

This is wired into `lake exe verify -- lirpa-attention [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  -- Every input coordinate gets the box $[x_i - \varepsilon, x_i + \varepsilon]$; the
  -- graph has 4 inputs, ids `0 .. 3`.
  let ps := ExampleInputs.seedNaturalInputBox 0 4 0.5 seedParamsFloat
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 3) path

end NN.Verification.LiRPA.Attention
