/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.LiRPA.ExampleInputs

/-!
# LiRPA transformer encoder certificate checker

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


namespace NN.Verification.LiRPA.TransformerEncoder

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open Spec TorchLean
open TorchLean.Tensor

/-- Small fixed graph with residual + layernorm + FFN (see module doc). -/
def buildGraph : Graph :=
  let nModel := 4
  let scoresDim := 5
  let nHidden := 6
  let inputNode : Node := { id := 0, parents := #[], kind := .input, outShape := [nModel] }
  let scoreNode : Node := { id := 1, parents := #[0], kind := .linear, outShape := [scoresDim] }
  let softmaxNode : Node :=
    { id := 2, parents := #[1], kind := .softmax (axis := 0), outShape := [scoresDim] }
  let attentionValueNode : Node :=
    { id := 3, parents := #[2], kind := .matmul, outShape := [nModel] }
  let attentionResidualNode : Node :=
    { id := 4, parents := #[0, 3], kind := .add, outShape := [nModel] }
  let firstLayerNormNode : Node :=
    { id := 5, parents := #[4], kind := .layernorm (axis := 0), outShape := [nModel] }
  let feedForwardHiddenNode : Node :=
    { id := 6, parents := #[5], kind := .linear, outShape := [nHidden] }
  let feedForwardReluNode : Node :=
    { id := 7, parents := #[6], kind := .relu, outShape := [nHidden] }
  let feedForwardOutputNode : Node :=
    { id := 8, parents := #[7], kind := .linear, outShape := [nModel] }
  let feedForwardResidualNode : Node :=
    { id := 9, parents := #[5, 8], kind := .add, outShape := [nModel] }
  let finalLayerNormNode : Node :=
    { id := 10, parents := #[9], kind := .layernorm (axis := 0), outShape := [nModel] }
  { nodes :=
      #[ inputNode
       , scoreNode
       , softmaxNode
       , attentionValueNode
       , attentionResidualNode
       , firstLayerNormNode
       , feedForwardHiddenNode
       , feedForwardReluNode
       , feedForwardOutputNode
       , feedForwardResidualNode
       , finalLayerNormNode ] }

/-- Seed deterministic parameters for the `.linear` / `.matmul` nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let nModel := 4; let scoresDim := 5; let nHidden := 6
  let scoreWeight : Tensor Float [scoresDim, nModel] :=
    Tensor.generate [scoresDim, nModel] fun
      | [i, j] => Float.ofNat (1 + i + 2 * j)
      | _ => 0.0
  let scoreBias : Tensor Float [scoresDim] :=
    Tensor.generate [scoresDim] fun
      | [i] => 0.1 * Float.ofNat i
      | _ => 0.0
  let valueWeight : Tensor Float [nModel, scoresDim] :=
    Tensor.generate [nModel, scoresDim] fun
      | [i, j] => Float.ofNat (2 + i + j)
      | _ => 0.0
  let feedForwardHiddenWeight : Tensor Float [nHidden, nModel] :=
    Tensor.generate [nHidden, nModel] fun
      | [i, j] => Float.ofNat (1 + ((i + j) % 3))
      | _ => 0.0
  let feedForwardHiddenBias : Tensor Float [nHidden] :=
    Tensor.generate [nHidden] fun
      | [i] => 0.05 * Float.ofNat i
      | _ => 0.0
  let feedForwardOutputWeight : Tensor Float [nModel, nHidden] :=
    Tensor.generate [nModel, nHidden] fun
      | [i, j] => Float.ofNat (2 + ((i + j) % 4))
      | _ => 0.0
  let feedForwardOutputBias : Tensor Float [nModel] :=
    Tensor.generate [nModel] fun
      | [i] => 0.02 * Float.ofNat i
      | _ => 0.0
  let emptyStore : ParamStore Float := {}
  let withScoreLinear :=
    { emptyStore with
      linearWB :=
        emptyStore.linearWB.insert 1
          { m := scoresDim
            n := nModel
            w := scoreWeight
            b := scoreBias } }
  let withValueProjection :=
    { withScoreLinear with
      matmulW :=
        withScoreLinear.matmulW.insert 3
          ({ m := nModel, n := scoresDim, w := valueWeight }) }
  let withFeedForwardHidden :=
    { withValueProjection with
      linearWB :=
        withValueProjection.linearWB.insert 6
          ({ m := nHidden, n := nModel, w := feedForwardHiddenWeight,
             b := feedForwardHiddenBias }) }
  let withFeedForwardOutput :=
    { withFeedForwardHidden with
      linearWB :=
        withFeedForwardHidden.linearWB.insert 8
          ({ m := nModel, n := nHidden, w := feedForwardOutputWeight,
             b := feedForwardOutputBias }) }
  withFeedForwardOutput

/--
Check an IBP certificate JSON against this transformer-encoder graph.

This is wired into `lake exe verify -- lirpa-encoder [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  -- Every input coordinate gets the box $[x_i - \varepsilon, x_i + \varepsilon]$; the
  -- graph has 4 inputs, ids `0 .. 3`.
  let ps := ExampleInputs.seedNaturalInputBox 0 4 0.5 seedParamsFloat
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 10) path

end NN.Verification.LiRPA.TransformerEncoder
