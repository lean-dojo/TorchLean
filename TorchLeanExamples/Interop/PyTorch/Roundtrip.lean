/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
import TorchLean.NN.Json
import TorchLean.NN.Module
import TorchLean.NN.Builder
import TorchLean.NN.PyTorch.Import.State
public import TorchLean.Tensor.Tactic.Init

/-!
# PyTorch state dicts in Lean models

This module reads the checked-in PyTorch state dicts of three small models (an MLP, a CNN and a
Transformer encoder block) and runs the ordinary `nn` model of each on a fixed input. The scripts
beside the fixtures write them with `python/torchlean/state.py` from the float64 copy of each
module's `state_dict()`. `TorchLean.PyTorch.state` reads a fixture into the model's
`TensorPack Float`, tensor `i` from the entry the PyTorch module names for it, such as `fc1.weight`,
and `nn.Module.instantiate` takes that state in place of the seeded initialization. The three
outputs are checked against PyTorch's by `TorchLeanTest/Interop/PyTorchRoundtrip.lean`.

Run via the TorchLean example runner:

`scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp`

Choose `mlp`, `cnn`, or `transformer` for `--model`.
-/

namespace TorchLean.Examples.PyTorchRoundtrip

open Lean
open Activation.Kind

/-! ## CLI model selection -/

/-- Which example model the round-trip driver should import. -/
inductive Model where
  | mlp
  | cnn
  | transformer
  deriving Repr, DecidableEq

/-- Parse the `--model` CLI flag accepted by the round-trip example. -/
def Model.parse? (s : String) : Option Model :=
  match s.toLower with
  | "mlp" => some .mlp
  | "cnn" => some .cnn
  | "transformer" => some .transformer
  | _ => none

private def usage : String :=
  String.intercalate "\n"
    [ "PyTorch round-trip example (TorchLean)"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp"
    , ""
    , "Notes:"
    , "  - `--model` accepts mlp, cnn, or transformer (default: mlp)."
    , ("  - The command reads the PyTorch state dict under " ++
      "`TorchLeanExamples/Interop/PyTorch/<Model>/` into the model's `TensorPack Float`.")
    , "  - It prints the output of the ordinary executable `nn` model on a fixed input."
    ]

/-! ## Paths, fixed example dimensions and PyTorch names -/

private def stateDictionaryPath : Model → System.FilePath
  | .mlp => "TorchLeanExamples/Interop/PyTorch/MLP/mlp.json"
  | .cnn => "TorchLeanExamples/Interop/PyTorch/CNN/cnn.json"
  | .transformer => "TorchLeanExamples/Interop/PyTorch/Transformer/transformer_encoder.json"

-- MLP dimensions (matches `TestMLP` in `train_mlp.py`).
private def mlpInputWidth : Nat := 2
private def mlpHiddenWidth : Nat := 3
private def mlpOutputWidth : Nat := 1

/-- The `nn` model of `TestMLP`: `Linear(2, 3)`, ReLU and `Linear(3, 1)`. -/
private def mlpModel :=
  nn.build 0 <| nn.Sequential![
    nn.linear mlpInputWidth mlpHiddenWidth,
    nn.op .relu,
    nn.linear mlpHiddenWidth mlpOutputWidth
  ]

/-- The names `TestMLP.state_dict()` gives the state of `mlpModel`, in its order. -/
private def mlpNames : List String :=
  ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]

-- CNN dimensions and hyperparameters (matches `TestCNN` in `train_cnn.py`).
private def cnnInputChannels : Nat := 1
private def cnnOutputChannels : Nat := 2
private def cnnInputHeight : Nat := 8
private def cnnInputWidth : Nat := 8
private def cnnKernelHeight : Nat := 3
private def cnnKernelWidth : Nat := 3
private def cnnPoolKernelHeight : Nat := 2
private def cnnPoolKernelWidth : Nat := 2

private def cnnInputSpatial : Tensor Nat [2] := [cnnInputHeight, cnnInputWidth]

private def cnnConvolution : nn.Conv.Config 2 :=
  { outChannels := cnnOutputChannels
    kernelSize := [cnnKernelHeight, cnnKernelWidth]
    stride := [1, 1]
    padding := [1, 1] }

private def cnnPooling : Window 2 :=
  { kernelSize := [cnnPoolKernelHeight, cnnPoolKernelWidth]
    stride := [2, 2] }

private def cnnAfterFirstConv : Tensor Nat [2] :=
  cnnConvolution.convOutputSpatial cnnInputSpatial

private def cnnAfterFirstPooling : Tensor Nat [2] :=
  cnnPooling.poolOutputSpatial cnnAfterFirstConv

private def cnnAfterSecondConv : Tensor Nat [2] :=
  cnnConvolution.convOutputSpatial cnnAfterFirstPooling

private def cnnAfterSecondPooling : Tensor Nat [2] :=
  cnnPooling.poolOutputSpatial cnnAfterSecondConv

private def cnnFlattenedWidth : Nat :=
  cnnOutputChannels * cnnAfterSecondPooling.prod

private def cnnModel := by
  letI : NeZero cnnInputChannels := ⟨by decide⟩
  letI : NeZero cnnOutputChannels := ⟨by decide⟩
  exact nn.build 0 <| nn.Sequential![
    nn.conv cnnInputSpatial cnnConvolution (inputChannels := cnnInputChannels),
    nn.op .relu,
    nn.maxPool cnnAfterFirstConv cnnPooling (channels := cnnOutputChannels),
    nn.conv cnnAfterFirstPooling cnnConvolution (inputChannels := cnnOutputChannels),
    nn.op .relu,
    nn.maxPool cnnAfterSecondConv cnnPooling (channels := cnnOutputChannels),
    nn.op (.reshape (Shape.size_vector _).symm),
    nn.linear cnnFlattenedWidth cnnOutputChannels
  ]

/-- The names `TestCNN.state_dict()` gives the state of `cnnModel`, in its order. -/
private def cnnNames : List String :=
  ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "fc.weight", "fc.bias"]

-- Transformer dimensions (matches `TinyTransformerEncoder` in `export_transformer.py`).
private def transformerSeqLength : Nat := 1
private def transformerModelWidth : Nat := 2
private def transformerHeadCount : Nat := 1
private def transformerFFNWidth : Nat := 2

private def transformerModel := by
  letI : NeZero transformerSeqLength := ⟨by decide⟩
  letI : NeZero transformerModelWidth := ⟨by decide⟩
  exact nn.build 0 <| nn.transformerBlock
    (sequenceLength := transformerSeqLength)
    (modelWidth := transformerModelWidth)
    { headCount := transformerHeadCount
      headWidth := transformerModelWidth / transformerHeadCount
      feedForwardWidth := transformerFFNWidth
      activation := relu }

/-- The names `TinyTransformerEncoder.state_dict()` gives the state of `transformerModel`, in its
order: the query, key, value and output projections, the first norm, the two feed-forward layers
and the second norm. The projections are `(input, output)` matrices in both. -/
private def transformerNames : List String :=
  ["layers.0.mha.q_weight", "layers.0.mha.k_weight", "layers.0.mha.v_weight",
    "layers.0.mha.out_weight", "layers.0.norm1.weight", "layers.0.norm1.bias",
    "layers.0.ffn.fc1.weight", "layers.0.ffn.fc1.bias", "layers.0.ffn.fc2.weight",
    "layers.0.ffn.fc2.bias", "layers.0.norm2.weight", "layers.0.norm2.bias"]

/-! ## Import actions -/

/-- Read the state dict of `model` into `TensorPack Float shapes`, tensor `i` from the entry
`names[i]`, and fail with the importer's message when it does not fit. -/
private def loadState (model : Model) (shapes : List Shape) (names : List String) :
    IO (TensorPack Float shapes) := do
  IO.ofExcept (PyTorch.state shapes names
    (← Json.readFile (stateDictionaryPath model)))

/-- Load the MLP state dict and evaluate `[0.5, 0.8]` on CPU with scalar `Float`. -/
public def mlpOutput : IO (Tensor Float [1]) := do
  let state ← loadState .mlp (nn.Sequential.stateShapes mlpModel) mlpNames
  let module ← nn.Module.instantiate mlpModel (α := Float) (initialState? := some state)
  let input : Tensor Float [mlpInputWidth] := [0.5, 0.8]
  module.forward (mode := some .eval) input

private def importMLP : IO Unit := do
  let y ← mlpOutput
  IO.println "== MLP import example =="
  IO.println s!"Loaded: {stateDictionaryPath .mlp}"
  IO.println "Output (executable `nn` module on CPU, Float):"
  IO.println (reprStr y)

/--
Load the CNN state dict and evaluate the row-major input `1..64` on CPU with scalar `Float`.
-/
public def cnnOutput : IO (Tensor Float [2]) := do
  let state ← loadState .cnn (nn.Sequential.stateShapes cnnModel) cnnNames
  -- Deterministic input matching the Python training script: values 1..64 laid out row-major.
  let input : Tensor Float [cnnInputChannels, cnnInputHeight, cnnInputWidth] :=
    Tensor.generate [cnnInputChannels, cnnInputHeight, cnnInputWidth] fun coordinates =>
      Float.ofNat (coordinates.getD 1 0 * cnnInputWidth + coordinates.getD 2 0 + 1)
  let module ← nn.Module.instantiate cnnModel (α := Float) (initialState? := some state)
  module.forward (mode := some .eval) input

private def importCNN : IO Unit := do
  let y ← cnnOutput
  IO.println "== CNN import example =="
  IO.println s!"Loaded: {stateDictionaryPath .cnn}"
  IO.println "Output (executable `nn` module on CPU, Float):"
  IO.println (reprStr y)

/--
Load the encoder state dict and evaluate the token `[[1.5, 1.5]]` on CPU with scalar `Float`.
-/
public def transformerOutput : IO (Tensor Float [1, 2]) := do
  let state ← loadState .transformer (nn.Sequential.stateShapes transformerModel) transformerNames
  let input : Tensor Float [transformerSeqLength, transformerModelWidth] := [[1.5, 1.5]]
  let module ← nn.Module.instantiate transformerModel (α := Float) (initialState? := some state)
  module.forward (mode := some .eval) input

private def importTransformer : IO Unit := do
  let y ← transformerOutput
  IO.println "== Transformer import example =="
  IO.println s!"Loaded: {stateDictionaryPath .transformer}"
  IO.println "Output (executable `nn` module on CPU, Float):"
  IO.println (reprStr y)

/-! ## Public entrypoint called from the examples runner -/

/-- Run the import example of the selected model. Without `--model` it runs the MLP. -/
public def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let (modelArg?, args) ← CLI.orThrow "PyTorch.Roundtrip" <|
    CLI.takeFlagValue? args "model"
  CLI.requireNoArgs "PyTorch.Roundtrip" args

  let modelStr := modelArg?.getD "mlp"
  let some model := Model.parse? modelStr
    | throw <| IO.userError s!"Unknown --model {modelStr}\n\n{usage}"

  match model with
  | .mlp => importMLP
  | .cnn => importCNN
  | .transformer => importTransformer

end TorchLean.Examples.PyTorchRoundtrip
