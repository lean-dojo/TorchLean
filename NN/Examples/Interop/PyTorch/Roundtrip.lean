/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Runtime.PyTorch.Export.MLP
import NN.Runtime.PyTorch.Export.CNN
import NN.Runtime.PyTorch.Export.Transformer
import NN.Runtime.PyTorch.Import.MLP
import NN.Runtime.PyTorch.Import.CNN
import NN.Runtime.PyTorch.Import.Transformer
public import NN.Tensor
import NN.API.Json
import NN.API.Module
import NN.API.Seeded

/-!
# PyTorch Round-Trip Driver

This module assembles the state-dict round-trip examples.

It does **not** re-implement model math. Instead it wires together:

- PyTorch exporters (`MLP/Export`, `CNN/Export`, `Transformer/Export`)
- typed PyTorch JSON importers (`MLP/Import`, `CNN/Import`, `Transformer/Import`)
- native tensor operations for the MLP forward pass
- public executable `nn` modules for the CNN and Transformer forward passes

Run via the TorchLean example runner:

`scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp --action import`

Choose `mlp`, `cnn`, or `transformer` for `--model`, and `export` or `import` for `--action`.

Design goals:
- keep paths/dimensions centralized (no duplicated constants across examples),
- keep the example output deterministic and readable,
- keep this example import-safe (no root-level `main` that collides with other executables).
-/

namespace NN.Examples.Interop.PyTorch.Roundtrip

open Lean
open TorchLean

/-! ## CLI model/action selection -/

/-- Which example model the round-trip driver should export or import. -/
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

/-- Which round-trip action to run for the selected example model. -/
inductive Action where
  | export
  | import
  deriving Repr, DecidableEq

/-- Parse the `--action` CLI flag accepted by the round-trip example. -/
def Action.parse? (s : String) : Option Action :=
  match s.toLower with
  | "export" => some .export
  | "import" => some .import
  | _ => none

private def usage : String :=
  String.intercalate "\n"
    [ "PyTorch round-trip example (TorchLean)"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean pytorch_roundtrip --model mlp --action import"
    , ""
    , "Notes:"
    , "  - `--model` accepts mlp, cnn, or transformer (default: mlp)."
    , "  - `--action` accepts export or import (default: export)."
    , ("  - `export` writes readable reference PyTorch modules under " ++
      "`NN/Examples/Interop/PyTorch/<Model>/`.")
    , ("  - `import` reads the JSON weights under " ++
      "`NN/Examples/Interop/PyTorch/<Model>/` and runs a checked Lean forward pass.")
    , "  - MLP uses tensor operations; CNN and Transformer use ordinary executable `nn` modules."
    ]

/-! ## Paths and fixed example dimensions -/

private def modelDirectory : Model → System.FilePath
  | .mlp => "NN/Examples/Interop/PyTorch/MLP"
  | .cnn => "NN/Examples/Interop/PyTorch/CNN"
  | .transformer => "NN/Examples/Interop/PyTorch/Transformer"

private def stateDictionaryPath : Model → System.FilePath
  | .mlp => "NN/Examples/Interop/PyTorch/MLP/mlp.json"
  | .cnn => "NN/Examples/Interop/PyTorch/CNN/cnn.json"
  | .transformer => "NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json"

-- MLP dimensions (matches `train_mlp.py` and `Import.PyTorch.MLP` example)
private def mlpInputWidth : Nat := 2
private def mlpHiddenWidth : Nat := 3
private def mlpOutputWidth : Nat := 1

-- CNN dimensions and hyperparameters.
private def cnnInputChannels : Nat := 1
private def cnnOutputChannels : Nat := 2
private def cnnInputHeight : Nat := 8
private def cnnInputWidth : Nat := 8
private def cnnKernelHeight : Nat := 3
private def cnnKernelWidth : Nat := 3
private def cnnPoolKernelHeight : Nat := 2
private def cnnPoolKernelWidth : Nat := 2

private def cnnInputSpatial : Tensor Nat [2] := [cnnInputHeight, cnnInputWidth]

private def cnnConvolution : nn.Convolution.Config 2 :=
  { outChannels := cnnOutputChannels
    kernelSize := [cnnKernelHeight, cnnKernelWidth]
    stride := [1, 1]
    padding := [1, 1] }

private def cnnPooling : nn.Pooling.Config 2 :=
  { kernelSize := [cnnPoolKernelHeight, cnnPoolKernelWidth]
    stride := [2, 2] }

private def cnnAfterFirstConvolution : Tensor Nat [2] :=
  cnnConvolution.outputSpatial cnnInputSpatial

private def cnnAfterFirstPooling : Tensor Nat [2] :=
  cnnPooling.outputSpatial cnnAfterFirstConvolution

private def cnnAfterSecondConvolution : Tensor Nat [2] :=
  cnnConvolution.outputSpatial cnnAfterFirstPooling

private def cnnAfterSecondPooling : Tensor Nat [2] :=
  cnnPooling.outputSpatial cnnAfterSecondConvolution

private def cnnFlattenedWidth : Nat :=
  cnnOutputChannels * cnnAfterSecondPooling.prod

private def cnnModel := by
  letI : NeZero cnnInputChannels := ⟨by decide⟩
  letI : NeZero cnnOutputChannels := ⟨by decide⟩
  exact nn.build 0 <| nn.Sequential![
    nn.conv cnnInputSpatial cnnConvolution (inputChannels := cnnInputChannels),
    nn.relu,
    nn.maxPool cnnAfterFirstConvolution cnnPooling (channels := cnnOutputChannels),
    nn.conv cnnAfterFirstPooling cnnConvolution (inputChannels := cnnOutputChannels),
    nn.relu,
    nn.maxPool cnnAfterSecondConvolution cnnPooling (channels := cnnOutputChannels),
    nn.flatten,
    nn.linear cnnFlattenedWidth cnnOutputChannels
  ]

-- Transformer dimensions (matches the PyTorch round-trip reference).
private def transformerSequenceLength : Nat := 1
private def transformerModelWidth : Nat := 2
private def transformerHeadCount : Nat := 1
private def transformerFeedForwardWidth : Nat := 2
private def transformerLayerCount : Nat := 1

private def transformerModel := by
  letI : NeZero transformerSequenceLength := ⟨by decide⟩
  letI : NeZero transformerModelWidth := ⟨by decide⟩
  exact nn.build 0 <| nn.transformerEncoderBlock
    (sequenceLength := transformerSequenceLength)
    (modelWidth := transformerModelWidth)
    { headCount := transformerHeadCount
      headWidth := transformerModelWidth / transformerHeadCount
      feedForwardWidth := transformerFeedForwardWidth
      activation := .relu }

/-! ## Small IO helpers -/

private def writePythonFile (directory : System.FilePath) (baseName : String)
    (content : String) : IO Unit := do
  IO.FS.createDirAll directory
  IO.FS.writeFile (directory / s!"{baseName}.py") content

/-! ## Export actions -/

private def exportMLP : IO Unit := do
  let directory := modelDirectory .mlp
  let source := Export.PyTorch.MLP.script
    (inputWidth := mlpInputWidth) (hiddenWidth := mlpHiddenWidth)
    (outputWidth := mlpOutputWidth) "TestMLP"
  writePythonFile directory "TestMLP_PyTorch" source
  -- An existing state dict is optional, but an invalid one must fail visibly.
  if ← (stateDictionaryPath .mlp).pathExists then
    let json ← TorchLean.Json.readFile (stateDictionaryPath .mlp)
    let some stateDictionary :=
      Import.PyTorch.MLP.load
        mlpInputWidth mlpHiddenWidth mlpOutputWidth json
      | throw <| IO.userError "MLP JSON present but failed to parse as an MLP state_dict"
    let sourceWithWeights :=
      Export.PyTorch.MLP.weights
        stateDictionary.inputWeight stateDictionary.inputBias
        stateDictionary.outputWeight stateDictionary.outputBias
        "TestMLP"
    writePythonFile directory "TestMLP_WithWeights" sourceWithWeights
  IO.println "Exported MLP PyTorch files under NN/Examples/Interop/PyTorch/MLP/."

private def exportCNN : IO Unit := do
  let directory := modelDirectory .cnn
  let firstConvolution : Export.PyTorch.CNN.Convolution 2 :=
    { inputChannels := cnnInputChannels
      outputChannels := cnnConvolution.outChannels
      kernel := cnnConvolution.kernelSize
      stride := cnnConvolution.stride
      padding := cnnConvolution.padding }
  let pooling : Export.PyTorch.CNN.Window 2 :=
    { kernel := cnnPooling.kernelSize
      stride := cnnPooling.stride
      padding := cnnPooling.padding }
  let config : Export.PyTorch.CNN.Config 2 :=
    { className := "TestCNN"
      inputChannels := cnnInputChannels
      inputSpatial := [cnnInputHeight, cnnInputWidth]
      firstConvolution
      firstPooling := pooling
      secondConvolution := { firstConvolution with inputChannels := cnnOutputChannels }
      secondPooling := pooling
      flattenedWidth := cnnFlattenedWidth
      outputWidth := cnnOutputChannels }
  let source ← IO.ofExcept (Export.PyTorch.CNN.source config)
  writePythonFile directory "TestCNN_PyTorch" source
  -- An existing state dict is optional, but an invalid one must fail visibly.
  if ← (stateDictionaryPath .cnn).pathExists then
    let json ← TorchLean.Json.readFile (stateDictionaryPath .cnn)
    let some stateDictionary :=
      Import.PyTorch.CNN.load
        cnnInputChannels cnnOutputChannels cnnKernelHeight cnnKernelWidth cnnFlattenedWidth json
      | throw <| IO.userError "CNN JSON present but failed to parse as a CNN state_dict"
    let sourceWithWeights ← IO.ofExcept <|
      Export.PyTorch.CNN.weights config
        (Export.PyTorch.tensorLiteral stateDictionary.firstConvolutionWeight)
        (Export.PyTorch.tensorLiteral stateDictionary.firstConvolutionBias)
        (Export.PyTorch.tensorLiteral stateDictionary.secondConvolutionWeight)
        (Export.PyTorch.tensorLiteral stateDictionary.secondConvolutionBias)
        (Export.PyTorch.tensorLiteral stateDictionary.classifierWeight)
        (Export.PyTorch.tensorLiteral stateDictionary.classifierBias)
    writePythonFile directory "TestCNN_WithWeights" sourceWithWeights
  IO.println "Exported CNN PyTorch files under NN/Examples/Interop/PyTorch/CNN/."

private def exportTransformer : IO Unit := do
  let directory := modelDirectory .transformer
  let source :=
    Export.PyTorch.Transformer.source
      transformerSequenceLength transformerModelWidth transformerHeadCount
      transformerFeedForwardWidth transformerLayerCount
      "TestTransformerEncoder"
  writePythonFile directory "TestTransformer_Encoder" source
  -- An existing state dict is optional, but an invalid one must fail visibly.
  if ← (stateDictionaryPath .transformer).pathExists then
    let json ← TorchLean.Json.readFile (stateDictionaryPath .transformer)
    let some stateDictionary :=
      Import.PyTorch.Transformer.load
        transformerModelWidth transformerFeedForwardWidth json
      | throw <| IO.userError
          "Transformer JSON present but failed to parse as a Transformer state_dict"
    let sourceWithWeights :=
      Export.PyTorch.Transformer.weights
        transformerSequenceLength transformerModelWidth transformerHeadCount
        transformerFeedForwardWidth
        stateDictionary.queryWeight stateDictionary.keyWeight stateDictionary.valueWeight
        stateDictionary.outputWeight
        stateDictionary.feedForwardInputWeight stateDictionary.feedForwardOutputWeight
        stateDictionary.feedForwardInputBias stateDictionary.feedForwardOutputBias
        stateDictionary.norm1Scale stateDictionary.norm1Bias
        stateDictionary.norm2Scale stateDictionary.norm2Bias
        "TestTransformerEncoder"
    writePythonFile directory "TestTransformer_Encoder_WithWeights" sourceWithWeights
  IO.println ("Exported Transformer encoder PyTorch files under "
    ++ "NN/Examples/Interop/PyTorch/Transformer/.")

/-! ## Import actions -/

/-- Load the MLP JSON fixture and evaluate `[0.5, 0.8]` with native `Float` tensor operations. -/
public def mlpOutput : IO (Tensor Float [1]) := do
  let json ← TorchLean.Json.readFile (stateDictionaryPath .mlp)
  let some stateDictionary :=
    Import.PyTorch.MLP.load
      mlpInputWidth mlpHiddenWidth mlpOutputWidth json
    | throw <| IO.userError "Failed to load MLP state dict"

  let input : Tensor Float [mlpInputWidth] := [0.5, 0.8]
  pure <| Import.PyTorch.MLP.forward stateDictionary input

private def importMLP : IO Unit := do
  let y ← mlpOutput
  IO.println "== MLP import example =="
  IO.println s!"Loaded: {stateDictionaryPath .mlp}"
  IO.println "Output (native tensor operations, Float):"
  IO.println (reprStr y)

/--
Load the CNN JSON fixture and evaluate the row-major input `1..64` on CPU with scalar `Float`.
-/
public def cnnOutput : IO (Tensor Float [2]) := do
  let json ← TorchLean.Json.readFile (stateDictionaryPath .cnn)
  let some stateDictionary :=
    Import.PyTorch.CNN.load
      cnnInputChannels cnnOutputChannels cnnKernelHeight cnnKernelWidth cnnFlattenedWidth json
    | throw <| IO.userError "Failed to load CNN state dict"

  -- Deterministic input matching the Python training script: values 1..64 laid out row-major.
  let input : Tensor Float [cnnInputChannels, cnnInputHeight, cnnInputWidth] :=
    Tensor.generate [cnnInputChannels, cnnInputHeight, cnnInputWidth] fun coordinates =>
      Float.ofNat (coordinates.getD 1 0 * cnnInputWidth + coordinates.getD 2 0 + 1)

  let module ← nn.Module.instantiate cnnModel (α := Float)
  module.setState <|
    nn.State.empty
      |>.push stateDictionary.firstConvolutionWeight
      |>.push stateDictionary.firstConvolutionBias
      |>.push stateDictionary.secondConvolutionWeight
      |>.push stateDictionary.secondConvolutionBias
      |>.push stateDictionary.classifierWeight
      |>.push stateDictionary.classifierBias
  module.forward (mode := some .eval) input

private def importCNN : IO Unit := do
  let y ← cnnOutput
  IO.println "== CNN import example =="
  IO.println s!"Loaded: {stateDictionaryPath .cnn}"
  IO.println "Output (executable `nn` module on CPU, Float):"
  IO.println (reprStr y)

/--
Load the encoder JSON fixture and evaluate the token `[[1.5, 1.5]]` on CPU with scalar `Float`.
-/
public def transformerOutput : IO (Tensor Float [1, 2]) := do
  let json ← TorchLean.Json.readFile (stateDictionaryPath .transformer)
  let some stateDictionary :=
    Import.PyTorch.Transformer.load
      transformerModelWidth transformerFeedForwardWidth json
    | throw <| IO.userError "Failed to load Transformer encoder state dict"

  let input : Tensor Float [transformerSequenceLength, transformerModelWidth] := [[1.5, 1.5]]
  let module ← nn.Module.instantiate transformerModel (α := Float)
  module.setState <|
    nn.State.empty
      |>.push stateDictionary.queryWeight
      |>.push stateDictionary.keyWeight
      |>.push stateDictionary.valueWeight
      |>.push stateDictionary.outputWeight
      |>.push stateDictionary.norm1Scale
      |>.push stateDictionary.norm1Bias
      |>.push stateDictionary.feedForwardInputWeight
      |>.push stateDictionary.feedForwardInputBias
      |>.push stateDictionary.feedForwardOutputWeight
      |>.push stateDictionary.feedForwardOutputBias
      |>.push stateDictionary.norm2Scale
      |>.push stateDictionary.norm2Bias
  module.forward (mode := some .eval) input

private def importTransformer : IO Unit := do
  let y ← transformerOutput
  IO.println "== Transformer import example =="
  IO.println s!"Loaded: {stateDictionaryPath .transformer}"
  IO.println "Output (executable `nn` module on CPU, Float):"
  IO.println (reprStr y)

/-! ## Public entrypoint called from the examples runner -/

/-- Run the selected import or export action; omitted flags select MLP export. -/
public def main (args : List String) : IO Unit := do
  let args := TorchLean.CLI.dropDashDash args
  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return
  let (modelArg?, args) ← TorchLean.CLI.orThrow "PyTorch.Roundtrip" <|
    TorchLean.CLI.takeFlagValue? args "model"
  let (actionArg?, args) ← TorchLean.CLI.orThrow "PyTorch.Roundtrip" <|
    TorchLean.CLI.takeFlagValue? args "action"
  TorchLean.CLI.requireNoArgs "PyTorch.Roundtrip" args

  let modelStr := modelArg?.getD "mlp"
  let actionStr := actionArg?.getD "export"
  let some model := Model.parse? modelStr
    | throw <| IO.userError s!"Unknown --model {modelStr}\n\n{usage}"
  let some action := Action.parse? actionStr
    | throw <| IO.userError s!"Unknown --action {actionStr}\n\n{usage}"

  match action, model with
  | .export, .mlp => exportMLP
  | .export, .cnn => exportCNN
  | .export, .transformer => exportTransformer
  | .import, .mlp => importMLP
  | .import, .cnn => importCNN
  | .import, .transformer => importTransformer

end NN.Examples.Interop.PyTorch.Roundtrip
