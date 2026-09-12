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
import NN.API
import NN.Examples.Support
public import NN.Tensor

/-!
# PyTorch Round-Trip Driver

This module assembles the state-dict round-trip examples.

It does **not** re-implement model math. Instead it wires together:

- PyTorch exporters (`MLP/Export`, `CNN/Export`, `Transformer/Export`)
- typed PyTorch JSON importers (`MLP/Import`, `CNN/Import`, `Transformer/Import`)
- native tensor operations for the MLP forward pass
- public executable `nn` modules for the CNN and Transformer forward passes

Run via the TorchLean example runner:

`lake exe torchlean pytorch_roundtrip --model mlp|cnn|transformer --action export|import`

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
    , "  lake exe torchlean pytorch_roundtrip --model mlp|cnn|transformer --action export|import"
    , ""
    , "Notes:"
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
private def cnnFirstStride : Nat := 1
private def cnnFirstPadding : Nat := 1
private def cnnSecondStride : Nat := 1
private def cnnSecondPadding : Nat := 1
private def cnnPoolKernelHeight : Nat := 2
private def cnnPoolKernelWidth : Nat := 2
private def cnnFirstPoolStride : Nat := 2
private def cnnSecondPoolStride : Nat := 2

private def cnnInputSpatial : Tensor Nat [2] := [cnnInputHeight, cnnInputWidth]

private def cnnFirstConvolution : nn.Convolution.Config 2 :=
  { outChannels := cnnOutputChannels
    kernelSize := [3, 3]
    stride := [1, 1]
    padding := [cnnFirstPadding, cnnFirstPadding] }

private def cnnFirstPooling : nn.Pooling.Config 2 :=
  { kernelSize := [2, 2]
    stride := [2, 2] }

private def cnnAfterFirstConvolution : Tensor Nat [2] :=
  cnnFirstConvolution.output cnnInputSpatial

private def cnnAfterFirstPooling : Tensor Nat [2] :=
  cnnFirstPooling.output cnnAfterFirstConvolution

private def cnnSecondConvolution : nn.Convolution.Config 2 :=
  { outChannels := cnnOutputChannels
    kernelSize := [3, 3]
    stride := [1, 1]
    padding := [cnnSecondPadding, cnnSecondPadding] }

private def cnnSecondPooling : nn.Pooling.Config 2 :=
  { kernelSize := [2, 2]
    stride := [2, 2] }

private def cnnAfterSecondConvolution : Tensor Nat [2] :=
  cnnSecondConvolution.output cnnAfterFirstPooling

private def cnnAfterSecondPooling : Tensor Nat [2] :=
  cnnSecondPooling.output cnnAfterSecondConvolution

private def cnnFlattenedWidth : Nat :=
  cnnOutputChannels * cnnAfterSecondPooling.prod

private def cnnModel := by
  letI : NeZero cnnInputChannels := ⟨by decide⟩
  letI : NeZero cnnOutputChannels := ⟨by decide⟩
  exact nn.build 0 <| nn.Sequential![
    nn.conv cnnInputSpatial cnnFirstConvolution (inputChannels := cnnInputChannels),
    nn.relu,
    nn.maxPool cnnAfterFirstConvolution cnnFirstPooling (channels := cnnOutputChannels),
    nn.conv cnnAfterFirstPooling cnnSecondConvolution (inputChannels := cnnOutputChannels),
    nn.relu,
    nn.maxPool cnnAfterSecondConvolution cnnSecondPooling (channels := cnnOutputChannels),
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
  let source := Export.PyTorch.MLP.completeSource
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
      Export.PyTorch.MLP.withParameters
        stateDictionary.inputWeight stateDictionary.inputBias
        stateDictionary.outputWeight stateDictionary.outputBias
        "TestMLP"
    writePythonFile directory "TestMLP_WithWeights" sourceWithWeights
  IO.println "Exported MLP PyTorch files under NN/Examples/Interop/PyTorch/MLP/."

private def exportCNN : IO Unit := do
  let directory := modelDirectory .cnn
  let firstConvolution : Export.PyTorch.CNN.ConvolutionConfig 2 :=
    { inputChannels := cnnInputChannels
      outputChannels := cnnOutputChannels
      kernel := [cnnKernelHeight, cnnKernelWidth]
      stride := [cnnFirstStride, cnnFirstStride]
      padding := [cnnFirstPadding, cnnFirstPadding] }
  let firstPooling : Export.PyTorch.CNN.PoolingConfig 2 :=
    { kernel := [cnnPoolKernelHeight, cnnPoolKernelWidth]
      stride := [cnnFirstPoolStride, cnnFirstPoolStride]
      padding := [0, 0] }
  let secondConvolution : Export.PyTorch.CNN.ConvolutionConfig 2 :=
    { inputChannels := cnnOutputChannels
      outputChannels := cnnOutputChannels
      kernel := [cnnKernelHeight, cnnKernelWidth]
      stride := [cnnSecondStride, cnnSecondStride]
      padding := [cnnSecondPadding, cnnSecondPadding] }
  let secondPooling : Export.PyTorch.CNN.PoolingConfig 2 :=
    { kernel := [cnnPoolKernelHeight, cnnPoolKernelWidth]
      stride := [cnnSecondPoolStride, cnnSecondPoolStride]
      padding := [0, 0] }
  let config : Export.PyTorch.CNN.Config 2 :=
    { className := "TestCNN"
      inputChannels := cnnInputChannels
      inputSpatial := [cnnInputHeight, cnnInputWidth]
      firstConvolution
      firstPooling
      secondConvolution
      secondPooling
      flattenedWidth := cnnFlattenedWidth
      outputWidth := cnnOutputChannels }
  let source ←
    match Export.PyTorch.CNN.classSource config with
    | .ok code => pure code
    | .error message => throw <| IO.userError message
  writePythonFile directory "TestCNN_PyTorch" source
  -- An existing state dict is optional, but an invalid one must fail visibly.
  if ← (stateDictionaryPath .cnn).pathExists then
    let json ← TorchLean.Json.readFile (stateDictionaryPath .cnn)
    let some stateDictionary :=
      Import.PyTorch.CNN.load
        cnnInputChannels cnnOutputChannels cnnKernelHeight cnnKernelWidth cnnFlattenedWidth json
      | throw <| IO.userError "CNN JSON present but failed to parse as a CNN state_dict"
    let sourceWithWeights ←
      match Export.PyTorch.CNN.withParameters config
        (Export.PyTorch.tensorToPyString stateDictionary.firstConvolutionWeight)
        (Export.PyTorch.tensorToPyString stateDictionary.firstConvolutionBias)
        (Export.PyTorch.tensorToPyString stateDictionary.secondConvolutionWeight)
        (Export.PyTorch.tensorToPyString stateDictionary.secondConvolutionBias)
        (Export.PyTorch.tensorToPyString stateDictionary.classifierWeight)
        (Export.PyTorch.tensorToPyString stateDictionary.classifierBias)
      with
      | .ok code => pure code
      | .error message => throw <| IO.userError message
    writePythonFile directory "TestCNN_WithWeights" sourceWithWeights
  IO.println "Exported CNN PyTorch files under NN/Examples/Interop/PyTorch/CNN/."

private def exportTransformer : IO Unit := do
  let directory := modelDirectory .transformer
  let source :=
    Export.PyTorch.Transformer.classSource
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
      Export.PyTorch.Transformer.withParameters
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

private def runExport (model : Model) : IO Unit := do
  match model with
  | .mlp => exportMLP
  | .cnn => exportCNN
  | .transformer => exportTransformer

/-! ## Import actions -/

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
  module.predict input

private def importCNN : IO Unit := do
  let y ← cnnOutput
  IO.println "== CNN import example =="
  IO.println s!"Loaded: {stateDictionaryPath .cnn}"
  IO.println "Output (executable `nn` module on CPU, Float):"
  IO.println (reprStr y)

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
  module.predict input

private def importTransformer : IO Unit := do
  let y ← transformerOutput
  IO.println "== Transformer import example =="
  IO.println s!"Loaded: {stateDictionaryPath .transformer}"
  IO.println "Output (executable `nn` module on CPU, Float):"
  IO.println (reprStr y)

private def runImport (model : Model) : IO Unit := do
  match model with
  | .mlp => importMLP
  | .cnn => importCNN
  | .transformer => importTransformer

/-! ## Public entrypoint called from the examples runner -/

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

  match action with
  | .export => runExport model
  | .import => runImport model

end NN.Examples.Interop.PyTorch.Roundtrip
