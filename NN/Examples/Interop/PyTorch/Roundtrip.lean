/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.API.Common
import NN.Examples.Interop.PyTorch.Export
import NN.Examples.Interop.PyTorch.Import

/-!
# PyTorch Round-Trip Driver

This module is the single source of truth for the state-dict round-trip demos.

It intentionally does **not** re-implement model math. Instead it wires together the existing:

- PyTorch exporters beside the demo fixtures (`MLP/Export`, `CNN/Export`, `Transformer/Export`)
- PyTorch JSON importers beside the demo fixtures (`MLP/Import`, `CNN/Import`, `Transformer/Import`)
- Spec models (`NN/Spec/Models/*`) for running a small forward pass in Lean

Run via the TorchLean example runner:

`lake exe torchlean pytorch_roundtrip --model mlp|cnn|transformer --action export|import`

Design goals:
- keep paths/dimensions centralized (no duplicated constants across demos),
- keep the example output deterministic and readable,
- keep this demo import-safe (no root-level `main` that collides with other executables).
-/

namespace NN.Examples.Interop.PyTorch.Roundtrip

open Lean

/-! ## CLI model/action selection -/

/-- Which demo model the round-trip driver should export or import. -/
inductive Model where
  | mlp
  | cnn
  | transformer
  deriving Repr, DecidableEq

/-- Parse the `--model` CLI flag accepted by the round-trip demo. -/
def Model.parse? (s : String) : Option Model :=
  match s.toLower with
  | "mlp" => some .mlp
  | "cnn" => some .cnn
  | "transformer" => some .transformer
  | _ => none

/-- Which round-trip action to run for the selected demo model. -/
inductive Action where
  | export
  | import
  deriving Repr, DecidableEq

/-- Parse the `--action` CLI flag accepted by the round-trip demo. -/
def Action.parse? (s : String) : Option Action :=
  match s.toLower with
  | "export" => some .export
  | "import" => some .import
  | _ => none

private def usage : String :=
  String.intercalate "\n"
    [ "PyTorch round-trip demo (TorchLean)"
    , ""
    , "Usage:"
    , "  lake exe torchlean pytorch_roundtrip --model mlp|cnn|transformer --action export|import"
    , ""
    , "Notes:"
    , "  - `export` writes readable reference PyTorch modules under `NN/Examples/Interop/PyTorch/<Model>/`."
    , ("  - `import` reads the JSON weights under " ++
      "`NN/Examples/Interop/PyTorch/<Model>/` and runs a Lean forward pass.")
    ]

/-! ## Paths and fixed demo dimensions -/

private def dirOf : Model → System.FilePath
  | .mlp => "NN/Examples/Interop/PyTorch/MLP"
  | .cnn => "NN/Examples/Interop/PyTorch/CNN"
  | .transformer => "NN/Examples/Interop/PyTorch/Transformer"

private def jsonOf : Model → System.FilePath
  | .mlp => "NN/Examples/Interop/PyTorch/MLP/mlp.json"
  | .cnn => "NN/Examples/Interop/PyTorch/CNN/cnn.json"
  | .transformer => "NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json"

-- MLP dims (matches `train_mlp.py` and `Import.MLPPyTorch` demo)
private def mlpInDim : Nat := 2
private def mlpHidDim : Nat := 3
private def mlpOutDim : Nat := 1

-- CNN dims/hparams (matches `train_cnn.py` and `Import.CNNPyTorch` demo)
private def cnnInC : Nat := 1
private def cnnOutC : Nat := 2
private def cnnInH : Nat := 8
private def cnnInW : Nat := 8
private def cnnKH : Nat := 3
private def cnnKW : Nat := 3
private def cnnStride1 : Nat := 1
private def cnnPadding1 : Nat := 1
private def cnnStride2 : Nat := 1
private def cnnPadding2 : Nat := 1
private def cnnPoolKH : Nat := 2
private def cnnPoolKW : Nat := 2
private def cnnPoolStride1 : Nat := 2
private def cnnPoolStride2 : Nat := 2

private def cnnFlatSize : Nat :=
  _root_.Models.CNN.featSize cnnOutC cnnInH cnnInW cnnKH cnnKW cnnStride1 cnnPadding1 cnnStride2
    cnnPadding2
    cnnPoolKH cnnPoolKW cnnPoolStride1 cnnPoolStride2

-- Transformer dims (matches `train_transformer.py` and `Import.TransformerPyTorch` demo)
private def trSeqLen : Nat := 1
private def trEmbedDim : Nat := 2
private def trHeadCount : Nat := 1
private def trHiddenDim : Nat := 2
private def trNumLayers : Nat := 1

/-! ## Small IO helpers -/

private def writePy (dir : System.FilePath) (base : String) (content : String) : IO Unit := do
  IO.FS.createDirAll dir
  IO.FS.writeFile (dir / s!"{base}.py") content

private def readJsonFile (path : System.FilePath) : IO Json := do
  let jsonStr ← IO.FS.readFile path
  match Json.parse jsonStr with
  | .ok j => pure j
  | .error msg => throw <| IO.userError s!"bad json in {path}: {msg}"

/-! ## Export actions -/

private def exportMLP : IO Unit := do
  let dir := dirOf .mlp
  let stub := Export.MLPPyTorch.generateCompleteMLPExport (inDim := mlpInDim) (hidDim := mlpHidDim)
    (outDim := mlpOutDim) "TestMLP"
  writePy dir "TestMLP_PyTorch" stub
  -- If we have a JSON state_dict handy, also emit a runnable "with weights" helper.
  try
    let j ← readJsonFile (jsonOf .mlp)
    let some sd := Import.MLPPyTorch.loadMlpStateDict mlpInDim mlpHidDim mlpOutDim j
      | throw <| IO.userError "MLP JSON present but failed to parse as an MLP state_dict"
    let codeW := Export.MLPPyTorch.generateMLPWithWeights sd.w1 sd.b1 sd.w2 sd.b2 "TestMLP"
    writePy dir "TestMLP_WithWeights" codeW
  catch _ =>
    pure ()
  IO.println "Exported MLP PyTorch files under NN/Examples/Interop/PyTorch/MLP/."

private def exportCNN : IO Unit := do
  let dir := dirOf .cnn
  let stub :=
    Export.CNNPyTorch.generateCNNPyTorchClass
      cnnInC cnnOutC cnnInH cnnInW cnnKH cnnKW
      cnnStride1 cnnPadding1 cnnStride2 cnnPadding2
      cnnPoolKH cnnPoolKW cnnPoolStride1 cnnPoolStride2 cnnFlatSize
      "TestCNN"
  writePy dir "TestCNN_PyTorch" stub
  -- If we have a JSON state_dict handy, also emit a runnable "with weights" helper.
  try
    let j ← readJsonFile (jsonOf .cnn)
    let some sd := Import.CNNPyTorch.loadCnnStateDict cnnInC cnnOutC cnnKH cnnKW cnnFlatSize j
      | throw <| IO.userError "CNN JSON present but failed to parse as a CNN state_dict"
    let codeW :=
      Export.CNNPyTorch.generateCNNWithWeights
        (Export.PyTorch.tensor4DToPy sd.convW1) (Export.PyTorch.tensor1DToPy sd.convB1)
        (Export.PyTorch.tensor4DToPy sd.convW2) (Export.PyTorch.tensor1DToPy sd.convB2)
        (Export.PyTorch.tensor2DToPy sd.linearW) (Export.PyTorch.tensor1DToPy sd.linearB)
        cnnInC cnnOutC cnnInH cnnInW cnnKH cnnKW
        cnnStride1 cnnPadding1 cnnStride2 cnnPadding2
        cnnPoolKH cnnPoolKW cnnPoolStride1 cnnPoolStride2 cnnFlatSize
        "TestCNN"
    writePy dir "TestCNN_WithWeights" codeW
  catch _ =>
    pure ()
  IO.println "Exported CNN PyTorch files under NN/Examples/Interop/PyTorch/CNN/."

private def exportTransformer : IO Unit := do
  let dir := dirOf .transformer
  let stub :=
    Export.TransformerPyTorch.generateTransformerEncoderPyTorchClass
      trSeqLen trEmbedDim trHeadCount trHiddenDim trNumLayers
      "TestTransformerEncoder"
  writePy dir "TestTransformer_Encoder" stub
  -- If we have a JSON state_dict handy, also emit a runnable "with weights" helper.
  try
    let j ← readJsonFile (jsonOf .transformer)
    let some sd := Import.TransformerPyTorch.loadTransformerEncoderStateDict trEmbedDim trHeadCount
      trHiddenDim j
      | throw <| IO.userError "Transformer JSON present but failed to parse as a Transformer state_dict"
    let codeW :=
      Export.TransformerPyTorch.generateTransformerEncoderWithWeights
        trSeqLen trEmbedDim trHeadCount trHiddenDim
        sd.Wq sd.Wk sd.Wv sd.Wo
        sd.W1 sd.W2 sd.b1 sd.b2
        sd.norm1_gamma sd.norm1_beta sd.norm2_gamma sd.norm2_beta
        "TestTransformerEncoder"
    writePy dir "TestTransformer_Encoder_WithWeights" codeW
  catch _ =>
    pure ()
  IO.println "Exported Transformer encoder PyTorch files under NN/Examples/Interop/PyTorch/Transformer/."

private def runExport (m : Model) : IO Unit := do
  match m with
  | .mlp => exportMLP
  | .cnn => exportCNN
  | .transformer => exportTransformer

/-! ## Import actions (Lean forward pass) -/

private def importMLP : IO Unit := do
  let j ← readJsonFile (jsonOf .mlp)
  let some sd := Import.MLPPyTorch.loadMlpStateDict mlpInDim mlpHidDim mlpOutDim j
    | throw <| IO.userError "Failed to load MLP state dict"

  let x : _root_.Spec.Tensor Float (NN.Tensor.Shape.Vec mlpInDim) := tensor! [0.5, 0.8]
  let y := Import.MLPPyTorch.forward sd x

  IO.println "== MLP import demo =="
  IO.println s!"Loaded: {jsonOf .mlp}"
  IO.println "Output (Lean, Float):"
  NN.Tensor.print y

private def importCNN : IO Unit := do
  let j ← readJsonFile (jsonOf .cnn)
  let some sd := Import.CNNPyTorch.loadCnnStateDict cnnInC cnnOutC cnnKH cnnKW cnnFlatSize j
    | throw <| IO.userError "Failed to load CNN state dict"

  let hInC : cnnInC ≠ 0 := by decide
  let hOutC : cnnOutC ≠ 0 := by decide
  let hKH : cnnKH ≠ 0 := by decide
  let hKW : cnnKW ≠ 0 := by decide
  let hPoolH : cnnPoolKH ≠ 0 := by decide
  let hPoolW : cnnPoolKW ≠ 0 := by decide
  let hPoolStride1 : cnnPoolStride1 ≠ 0 := by decide
  let hPoolStride2 : cnnPoolStride2 ≠ 0 := by decide

  let conv1 : _root_.Spec.Conv2DSpec cnnInC cnnOutC cnnKH cnnKW cnnStride1 cnnPadding1 Float hInC
    hKH hKW :=
    { kernel := sd.convW1, bias := sd.convB1 }
  let conv2 : _root_.Spec.Conv2DSpec cnnOutC cnnOutC cnnKH cnnKW cnnStride2 cnnPadding2 Float hOutC
    hKH hKW :=
    { kernel := sd.convW2, bias := sd.convB2 }
  let pool1 : _root_.Spec.MaxPool2DSpec cnnPoolKH cnnPoolKW cnnPoolStride1 hPoolH hPoolW hPoolStride1 :=
    { kernelHeight := cnnPoolKH, kernelWidth := cnnPoolKW, stride := cnnPoolStride1 }
  let pool2 : _root_.Spec.MaxPool2DSpec cnnPoolKH cnnPoolKW cnnPoolStride2 hPoolH hPoolW hPoolStride2 :=
    { kernelHeight := cnnPoolKH, kernelWidth := cnnPoolKW, stride := cnnPoolStride2 }
  let linear : _root_.Spec.LinearSpec Float cnnFlatSize cnnOutC := { weights := sd.linearW, bias :=
    sd.linearB }

  let net :=
    _root_.Models.cnnWithReluSpec (α := Float)
      (inH := cnnInH) (inW := cnnInW)
      conv1 conv2 pool1 pool2 linear

  -- Deterministic input matching the Python training script: values 1..64 laid out row-major.
  let x : _root_.Spec.Tensor Float (NN.Tensor.Shape.Image cnnInC cnnInH cnnInW) :=
    _root_.Spec.Tensor.dim (fun _ =>
      _root_.Spec.Tensor.dim (fun i =>
        _root_.Spec.Tensor.dim (fun j =>
          _root_.Spec.Tensor.scalar (Float.ofNat (i.val * cnnInW + j.val + 1)))))

  let y := ModSpec.SpecChain.forward (α := Float) net x

  IO.println "== CNN import demo =="
  IO.println s!"Loaded: {jsonOf .cnn}"
  IO.println "Output (Lean, Float):"
  NN.Tensor.print y

private def importTransformer : IO Unit := do
  let j ← readJsonFile (jsonOf .transformer)
  let some sd := Import.TransformerPyTorch.loadTransformerEncoderStateDict trEmbedDim trHeadCount
    trHiddenDim j
    | throw <| IO.userError "Failed to load Transformer encoder state dict"

  let layer : _root_.Spec.TransformerEncoderLayer trHeadCount trEmbedDim trHiddenDim Float :=
    { mha := { Wq := sd.Wq, Wk := sd.Wk, Wv := sd.Wv, Wo := sd.Wo }
      ffn := { W1 := sd.W1, W2 := sd.W2, b1 := sd.b1, b2 := sd.b2 }
      norm1_gamma := sd.norm1_gamma
      norm1_beta := sd.norm1_beta
      norm2_gamma := sd.norm2_gamma
      norm2_beta := sd.norm2_beta }
  let encoder : _root_.Spec.TransformerEncoder trNumLayers trHeadCount trEmbedDim trHiddenDim Float
    :=
    { layers := [layer] }

  let x : _root_.Spec.Tensor Float (NN.Tensor.Shape.Mat trSeqLen trEmbedDim) := tensor! [[1.5, 1.5]]
  let y := _root_.Spec.TransformerEncoder.forward (seqLen := trSeqLen) (embedDim := trEmbedDim)
    encoder x (by decide) (by decide)

  IO.println "== Transformer import demo =="
  IO.println s!"Loaded: {jsonOf .transformer}"
  IO.println "Output (Lean, Float):"
  NN.Tensor.print y

private def runImport (m : Model) : IO Unit := do
  match m with
  | .mlp => importMLP
  | .cnn => importCNN
  | .transformer => importTransformer

/-! ## Public entrypoint called from the examples zoo runner -/

public def main (args : List String) : IO Unit := do
  let args := NN.API.CLI.dropDashDash args
  let (modelArg?, args) ← NN.API.Common.orThrow "PyTorch.Roundtrip" <|
    NN.API.CLI.takeFlagValueOnce args "model"
  let (actionArg?, args) ← NN.API.Common.orThrow "PyTorch.Roundtrip" <|
    NN.API.CLI.takeFlagValueOnce args "action"
  NN.API.Common.orThrow "PyTorch.Roundtrip" <| NN.API.CLI.requireNoArgs args

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
