/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Verification.Lowering
public import NN.Runtime.PyTorch.Export.IRPyTorch

/-!
# TorchLean IR to PyTorch

Tutorial: TorchLean → IR (`NN.IR.Graph`) → emitted PyTorch code.

Run:
  `lake exe torchlean torch_ir_pytorch --arch linear > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch sum > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mha > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py`
Then:
  `python3 exported_model.py`

The command emits the selected architecture and initialized parameters; it does not train them.
Python execution requires PyTorch. Exporting or running the file does not by itself prove
TorchLean/PyTorch numerical parity.
-/

@[expose] public section


namespace NN.Examples.DeepDives.TorchIRPyTorch

open TorchLean

/-- Command name used in diagnostics and by the top-level example runner. -/
def exeName : String := "torch_ir_pytorch"

/-! ## Architectures -/

def archLinear : nn.Builder (nn.Sequential [2] [1]) :=
  nn.linear 2 1

/-- A `2 -> 3 -> 1` MLP with ReLU, the smallest architecture with a nonlinearity. -/
def archMLP : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 3,
    nn.relu,
    nn.linear 3 1
  ]

/-- A bare reduction, included because it exports to `torch.sum` rather than to a module. -/
def archSumReduce : nn.Builder (nn.Sequential [4] []) :=
  nn.sum (shape := [4])

/-- A `3 -> 2 -> 3` autoencoder with a tanh bottleneck. -/
def archAutoencoder : nn.Builder (nn.Sequential [3] [3]) :=
  nn.Sequential![
    nn.linear 3 2,
    nn.tanh,
    nn.linear 2 3
  ]

/-- Two-head self-attention over a length-four sequence of width eight. -/
def archMHA :
    nn.Builder (nn.Sequential [1, 4, 8] [1, 4, 8]) :=
  nn.multiHeadAttention { headCount := 2, headWidth := 4 }
    (batchShape := [1]) (sequenceLength := 4) (modelWidth := 8)

/-- A causal mask: position `i` may attend only to positions `j ≤ i`. -/
def archMHAMask : Tensor Bool [4, 4] :=
  Spec.causalMask 4

/--
The same attention block with the causal mask applied, so the export can be compared against
`torch.nn.MultiheadAttention` with `attn_mask`.
-/
def archMHAMasked :
    nn.Builder (nn.Sequential [1, 4, 8] [1, 4, 8]) :=
  nn.multiHeadAttention { headCount := 2, headWidth := 4 }
    (mask := some archMHAMask) (batchShape := [1])
    (sequenceLength := 4) (modelWidth := 8)

/--
A full encoder block: attention, residual, LayerNorm, feed-forward, residual, LayerNorm.

Deliberately tiny (one head of width two) so the emitted Python stays readable.
-/
def archTransformer :
    nn.Builder (nn.Sequential [1, 2, 2] [1, 2, 2]) :=
  nn.transformerEncoderBlock
    { headCount := 1
    , headWidth := 2
    , feedForwardWidth := 2 }
    (batchShape := [1]) (sequenceLength := 2) (modelWidth := 2)

/-! ## CLI parsing -/

def usage : String :=
  String.intercalate "\n"
    [ "TorchLean → IR → PyTorch exporter"
    , ""
    , "Usage:"
    , "  lake exe torchlean torch_ir_pytorch --arch linear > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp --seed 123 > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch sum > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mha > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py"
    , ""
    , "Then: python3 exported_model.py"
    ]

/-! ## Export driver -/

/-- Lower a sequential model and its initial state, then write generated Python to stdout. -/
def emitSeq {σ τ : Shape} (className : String) (model : nn.Sequential σ τ) : IO Unit := do
  let lowered ←
    match Verification.lowerForwardToIR model (nn.initialState model) with
    | .error e => throw <| IO.userError e
    | .ok c => pure c

  let code ←
    match Export.IRPyTorch.emit
        (g := lowered.graph) (ps := lowered.ps) (inputId := lowered.inputId) (outputId :=
          lowered.outputId)
        (options := { className := className }) with
    | .error e => throw <| IO.userError e
    | .ok s => pure s

  IO.println code

/--
Entry point. Writes Python to stdout for redirection to a file and execution under PyTorch.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  let help := args.contains "--help" || args.contains "-h"
  if help then
    IO.println usage
  else
    let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args (default := 0)
    let (arch, rest) ← CLI.orThrow exeName <|
      CLI.takeFlagValue args "arch" (default := "mlp")
    CLI.requireNoArgs exeName rest
    if arch == "linear" then
      emitSeq (className := "TorchLeanLinear") (nn.build seed archLinear)
    else if arch == "mlp" then
      emitSeq (className := "TorchLeanMLP") (nn.build seed archMLP)
    else if arch == "sum" then
      emitSeq (className := "TorchLeanSumReduce") (nn.build seed archSumReduce)
    else if arch == "autoencoder" then
      emitSeq (className := "TorchLeanAutoencoder") (nn.build seed archAutoencoder)
    else if arch == "mha" then
      emitSeq (className := "TorchLeanMHA") (nn.build seed archMHA)
    else if arch == "mha-mask" then
      emitSeq (className := "TorchLeanMHAMasked") (nn.build seed archMHAMasked)
    else if arch == "transformer" then
      emitSeq (className := "TorchLeanTransformerBlock") (nn.build seed archTransformer)
    else
      throw <| IO.userError
        (s!"unknown --arch {arch} (supported: linear | mlp | sum | autoencoder | " ++
          s!"mha | mha-mask | transformer)")

end NN.Examples.DeepDives.TorchIRPyTorch
