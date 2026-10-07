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
  `scripts/lake.sh exe torchlean torch_ir_pytorch --arch linear > exported_model.py`
  `scripts/lake.sh exe torchlean torch_ir_pytorch --arch mlp > exported_model.py`
  `scripts/lake.sh exe torchlean torch_ir_pytorch --arch sum > exported_model.py`
  `scripts/lake.sh exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py`
  `scripts/lake.sh exe torchlean torch_ir_pytorch --arch mha > exported_model.py`
  `scripts/lake.sh exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py`
  `scripts/lake.sh exe torchlean torch_ir_pytorch --arch transformer > exported_model.py`
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

def linear : nn.Builder (nn.Sequential [2] [1]) :=
  nn.linear 2 1

/-- A `2 -> 3 -> 1` MLP with ReLU, the smallest architecture with a nonlinearity. -/
def mlp : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 3,
    nn.relu,
    nn.linear 3 1
  ]

/-- A bare reduction, included because it exports to `torch.sum` rather than to a module. -/
def sum : nn.Builder (nn.Sequential [4] []) :=
  nn.sum (shape := [4])

/-- A `3 -> 2 -> 3` autoencoder with a tanh bottleneck. -/
def autoencoder : nn.Builder (nn.Sequential [3] [3]) :=
  nn.Sequential![
    nn.linear 3 2,
    nn.tanh,
    nn.linear 2 3
  ]

/-- Two-head self-attention over a length-four sequence of width eight, with an optional mask.
For causal attention, pass `some (Spec.causalMask 4)`; position `i` then attends only to `j ≤ i`. -/
def attention (mask : Option (Tensor Bool [4, 4]) := none) :
    nn.Builder (nn.Sequential [1, 4, 8] [1, 4, 8]) :=
  nn.attention { headCount := 2, headWidth := 4 }
    (mask := mask) (batchShape := [1]) (sequenceLength := 4) (modelWidth := 8)

/--
A full encoder block: attention, residual, LayerNorm, feed-forward, residual, LayerNorm.

Deliberately tiny (one head of width two) so the emitted Python stays readable.
-/
def transformer :
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
    , "  scripts/lake.sh exe torchlean torch_ir_pytorch --arch linear > exported_model.py"
    , "  scripts/lake.sh exe torchlean torch_ir_pytorch --arch mlp > exported_model.py"
    , "  scripts/lake.sh exe torchlean torch_ir_pytorch --arch mlp --seed 123 > exported_model.py"
    , "  scripts/lake.sh exe torchlean torch_ir_pytorch --arch sum > exported_model.py"
    , "  scripts/lake.sh exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py"
    , "  scripts/lake.sh exe torchlean torch_ir_pytorch --arch mha > exported_model.py"
    , "  scripts/lake.sh exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py"
    , "  scripts/lake.sh exe torchlean torch_ir_pytorch --arch transformer > exported_model.py"
    , ""
    , "Then: python3 exported_model.py"
    ]

/-! ## Export driver -/

/-- Lower a sequential model and its initial state, then write generated Python to stdout. -/
def emit {σ τ : Shape} (className : String) (model : nn.Sequential σ τ) : IO Unit := do
  let lowered ← IO.ofExcept <| Verification.lowerForwardToIR model (nn.initialState model)
  let code ← IO.ofExcept <| Export.IRPyTorch.emit
    (g := lowered.graph) (ps := lowered.ps) (inputId := lowered.inputId)
    (outputId := lowered.outputId) (options := { className := className })

  IO.println code

/--
Entry point. Writes Python to stdout for redirection to a file and execution under PyTorch.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
  else
    let (seed, args) ← CLI.orThrow exeName <| CLI.takeSeed args (default := 0)
    let (arch, rest) ← CLI.orThrow exeName <|
      CLI.takeFlagValue args "arch" (default := "mlp")
    CLI.requireNoArgs exeName rest
    if arch == "linear" then
      emit (className := "TorchLeanLinear") (nn.build seed linear)
    else if arch == "mlp" then
      emit (className := "TorchLeanMLP") (nn.build seed mlp)
    else if arch == "sum" then
      emit (className := "TorchLeanSumReduce") (nn.build seed sum)
    else if arch == "autoencoder" then
      emit (className := "TorchLeanAutoencoder") (nn.build seed autoencoder)
    else if arch == "mha" then
      emit (className := "TorchLeanMHA") (nn.build seed attention)
    else if arch == "mha-mask" then
      emit (className := "TorchLeanMHAMasked")
        (nn.build seed (attention (mask := some (Spec.causalMask 4))))
    else if arch == "transformer" then
      emit (className := "TorchLeanTransformerBlock") (nn.build seed transformer)
    else
      throw <| IO.userError
        (s!"unknown --arch {arch} (supported: linear | mlp | sum | autoencoder | " ++
          s!"mha | mha-mask | transformer)")

end NN.Examples.DeepDives.TorchIRPyTorch
