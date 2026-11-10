/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.Verify.Lowering
public import TorchLean.NN.PyTorch.Export.IR
public import TorchLean.NN.Data.Source
public import TorchLean.NN.Builder
public import TorchLean.NN.Training.Reporting

/-!
# TorchLean IR to PyTorch

Tutorial: TorchLean → operator graph (`LoweredModel.toImported`) → emitted PyTorch code.

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

`--arch` defaults to `mlp`, and `--seed N` (default `0`) seeds the initialization. The command
writes a standalone Python script with the selected architecture and its initialized, untrained
parameters as an `nn.Module` class. Running the file needs PyTorch: its `__main__` block computes
the MSE loss of the model on a random input and target, takes one Adam step when the model has
parameters, and prints that loss. Neither the export nor the script checks that PyTorch reproduces
TorchLean's numbers.
-/

@[expose] public section


namespace TorchLean.Examples.TorchIRPyTorch

open Activation.Kind

/-- Command name used in diagnostics and by the top-level example runner. -/
def exeName : String := "torch_ir_pytorch"

/-! ## Architectures -/

/-- A single `2 -> 1` linear layer. -/
def linear : nn.Builder (nn.Sequential [2] [1]) :=
  nn.linear 2 1

/-- A `2 -> 3 -> 1` MLP with ReLU, the smallest architecture with a nonlinearity. -/
def mlp : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 3,
    nn.op .relu,
    nn.linear 3 1
  ]

/-- Sum of the four input entries to a scalar: an architecture without parameters, whose exported
module calls `torch.sum`. -/
def sum : nn.Builder (nn.Sequential [4] []) :=
  nn.sum (shape := [4])

/-- A `3 -> 2 -> 3` autoencoder with a tanh bottleneck. -/
def autoencoder : nn.Builder (nn.Sequential [3] [3]) :=
  nn.Sequential![
    nn.linear 3 2,
    nn.op .tanh,
    nn.linear 2 3
  ]

/-- Two-head self-attention over a length-four sequence of width eight, with an optional mask. With
`some (Attention.causalMask 4)` the attention is causal: position `i` attends only to `j ≤ i`. -/
def attention (mask : Option (Tensor Bool [4, 4]) := none) :
    nn.Builder (nn.Sequential [1, 4, 8] [1, 4, 8]) :=
  nn.attention { headCount := 2, headWidth := 4 }
    (mask := mask) (batchShape := [1]) (sequenceLength := 4) (modelWidth := 8)

/--
A full encoder block: attention, residual, LayerNorm, ReLU feed-forward, residual, LayerNorm.

Deliberately tiny (one head of width two) so the emitted Python stays readable.
We choose ReLU explicitly because the lowering to the verifier IR rejects GELU.
-/
def transformer :
    nn.Builder (nn.Sequential [1, 2, 2] [1, 2, 2]) :=
  nn.transformerBlock
    { headCount := 1
    , headWidth := 2
    , feedForwardWidth := 2
    , activation := relu }
    (batchShape := [1]) (sequenceLength := 2) (modelWidth := 2)

/-! ## CLI parsing -/

/-- Help text of `torch_ir_pytorch`, with an example command line for each `--arch` value. -/
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

/-- Lower a model with its initial state to verifier IR, convert it to an operator graph with
`LoweredModel.toImported`, and print the generated PyTorch module, named `className`, to stdout. -/
def emit {σ τ : Shape} (className : String) (model : nn.Sequential σ τ) : IO Unit := do
  let lowered ← IO.ofExcept <| Verification.lowerForwardToIR model (nn.initialState model)
  let imported := lowered.toImported
  let code ← IO.ofExcept <| PyTorch.emit imported (options := { className := className })
  IO.println code

/--
Entry point. `--arch` picks the architecture (default `mlp`) and `--seed` its initialization
(default `0`), and the generated Python goes to stdout for redirection to a file.
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
        (nn.build seed (attention (mask := some (Attention.causalMask 4))))
    else if arch == "transformer" then
      emit (className := "TorchLeanTransformerBlock") (nn.build seed transformer)
    else
      throw <| IO.userError
        (s!"unknown --arch {arch} (supported: linear | mlp | sum | autoencoder | " ++
          s!"mha | mha-mask | transformer)")

end TorchLean.Examples.TorchIRPyTorch
