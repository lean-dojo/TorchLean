/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Quickstart.Common

/-!
# Quickstart: Training a Small MLP

Regression with a two-layer MLP on a synthetic grid:

1. define the model with `nn.Sequential!`,
2. build an in-memory dataset with `Data.fromTensors`,
3. create a `Trainer` with an objective and optimizer,
4. call `trainer.train`, then predict with the result.

Run:

- `scripts/lake.sh exe torchlean quickstart_mlp`
- `scripts/lake.sh exe torchlean quickstart_mlp --steps 20 --seed 3`

Flags: `--steps N`, `--seed S`, `--arithmetic`, `--execution`, `--device`, and `--show-backend`.
-/

@[expose] public section

namespace NN.Examples.Quickstart.SimpleMlpTrain

open TorchLean

/-- Command name used in diagnostics and by the top-level example runner. -/
def exeName : String := "quickstart_mlp"

/-- A two-layer ReLU MLP `2 -> 8 -> 1`. -/
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

/-- Piecewise-linear regression target `0.8 relu(x₁ + x₂) - 0.4 relu(x₂ - x₁) + 0.2`. -/
def target (x : Tensor Float [2]) : Tensor Float [1] :=
  let features := Tensor.relu ([x[0] + x[1], x[1] - x[0]] : Tensor Float [2])
  [0.8 * features[0] - 0.4 * features[1] + 0.2]

/-- Twenty-five grid points in `[-1, 1]²` with their targets. -/
def data : Trainer.Dataset [2] [1] :=
  let inputs := Data.Synthetic.squareGrid (-1.0) 1.0 5
  Data.fromTensors inputs (Tensor.mapLeading [5 * 5] target inputs)

/-- Command-line help for the simple MLP quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean simple MLP quickstart"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean quickstart_mlp [--steps N] [--seed S]"
    , "    [--arithmetic native|ieee] [--execution eager|typed-graph] [--device cpu|cuda]"
    , "    [--show-backend]"
    ]

/--
Entry point. Parses the training flags, then runs the loop; `--steps` defaults to 200, which is
enough for the printed loss to visibly fall without the demo taking long.
-/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let flags ← parseFlags exeName args (defaultSteps := 200)

  let trainer := Trainer.new model
    { flags.runtime with
        objective := .meanSquaredError
        optimizer := optim.adam { learningRate := 0.03 }
        seed := flags.seed }

  IO.println s!"== Quickstart: simple MLP training (seed={flags.seed}, steps={flags.steps}) =="
  let heldout : Tensor Float [2] := [0.25, -0.75]
  IO.println s!"target(heldout)    = {reprStr (target heldout)}"
  let untrained ← trainer.predict heldout
  IO.println s!"untrained(heldout) = {reprStr untrained}"

  let trained ← trainer.train data { steps := flags.steps, logEvery := 25 }
  trained.printSummary
  trained.printPrediction "trained(heldout)" heldout

end NN.Examples.Quickstart.SimpleMlpTrain
