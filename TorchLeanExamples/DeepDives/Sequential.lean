/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Train
public import TorchLeanExamples.DeepDives.Sequential.MLP
public import TorchLean.NN.CLI.Trainer
public import TorchLean.NN.Data.Dataset

/-!
# Sequential tutorial

A sequential model (`nn.Sequential`) is data: its layers, the ordered shapes of their parameters
(`stateShapes`), and each layer's forward program. The same value has a pure meaning,
`nn.Sequential.eval`, and runs on the eager runtime and the typed graph.

The tutorial takes the MLP `Examples.Sequential.MLP.model 2 3 1`
(`Linear(2, 3) → ReLU → Linear(3, 1)`, weight seeds 0 and 1) and prints its pure forward value on
the initial parameters, which `Examples.Sequential.MLP.eval_model` proves is
`linear ∘ ReLU ∘ linear`. It then trains the same model with SGD (learning rate 0.1) for three steps
on one sample, with mean squared error as the loss.

Run:

```bash
scripts/lake.sh exe torchlean sequential
```

The standard TorchLean runtime flags such as `--arithmetic ieee` or `--device cuda` apply.
-/

@[expose] public section

namespace TorchLean.Examples.Sequential

open Trainer.Objective
open TorchLean.Tensor

/-- Command-line help for the sequential tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean sequential tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean sequential [options]"
    , ""
    , "Options:"
    , "  --arithmetic native|ieee"
    , s!"  --device auto|{Backend.Device.help}"
    , "  --show-backend                    print backend capsules as they are selected"
    ]

/-- The MLP `Linear(2, 3) → ReLU → Linear(3, 1)`, with parameter shapes
`[[3, 2], [3], [1, 3], [1]]`. -/
def model : nn.Sequential [2] [1] :=
  MLP.model 2 3 1

/-- One sample: input `[0.5, 0.8]`, target `[1.0]`. -/
def dataset : Trainer.Dataset [2] [1] :=
  let inputs : Tensor Float [1, 2] := [[0.5, 0.8]]
  let targets : Tensor Float [1, 1] := [[1.0]]
  Data.fromTensors inputs targets

/-- Print the pure forward value of `model` at input `[0.5, 0.8]` on its initial parameters, then
train it for three steps. -/
def run (args : List String) : IO Unit := do
  let input : Tensor Float [2] := [0.5, 0.8]
  IO.println s!"pure forward (nn.Sequential.eval, Float) = {model.eval model.initState input}"
  let runConfig ← Trainer.RunConfig.parseCommandLine "sequential" args
    { optimizer := optim.sgd { lr := 0.1 } }
  let trainer :=
    Trainer.create model <|
    { runConfig with objective := mse }
  trainer.printSummary
  let trained ← trainer.train dataset { steps := 3, logTitle := "Sequential tutorial" }
  trained.printSummary

/-- Entry point: print `usage` for `--help` or `-h`, otherwise run the tutorial. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  IO.println "== Sequential tutorial =="
  run args

end TorchLean.Examples.Sequential
