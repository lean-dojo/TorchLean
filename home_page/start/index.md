---
title: Getting Started
layout: default
---

# Getting Started

After [installing TorchLean]({{ '/installation/' | relative_url }}), let's train a small model
and try interval bound propagation:

```bash
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 50
scripts/lake.sh exe verify -- torchlean-ibp
```

The quickstart initializes parameters, executes a binary32 forward pass, computes a loss, runs
reverse mode, and updates the parameters. The verifier command lowers a TorchLean model to its
operation graph and propagates an input interval through that graph.

To see the command-line entry points:

```bash
scripts/lake.sh exe torchlean --help
scripts/lake.sh exe verify --help
```

## Quickstart In Lean

Here's the same model written as Lean code (it's the example from the repository README).
Application code writes tensor types as `Tensor α [dims...]` with the element type first, so
`Tensor Float [4, 2]` is a four-by-two tensor of `Float` values. `nn.Sequential!` is scoped syntax,
so the file needs `open TorchLean`.

```lean
import TorchLean.NN.Data.Dataset
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
open TorchLean
open Trainer.Objective (mse)

/-- A two-layer regression model. The dimensions are checked when the layers are composed. -/
def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.op .relu,
    nn.linear 8 1
  ]

-- Four input rows, each containing two features.
def xs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

-- One regression target for each input row.
def ys : Tensor Float [4, 1] :=
  [[0.2], [1.0], [1.0], [1.8]]

-- The leading `4` counts samples, and each sample has shapes `[2]` and `[1]`.
def data : Trainer.Dataset [2] [1] := Data.fromTensors xs ys

def trainOnce : IO Unit := do
  let trainer :=
    Trainer.create model
      { objective := mse
        optimizer := optim.sgd { lr := 0.05 } }
  -- Inspect the initialized model before any parameter updates.
  let initialPrediction ← trainer.predict ([0.5, -0.25] : Tensor Float [2])
  IO.println s!"initial={reprStr initialPrediction}"
  -- Each step averages 16 sample gradients at one parameter point, then updates once.
  -- Training returns a result that retains the updated parameters and run report.
  let trained ← trainer.train data { steps := 200, samplesPerStep := 16, logEvery := 25 }
  trained.printSummary
```

`Trainer.create` takes the model and a configuration. `trainer.train` takes the dataset and a
`TrainOptions` record, in which `steps` is required and `samplesPerStep` says how many sample
gradients are averaged before each update. Its result retains the updated parameter state and run
report. The trainer uses `Float32` by default.

## Choosing Your Precision

Choose a FloatLib binary format, including custom precision, as the element type of your tensors.
Binary32 is a familiar starting point, but binary128 and custom widths use the same API. FloatLib
names the interchange formats `Binary32`, `Binary64` and `Binary128`, and `ExecFloat.Binary` builds
any other width:

```lean
import TorchLean.NN.Sequential.Training
open TorchLean
open FloatLib.Floats

abbrev Custom :=
  ExecFloat.Binary (exponentBits := 11) (fractionBits := 80)

def ordinaryInput : Tensor Binary32 [2] := [1.5, 2.25]

def preciseInput : Tensor Binary128 [2] :=
  [1.5, Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)]

def customInput : Tensor Custom [2] :=
  [1.5, Rat.cast (1 + 1 / (2 ^ 70 : Nat) : Rat)]
```

The wider inputs retain fractions that binary64 loses at 1. A rational cast rounds directly into
the chosen format, whose widths must satisfy FloatLib's constraints: at least two exponent bits, a
positive fraction width, and a valid exponent bias. Larger formats cost more memory and work.

The [tensor chapter]({{ '/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/' | relative_url }})
runs an affine model and its derivative at the chosen precision. The
[typed training example](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Quickstart/TypedTraining.lean)
keeps the same scalar type through an SGD update. To use a supervised trainer on CPU with that
precision, call `trainer.openTyped (α := Binary128)`.

For a longer run, pick a [model example]({{ '/examples/' | relative_url }}). If you want to
understand the training loop first, start with [Building Models]({{ '/blueprint/Building-Models/' | relative_url }}).
