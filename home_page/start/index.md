---
title: Getting Started
layout: default
---

# Getting Started

After [installing TorchLean]({{ '/installation/' | relative_url }}), let's train a small model
and try interval bound propagation:

```bash
scripts/lake.sh build
scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 10 --arithmetic ieee --execution eager
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

The first lists runnable examples: quickstarts, supervised models, text models, diffusion, FNO
Burgers, reinforcement learning, data loaders, PyTorch interop, graph examples, and floating-point
checks. The second lists checker entry points: TorchLean IBP/CROWN paths, LiRPA-style fixtures,
PINN certificates, ODE enclosures, VNN-COMP-style MNIST queries, 3D projection certificates, spline
certificates, and two-stage Lyapunov experiments.

## Quickstart In Lean

The same model written as Lean code is the example from the repository README. Application code
writes tensor types as `Tensor α [dims...]` with the element type first, so `Tensor Float [4, 2]`
is a four-by-two tensor of `Float` values. `nn.Sequential!` is scoped syntax, so the file needs
`open TorchLean`.

```lean
import NN.API
open TorchLean
open Trainer.Objective (mse)

/-- A two-layer regression model. The dimensions are checked when the layers are composed. -/
def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

-- Four input rows, each containing two features.
def xs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

-- One regression target for each input row.
def ys : Tensor Float [4, 1] :=
  [[0.2], [1.0], [1.0], [1.8]]

-- The leading `4` counts samples; each sample has shapes `[2]` and `[1]`.
def data : Trainer.Dataset [2] [1] := Data.fromTensors xs ys

def trainOnce : IO Unit := do
  -- Select the loss and train through a typed graph with FloatLib binary32 arithmetic.
  let trainer :=
    Trainer.new model
      { objective := mse
        optimizer := optim.sgd { learningRate := 0.05 }
        execution := typedGraph
        device := cpu
        arithmetic := ieee }
  -- Inspect the initialized model before any parameter updates.
  let initialPrediction ← trainer.predict ([0.5, -0.25] : Tensor Float [2])
  IO.println s!"initial={reprStr initialPrediction}"
  -- Each step averages 16 sample gradients at one parameter point, then updates once.
  -- Training returns a result that retains the updated parameters and run report.
  let trained ← trainer.train data { steps := 200, samplesPerStep := 16, logEvery := 25 }
  trained.printSummary
```

`Trainer.new` takes the model and a configuration. `trainer.train` takes the dataset and a
`TrainOptions` record; `steps` is required and `samplesPerStep` says how many sample gradients
are averaged before each update. The result retains the updated parameter state and run report.

## Choosing Your Precision

Choose a FloatLib binary format, including custom precision, directly in your tensors. Binary32
is a familiar starting point; binary128 and custom widths use the same API:

```lean
import NN.API
open TorchLean
open FloatLib.Floats

abbrev Binary32 :=
  ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)

abbrev Binary128 :=
  ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)

abbrev Custom :=
  ExecFloat.Binary (exponentBits := 11) (fractionBits := 80)

def ordinaryInput : Tensor Binary32 [2] := [1.5, 2.25]

def preciseInput : Tensor Binary128 [2] :=
  [1.5, Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)]

def customInput : Tensor Custom [2] :=
  [1.5, Rat.cast (1 + 1 / (2 ^ 70 : Nat) : Rat)]
```

The wider inputs retain fractions that binary64 loses at 1. A rational cast rounds directly into
the chosen format. Its widths must satisfy FloatLib's constraints: at least two exponent bits,
a positive fraction width, and a valid exponent bias. Larger formats cost more memory and work.

Use `nn.State Binary128 (nn.stateShapes model)` and `nn.lowerToTypedGraph model (α := Binary128)`
to keep that scalar through parameters, activations, and derivatives on the typed CPU path. The
[tensor guide]({{ '/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/' | relative_url }})
works through an affine model with exact output and derivative checks. The supervised trainer
above uses `Float` datasets and reports. To retain the chosen scalar through a supervised session,
use `trainer.openTyped (α := Binary128)`: samples, predictions, losses, reports, and model-state
checkpoints keep that type. Typed sessions run on CPU and reject custom backend profiles. Their
results have no attached verifier, so `Result.verify` returns an error. Seeded initialization and
optimizer/scheduler settings still begin with
`Float` values; supply typed initial state when those initial values need extra precision.
The eager CUDA runtime uses LibTorch binary32 buffers; its separate
matrix-multiplication interface also supports binary64.

For an explicit training loop, take the parameter gradients returned by
`nn.TypedGraphModel.vjp` and pass them to `nn.sgdStep model learningRate state gradient`.
The learning rate, state, and updates use the selected scalar. The
[typed training example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TypedTraining.lean)
shows this loop for half squared error. It uses immutable CPU state, preserves frozen entries,
and rejects models with buffer-update hooks such as BatchNorm; it does not provide their
running-statistics updates or extend the trainer's checkpoint interface.

## Where To Go Next

1. [Installation]({{ '/installation/' | relative_url }}) covers Linux, macOS, Windows/WSL, CUDA,
   the LibTorch GPU backend, and backend capsules.
2. [Building Models]({{ '/blueprint/Building-Models/' | relative_url }}) introduces typed tensors,
   layers, parameter packs, datasets, losses, optimizers, and the trainer.
3. [Runtime and Interop]({{ '/blueprint/Runtime___-Autograd___-and-Interop/' | relative_url }})
   explains eager and typed graph execution, autograd, runtime artifacts, PyTorch interop boundaries,
   data streams, and backend selection.
4. [Semantics and Graphs]({{ '/blueprint/Semantics-and-Graphs/' | relative_url }}) explains the
   graph IR, graph denotation, shape checks, named operations, and why verifiers reuse the same
   graph rather than inventing a second model language.
5. [Floating Point and Native Boundaries]({{ '/blueprint/Floating-Point-and-Native-Boundaries/' | relative_url }})
   separates real-valued specifications, executable Float32 models, CUDA/native execution, and
   external producer assumptions.
6. [Verification and Certificates]({{ '/blueprint/Verification-and-Certificates/' | relative_url }})
   covers IBP/CROWN bounds, imported artifacts, optimizer laws, autograd proof APIs, scientific
   ML certificates, and trust boundaries.
7. [Examples]({{ '/examples/' | relative_url }}) collects runnable model, scientific ML,
   verification, text, diffusion, geometry, and Bug Zoo workflows.

## Common Next Steps

- Train a model: `scripts/lake.sh exe torchlean quickstart_mlp --device cpu --steps 100 --arithmetic ieee`.
- Inspect a scientific ML run: [Scientific ML]({{ '/examples/scientific-ml/' | relative_url }}).
- Check a certificate or bound pass: [Verification Bounds]({{ '/examples/verification/' | relative_url }}).
- Start application code with `import NN.API; open TorchLean`.
- Follow declaration and proof dependencies: [Formalization graph]({{ '/blueprint/Dependency-Graph/' | relative_url }}).
- Explore module dependencies: [Import graphs]({{ '/graphs/' | relative_url }}).
- Understand CUDA assumptions: [GPU and CUDA Boundaries]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }}).
