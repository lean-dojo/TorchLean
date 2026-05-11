import VersoManual

open Verso.Genre Manual

#doc (Manual) "Training From Scratch" =>
%%%
tag := "training-from-scratch"
%%%

A TorchLean training file usually has five blocks: constants, model, task, data, and fit/report.
Once you can recognize those blocks, most examples in the repository become readable.

The smallest workflow is:

1. define a typed model,
2. attach a loss by choosing a task,
3. choose an optimizer,
4. build or load a dataset,
5. run training and inspect the report.

The goal is not to prove a theorem yet. The goal is to make the objects that later become graph
artifacts and verification targets feel like ordinary model code.

# A Minimal Complete Script

The quickest way to get a feel for the public API is to run the focused MLP example:

- [NN.Examples.Quickstart.SimpleMlpTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)
- [NN.Examples.Quickstart.SimpleCnnTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [NN.Examples.Quickstart.ResnetBasicblockTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/ResnetBasicblockTrain.lean)
- [NN.Examples.Quickstart.AutogradBasics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

```
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- --steps 200 --dtype float --backend eager
```

A few details to notice in that file:

- `import NN` is the supported tutorial entrypoint.
- `open NN.API` gives you the “frontend” namespaces: `nn`, `train`, `optim`, `Data`, and CLI helpers.
- The model is built by `nn.build seed mkModel`, which fixes initialization and returns the
  trainable model package.
- The dataset is ordinary Lean data with tensor shapes.
- The report prints loss before and after training, then probes the trained model on a few inputs.

# Building Models With `nn`

Here is the core model definition pattern used across the tutorials:

```
open Spec
open NN.Tensor
open NN.API

def mkModel : nn.M (nn.Sequential (Shape.Vec 2) (Shape.Vec 1)) :=
  nn.sequential![
    nn.linear 2 8 (pfx := Spec.Shape.scalar),
    nn.relu,
    nn.linear 8 1 (pfx := Spec.Shape.scalar)
  ]

def task (seed : Nat) :=
  train.regression (nn.build seed mkModel)
```

Think of `nn.M` as a small “model builder” monad: it allocates parameters and wires layers
together, while keeping the end result purely functional.

## PyTorch Similarity

The PyTorch training pattern and the TorchLean training pattern are intentionally close:

```
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(2, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 1),
)
opt = torch.optim.Adam(model.parameters(), lr=0.03)
```

TorchLean writes the same structure in Lean, but makes the parameter bundle explicit:

```
open Spec
open NN.Tensor
open NN.API

def mkModel : nn.M (nn.Sequential (Shape.Vec 2) (Shape.Vec 1)) :=
  nn.sequential![
    nn.linear 2 8 (pfx := Spec.Shape.scalar),
    nn.relu,
    nn.linear 8 1 (pfx := Spec.Shape.scalar)
  ]

def task (seed : Nat) :=
  train.regression (nn.build seed mkModel)
```

The loop has the same rhythm as PyTorch: forward pass, loss, backward pass, optimizer step.
TorchLean's difference is where the state lives. Parameters, optimizer state, runtime mode, and logs
are explicit values rather than hidden fields of a module object.

# Workflow Lessons

The first training example is small, but it is already exercising the decisions that matter later:

- parameters are explicit data, so compilation and verification can find them without inspecting a
  hidden Python object;
- the model builder records structure separately from scalar execution, so the same architecture can
  be reused across backends;
- losses and optimizers are ordinary Lean functions over typed tensors, not implicit methods attached
  to mutable modules;
- the tutorial path and the verifier path share the same public model definitions instead of using
  unrelated examples.

A successful training run establishes that the model, data, loss, optimizer, and selected backend
execute together and produce a report. It does not prove convergence, robustness, or correctness of
the CUDA backend. Those claims need later checker or theorem support.

For a slightly more advanced path after the MLP demo, try the data-backed training tutorials:

- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)
- [CIFAR image loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean)

If you want more than an MLP:

- [NN.Examples.Quickstart.SimpleCnnTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [NN.Examples.Quickstart.ResnetBasicblockTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/ResnetBasicblockTrain.lean)

# Autograd and Training

TorchLean’s differentiation is functional: tensors do not carry mutable `.grad` fields. Instead,
TorchLean provides `autograd` primitives that transform functions.

For the smallest autograd-only demo, see:

- [NN.Examples.Quickstart.AutogradBasics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

If you want to understand how the tutorial files stay short, look at the tiny glue layer that
authors the examples in a chosen scalar backend:

- `NN.API.Common.castTensor`
- `NN.API.Common.tensorF` / `vecF` / `matF`
- `NN.API.Common.runWithDType` / `runWithRuntimeDType`
- `NN.API.Common.demoMain` / `demoMainRuntime`

That layer is what turns “one tutorial file” into “the same tutorial file can run over `Float`,
`IEEE32Exec`, or scalars used in proofs when appropriate”.

Runnable example patterns that depend on it:

- [Float32 modes example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Floats/Float32Modes.lean)
- [NN.Examples.Quickstart.SimpleMlpTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)

There is also a small synthetic data layer that keeps examples self-contained without importing
external datasets or Python preprocessing:

- `NN.API.Samples.grid2` / `grid2Square` for 2D regression grids,
- `NN.API.Samples.regression2to1Float` for simple supervised pairs,
- `NN.API.Samples.Image2D.bandDataset` / `namedBandSamples` for tiny CHW classification fixtures,
- `NN.API.Samples.oneHotFloat` / `classification` / `labeled` for label packing,
- `NN.API.Samples.imageCHWFloat` for parsing flat image buffers,
- `NN.API.Samples.Runtime.*` and `NN.API.Samples.Lit.*` for runtime-cast and literal-driven variants.

That keeps the quickstart tutorials focused on the training loop instead of on repetitive data setup.

## A Direct Loop Comparison

PyTorch usually looks like this:

```
for step in range(steps):
    opt.zero_grad()
    pred = model(x)
    loss = mse(pred, y)
    loss.backward()
    opt.step()
```

TorchLean’s equivalent is the same loop shape, but the gradients and the model state are explicit in
the API:

```
let loop ← train.stepper (task := task) runner (optim.adam 0.03)

for step in [0:steps] do
  let sample := getSample step
  let loss ← train.step (task := task) loop sample
  if step % 25 = 0 then
    IO.println s!"step {step}: loss={loss}"
```

That correspondence is deliberate. TorchLean keeps the training rhythm familiar while making the
state explicit enough for later graph, runtime, and proof chapters.

# The Anatomy Of A Training File

Most training examples have the same five blocks.

## 1. Constants

```
def inDim : Nat := 2
def hidden : Nat := 8
def outDim : Nat := 1
```

These are ordinary Lean definitions. They are not hidden inside a module object, so the shapes that
depend on them are visible to the type checker.

## 2. Model builder

```
def mkModel : nn.M (nn.Sequential (Shape.Vec inDim) (Shape.Vec outDim)) :=
  nn.sequential![
    nn.linear inDim hidden (pfx := Spec.Shape.scalar),
    nn.relu,
    nn.linear hidden outDim (pfx := Spec.Shape.scalar)
  ]
```

This is the architecture. It does not train anything by itself.

## 3. Task

```
def task (seed : Nat) :=
  train.regression (nn.build seed mkModel)
```

This is the model plus the loss convention. For classification examples, the analogous entry point
is `train.classificationOneHot`.

## 4. Dataset or loader

```
let dataset := buildDataset (α := α)
```

For tiny examples this is an in memory dataset. For real examples it is often loaded from CSV or
NPY, then wrapped in a `Data.batchLoader`.

## 5. Fit and report

```
let cfg := train.steps steps (optimizer := optim.adam 0.03) (logEvery := 25)
let _report <- train.fitDataset (task := task) runner cfg dataset
```

For loader based training, the last line becomes:

```
let (_report, _loader') <- train.fitLoaderWith (task := task) runner cfg loader hooks
```

The shape of the training file is stable even when the model family changes. MLPs, CNNs, residual
blocks, transformer blocks, and small scientific ML examples all follow this pattern.

For training loops, the public entrypoint is `train.run`. It parses standard flags (dtype, backend,
CUDA, etc.), constructs a runner, and then calls your callback at a concrete runtime scalar type.

Inside the callback, typical code uses helpers like:

- `train.fitDataset` for full fitting, or
- `train.stepper` + `train.step` for manual loops that feel closer to PyTorch.

A training file should make four things visible:

- the tensor shapes;
- the minibatch shape;
- the selected backend;
- the report produced by training.

If you want the autograd side of the same example, jump to:

- [NN.Examples.Quickstart.AutogradBasics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

# Datasets and Minibatching

TorchLean supports fully in-memory datasets (TensorDataset-style), plus typed minibatching.

Examples:

- [NN.Examples.Quickstart.MinibatchMlpTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/MinibatchMlpTrain.lean)
- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)

For the data boundary contract (`.npy`, small numeric CSV, UTF-8 text), see:

- [NN/API/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Data/README.md)
- [NN/Examples/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/README.md)

# Explicit Training Loops

If you want something that looks closer to “manual PyTorch training code”, use the quickstart
training files directly. They keep the same explicit rhythm (build model, choose optimizer, run
steps), while the PyTorch interop folder stays focused on actual Python import/export:

- [NN.Examples.Quickstart.SimpleMlpTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)
- [NN.Examples.Quickstart.SimpleCnnTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [NN.Examples.Quickstart.AutogradBasics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

Once you are comfortable running and training models, continue with “Runtime and Autograd” for the
execution modes (eager vs compiled graphs), what gets cached, and where the trust boundary sits.
