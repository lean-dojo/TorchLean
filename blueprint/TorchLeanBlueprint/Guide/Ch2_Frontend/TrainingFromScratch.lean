import VersoManual

open Verso.Genre Manual

#doc (Manual) "Training From Scratch" =>
%%%
tag := "training-from-scratch"
%%%

A TorchLean training file usually has five blocks: constants, model, trainer, data, and train/report.
Once you can recognize those blocks, most examples in the repository become readable.

The smallest workflow is:

1. define a typed model,
2. attach a loss by choosing a trainer,
3. choose an optimizer,
4. build or load a dataset,
5. run training and inspect the report.

No theorem is needed yet. The objects that later become graph artifacts and verification targets
should still feel like ordinary model code.

# A Minimal Complete Script

The shortest way to get a feel for the public API is to run the focused MLP example:

- [NN.Examples.Quickstart.SimpleMlpTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)
- [NN.Examples.Quickstart.SimpleCnnTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [NN.Examples.Quickstart.AutogradBasics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

```
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean
lake env lean --run NN/Examples/Quickstart/SimpleMlpTrain.lean -- --steps 200 --dtype float --backend eager
```

A few details to notice in that file:

- `import NN` is the supported tutorial entrypoint.
- `open TorchLean` gives you the public namespaces: `nn`, `Trainer`, `optim`, `Data`, and CLI parsers.
- `Trainer.new mkModel { task := .regression, seed := seed }` fixes initialization and returns the training task.
- The dataset is ordinary Lean data with tensor shapes.
- The report prints loss before and after training, then probes the trained model on a few inputs.

# Building Models With `nn`

Here is the core model definition pattern used across the tutorials:

```
import NN
open TorchLean

def mkModel : nn.M (nn.Sequential (Shape.vec 2) (Shape.vec 1)) :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

def trainer (seed : Nat) :=
  Trainer.new mkModel { task := .regression, seed := seed }
```

Think of `nn.M` as a small “model builder” monad: it allocates parameters and wires layers
together, while keeping the end result purely functional.

## PyTorch Similarity

The PyTorch training pattern and the TorchLean training pattern are close by design:

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
import NN
open TorchLean

def mkModel : nn.M (nn.Sequential (Shape.vec 2) (Shape.vec 1)) :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

def trainer (seed : Nat) :=
  Trainer.new mkModel { task := .regression, seed := seed }
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

For a slightly more advanced path after the MLP example, try the data-backed training tutorials:

- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)
- [CIFAR image loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean)

If you want more than an MLP:

- [NN.Examples.Quickstart.SimpleCnnTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [NN.Examples.Models.Vision.Cnn API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Vision/Cnn.lean)

# Autograd and Training

TorchLean’s differentiation is functional: tensors do not carry mutable `.grad` fields. Instead,
TorchLean provides `autograd` primitives that transform functions.

For the smallest autograd-only example, see:

- [NN.Examples.Quickstart.AutogradBasics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

The current tutorial files stay short by using the public API directly:

- `Data.*` builds shape-indexed datasets from small tensors, generated grids, CSV files, or image
  loaders.
- `Trainer.new` and `trainer.train` own the runtime scalar and backend
  selection.
- `Tensor`, `TensorPack`, and the `tensor!` / `tensorND!` / `tensorpack!` macros keep examples
  typed without making each tutorial reopen the runtime callback layer.

That is the important split: public tutorials should describe the model, data, optimizer, and training
options. The `NN.API.Runtime` callback layer still exists for runtime implementers and proof
work, but it is no longer the path a first example has to copy.

Runnable examples that show the public style:

- [Float32 modes example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Floats/Float32Modes.lean)
- [NN.Examples.Quickstart.SimpleMlpTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)

There is also a small public data layer that keeps examples self-contained without importing
external datasets or Python preprocessing:

- `Data.regression2to1Grid` for 2D regression grids,
- `Data.tensorDataset` for already-materialized input/target tensors,
- `Data.tabularCsvDataset` for small supervised CSV files,
- `Data.cifar10Dataset` and related loader APIs for image examples,
- `Trainer.Probe.*` constructors for before/after prediction reports.

That keeps the introductory tutorials focused on the training loop instead of on repetitive data setup.

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
let loop ← Trainer.Advanced.stepper (task := task) runner (optim.adam { lr := 0.03 })

for step in [0:steps] do
  let sample := getSample step
  let loss ← Trainer.Advanced.step (task := task) loop sample
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
def mkModel : nn.M (nn.Sequential (Shape.vec inDim) (Shape.vec outDim)) :=
  nn.Sequential![
    nn.Linear inDim hidden,
    nn.ReLU,
    nn.Linear hidden outDim
  ]
```

This is the architecture. It does not train anything by itself.

## 3. Trainer

```
def trainer (seed : Nat) :=
  Trainer.new mkModel { task := .regression, seed := seed }
```

This is the public model-plus-loss choice. For classification examples, use
`Trainer.new model { task := .classification }`.

## 4. Dataset or loader

```
let data : Trainer.Dataset (Shape.vec inDim) (Shape.vec outDim) :=
  Data.tensorDataset xs ys
```

For tiny examples this is an in memory dataset. For real examples it is often loaded from CSV or
NPY, then wrapped in a `Data.batchLoader`.

## 5. Train and report

```
let trainer := Trainer.new mkModel
  { task := .regression, optimizer := optim.adam { lr := 0.03 }, seed := seed }
let trained ← trainer.train data { steps := steps, batchSize := 16 }
trained.printSummary
let yhat ← trained.eval (tensorND! [2] [0.25, -0.75])
IO.println s!"heldout={yhat}"
```

The public shape stays stable even when the model family changes: define a model, define data,
choose a trainer, pass one config, call `trainer.train`, then reuse the trained handle for inference.

If a file needs callbacks, custom step scheduling, or proof-facing runtime access, it can drop into
`Trainer.Advanced`. That should read as advanced code. The beginner path should remain the same
four-value shape: model, data, trainer, config.

A training file should make four things visible:

- the tensor shapes;
- the minibatch shape;
- the selected backend;
- the trained handle produced by training.

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

# Step Streams

Not every training job is a finite dataset pass.  Some workloads produce the next batch from a
rule, a simulator, a replay buffer, or a file-backed token window.  For those cases TorchLean has a
typed step stream:

```
Trainer.Advanced.StepBatchStream α inputShapes
```

A `StepBatchStream` is a function from the optimizer step number to the already-collated input list
for the module.  The important point is that the shape list is still checked by Lean.  A stream for
`[x, y]` samples and a stream for `[tokens, targets]` samples have different types, even if both are
loaded from external files.

The corresponding training APIs are:

```
Trainer.Advanced.trainModuleStreamStepsWith
Trainer.Advanced.trainModuleStreamStepsReport
Trainer.Advanced.trainModuleStreamStepsCurveFloat
```

Use a loader when the dataset is a fixed finite table. Use a step stream when the batch is produced
on demand: RL rollouts, collocation points, generated windows from a large text file, or synthetic
diagnostic inputs. The public loop is model-agnostic; only the stream knows where the next
batch comes from.

# Explicit Training Loops

If you want something that looks closer to “manual PyTorch training code”, use the quickstart
training files directly. They keep the same explicit rhythm (build model, choose optimizer, run
steps), while the PyTorch interop folder stays focused on actual Python import/export:

- [NN.Examples.Quickstart.SimpleMlpTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean)
- [NN.Examples.Quickstart.SimpleCnnTrain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean)
- [NN.Examples.Quickstart.AutogradBasics API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)

Once you are comfortable running and training models, continue with “Runtime and Autograd” for the
execution modes (eager vs compiled graphs), what gets cached, and where the trust boundary sits.
