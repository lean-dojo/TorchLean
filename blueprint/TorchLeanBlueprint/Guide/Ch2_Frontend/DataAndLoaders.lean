import VersoManual

open Verso.Genre Manual

#doc (Manual) "Datasets, Loaders, and Minibatches" =>
%%%
tag := "datasets-loaders"
%%%

A model tutorial is only useful for a few minutes if it trains on one hard-coded tensor. Data enters
TorchLean as in-memory samples, file-backed tensors, deterministic loaders, and minibatches with
shapes the model can see. The surface is close by design to `torch.utils.data`, but the samples
carry Lean shapes.

The reader model is:

```
file or in-memory tensors
  -> typed dataset
  -> batch loader
  -> training loop
  -> report or saved curve
```

# Samples Have Shapes

A supervised sample has two tensors:

```
(x : Tensor alpha inputShape, y : Tensor alpha targetShape)
```

TorchLean packages that idea with the public `Data` and `sample` namespaces. In the MLP quickstart,
the dataset is built from two batched tensors:

```
import NN
open TorchLean

def X : Tensor.T Float (shape![25, 2]) :=
  Samples.grid2Square (-1.0) 1.0 5

def Y : Tensor.T Float (shape![25, 1]) :=
  Samples.regression2to1Float X target

def dataset : Trainer.Dataset (Shape.vec 2) (Shape.vec 1) :=
  Data.tensorDataset X Y
```

The leading dimension is the sample dimension. The loader then turns per sample shapes into batched
shapes.

# TensorDataset Style Data

For small tutorials, the simplest path is fully in memory:

```
let dataset :=
  Data.Bands.dataset
```

This is the TorchLean analogue of a small PyTorch `TensorDataset`: a finite dataset whose elements
are already tensors. It is excellent for introductory examples, tests, and examples where the point is the
model or proof interface rather than file IO.

The image band dataset used by the CNN tutorial is exposed through the public `Data` API and is
used directly by
[NN.Examples.Quickstart.SimpleCnnTrain](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleCnnTrain.lean).

# File Sources

For real data, TorchLean uses small, predictable file contracts:

- `.npy` for numeric tensors;
- small numeric CSV for tabular examples;
- text files for sequence examples;
- conversion scripts for formats such as image folders, `.mat`, `.pt`, `.pth`, or `.npz`.

The public source types are:

- `Data.TensorSource` for one tensor file;
- `Data.SupervisedSource` for `(X, Y)` tensor files;
- `Data.LabeledSource` for inputs plus integer labels;
- `Data.TabularSupervisedSource` for numeric CSV with input and target columns.

The important behavior is that loading returns an error when the file or shape does not match the
declared contract. A loader failure is not delayed until the model sees a bad batch.

The data contract references are:

- [NN/API/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Data/README.md)
- [NN/Examples/Data/README.md](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/README.md)
- [CSV loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [NPY loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean)
- [CIFAR image loader example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean)

# Minibatches and Epochs

The basic minibatch API is:

```
let loader :=
  Data.batchLoader dataset batch (shuffle := true) (seed := seed) (dropLast := true)
```

The arguments mean what a PyTorch reader expects:

- `batch` is the number of samples per minibatch;
- `shuffle := true` permutes the dataset at epoch boundaries;
- `seed` makes the order reproducible;
- `dropLast := true` keeps every batch at the same static shape.

That last point matters in Lean. A batch of size `5` has a different type from a batch of size `4`.
For tutorials that want one model type throughout training, `dropLast := true` is usually the clean
choice.

## Why `dropLast` Matters In Lean

In PyTorch, a final batch of size `3` after several batches of size `8` is usually fine. In
TorchLean, the batch dimension can appear in the type. If the model is written for
`Shape.mat 8 inDim`, then a final `Shape.mat 3 inDim` batch is not the same type. `dropLast := true`
keeps the tutorial simple by making every batch have the same static shape.

More flexible loaders can still be written, but then the file has to say how it handles the changing
batch dimension. That is the tradeoff: less implicit convenience, more visible shape information.

When a full epoch is needed directly, `Data.BatchLoader.epoch` materializes the batches and returns
the updated loader state. Most public examples stay one level higher and batch the dataset first:

```
let data := Data.batchDataset batch baseData (shuffle := true) (seed := seed)
let trained ← trainer.train data { steps := 200, batchSize := 16 }
trained.printSummary
```

That is the standard user-facing path. The loader still exists under the hood, but the public
example does not need to own runner state, callbacks, or a separate epoch loop.

# A Complete Minibatch Shape

Suppose a CSV row has:

- two input columns,
- one target column,
- and we train with `batch = 5`.

The per sample task is:

```
Shape.vec 2 -> Shape.vec 1
```

The minibatch model is:

```
Shape.mat 5 2 -> Shape.mat 5 1
```

That is why the quickstart writes:

```
def mkModel {batch : Nat} :
    nn.M (nn.Sequential (Shape.mat batch inDim) (Shape.mat batch outDim)) :=
  nn.Sequential![
    nn.Linear inDim hidDim,
    nn.ReLU,
    nn.Linear hidDim outDim
  ]
```

The model says it consumes a batch. The dataset says it contains individual samples. The loader
connects those two views.

# Hooks And Curves

Good runnable commands should still leave an artifact behind. The public trainer result writes a
two-point TrainLog when JSON logging is enabled, and it still exposes that same before/after summary
as its quick terminal report.

```
let trained ← trainer.train data
  { steps := 200
    log := .json outPath }
```

The returned `trained` value is not only a string summary. It also keeps the trained runtime handle alive,
so public examples can immediately run `trained.eval ...` without reopening the advanced runner
API.

Lower-level callbacks still exist for runtime-module tutorials and custom training loops:

```
Trainer.Advanced.onTrainStart do
  Trainer.Advanced.Report.reportMeanLoss (task := task) runner dataset "before"
```

The model zoo examples also accept a log path through the shared CLI flags. The JSON log is meant to
be plotted or checked later, so the examples can answer the practical question: did the model learn
on the dataset we gave it?

# Text and Sequence Data

Text models use the same principle, but the sample builder is different. The sequence examples read
a corpus, tokenize or encode it, and create causal samples. The useful files to read next are:

- [NN/API/Text.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Text.lean)
- [NN/Examples/Models/Sequence/Transformer.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/Transformer.lean)
- [NN/Examples/Models/Sequence/TextGpt2.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/TextGpt2.lean)

The shapes are still the guide. A language model sample is not "just a list of tokens"; it is a
tensor with a sequence length and a target convention.

# What the Data Layer Guarantees

The data layer does not prove that a dataset is scientifically meaningful. It does something more
modest and more useful for the rest of the stack:

- it checks that tensors entering training have the shapes the model expects;
- it keeps minibatches reproducible when the seed is fixed;
- it gives training loops stable batch shapes when the example requests them;
- it produces logs and reports that can be inspected later.

That is the right amount of machinery for a tutorial book: enough to train real examples, but small
enough that the reader can still see the path from file to tensor to model update.
