import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Datasets, Loaders, and Minibatches" =>
%%%
tag := "datasets-loaders"
%%%

A model tutorial is only useful for a few minutes if it trains on one hard-coded tensor. This page
explains how data enters TorchLean: as in-memory samples, file-backed tensors, deterministic
loaders, and minibatches with shapes the model can see. The surface is intentionally close to
`torch.utils.data`, but the samples carry Lean shapes.

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
let X : Spec.Tensor Float (shape![25, 2]) :=
  API.Samples.grid2Square (-1.0) 1.0 5

let Y : Spec.Tensor Float (shape![25, 1]) :=
  API.Samples.regression2to1Float X target

Data.supervisedDim0F (α := α) X Y
```

The leading dimension is the sample dimension. The loader then turns per sample shapes into batched
shapes.

# TensorDataset Style Data

For small tutorials, the simplest path is fully in memory:

```
let dataset :=
  Data.labeled (α := α) (σ := Shape.CHW 1 4 4) 2 samplesF
```

This is the TorchLean analogue of a small PyTorch `TensorDataset`: a finite dataset whose elements
are already tensors. It is excellent for quickstarts, tests, and examples where the point is the
model or proof interface rather than file IO.

The image band dataset used by the CNN tutorial lives in
[NN.API.Samples.Bands](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Samples/Bands.lean).

# File Sources

For real data, TorchLean uses small, predictable file contracts:

- `.npy` for numeric tensors;
- small numeric CSV for tabular examples;
- text files for sequence demos;
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
`Shape.Mat 8 inDim`, then a final `Shape.Mat 3 inDim` batch is not the same type. `dropLast := true`
keeps the tutorial simple by making every batch have the same static shape.

More flexible loaders can still be written, but then the file has to say how it handles the changing
batch dimension. That is the tradeoff: less implicit convenience, more visible shape information.

When a full epoch is needed directly, `Data.BatchLoader.epoch` materializes the batches and returns
the updated loader state. Most examples call the higher level training helpers instead:

```
let cfg := train.epochs epochs (optimizer := optim.adam 0.05)
let (_report, _loader') <- train.fitLoaderWith (task := task) runner cfg loader hooks
```

That is the standard multi epoch path. It is not repeated training on one fixed batch. Each epoch
uses the loader, optionally reshuffles, and feeds the minibatches to the same task.

# A Complete Minibatch Shape

Suppose a CSV row has:

- two input columns,
- one target column,
- and we train with `batch = 5`.

The per sample task is:

```
Shape.Vec 2 -> Shape.Vec 1
```

The minibatch model is:

```
Shape.Mat 5 2 -> Shape.Mat 5 1
```

That is why the quickstart writes:

```
def mkModel {batch : Nat} :
    nn.M (nn.Sequential (Shape.Mat batch inDim) (Shape.Mat batch outDim)) :=
  nn.sequential![
    nn.linear inDim hidDim (pfx := Shape.Vec batch),
    nn.relu,
    nn.linear hidDim outDim (pfx := Shape.Vec batch)
  ]
```

The model says it consumes a batch. The dataset says it contains individual samples. The loader
connects those two views.

# Hooks and Curves

Good examples should show training progress. TorchLean uses callbacks for that:

```
let hooks : train.Callbacks alpha :=
  train.logLossEvery 5
```

More structured examples attach reports at the start, end, or after each epoch:

```
train.onTrainStart do
  train.Report.reportMeanLoss (task := task) runner dataset "before"
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
