/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Loaders
public import NN.API.Trainer.Dataset
public import NN.API.Data.Sources

/-!
# Datasets

Dataset constructors and file-backed loaders used by `TorchLean.Trainer`.
-/

@[expose] public section

namespace TorchLean

namespace Data

/--
Runtime-polymorphic supervised dataset for `Float` tensors.

This is the typed counterpart of PyTorch's `TensorDataset(inputs, targets)`. The common leading
dimension counts samples; the remaining input and target dimensions are inferred and retained in
the result type.

Use this for most tutorials and file-loader paths: Float data is cast into the arithmetic
representation selected by the command with `--arithmetic`.

Example:
```lean
-- Four samples of a two-feature regression problem. The leading `4` counts samples.
def inputs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

def targets : Tensor Float [4, 1] := [[0.0], [1.0], [1.0], [0.0]]

-- The sample axis leaves the type; what stays is the shape of one sample.
def dataset : Trainer.Dataset [2] [1] := Data.fromTensors inputs targets
```
-/
def fromTensors
    {n : Nat} {σ τ : Spec.Shape}
    (inputs : Tensor Float (σ.prependDim n))
    (targets : Tensor Float (τ.prependDim n)) :
    Trainer.Dataset σ τ :=
  { materialize := fun {_α} _ =>
      pure <| SampleStream.fromFunction n fun sampleIndex =>
        { input := Tensor.map Runtime.ofFloat (Spec.get inputs sampleIndex)
          target := Tensor.map Runtime.ofFloat (Spec.get targets sampleIndex) } }

/--
Runtime-polymorphic supervised dataset from an explicit sample builder.

Use this when Lean code generates samples directly rather than loading them from batched tensors,
CSV, or NPY files. Sequence windows, synthetic PDE batches, and task-specific examples can keep
their own sample logic while still returning a standard `Trainer.Dataset`.

Example:
```lean
-- Samples built in Lean instead of read from disk. `α` stays abstract, so one builder serves a
-- `.native` run and an `.ieee` run.
def dataset : Trainer.Dataset [1] [1] :=
  Data.generate fun {_α} _ _ _ =>
    Array.ofFn (n := 8) fun (i : Fin 8) =>
      let x := i.val.toFloat
      { input := Tensor.map Runtime.ofFloat ([x] : Tensor Float [1])
        target := Tensor.map Runtime.ofFloat ([2.0 * x] : Tensor Float [1]) }
```
-/
def generate
    {σ τ : Spec.Shape}
    (generateSamples : {α : Type} → [TorchLean.Storage α] →
      [Context α] → [Runtime.FromFloat α] →
      Array (Sample.Supervised α σ τ)) :
    Trainer.Dataset σ τ :=
  { materialize := fun {α} _ _ =>
      pure <| TorchLean.Data.SampleStream.fromArray (generateSamples (α := α)) }

/-- Runtime-polymorphic dataset from an indexed stream of Float samples.

Samples are constructed and cast when accessed, so generated windows need not be materialized
as an array before training begins.
-/
def fromStream {σ τ : Spec.Shape}
    (values : SampleStream (Sample.Supervised Float σ τ)) : Trainer.Dataset σ τ :=
  { materialize := fun {_} _ _ =>
      pure <| values.map fun sample =>
        Sample.map (Tensor.map Runtime.ofFloat) (Tensor.map Runtime.ofFloat) sample }

/--
Runtime-polymorphic dataset from an in-memory array of `Float` supervised samples.

External data naturally enters Lean as `Float`. Conversion to the trainer-selected arithmetic
representation remains inside the dataset boundary.

Example:
```lean
def samples : Array (Sample.Supervised Float [2] [1]) :=
  #[{ input := [0.0, 1.0], target := [1.0] },
    { input := [1.0, 1.0], target := [0.0] }]

def dataset : Trainer.Dataset [2] [1] := Data.fromSamples samples
```
-/
def fromSamples
    {σ τ : Spec.Shape}
    (values : Array (Sample.Supervised Float σ τ)) :
    Trainer.Dataset σ τ :=
  fromStream (SampleStream.fromArray values)

/--
Build a dataset from one concrete `Float` sample.

Example:
```lean
-- One sample is enough to smoke-test a model end to end.
def dataset : Trainer.Dataset [2] [1] :=
  Data.fromSample { input := [1.25, -2.5], target := [3.75] }
```
-/
def fromSample
    {σ τ : Spec.Shape}
    (sample : Sample.Supervised Float σ τ) :
    Trainer.Dataset σ τ :=
  fromStream (SampleStream.fromFunction 1 fun _ => sample)

/--
Defer construction of one `Float` sample until the dataset is materialized.

Use this when the sample comes from a file-backed or runtime-loaded Float boundary. The public
trainer still owns the arithmetic/backend choice through `Trainer.RunConfig` and
`Trainer.TrainOptions`.

Example:
```lean
-- The file is read when the trainer materializes the dataset, not while this definition is
-- elaborated, so building the module never touches the disk.
def dataset (path : System.FilePath) : Trainer.Dataset [2] [1] :=
  Data.defer do
    let input : Tensor Float [2] ← Tensor.load path
    pure { input := input, target := [3.75] }
```
-/
def defer
    {σ τ : Spec.Shape}
    (loadSample : IO (Sample.Supervised Float σ τ)) :
    Trainer.Dataset σ τ :=
  { materialize := fun {_} _ _ => do
      let sample ← loadSample
      pure <| TorchLean.Data.SampleStream.fromArray
        #[ Sample.map
            (Tensor.map Runtime.ofFloat)
            (Tensor.map Runtime.ofFloat)
            sample ] }

/--
Convert an unbatched supervised dataset into a fixed-size batched dataset.

Public adapter for examples that want to minibatch the dataset before training and let the model own
the batch axis. The returned dataset prepends `batch` to each sample shape, so it can be passed
directly to `Trainer.new` with a batched model.

Example:
```lean
-- Sixteen single samples become four batched samples, and the batch axis lands in front of both
-- shapes so a batched model accepts the result unchanged.
def batched (dataset : Trainer.Dataset [2] [1]) :
    Trainer.Dataset [4, 2] [4, 1] :=
  Data.batch 4 dataset (shuffle := true) (seed := 0)
```
-/
def batch
    {σ τ : Spec.Shape} (batchSize : Nat)
    (dataset : Trainer.Dataset σ τ)
    (shuffle : Bool := true) (seed : Nat := 0) :
    Trainer.Dataset (σ.prependDim batchSize) (τ.prependDim batchSize) :=
  { materialize := fun {α} _ => do
      let samples ← dataset.materialize (α := α)
      let samples :=
        if shuffle then
          SampleStream.shuffled seed samples
        else
          samples
      match collateStream (α := α) batchSize samples with
      | .ok batchedSamples => pure batchedSamples
      | .error message => throw <| IO.userError s!"Data.batch: {message}" }

/-- Named train/test views produced by a dataset split. -/
structure DatasetSplit (input target : Spec.Shape) where
  /-- Samples selected for training. -/
  train : Trainer.Dataset input target
  /-- Samples held out for testing or validation. -/
  test : Trainer.Dataset input target

/--
Split a public dataset into deterministic train/test views.

Dataset-level analogue of `torch.utils.data.random_split`: the split happens after the
trainer materializes its selected arithmetic representation, but callers stay on ordinary
`Trainer.Dataset` values. Each view materializes the source independently; the source must return
the same sample order for a fixed seed to keep training and test membership disjoint. Materialize
changing or effectful sources into `Data.fromSamples` before splitting them.

Example:
```lean
-- Deterministic given the seed, so the held-out samples are the same on every machine.
def split (dataset : Trainer.Dataset [2] [1]) : Data.DatasetSplit [2] [1] :=
  Data.randomSplit 3 dataset (seed := 17)

def heldOut (dataset : Trainer.Dataset [2] [1]) : Trainer.Dataset [2] [1] :=
  (split dataset).test
```
-/
def randomSplit
    {σ τ : Spec.Shape} (trainSize : Nat)
    (dataset : Trainer.Dataset σ τ) (seed : Nat := 0) :
    DatasetSplit σ τ :=
  let selectPartition (selectTrainingSamples : Bool) :
      Trainer.Dataset σ τ :=
    { materialize := fun {α} _ _ => do
        let samples ← dataset.materialize (α := α)
        if trainSize > samples.size then
          throw <| IO.userError
            (s!"Data.randomSplit: requested split {trainSize}, "
              ++ s!"but dataset only has {samples.size} samples")
        let split := SampleStream.randomSplitAt seed trainSize samples
        pure <| if selectTrainingSamples then split.selected else split.remaining }
  { train := selectPartition true, test := selectPartition false }

/--
Load a numeric CSV table as a dataset of fixed-size tabular regression batches.

Each CSV row is interpreted as `inputWidth` feature columns followed by `targetWidth` target
columns.
The returned dataset already has the leading batch dimension expected by a model with input
shape `[batchSize, inputWidth]` and output shape `[batchSize, targetWidth]`.

Example:
```lean
-- Three columns per row: two features then one target, delivered as batches of eight.
def dataset : Trainer.Dataset [8, 2] [8, 1] :=
  Data.fromCsv "data/table.csv" (batchSize := 8) (inputWidth := 2) (targetWidth := 1)
```
-/
def fromCsv
    (path : System.FilePath) (batchSize inputWidth targetWidth : Nat)
    (csvOptions : CsvOptions := {}) (shuffle : Bool := true)
    (seed : Nat := 0) :
    Trainer.Dataset [batchSize, inputWidth] [batchSize, targetWidth] :=
  { materialize := fun {α} _ => do
      let source := TabularSupervisedSource.fromCsv path inputWidth targetWidth csvOptions
      let samples ← source.load (α := α)
      let samples :=
        if shuffle then
          SampleStream.shuffled seed samples
        else
          samples
      match collateStream (α := α) batchSize samples with
      | .ok batchedSamples => pure batchedSamples
      | .error message => throw <| IO.userError s!"Data.fromCsv: {message}" }

/--
Runtime-polymorphic supervised regression dataset from a tensor source.

Public file-data analogue of `torch.utils.data.TensorDataset(X, Y)` for examples whose targets are
tensors rather than class labels. The source records where batched features and targets live; the
trainer materializes them in the selected arithmetic representation.

Example:
```lean
def source : Data.SupervisedSource :=
  Data.SupervisedSource.fromFiles "data/inputs.npy" "data/targets.npy"
    (sampleCount := 64) (input := [16]) (target := [1])

def dataset : Trainer.Dataset [16] [1] := Data.fromSupervisedSource source
```
-/
def fromSupervisedSource (source : SupervisedSource) :
    Trainer.Dataset source.input source.target :=
  { materialize := fun {α} _ => do
      source.load (α := α) }

/--
Runtime-polymorphic one-hot classification dataset from a tensor source.

Public file-data analogue of `torch.utils.data.TensorDataset`: the source records where features and
integer labels live, and the trainer materializes them in the selected arithmetic representation.

Example:
```lean
def source : Data.LabeledSource :=
  Data.LabeledSource.fromFiles "data/images.npy" "data/labels.npy"
    (sampleCount := 64) (input := [3, 8, 8]) (classCount := 10)

-- Labels are one-hot encoded on load, so the target shape is `[classCount]`.
def dataset : Trainer.Dataset [3, 8, 8] [10] := Data.fromLabeledSource source
```
-/
def fromLabeledSource (source : LabeledSource) :
    Trainer.Dataset source.input [source.classCount] :=
  { materialize := fun {α} _ => do
      source.load (α := α) }

end Data

end TorchLean
