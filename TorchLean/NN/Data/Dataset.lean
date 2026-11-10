/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Data.Loader
public import TorchLean.NN.Trainer.Dataset

/-!
# Datasets

Constructors for `Trainer.Dataset`, the data a `TorchLean.Trainer` trains on. A dataset holds
`Float` data or a sample builder generic in the scalar type, and the trainer materializes it in
the arithmetic the run selects. `fromTensors`, `fromSamples` and `generate` build datasets in
memory, `fromCSV`, `fromSupervisedSource` and `fromLabeledSource` read files, and `batch` and
`randomSplit` transform a dataset.
-/

@[expose] public section

namespace TorchLean

namespace Data

/--
Dataset from an indexed stream of `Float` samples. Samples are built and cast when accessed, so
generated windows need not be collected into an array before training begins.
-/
def fromStream {σ τ : Shape}
    (values : SampleStream (Sample.Supervised Float σ τ)) : Trainer.Dataset σ τ :=
  { materialize := fun {_} _ _ =>
      pure <| values.map fun sample =>
        Sample.map (Tensor.map ofFloat) (Tensor.map ofFloat) sample }

/--
Supervised dataset from batched `Float` tensors, the typed counterpart of PyTorch's
`TensorDataset(inputs, targets)`. The common leading dimension counts samples, and the remaining
input and target dimensions stay in the result type.

Most tutorials and file loaders use this: the `Float` data is cast into the arithmetic that the
command selects with `--arithmetic`.

Example:
```lean
-- Four samples of a two-feature regression problem. The leading `4` counts samples.
def inputs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

def targets : Tensor Float [4, 1] := [[0.0], [1.0], [1.0], [0.0]]

-- The sample axis leaves the type, and what stays is the shape of one sample.
def dataset : Trainer.Dataset [2] [1] := Data.fromTensors inputs targets
```
-/
def fromTensors
    {n : Nat} {σ τ : Shape}
    (inputs : Tensor Float (Shape.dim n σ))
    (targets : Tensor Float (Shape.dim n τ)) :
    Trainer.Dataset σ τ :=
  fromStream <| SampleStream.mk n fun sampleIndex =>
    { input := Tensor.unstack inputs sampleIndex
      target := Tensor.unstack targets sampleIndex }

/--
Supervised dataset from a sample builder that is generic in the scalar type `α`.

Use this when Lean code generates the samples itself. Sequence windows, synthetic PDE batches and
task-specific examples keep their own sample logic and still return a standard `Trainer.Dataset`.

Example:
```lean
-- The samples are computed in Lean, with nothing read from disk. `α` stays abstract, so one
-- builder serves a `native` run and an `ieee` run.
def dataset : Trainer.Dataset [1] [1] :=
  Data.generate fun {_α} _ _ _ =>
    Array.ofFn (n := 8) fun (i : Fin 8) =>
      let x := i.val.toFloat
      { input := Tensor.map ofFloat ([x] : Tensor Float [1])
        target := Tensor.map ofFloat ([2.0 * x] : Tensor Float [1]) }
```
-/
def generate
    {σ τ : Shape}
    (generateSamples : {α : Type} → [TorchLean.Storage α] →
      [Context α] → [FromFloat α] →
      Array (Sample.Supervised α σ τ)) :
    Trainer.Dataset σ τ :=
  { materialize := fun {α} _ _ =>
      pure <| TorchLean.Data.SampleStream.fromArray (generateSamples (α := α)) }

/--
Dataset from an in-memory array of `Float` supervised samples.

External data enters Lean as `Float`, and the conversion to the trainer's arithmetic happens
inside the dataset.

Example:
```lean
def samples : Array (Sample.Supervised Float [2] [1]) :=
  #[{ input := [0.0, 1.0], target := [1.0] },
    { input := [1.0, 1.0], target := [0.0] }]

def dataset : Trainer.Dataset [2] [1] := Data.fromSamples samples
```
-/
def fromSamples
    {σ τ : Shape}
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
    {σ τ : Shape}
    (sample : Sample.Supervised Float σ τ) :
    Trainer.Dataset σ τ :=
  fromStream (SampleStream.mk 1 fun _ => sample)

/--
Defer building one `Float` sample until the dataset is materialized, for a sample read from a
file or produced at run time. The trainer keeps the arithmetic and backend choice through
`Trainer.RunConfig` and `Trainer.TrainOptions`.

Example:
```lean
-- The file is read when the trainer materializes the dataset, so elaborating this definition
-- never touches the disk.
def dataset (path : System.FilePath) : Trainer.Dataset [2] [1] :=
  Data.defer do
    let input : Tensor Float [2] ← Tensor.load path
    pure { input := input, target := [3.75] }
```
-/
def defer
    {σ τ : Shape}
    (loadSample : IO (Sample.Supervised Float σ τ)) :
    Trainer.Dataset σ τ :=
  { materialize := fun {_} _ _ => do
      let sample ← loadSample
      pure <| TorchLean.Data.SampleStream.fromArray
        #[ Sample.map
            (Tensor.map ofFloat)
            (Tensor.map ofFloat)
            sample ] }

/-- Shuffle when `shuffle` is set, then collate, naming `context` in batch validation errors. -/
def collateShuffled {α : Type} [Storage α] {σ τ : Shape}
    (context : String) (batchSize : Nat) (shuffle : Bool) (seed : Nat)
    (samples : SampleStream (Sample.Supervised α σ τ)) :
    IO (SampleStream (Sample.Supervised α (Shape.dim batchSize σ) (Shape.dim batchSize τ))) :=
  let samples := if shuffle then SampleStream.shuffled seed samples else samples
  IO.ofExcept <| (collateStream batchSize samples).mapError fun message => s!"{context}: {message}"

/--
Group an unbatched supervised dataset into `batchSize`-sample batches, for examples whose model
owns the batch axis.

The result prepends `batchSize` to both sample shapes, so it goes straight to `Trainer.create` with
a batched model. Samples are shuffled by `seed` unless `shuffle := false`, and a final partial batch
is dropped.

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
    {σ τ : Shape} (batchSize : Nat)
    (dataset : Trainer.Dataset σ τ)
    (shuffle : Bool := true) (seed : Nat := 0) :
    Trainer.Dataset (Shape.dim batchSize σ) (Shape.dim batchSize τ) :=
  { materialize := fun {α} _ => do
      let samples ← dataset.materialize (α := α)
      collateShuffled "Data.batch" batchSize shuffle seed samples }

/-- Named train/test views produced by a dataset split. -/
structure DatasetSplit (input target : Shape) where
  /-- Samples selected for training. -/
  train : Trainer.Dataset input target
  /-- Samples held out for testing or validation. -/
  test : Trainer.Dataset input target

/--
Split a dataset into deterministic train and test views, with `trainSize` samples for training.

Dataset-level analogue of `torch.utils.data.random_split`: the split happens after the trainer
materializes its arithmetic, and callers keep ordinary `Trainer.Dataset` values. Each view
materializes the source on its own, so the source must return the same samples in the same order
every time for the two views to stay disjoint. Materialize a changing or effectful source with
`Data.fromSamples` before splitting it. A `trainSize` above the sample count is an error.

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
    {σ τ : Shape} (trainSize : Nat)
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
Load a numeric CSV table as a dataset of `batchSize`-row regression batches. Each row holds
`inputWidth` feature columns followed by `targetWidth` target columns, and the batches fit a model
with input shape `[batchSize, inputWidth]` and output shape `[batchSize, targetWidth]`. Shuffling
and the dropped partial batch follow `batch`.

Example:
```lean
-- Three columns per row: two features then one target, delivered as batches of eight.
def dataset : Trainer.Dataset [8, 2] [8, 1] :=
  Data.fromCSV "data/table.csv" (batchSize := 8) (inputWidth := 2) (targetWidth := 1)
```
-/
def fromCSV
    (path : System.FilePath) (batchSize inputWidth targetWidth : Nat)
    (csvOptions : CSV.Options := {}) (shuffle : Bool := true)
    (seed : Nat := 0) :
    Trainer.Dataset [batchSize, inputWidth] [batchSize, targetWidth] :=
  { materialize := fun {α} _ => do
      let source : TabularSupervisedSource := { path, inputWidth, targetWidth, csvOptions }
      let samples ← source.load (α := α)
      collateShuffled "Data.fromCSV" batchSize shuffle seed samples }

/--
Supervised regression dataset from a `SupervisedSource`, the file-backed analogue of
`torch.utils.data.TensorDataset(X, Y)` for tensor targets. The source records where the batched
features and targets live, and the trainer loads them in the arithmetic it selects.

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
One-hot classification dataset from a `LabeledSource`, the file-backed analogue of
`torch.utils.data.TensorDataset` with integer labels. The trainer loads the features and labels
in the arithmetic it selects.

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
