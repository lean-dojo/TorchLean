import VersoManual
import NN.API
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Data and Loaders" =>
%%%
tag := "datasets-loaders"
file := "From-Files-To-Typed-Minibatches"
%%%

A model type tells Lean the shape of one input and one output. A dataset must eventually provide
values of exactly those shapes, but real data begins in a less orderly form: rows in a CSV file,
arrays in an NPY file, text tokens, simulator output, or tensors generated on demand.

TorchLean treats loading as a boundary. Parsing and dimension checks happen before a value becomes
a typed training sample. Once the boundary succeeds, the training loop does not need to ask on
every step whether a row had the right number of columns.

It helps to follow one row through that boundary. Two feature columns and one target column
become a pair of tensors with shapes `[2]` and `[1]`. A loader may then join five such pairs
into one item with shapes `[5, 2]` and `[5, 1]`. Neither the file's row count nor the batch size
changes what one feature means. Keeping source rows, dataset items, and model axes separate is
what lets the same training loop consume an in-memory table or a checked file.

Named Lean blocks and their outputs are checked when this page is built. Shell and Python blocks
are reproduction instructions or recorded transcripts. Comparing the same twenty-five rows through
three file boundaries checks this example; agreement of printed losses alone does not establish
identical tensor bits or parser equivalence.

# The Dataset Type

The trainer-facing type is:

```
-- The two shape parameters describe one input item and one
-- target item.
Trainer.Dataset input target
```

Its shape parameters describe one training item. The element type is intentionally absent. The
structure has a
single field:

```lean (name := dlDatasetType)
-- Materialization chooses the scalar type and returns a
-- stream through IO.
#check @Trainer.Dataset.materialize
```

```leanOutput dlDatasetType (whitespace := lax)
@Trainer.Dataset.materialize : {input target : Shape} →
  Trainer.Dataset input target →
    {α : Type} →
      [inst : Storage α] →
        [Context α] →
          [Runtime.FromFloat α] →
            IO (Data.SampleStream (Sample.Supervised α input target))
```

Read that signature from the inside out. A dataset is a function waiting for a scalar type. It
returns samples in `IO`, because opening a file, parsing values, and reporting a malformed row all
belong there. The scalar type `α` stays universally quantified, so one dataset value serves a
native `Float32` run and an `IEEE32Exec` run without being rebuilt. Materialization chooses one
arithmetic semantics for every numeric input and target in that run; it is not a per-column dtype
schema.

The same host-`Float` data source can therefore feed several arithmetic implementations through
one materialization interface. A proof-level real tensor cannot enter this IO training loop:
`Runtime.FromFloat α` has no instance for `ℝ`.

The signature also separates constructing a dataset from asking it for data. A value of
`Trainer.Dataset input target` records how materialization should happen. Calling `materialize`
supplies the scalar instances and runs the action; the resulting stream then supplies individual
samples by index. Some constructors can create those samples lazily. Materializing a dataset
therefore does not universally mean allocating an array containing every tensor in advance,
although a file parser may need to read its source before returning the stream.

For tensors already in memory, the public constructor shows how the sample shapes are inferred:

```lean (name := dlFromTensorsType)
-- The shared leading n pairs every input row with exactly
-- one target row.
#check @Data.fromTensors
```

```leanOutput dlFromTensorsType (whitespace := lax)
@Data.fromTensors : {n : ℕ} →
  {σ τ : Shape} →
    Tensor Float (σ.prependDim n) →
      Tensor Float (τ.prependDim n) → Trainer.Dataset σ τ
```

The common dimension `n` is the number of samples. It is removed from the dataset's input and
target types, so the caller does not repeat `[2]` and `[1]` after those shapes are already known
from tensors of shape `[4, 2]` and `[4, 1]`. Because `n` appears in both arguments, a pair of
tensors whose sample counts disagree is rejected before any file is read.

In the materialization signature, `Storage α` concerns how a sample's scalars are represented,
`Context α` supplies the operations available to a generated sample, and `Runtime.FromFloat α`
handles conversion from host-authored numeric data. None of those arguments changes the sample
shapes. This distinction is useful when comparing arithmetic modes: a change in scalar
representation should not also silently change the number of features, target layout, or
selection of rows.

# Data Validation Stages

Data failures are easier to diagnose when each fact is tied to the boundary where it becomes known:

:::table +header
*
  * Stage
  * What is known
  * Typical failure
*
  * source description
  * paths, format, requested dimensions, tokenizer or preprocessing choice
  * unsupported format or incomplete configuration
*
  * parsing
  * bytes or text can be decoded into values
  * missing file, malformed number, corrupt header
*
  * shape validation
  * element count and sample dimensions agree
  * short row, mismatched leading counts, invalid token id
*
  * arithmetic materialization
  * numeric values are converted to the selected runtime arithmetic, possibly with rounding
  * unsupported source dtype; narrowing can change values without a diagnostic
*
  * batching
  * each emitted item has the model's input and target shape
  * final partial batch under a fixed-size policy
*
  * training
  * the runner receives typed samples
  * numerical failure, unsupported operation, or backend rejection
:::

The runs later in this chapter illustrate file, parsing, shape, dtype, and layout failures:

```
data_csv: missing CSV dataset: NN/Examples/Data/small_regression.csv
TabularSupervisedSource.load: csv row 2, col 2: unparsed suffix: oops
TabularSupervisedSource.load: csv: expected 1 values for shape [1], got 0
data_npy: npy: unsupported dtype: <i4
SupervisedSource.load: npy: prefix row loading requires C-order NPY arrays
data_npy: X.npy: expected shape (N,2), got #[25, 1]
```

Moving a check earlier improves the error message, but it does not change its logical strength. A
CSV width check proves that accepted rows have the requested number of fields. It says nothing
about whether the second field is the physical quantity the experiment intended to measure. That
distinction is the whole reason this chapter separates the words "parsed", "well-shaped", and
"correct".

# Sample Shapes In Memory

The XOR table makes the distinction between a stored tensor and a training item explicit. Four
rows share one input tensor and one target tensor:

```lean (name := dlXor)
-- Keep all four XOR inputs and their targets in matching
-- leading-row order.
def dlXs : Tensor Float [4, 2] :=
  [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

def dlYs : Tensor Float [4, 1] :=
  [[0.0], [1.0], [1.0], [0.0]]

def dlXorData := Data.fromTensors dlXs dlYs

#check dlXorData
```

```leanOutput dlXor
dlXorData : Trainer.Dataset [2] [1]
```

No annotation was written on `dlXorData`. The leading dimension of `dlXs` and `dlYs` is the sample
count, `Data.fromTensors` checks that both counts agree, and the item shapes are what remain:

```
-- fromTensors removes the shared sample-count axis from the
-- shape of each item.
whole input tensor    [4, 2]
one input sample         [2]

whole target tensor   [4, 1]
one target sample        [1]
```

Nothing has been read from the dataset yet. Asking for the samples means choosing a scalar type and
running the `IO` action, which is exactly what the trainer does on your behalf:

```lean (name := dlSample1)
-- Materialize at Float, then read the second paired sample
-- using zero-based indexing.
#eval show IO Unit from do
  let stream ← dlXorData.materialize (α := Float)
  IO.println s!"samples = {stream.size}"
  match stream.get? 1 with
  | some sample =>
      IO.println s!"input  = {sample.input}"
      IO.println s!"target = {sample.target}"
  | none => IO.println "index 1 is out of range"
```

```leanOutput dlSample1 (whitespace := lax)
samples = 4
input  = [0.000000, 1.000000]
target = [1.000000]
```

`get? 1` uses a zero-based stream position, so the displayed input is the second XOR row,
`[0, 1]`. Its target is `[1]`, paired before the sample reaches training. The optional result
handles an invalid stream position; the tensors inside a successful sample already have their
promised dimensions. This separates a missing item from a malformed item. The example's
`none` branch concerns whether a second sample exists, not whether its feature vector has length
two.

This is the same role played by PyTorch's `TensorDataset`, whose indexing returns the pair of
tensors for one sample {Informal.citep pytorch2019}[]:

```
>>> # Select the second input row together with its
>>> # corresponding target.
>>> import torch
>>> from torch.utils.data import TensorDataset, DataLoader, random_split
>>> xs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
>>> ys = torch.tensor([[0.], [1.], [1.], [0.]])
>>> ds = TensorDataset(xs, ys)
>>> print("len(ds) =", len(ds), "sample 1 =", [t.tolist() for t in ds[1]])
len(ds) = 4 sample 1 = [[0.0, 1.0], [1.0]]
```

The values agree. The difference is where the sample shape lives: PyTorch keeps it in the tensors
and TorchLean also keeps it in the type, so `Trainer.Dataset [2] [1]` cannot be handed to a model
that expects three features.

Sample counts that disagree are rejected while the module elaborates, before any training code
exists {Informal.citep lean4}[]:

```lean +error (name := dlMismatched)
-- Three target rows cannot be paired with four input rows
-- by this constructor.
def dlYsShort : Tensor Float [3, 1] :=
  [[0.0], [1.0], [1.0]]

def dlMismatched := Data.fromTensors dlXs dlYsShort
```

```leanOutput dlMismatched (whitespace := lax)
Application type mismatch: The argument
  dlYsShort
has type
  Tensor Float [3, 1]
but is expected to have type
  Tensor Float (Shape.prependDim ?m.6 4)
in the application
  Data.fromTensors dlXs dlYsShort
```

The metavariable in that message is the unsolved sample shape. Lean already committed to a sample
count of `4` from the first argument, so the second tensor cannot supply three rows no matter what
its item shape turns out to be. If the mismatch instead arrives from runtime files, the loader
reports it as an `IO` error when materializing the dataset; the shape constraint is checked later.

Use the smallest constructor that matches where the samples come from:

- `Data.fromTensors inputs targets` splits two batched tensors along the sample axis;
- `Data.fromSamples values` wraps an in-memory array of concrete `Float` samples;
- `Data.fromSample sample` wraps one concrete `Float` sample, which is enough to smoke-test a model;
- `Data.defer action` materializes one concrete `Float` sample from an `IO` action;
- `Data.generate builder` constructs samples directly in the arithmetic runtime selected by the
  trainer.

All of them return the same `Trainer.Dataset input target` type. The separate names describe the
data boundary, not different dataset representations. Writing the samples out by hand looks like
this:

```lean (name := dlPair)
-- Each record owns its input-target pair before the records
-- are collected into a dataset.
def dlPair : Trainer.Dataset [2] [1] :=
  Data.fromSamples
    #[{ input := [0.0, 1.0], target := [1.0] },
      { input := [1.0, 1.0], target := [0.0] }]

#eval show IO Unit from do
  let stream ← dlPair.materialize (α := Float)
  IO.println s!"samples = {stream.size}"
```

```leanOutput dlPair
samples = 2
```

The count is two because the array contains two supervised records. Each record keeps its input
and target together, which prevents shuffling from independently reordering the two sides.
Choosing `Data.fromSamples` is useful when labels are computed one record at a time or samples
already arrive as pairs. `Data.fromTensors` instead starts with two stacked tensors and uses
their common leading index to construct those same pairs. Both routes eventually expose the
same per-item interface to the trainer.

The `Data.defer` variant is the one to reach for when the sample lives in a file. Note that the
file is read when the trainer materializes the dataset, not while the module elaborates, so
defining this dataset does not read the file:

```
-- Delay file access until materialization while retaining
-- the required input shape.
def dlOneSample (path : System.FilePath) : Trainer.Dataset [2] [1] :=
  Data.defer do
    let input : Tensor Float [2] ← Tensor.load path
    pure { input := input, target := [3.75] }
```

Held-out data is a dataset operation rather than a training-loop operation, and it is deterministic
given its seed:

```lean (name := dlSplit)
-- A seeded split assigns three of the four paired items to
-- training.
def dlSplit : Data.DatasetSplit [2] [1] :=
  Data.randomSplit 3 dlXorData (seed := 17)

#eval show IO Unit from do
  let train ← dlSplit.train.materialize (α := Float)
  let test ← dlSplit.test.materialize (α := Float)
  IO.println s!"train = {train.size}, test = {test.size}"
```

```leanOutput dlSplit
train = 3, test = 1
```

`torch.utils.data.random_split` plays the same role and takes the sizes of every part rather than
the size of the first part:

```
>>> # Request three training samples and one held-out sample
>>> # from the same dataset.
>>> tr, te = random_split(ds, [3, 1],
...     generator=torch.Generator().manual_seed(17))
>>> print("train =", len(tr), "test =", len(te))
train = 3 test = 1
```

A single held-out sample is useful for a plumbing check but gives little information about
generalization. The seeded split is reproducible only if materializations return the same ordered
samples: `train` and `test` each materialize the source separately. For a changing source, capture
one snapshot before splitting. Statistical evaluation and formal
{ref "certificates"}[certificates and bounds] answer different questions.

The displayed `3, 1` split describes membership counts, not the particular examples assigned
to either side. The seed fixes that membership only relative to an ordered source. If a file
changes between materializing `train` and `test`, applying the same permutation to its new row
positions need not select complementary examples anymore. Capturing the source once makes the
index-based split refer to one collection, which is the fact needed to interpret the two views
as training and held-out portions of the same dataset.

# Loading A Regression Table

The data-generation script writes a 25-row regression table with columns `x1,x2,y`, along with
equivalent NPY arrays:

```terminal +output
$ python3 NN/Examples/Data/generate_small_data.py
wrote NN/Examples/Data/small_regression.csv rows=25
wrote NN/Examples/Data/small_regression_X.npy shape=(25, 2) dtype=float32
wrote NN/Examples/Data/small_regression_y.npy shape=(25, 1) dtype=float32
```

The rows are a deterministic 5 by 5 grid over $`[-1,1]^2` with targets
$`y = 0.7x_1 - 0.4x_2 + 0.5x_1x_2`. This fixed target lets us compare loading paths on a
controlled regression problem:

```terminal +output
$ lake exe torchlean data_csv --device cpu --batch 5 --steps 5 --seed 2026
== CSV loader training tutorial ==
model:
Sequential: [5, 2] -> [5, 1], layers=3, params=33, state=33
  [0] Linear(2, 8): [5, 2] -> [5, 8] params=24, state=24 [[8, 2], [8]]
  [1] ReLU: [5, 8] -> [5, 8] params=0, state=0 []
  [2] Linear(8, 1): [5, 8] -> [5, 1] params=9, state=9 [[1, 8], [1]]
data_dir = NN/Examples/Data
csv_path  = NN/Examples/Data/small_regression.csv
seed      = 2026
train     = Adam(lr=0.05), steps=5, batch_size=5, shuffle=true, drop_last=true
dataset size = 5
mean_loss(before training) = 0.210192
mean_loss(after training) = 0.055578
steps=5 arithmetic=native scalar=Float32 loss=0.210192 -> 0.055578
predict(batch=heldout) = [[0.303611], [0.303611], [0.303611], [0.303611], [0.303611]]
```

`dataset size = 5` counts materialized minibatches: twenty-five source rows become five items,
each with input shape `[5, 2]` and target shape `[5, 1]`. The prediction contains five identical
rows because the probe `Tensor.full [5, 2] 0.25` repeats the same point across the batch axis.
This particular model requires that axis in its input type.

The loss is reduced over each batch item and the reported dataset mean is taken over the five
items. Because every item here contains five rows, those groups have equal weight in the report.
That interpretation depends on the full-batch policy: an arbitrary collection of unequal-sized
items would need an explicit choice between averaging item losses and weighting by their sample
counts. The printed `batch_size=5` and `drop_last=true` tell us which grouping produced this
transcript.

The loader that produced that run is six lines of
{src "NN/Examples/Data/Loaders/Csv.lean"}[`NN/Examples/Data/Loaders/Csv.lean`]:

```
-- CSV widths describe the columns per row; batchSize
-- determines the leading tensor axis.
let csvOptions : Data.CsvOptions := { skipHeader := true }
let data :=
  Data.fromCsv csvPath batchSize inputWidth outputWidth
    (csvOptions := csvOptions) (shuffle := true) (seed := seed)
Data.requireFile exeName "CSV dataset" csvPath missingCsvHint
let trained ← trainer.train data { steps := steps }
```

The two width arguments say how many columns belong to the input and how many to the target, and
a separate batch-size argument precedes those widths and becomes part of the resulting type:

```lean (name := dlFromCsvType)
-- The result already contains a batch axis in both input
-- and target shapes.
#check @Data.fromCsv
```

```leanOutput dlFromCsvType (whitespace := lax)
@Data.fromCsv : System.FilePath →
  (batchSize inputWidth targetWidth : ℕ) →
    optParam Data.CsvOptions { } →
      optParam Bool true →
        optParam ℕ 0 →
          Trainer.Dataset [batchSize, inputWidth] [batchSize, targetWidth]
```

Nothing about column meaning is inferred from the file. A header line is skipped only because
`skipHeader := true` says so, and the remaining parsing choices are the fields of
`Data.CsvOptions`:

:::table +header
*
  * Field
  * Default
  * Meaning
*
  * `delimiter`
  * `','`
  * cell separator
*
  * `skipHeader`
  * `false`
  * drop the first line before parsing rows
*
  * `trimCells`
  * `true`
  * trim ASCII whitespace around cells and rows
*
  * `allowEmptyLines`
  * `true`
  * ignore blank lines instead of failing on them
:::

Two of those defaults are deliberately permissive and one is deliberately strict. Trimming and
blank-line tolerance absorb the harmless variation that text editors and spreadsheet exports
introduce. Header skipping is off by default because guessing that the first row is a header would
silently drop a real sample from a headerless file, and a silently dropped sample is much harder to
notice than a parse error.

# File And Schema Errors

A missing file, an invalid numeric cell, and a short row fail at different stages. A missing file
is caught before parsing:

```terminal +output
$ lake exe torchlean data_csv --csv /tmp/no-such-data.csv
...
train     = Adam(lr=0.05), steps=30, batch_size=5, shuffle=true, drop_last=true
error: data_csv: missing CSV dataset: /tmp/no-such-data.csv
Generate the small regression CSV with:
  python3 NN/Examples/Data/generate_small_data.py
```

The program stops at `Data.requireFile`, with a hint naming the script that produces the file.

Now corrupt one cell. Replacing the value `-0.5` in the second data row with the text `oops` gives:

```terminal +output
error: TabularSupervisedSource.load: csv row 2, col 2: unparsed suffix: oops
```

Deleting that value instead, so the row has two fields where three are expected, gives a different
message from a later stage:

```terminal +output
error: TabularSupervisedSource.load: csv: expected 1 values for shape [1], got 0
```

The loader splits the row at the declared input width of two. Both remaining fields are consumed
as input, leaving no field for the target, so validation reports a missing target value. The
message omits the row index; locating the malformed row therefore requires another inspection of
the file.

NumPy rejects both files as well, and its width message is the better one:

```
>>> # A malformed row and a missing file fail at different
>>> # boundaries.
>>> np.loadtxt("short_row.csv", delimiter=",", skiprows=1, dtype=np.float32)
ValueError: the number of columns changed from 3 to 2 at row 2; use `usecols` to
select a subset and avoid this error
>>> np.loadtxt("does_not_exist.csv", delimiter=",")
FileNotFoundError
```

These two failures are different in kind:

- file existence is an operating-system boundary;
- row width is a data-schema boundary.

Neither is a theorem about the data-generating process. They establish that the accepted artifact
has the structure the model requested.

The short-row example shows why parsing and shape checking need different diagnostics. Every
remaining cell may be a valid number, so numeric parsing alone can succeed. Only after the
declared input width consumes the first two values does the missing target become apparent.
Increasing the model's tolerance for numeric text would not repair that schema error. Conversely,
changing the expected shape would not turn `oops` into a number. The boundary that rejects an
artifact tells the caller which part of its preparation needs attention.

# NPY And Other Numeric Sources

NPY preserves numeric dtype and array dimensions, so it is a cleaner boundary than CSV for numeric
tensors that are already prepared. A single tensor loads directly; dataset records describe paired
training data:

:::table +header
*
  * Artifact
  * Public boundary
*
  * one numeric tensor
  * `Tensor.load`
*
  * paired input and target tensors
  * `Data.SupervisedSource`
*
  * input tensors and integer labels
  * `Data.LabeledSource`
*
  * one CSV with input and target columns
  * `Data.TabularSupervisedSource`
:::

The expected result type supplies both the scalar type and the dimensions, which is why
`Tensor.load` takes no shape argument:

```lean (name := dlLoadType)
-- The expected tensor type supplies the shape that the
-- loaded payload must satisfy.
#check @Tensor.load
```

```leanOutput dlLoadType (whitespace := lax)
@Tensor.load : {α : Type} →
  [inst : Storage α] →
    [Runtime.FromFloat α] →
      {shape : Shape} →
        System.FilePath →
          optParam Data.CsvOptions { } → IO (Tensor α shape)
```

So a caller writes the shape once, in the type it wants back:

```
-- Accept this file only if its payload can be loaded as the
-- declared two-by-three tensor.
def dlMatrix : IO (Tensor Float [2, 3]) :=
  Tensor.load "data/matrix.npy"
```

The extension decides the parser, and the file metadata is checked against `shape` before the tensor
is returned. Paired training data uses a source record instead, which keeps the paths, the sample
count, and the per-sample dimensions in one value:

```lean (name := dlSourceType)
-- A file source carries the expected sample count and
-- per-sample shapes into materialization.
#check @Data.SupervisedSource.fromFiles
```

```leanOutput dlSourceType (whitespace := lax)
@Data.SupervisedSource.fromFiles : System.FilePath →
  System.FilePath →
    ℕ →
      Shape →
        Shape → optParam Data.CsvOptions { } → Data.SupervisedSource
```

Named arguments distinguish the two paths, the sample count, and the two sample shapes:

```
-- Describe the two file roles and their common sample count
-- before loading their values.
def dlSource : Data.SupervisedSource :=
  Data.SupervisedSource.fromFiles
    (inputPath := "data/x.npy") (targetPath := "data/y.npy")
    (sampleCount := 25) (input := [2]) (target := [1])

def dlNpyData : Trainer.Dataset [2] [1] :=
  Data.fromSupervisedSource dlSource
```

`Data.fromSupervisedSource` turns the source into a trainer dataset, and the two companion
constructors cover the other shapes of supervised file data: `Data.fromLabeledSource` one-hot
encodes checked integer labels, and `Data.fromCsv` splits each row at the declared input width.

## CSV, NPY, And PyTorch Data Comparison

The generator wrote the same twenty-five samples as a CSV table and as a pair of NPY arrays.
With matching training settings, the NPY run reproduces the displayed CSV losses and prediction:

```terminal +output
$ lake exe torchlean data_npy --device cpu --steps 5 --seed 2026
== NPY loader training tutorial ==
model:
Sequential: [5, 2] -> [5, 1], layers=3, params=33, state=33
  [0] Linear(2, 8): [5, 2] -> [5, 8] params=24, state=24 [[8, 2], [8]]
  [1] ReLU: [5, 8] -> [5, 8] params=0, state=0 []
  [2] Linear(8, 1): [5, 8] -> [5, 1] params=9, state=9 [[1, 8], [1]]
data_dir = NN/Examples/Data
x_path   = NN/Examples/Data/small_regression_X.npy
y_path   = NN/Examples/Data/small_regression_y.npy
seed     = 2026
train    = Adam(lr=0.05), steps=5, batch_size=5, shuffle=true, drop_last=true
X.npy dtype=<f4 shape=#[25, 2]
y.npy dtype=<f4 shape=#[25, 1]
dataset size = 5
mean_loss(before training) = 0.210192
mean_loss(after training) = 0.055578
steps=5 arithmetic=native scalar=Float32 loss=0.210192 -> 0.055578
predict(batch=heldout) = [[0.303611], [0.303611], [0.303611], [0.303611], [0.303611]]
```

The losses and prediction agree to the printed precision. Converting this input file to `float64`
also preserves the run's displayed values: its entries started as `float32` and survive the round
trip through the wider format {Informal.citep goldberg1991}[].

The three input files used from here to the end of this section are not in the repository. Two
use a dtype or memory layout that the loader does not support; the third is a widened copy of
the input. The following commands build all three from the generated `float32` array:

```terminal
# Keep the values from the original binary32 array while
# writing alternate NumPy storage dtypes.
python3 - <<'PY'
import numpy as np
X = np.load("NN/Examples/Data/small_regression_X.npy")
np.save("NN/Examples/Data/X_f64.npy", X.astype(np.float64))
np.save("NN/Examples/Data/X_i32.npy", X.astype(np.int32))
np.save("NN/Examples/Data/X_fortran.npy", np.asfortranarray(X))
PY
```

The loader takes `--x` exactly as written and does not join it to `data_dir`, so these paths are
spelled out in full:

```terminal +output
$ lake exe torchlean data_npy --steps 5 --seed 2026 \
    --x NN/Examples/Data/X_f64.npy
...
steps=5 arithmetic=native scalar=Float32 loss=0.210192 -> 0.055578
```

The loader does not issue a diagnostic for this narrowing. A `float64` source can be accepted
even when conversion to `float32` changes its values, so successful dtype validation does not
establish lossless conversion. The header records the source representation; `scalar=Float32`
records the arithmetic used for training.

For this widened copy, the source header changes but the original binary32 values remain
representable on conversion back. That explains the matching printed run without requiring all
binary64 inputs to behave the same way. A newly generated binary64 dataset could contain values
between adjacent binary32 numbers; materialization would then choose representable values in the
runtime scalar. The file dtype, conversion, and arithmetic label describe three successive steps
in the value's path from disk to a loss computation.

Selecting a different executable arithmetic keeps the dataset untouched, which is the whole reason
`materialize` is polymorphic in `α`:

```terminal +output
$ lake exe torchlean data_csv --batch 5 --steps 5 --seed 2026 --arithmetic ieee
...
steps=5 arithmetic=ieee scalar=IEEE32Exec loss=0.210192 -> 0.055578
```

One dataset value, two arithmetic implementations, and here the printed losses agree. The chapter on
{ref "floats"}[floating-point semantics] explains where the two stop agreeing and why the
`IEEE32Exec` path exists at all.

## NPY Header And Shape Errors

Three ways of feeding the NPY loader something structurally wrong, and the three messages they
produce:

```terminal +output
$ lake exe torchlean data_npy --x NN/Examples/Data/small_regression_y.npy
error: data_npy: X.npy: expected shape (N,2), got #[25, 1]

$ lake exe torchlean data_npy --x NN/Examples/Data/X_i32.npy
error: data_npy: npy: unsupported dtype: <i4

$ lake exe torchlean data_npy --x NN/Examples/Data/X_fortran.npy
error: SupervisedSource.load: npy: prefix row loading requires C-order NPY arrays
```

The first is a shape check, the second a dtype check, and the third a memory-order check. Saving
the result of `numpy.asfortranarray` sets the NPY header's `fortran_order` field to true.
TorchLean's prefix loader computes contiguous offsets for leading-axis slices, which requires
C-order storage. It rejects Fortran-order arrays at this boundary. Use
`numpy.save(path, numpy.ascontiguousarray(array))` to make the saved array C-order;
`numpy.save` can preserve Fortran order when given a Fortran-contiguous array.

TorchLean does not maintain a second parser for every ecosystem format. `.pt`, `.pth`, `.npz`, image
folders, and specialized scientific containers can be converted with Python into a small numeric
boundary such as NPY or CSV. The converter is an untrusted producer; the Lean loader checks the
resulting artifact. That division is deliberate: a Lean parser for every container format would be a
large amount of trusted code that nothing else in the library needs.

# Next-Token Datasets

Language models need a vocabulary, a tokenization rule, a context window, and a target shift. Those
are semantic choices, not generic CSV parsing.

A next-token dataset uses a context length $`L`. From the token sequence

$$`t_0,t_1,\ldots,t_n`

it forms input/target windows at starting positions for which the extra target token exists:

$$`
x_i=(t_i,\ldots,t_{i+L-1}),\qquad
y_i=(t_{i+1},\ldots,t_{i+L}).
`

Both shapes may be `[batch, L]`, so the types alone will not tell you whether the shift is there.
The shift is the learning problem {Informal.citep transformer2017}[], and TorchLean keeps it in one
named function rather than in each model's data preparation:

```lean (name := dlShift)
-- A five-token window supplies four inputs and their
-- one-position-shifted targets.
def dlWindow : Tensor Nat [5] := [10, 11, 12, 13, 14]

def dlShifted : Sample.Supervised Nat [4] [4] :=
  Data.CausalLM.tokenSample [] 4 dlWindow

#eval show IO Unit from do
  IO.println s!"input  = {dlShifted.input}"
  IO.println s!"target = {dlShifted.target}"
```

```leanOutput dlShift (whitespace := lax)
input  = [10, 11, 12, 13]
target = [11, 12, 13, 14]
```

Five tokens in, four positions out. The argument type requires one extra token, and the
implementation uses `castSucc` for inputs and `succ` for targets to preserve the intended shift:

```lean (name := dlShiftType)
-- Input and target have the same shape even though they
-- select different token positions.
#check dlShifted
```

```leanOutput dlShiftType
dlShifted : Sample.Supervised ℕ [4] [4]
```

The `+ 1` in that argument type is doing real work. A caller who has exactly `L` tokens cannot pass
them: the window has to be one longer than the sequence the model sees, and any padding needed to
reach that length is the corpus reader's decision, made before this function is called.

At input position zero the model sees token `10` and is asked to predict `11`; at position
three it sees `13` and is asked to predict `14`. The last token is a target without being the
last input, which is why five source tokens are needed. The printed type records the two
length-four tensors but not their positional relationship. That relationship comes from the
construction with `castSucc` and `succ`, so keeping it in one helper prevents two equal-shaped
slices from being accidentally exchanged.

PyTorch writes the same shift as two slices, and nothing in the shapes records which slice was
which:

```
>>> # Pair every input token with the token immediately
>>> # after it.
>>> tokens = torch.tensor([10, 11, 12, 13, 14])
>>> print("input  =", tokens[:-1].tolist())
>>> print("target =", tokens[1:].tolist())
input  = [10, 11, 12, 13]
target = [11, 12, 13, 14]
```

## Token ID Validation

An embedding lookup requires each token id to name a valid table row.
`Data.CausalLM.tokenBatch` validates the whole batch against the vocabulary size at
the corpus boundary and returns `Fin`-indexed tensors, so model code has no out-of-range case left
to handle:

```lean (name := dlTokenBatch)
-- Sample two windows and check their input and target token
-- bounds against vocabulary size 16.
#eval match Data.CausalLM.tokenBatch 16 2 4
    ([1, 2, 3, 4, 5, 6, 7, 8] : Tensor Nat [8]) 0 0 with
  | .ok sample => s!"input = {sample.input}"
  | .error e => e
```

```leanOutput dlTokenBatch
"input = [[4, 5, 6, 7], [4, 5, 6, 7]]"
```

Both rows select the same window. Offsets come from a deterministic keyed hash of the seed, step,
and row, with replacement, so different rows can select the same starting position. Repeated
windows are especially visible in this short corpus.

The bounds check applies to the sampled windows, including their extra target token. A successful
batch validates those selected identifiers; it does not scan every token in a large corpus to
certify all future batches. After validation, the `Fin vocabularySize` entries carry the bounds
that embedding lookup needs. This is also why changing the vocabulary mapping is more than
changing a model width: the same integer can acquire a different meaning even if it remains
within the accepted range.

A dataset that requires distinct windows
within a batch needs a sampler that draws without replacement.

Shrink the vocabulary below the largest id and the same call reports it instead of training on a bad
index:

```lean (name := dlTokenBad)
-- The same sampled windows contain token identifiers
-- outside the smaller vocabulary.
#eval match Data.CausalLM.tokenBatch 4 2 4
    ([1, 2, 3, 4, 5, 6, 7, 8] : Tensor Nat [8]) 0 0 with
  | .ok _ => "ok"
  | .error e => e
```

```leanOutput dlTokenBad
"tensor contains an index outside the valid range [0, 4)"
```

A tokenizer file or vocabulary mapping should therefore be stored with the run, because changing it
changes the meaning of every integer in the dataset, including whether that error fires.

## GPT-2 Byte-Level BPE Files

`NN.API.Text.Bpe` loads the conventional GPT-2 tokenizer pair directly in Lean. The pair implements
byte-pair encoding {Informal.citep sennrich2016}[] over a byte-level alphabet:

- `vocab.json` is a flat JSON object from byte-escaped token spellings to natural-number ids;
- `merges.txt` gives one adjacent symbol pair per non-comment line. File order determines rank, and
  a lower rank is applied first.

The two files form one artifact: using a vocabulary with a merge table from a different tokenizer
can change token boundaries or leave a merged piece absent from the vocabulary. Pass both paths and
record both files, preferably with content hashes, alongside a run. A malformed vocabulary or a
missing file produces an error. The merge parser rejects malformed non-comment lines with a line
number. Successful parsing still does not establish that these are the intended tokenizer files.

Encoding has four visible stages. It applies the GPT-2 pre-tokenizer, converts each fragment's UTF-8
bytes through GPT-2's reversible byte-to-Unicode table, repeatedly applies the best-ranked adjacent
merge, and finally looks up every piece in `vocab.json`. `GPT2BPE.encode` returns an error when a
piece has no id. `GPT2BPE.decode` performs the inverse vocabulary lookup and byte decoding, and
reports an unknown id or invalid UTF-8 rather than inventing text.

Here is a direct round-trip, which runs from the repository root once a matching tokenizer pair sits
at the shown paths. It is written with accented text on purpose, because that is where a byte-level
alphabet earns its keep:

```
-- Load the vocabulary and merge rules before converting
-- text into vocabulary identifiers.
import NN.API.Text.Bpe
open TorchLean.text

def inspectBpe : IO Unit := do
  let tok ← GPT2BPE.load
    "data/real/gpt2/vocab.json"
    "data/real/gpt2/merges.txt"
  match GPT2BPE.encode tok "naïve café" with
  | .error e => throw <| IO.userError e
  | .ok ids =>
      IO.println s!"ids = {repr ids}"
      match GPT2BPE.decode tok ids with
      | .error e => throw <| IO.userError e
      | .ok decoded => IO.println s!"decoded = {decoded}"
```

The generic-tokenizer adapter serves APIs whose interface cannot return tokenization errors, so it
maps failures to empty output. Use the error-reporting `GPT2BPE.encode` and `GPT2BPE.decode`
functions directly at an artifact-validation boundary.

## Unicode Tables And Tokenizer Reproducibility

GPT-2 pre-tokenization distinguishes Unicode letters, numbers, whitespace, contractions, and other
symbols. `NN.API.Text.Unicode` supplies checked-in Unicode 15.1 category ranges for `L*` and `N*`
plus the Unicode `White_Space` set. Tokenization therefore does not silently inherit an operating
system regex engine, but the checked-in tables are still a semantic dependency: category changes
between Unicode versions can change segmentation. Record the TorchLean revision as well as the BPE
files when exact reproducibility matters.

## Language Model Training With BPE

First prepare the example corpus, then run the CUDA trainer with both tokenizer paths:

```terminal
# Prepare the text corpus, then pass its path to the CUDA
# text-model training command.
python3 scripts/datasets/download_example_data.py --tiny-shakespeare

lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
  --data-file data/real/text/tiny_shakespeare.txt \
  --bpe-vocab data/real/gpt2/vocab.json \
  --bpe-merges data/real/gpt2/merges.txt \
  --allow-small-data --max-chars 20000 --steps 1 \
  --prompt "First Citizen:" --generate 0
```

The data script prepares the corpus; it does not supply the tokenizer pair. The command prints BPE
loading progress, a first shifted token window, and before/after loss. Both BPE flags are required
together. Omitting both selects the runner's byte-token path instead.

This runner is CUDA-only. It does not load OpenAI or Hugging Face model weights. Its BPE mode trains
a randomly initialized TorchLean Transformer with batch size two, a one-token context, and a local
projection of at most 512 observed GPT-2 ids; ids outside that retained set map to the local
fallback id. It exercises real file parsing, tokenization, shifted-window construction, training,
and decode plumbing, but it is neither GPT-2-small nor evidence of checkpoint-level tokenizer
equivalence.

# Tensor Batching And Gradient Accumulation

TorchLean uses “batch size” for two related operations, and confusing them is the most common way to
get a shape error out of an otherwise correct training script.

## Tensor minibatches

`Data.batch` changes the item shapes, so the batch axis becomes part of the type:

```lean (name := dlBatchType)
-- Batching pairs of items adds a leading two to both sample
-- shapes.
def dlXorBatches :=
  Data.batch 2 dlXorData (shuffle := true) (seed := 42)

#check dlXorBatches
```

```leanOutput dlBatchType (whitespace := lax)
dlXorBatches : Trainer.Dataset (Shape.prependDim [2] 2) (Shape.prependDim [1] 2)
```

The displayed type is unevaluated on purpose: `Shape.prependDim [2] 2` is definitionally `[2, 2]`,
and writing the annotation `Trainer.Dataset [2, 2] [2, 1]` on the definition is accepted and reads
better. Materializing shows both what the batching did and what the shuffle did:

```lean (name := dlBatchEval)
-- Read one shuffled batch as an input tensor and its
-- aligned target tensor.
#eval show IO Unit from do
  let stream ← dlXorBatches.materialize (α := Float)
  IO.println s!"batches = {stream.size}"
  match stream.get? 0 with
  | some batch =>
      IO.println s!"input  = {batch.input}"
      IO.println s!"target = {batch.target}"
  | none => IO.println "no batches"
```

```leanOutput dlBatchEval (whitespace := lax)
batches = 2
input  = [[1.000000, 0.000000], [0.000000, 1.000000]]
target = [[1.000000], [1.000000]]
```

Seed 42 happened to put both positive XOR examples in the first batch. That is a fair reminder that
shuffling does not stratify anything: even a uniform permutation groups the two
ones in one of the two batches with probability one third, not one half. A fixed source order,
seed, and shuffle implementation reproduce the grouping.

PyTorch reaches the same place with a `DataLoader`, and its shuffle is also seeded explicitly when
you ask for reproducibility {Informal.citep pytorch2019}[]:

```
>>> # Shuffle paired samples, then group them into batches
>>> # of two.
>>> dl = DataLoader(ds, batch_size=2, shuffle=True,
...     generator=torch.Generator().manual_seed(42))
>>> for i, (bx, by) in enumerate(dl):
...     print(f"batch {i}: x shape {list(bx.shape)} = {bx.tolist()}")
batch 0: x shape [2, 2] = [[0.0, 0.0], [1.0, 0.0]]
batch 1: x shape [2, 2] = [[0.0, 1.0], [1.0, 1.0]]
```

The permutations differ, because the two libraries use different generators, and neither ordering is
more correct than the other. What must not differ is the multiset of samples, and it does not.

The model must accept `[2, 2]` and return `[2, 1]`. One forward and backward pass processes the
whole tensor minibatch.

The first displayed batch contains two separate input rows and their two targets. Its outer
axis indexes the selected examples; its inner input axis still indexes the two XOR features.
A model accepting an unbatched `[2]` input cannot consume this value without a deliberate
change in its contract. This makes a batching error visible at the trainer boundary rather than
letting the model accidentally treat the number of examples as the feature width.

## Groups of unbatched samples

`Trainer.TrainOptions.samplesPerStep` counts dataset items per optimizer update, and it defaults to
one item:

```lean (name := dlSamplesPerStep)
-- One dataset item is the default amount consumed by each
-- optimizer update.
#eval ({ steps := 5 } : Trainer.TrainOptions).samplesPerStep
```

```leanOutput dlSamplesPerStep
1
```

On a model `[2] → [1]`, ordinary dataset items are individual samples. Every selected item is
differentiated at the same parameter point, the resulting gradient packs are averaged, and the
optimizer is applied once. The reported loss is the corresponding mean of the item losses. This path
does not insert a leading tensor axis.

For a `Data.batch`, each item already contains a fixed-size tensor minibatch. Keep
`TrainOptions.samplesPerStep := 1` to perform one vectorized device pass per update. A larger value
accumulates gradients across several tensor minibatches. On CUDA, generic accumulation currently
synchronizes the parameter pack for the host optimizer update, so one typed batch item per update is
the fast path for large batches.

# Partial Batches And Fixed Shapes

A model accepting `[5, 2]` cannot receive `[3, 2]`; these are different Lean types. The fixed-size
typed batching path therefore emits full batches only. Twenty-three samples at batch size five is
a small example that shows it, and this one is generated rather than loaded so the whole
experiment fits on the page:

```lean (name := dlRampBatches)
-- Twenty-three ordered samples reveal both full batches and
-- the dropped remainder.
def dlRamp : Trainer.Dataset [1] [1] :=
  Data.generate fun {_α} _ _ _ =>
    Array.ofFn (n := 23) fun (i : Fin 23) =>
      let x : Tensor Float [1] := [i.val.toFloat]
      let y : Tensor Float [1] := [2.0 * i.val.toFloat]
      { input := Tensor.map Runtime.ofFloat x
        target := Tensor.map Runtime.ofFloat y }

#eval show IO Unit from do
  let fivesData := Data.batch 5 dlRamp (shuffle := false)
  let thirtyData := Data.batch 30 dlRamp (shuffle := false)
  let fives ← fivesData.materialize (α := Float)
  let thirty ← thirtyData.materialize (α := Float)
  IO.println s!"batch 5  -> {fives.size} items"
  IO.println s!"samples used = {fives.size * 5}"
  IO.println s!"batch 30 -> {thirty.size} items"
```

```leanOutput dlRampBatches (whitespace := lax)
batch 5  -> 4 items
samples used = 20
batch 30 -> 0 items
```

Four full batches, twenty samples used, three samples left on the floor. PyTorch's `DataLoader`
makes the same policy a flag, and its default is the other choice:

```
drop_last=True: 4 batches, shapes [[5, 1], [5, 1], [5, 1], [5, 1]]
drop_last=False: 5 batches, shapes [[5, 1], [5, 1], [5, 1], [5, 1], [3, 1]]
batch_size=30, drop_last=True: 0 batches
```

The `[3, 1]` final batch in the second line is precisely what a shape-typed model cannot accept, so
`drop_last=False` has no counterpart here. Notice also that PyTorch agrees on the batch-size-30
case: zero batches, no error.

The empty dataset is a valid batching result, but a positive-step training request rejects it.
For example, a batch larger than this 25-row CSV cannot silently report a trained model:

```terminal +output
$ lake exe torchlean data_csv --batch 30 --steps 3 --seed 2026
...
error: Trainer.train: no training samples; check the dataset and batch size (including drop_last)
```

A zero-step report or checkpoint workflow may still use an empty dataset; its reported zero is an
empty-collection convention, not a fitted loss. Manual loader code can reject an empty epoch earlier
with a diagnostic that includes the batch size and row count:

```lean (name := dlEmptyEpoch)
-- A batch size larger than the stream cannot produce a
-- nonempty epoch with dropping enabled.
#eval show IO Unit from do
  let samples ← dlRamp.materialize (α := Float)
  let loader := Data.Loader.fromStream samples 30
  match Data.Loader.nextNonemptyEpoch "dlRamp" loader with
  | .ok epoch =>
      IO.println s!"batches = {epoch.batches.size}"
  | .error message => IO.println message
```

```leanOutput dlEmptyEpoch (whitespace := lax)
dlRamp: no full minibatch available (batch=30, rows=23)
```

The message includes both the batch size and the available row count. To retain incomplete
groups, the program needs a different data contract, such as:

- padding to five and carrying a validity mask;
- bucketing examples so each group has a fixed length;
- using a dynamic wrapper at a lower runtime layer;
- choosing a batch size that divides the dataset.

Each alternative changes the data contract, and a mask in particular changes the loss. TorchLean
refuses to silently change the shape of the last item.

Padding illustrates why retaining all rows is not just a container choice. If three real rows
are padded to five, an unmasked mean gives the two inserted rows a share of the objective.
A validity mask must therefore travel far enough to exclude them or assign the intended weight.
Dropping the incomplete group avoids inventing samples, but changes which rows contribute to
that epoch. The correct policy depends on the experiment and should be visible beside its
sample counts.

# Manual Streams And Epoch State

`Trainer.Dataset` delays the arithmetic choice. Lower-level manual code may already own a lazily
indexed stream at one selected element type:

```
-- A stream yields paired records without changing their
-- scalar type or tensor shapes.
Data.SampleStream (Sample.Supervised α input target)
```

Each item has named tensors `sample.input` and `sample.target`. `Sample.mapInput` and
`Sample.mapTarget` transform one side without exposing the heterogeneous argument pack used by the
graph runtime.

`Data.Loader.fromStream samples batch` constructs typed epoch state whose batch size appears in its
type, and `Data.Loader.nextEpoch name loader` returns the epoch's full batches together with the
loader state for the next epoch:

```lean (name := dlEpoch)
-- The returned loader carries the state needed to obtain
-- the following epoch.
#eval show IO Unit from do
  let samples ← dlRamp.materialize (α := Float)
  let loader := Data.Loader.fromStream samples 5
    (shuffle := true) (seed := 7)
  match Data.Loader.nextEpoch "dlRamp" loader with
  | .ok epoch =>
      IO.println s!"epoch batches = {epoch.batches.size}"
      IO.println s!"next seed = {epoch.nextLoader.seed}"
  | .error message => IO.println message
```

```leanOutput dlEpoch (whitespace := lax)
epoch batches = 4
next seed = 332879054
```

The loader starts with seed `7` and returns a next loader with seed `332879054`. Passing
`epoch.nextLoader` to the next epoch advances the shuffle state; reusing the original loader
would repeat its state. Epoch order remains a function of the initial seed.
Different seeds need not produce different permutations. There is no hidden global generator
whose position depends on unrelated code that happened to draw a random number earlier.

The returned seed is state for a later call, not an extra random seed to choose independently.
An epoch loop should retain `nextLoader` along with whatever records its progress through the
returned batches. Starting again from the original loader reconstructs the original epoch order;
continuing from the returned loader requests the next one. This distinction matters when a run
is interrupted: saving model weights alone does not identify which batch that loop should use
next.

Use the lower loader API for a custom epoch loop. Ordinary model training should prefer
`Data.batch`, `Data.fromCsv`, or another dataset constructor and let `trainer.train` own the loop.

# Generated Streams

Some workloads are not finite passes over a stored dataset. Physics-informed networks resample
collocation points {Informal.citep pinn2019}[], reinforcement-learning agents collect new
transitions into a replay buffer {Informal.citep dqn2015}[], and language models may generate
windows from a large file on demand. `dlRamp` above is the smallest instance of the pattern: its
samples exist only inside `materialize`, and `Data.generate` receives the arithmetic instances, so
the values are built directly in the runtime the trainer selected.

For step-indexed generation, `trainer.trainStream` takes a sample function
`Nat → Sample.Supervised Float σ τ`, and a `Trainer.Session` accepts whatever sample the caller
passes to `step`. The shapes stay fixed in the type while the values are generated or loaded lazily
at each step.

This gives the loop explicit control over the step number, the generator state, the file position,
the simulator state, and checkpoint restoration. A generated stream should log enough state to
reproduce a batch. “Seed 42” is insufficient if the generator also depends on an evolving
environment or a file cursor.

# Data Reproducibility

At minimum, record:

:::table +header
*
  * Choice
  * Example
*
  * model initialization
  * trainer seed
*
  * sample order
  * loader or shuffle seed
*
  * source identity
  * file path and preferably content hash
*
  * preprocessing
  * tokenizer, normalization, column split
*
  * batch policy
  * size, shuffling, dropping, padding
*
  * runtime
  * arithmetic semantics, backend, device
:::

The CSV example passes `2026` to both model initialization and shuffling for convenience. They are
conceptually separate choices and may be configured independently in a larger experiment. Running
the same command twice reproduces every digit, and changing the one seed changes both the
initialization and the sample order at once:

```terminal +output
$ lake exe torchlean data_csv --batch 5 --steps 5 --seed 2026
mean_loss(before training) = 0.210192
mean_loss(after training) = 0.055578
predict(batch=heldout) = [[0.303611], ...]

$ lake exe torchlean data_csv --batch 5 --steps 5 --seed 2026
mean_loss(before training) = 0.210192
mean_loss(after training) = 0.055578
predict(batch=heldout) = [[0.303611], ...]

$ lake exe torchlean data_csv --batch 5 --steps 5 --seed 7
mean_loss(before training) = 0.740953
mean_loss(after training) = 0.205089
predict(batch=heldout) = [[-0.058836], ...]
```

The gap between `0.055578` and `0.205089` after five steps reflects a combined change in
initialization and sample order. These commands do not isolate their individual effects.
Recording both seeds and repeating the experiment makes that limitation visible.

Training can write a JSON `TrainLog`:

```
-- Store the training measurements alongside a title that
-- identifies this dataset and run.
let trained ← trainer.train data
  { steps := 200
    log := .json outPath
    title := "small regression" }
```

The log is an execution artifact. It can support debugging and reproducibility, but it is not a
certificate of convergence or generalization.

For this example, a useful reading of the log is a chain of concrete identities: the paths
identify the source files, the declared widths identify the column split, the batch size explains
the dataset-item count, and the runtime label identifies the arithmetic that produced the loss.
Each entry answers a different reconstruction question. Together they make a discrepancy easier
to locate than a final loss alone, without changing the statistical meaning of that loss.

# CSV Training Example

Generate the small deterministic dataset once, then run the maintained loader-and-trainer example:

```terminal
# Generate the small regression files before invoking the
# example that reads their CSV rows.
python3 NN/Examples/Data/generate_small_data.py
lake exe torchlean data_csv \
  --device cpu --batch 5 --steps 5 --seed 2026
```

It exercises CSV parsing, deterministic shuffling, fixed-size typed batching, model execution,
autograd, Adam, and prediction. Swap `data_csv` for `data_npy` to run the same twenty-five samples
through the NPY boundary instead.

Sources:

- [`NN/API/Data/README.md`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Data/README.md);
- {src "NN/Examples/Data/README.md"}[`NN/Examples/Data/README.md`];
- [`Bpe.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Text/Bpe.lean);
- [`Unicode.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Text/Unicode.lean);
- {src "NN/Examples/Models/Sequence/TextGpt2.lean"}[`TextGpt2.lean`];
- [`Npy.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean);
- {src "NN/Examples/Data/Loaders/Cifar10Images.lean"}[`Cifar10Images.lean`].
