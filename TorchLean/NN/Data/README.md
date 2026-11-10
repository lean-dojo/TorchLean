# `TorchLean/NN/Data`

This folder holds the `Data` namespace, in the modules `TorchLean.NN.Data.*`: the constructors
and file sources for `Trainer.Dataset`, the one supervised-data type of application code:

```lean
Trainer.Dataset input target
```

The type carries the sample shapes and delays scalar conversion until the trainer picks its
runtime. `Data.fromTensors`, `Data.fromSamples`, `Data.fromSupervisedSource` and the other
constructors all return it.

For data already represented as Lean values:

- `Data.fromTensors xs ys` splits two batched `Float` tensors along their leading axis.
- `Data.fromSamples values` wraps an array of concrete `Float` samples.
- `Data.fromStream samples` wraps a lazily indexed stream of concrete `Float` samples.
- `Data.fromSample sample` wraps one concrete `Float` sample.
- `Data.defer action` materializes one concrete `Float` sample from an `IO` action.
- `Data.generate builder` constructs samples using the arithmetic selected by the trainer.

`Data.CausalLM.byteSamples` returns a `SampleStream`. With `α := Float`, pass it to
`Data.fromStream`, which builds each sample and converts it to the trainer's scalar on access.
`Data.fromSamples` takes an array of samples.

Manual loops that already own a runtime scalar can import `TorchLean.NN.Data.SampleStream`. A sample
stream is finite and lazily indexed: wrapping an array doesn't copy it, and slicing a batched tensor
happens when a sample is requested.

Fixed-size manual loops can turn that stream into a typed loader:

```lean
def loader : Data.Loader Float 32 [features] [targets] :=
  Data.Loader.fromStream samples 32 (shuffle := true) (seed := 7)
```

`Data.Loader.nextEpoch` returns full tensor batches and the loader state for the next deterministic
epoch. The batch size is part of the type, so the loader drops a final partial batch and has no
`dropLast` option that could contradict its result type. An empty stream or a zero batch size is an
error. A nonempty stream shorter than the batch gives an epoch with no batch, and
`(requireBatch := true)` rejects that case. `Data.collateStream` and `Data.batch` build batches on
access. Use `Data.Loader.firstBatch` to get one batch without building the rest of the epoch.

The data boundary reads three formats:

- `.npy` for numeric tensors
- numeric CSV for small tabular data
- UTF-8 text for language-model examples

For other formats, use the converter:

```bash
python3 scripts/datasets/torchlean_data_convert.py --help
```

TorchLean doesn't try to own every dataset format in the ML ecosystem. Python handles those
formats and writes a simple boundary file, and TorchLean checks its shape, element type, batch
structure and labels before the data reaches a trainer, graph export or verifier.

## Modules

- `Dataset.lean`: the `Trainer.Dataset` constructors (`fromTensors`, `fromSamples`, `generate`,
  `fromCSV`, `fromSupervisedSource`, `fromLabeledSource`) and the transforms `batch` and
  `randomSplit`.
- `SampleStream.lean`: `SampleStream α`, a finite stream that computes each sample on request.
- `Loader.lean`: `collate`, `collateStream` and the typed `Loader`.
- `Source.lean`: `SupervisedSource`, `LabeledSource`, `TabularSupervisedSource` and `Tensor.load`.
- `NPY.lean`, `CSV.lean` and `Parsing.lean`: the `.npy` and numeric CSV readers, and the small
  parsers they share with the command line and the verification text readers.
- `Monad.lean`: the `LawfulMonadAttach` instances of `EST` and `EIO`, so that the values an `IO`
  reader returns can be proved from its code.
- `CausalLM.lean`: causal language-model samples from token tensors and UTF-8 bytes
  (`tokenSample`, `tokenBatch`, `byteSample`, `byteSamples`, `byteBatch`).
- `Synthetic.lean`: deterministic tabular grids (`cartesianGrid`, `linspace`, `squareGrid`).
- `Image.lean`: writes normalized channel-first RGB tensors as PPM files.

The sample records `Sample.Supervised` and `Sample.Batch` are one level up, in
`TorchLean/NN/Sample.lean`.

## Data Sources

| Data shape | Use |
| --- | --- |
| one tensor file | `Tensor.load` |
| supervised `X.npy`, `Y.npy` | `Data.SupervisedSource` |
| image/classification labels | `Data.LabeledSource` |
| small numeric CSV | `Data.TabularSupervisedSource` |
| fixed-size tensor batches | `Data.batch` |
| generated or file-backed batches | typed step streams through the trainer API |
| text windows | `Text` tokenizers, then `Data.CausalLM` bounded-token samples |

The main Lean entry points are:

- `Tensor.load`: one NPY or numeric CSV file, with scalar type and shape selected by the result
  type.
- `Data.SupervisedSource`: two batched tensors with named input and target shapes.
- `Data.LabeledSource`: batched inputs plus labels checked on load and one-hot encoded on access.
- `Data.TabularSupervisedSource`: one CSV where each row contains `x..., y...`.
- `Data.batch`: fixed-size tensor minibatches represented as another `Trainer.Dataset`.

Load a single tensor without a source wrapper:

```lean
def matrix : IO (Tensor Float [2, 3]) :=
  Tensor.load "data/matrix.npy"
```

The file metadata must match `[2, 3]`, or `Tensor.load` raises a descriptive `IO` error.

NPY payloads are decoded into unboxed `FloatArray` storage and wrapped as tensors without a boxed
intermediate. The low-level readers `Data.NPY.readBlocks` and `Data.NPY.parseBlocks` can
decode C-order leading-axis blocks concurrently, with a `Parallelism` setting that defaults to
`.auto`, and return the blocks in file order. They reject scalar and Fortran-order files, which
the full reader handles. Signed zeros and finite values are preserved, and Lean's float
conversion canonicalizes NaNs.

A concrete supervised sample is a named tensor record:

```lean
def sample : Sample.Supervised Float [2] [1] :=
  { input := [1.0, 2.0]
    target := [3.0] }
```

Use `sample.input`, `sample.target`, `Sample.mapInput`, and `Sample.mapTarget` for preprocessing.
Manual stream code may use `SampleStream.map`, which transforms samples only when they are
requested.

## Boundary Discipline

Every loader should make these facts visible:

- the number of samples,
- the input shape after removing the batch axis,
- the target or label shape,
- the arithmetic interpretation,
- whether labels are one-hot encoded or integer ids,
- whether shuffling is deterministic and which seed controls it.

A classifier margin certificate, a PINN dataset containment check or an FNO prediction artifact
depends on the data shape and ordering supplied to its checker, so the loader is part of the trust
boundary. It makes the imported tensor shape, label interpretation and sample order visible before
the data reaches a trainer or checker. It doesn't certify that an external dataset is
scientifically correct.

## Common Paths

| Workflow | Typical data path |
| --- | --- |
| MLP/KAN tabular examples | numeric CSV or `X.npy`/`Y.npy` with shape `(N, features)` and `(N, targets)` |
| CNN/ViT examples | image tensors `(N, C, H, W)` plus label vector `(N,)` |
| GPT/Mamba examples | UTF-8 text, tokenizer assets, bounded-token windows, shifted targets |
| FNO Burgers | `.mat` or simulator output converted to `X.npy`/`Y.npy`, plus metadata for grid and split |
| PINN checks | JSON certificates and samples checked in `TorchLeanExamples/Verification/PINN` |
| VNN-COMP-style checks | weight and suite JSON read by `TorchLeanExamples/Verification/VNNComp` |

One boundary file can feed several layers. A supervised `.npy` pair can train a model, export
predictions and later provide a verification fixture, so the loader has to make shape and
provenance obvious before those later claims are made.

## What To Document

When adding a new data path, document:

- the external source or producer
- the command that creates the boundary file
- the expected tensor shapes after conversion
- label encoding and class count, if applicable
- whether shuffling is deterministic
- where generated logs, predictions or manifests are written

If the source is a public dataset or third-party artifact, update `docs/THIRD_PARTY_NOTICES.md`.
If the data feeds a checker, also update the relevant `TorchLean/Verify` or
`TorchLeanExamples/Verification` README so the checked predicate names the data boundary it relies
on.

Examples and conversion recipes live in:

```text
TorchLeanExamples/Data/README.md
```
