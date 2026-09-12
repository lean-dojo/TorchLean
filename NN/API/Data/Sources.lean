/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Arithmetic -- shake: keep
public import NN.API.Sample -- shake: keep
public import NN.Data.SampleStream -- shake: keep
public import NN.Data.IO.Csv -- shake: keep
public import NN.Data.IO.Npy -- shake: keep
public import NN.Tensor.Constructors -- shake: keep
public import NN.Tensor.Conversion -- shake: keep
public import NN.Data.IO -- shake: keep

/-!
# File-Backed Dataset Sources

Typed NPY and CSV sources for supervised and labeled datasets. This module owns the boundary
between external files and TorchLean tensors; importing `Data.SampleStream` does not pull
in these parsers.
-/

@[expose] public section

namespace TorchLean
namespace Data

export TorchLean.Data.IO
  (CsvOptions readCsvFloatRows readNpy readNpyLeadingAxisPrefix)

/-- Require that every path exists, adding `hint` to a missing-file error when supplied. -/
def requireFiles (exeName : String) (paths : Array System.FilePath) (hint : String := "") :
    IO Unit := do
  for path in paths do
    unless (← path.pathExists) do
      let suffix := if hint.isEmpty then "" else "\n" ++ hint
      throw <| IO.userError s!"{exeName}: missing data file: {path}{suffix}"

/--
Require one named data file to exist.

Example:
```lean
-- A missing fixture should fail with a sentence the reader can act on.
def checkFixtures : IO Unit :=
  Data.requireFile "cifar10_images" "input tensor" "data/cifar_x.npy"
    (hint := "run: python3 NN/Examples/Data/generate_small_data.py")
```
-/
def requireFile
    (exeName : String) (label : String) (path : System.FilePath) (hint : String := "") :
    IO Unit := do
  unless (← path.pathExists) do
    let suffix := if hint.isEmpty then "" else "\n" ++ hint
    throw <| IO.userError s!"{exeName}: missing {label}: {path}{suffix}"

/-- Require paired supervised input and target files to exist. -/
def requirePairedFiles
    (exeName : String)
    (inputLabel : String) (inputPath : System.FilePath)
    (targetLabel : String) (targetPath : System.FilePath)
    (hint : String := "") : IO Unit := do
  requireFile exeName inputLabel inputPath hint
  requireFile exeName targetLabel targetPath hint

/--
Read an `.npy` row count from its header while checking all trailing dimensions.

For shape `(N, d₁, ..., dₖ)`, this returns `N` exactly when the trailing shape is `tailShape`.
The payload is not read; the subsequent full or prefix load checks its required byte range.
-/
def availableNpyRows
    (path : System.FilePath) (tailShape : Shape) (expectedDesc : String) :
    IO Nat := do
  match ← TorchLean.Data.IO.readNpyHeader path with
  | .error message =>
      throw <| IO.userError s!"Data.availableNpyRows: {message}"
  | .ok data =>
      match data.shape[0]? with
      | some rows =>
          let trailingShape := data.shape.extract 1 data.shape.size
          if trailingShape.toList = tailShape.toList then
            pure rows
          else
            throw <| IO.userError
              s!"Data.availableNpyRows: expected {expectedDesc}, got {data.shape}"
      | none =>
          throw <| IO.userError
            s!"Data.availableNpyRows: expected {expectedDesc}, got {data.shape}"

/--
Validate untrusted flat payload length before crossing into total tensor code.

`Tensor.from` itself is intentionally total; only external metadata can fail.
-/
def Internal.tensorFromFlat
    (tag : String) (dims : Shape) (values : Array Float) :
    Except String (Tensor Float dims) :=
  if h : values.size = dims.size then
    .ok <| (Tensor.from values).reshape dims (by
      simpa [Spec.Shape.size] using h)
  else
    .error s!"{tag}: expected {dims.size} values for shape {dims}, got {values.size}"

namespace Internal

/-- How a file's physical shape is matched against the requested tensor shape. -/
inductive TensorShapeMatch where
  /-- Require every physical dimension to equal the requested dimension. -/
  | exact
  /-- Permit a larger first dimension and read its requested prefix. -/
  | leadingPrefix
deriving BEq, Repr

end Internal

/--
Load an arbitrary-rank tensor from a `.npy` file under an explicit shape-matching policy.

With `exact`, every dimension must match. With `leadingPrefix`, rank and trailing dimensions must
still match, while the file may contain more entries on its first axis. Prefix loading requires a
C-order NPY file because the requested values must form one contiguous prefix.
-/
def Internal.readNpyTensor (path : System.FilePath) (dims : Shape)
    (shapeMatch : Internal.TensorShapeMatch := .exact) :
    IO (Except String (Tensor Float dims)) := do
  match shapeMatch with
  | .exact =>
      let res ← readNpy path
      match res with
      | .error e => pure (.error e)
      | .ok data =>
          if data.shape.toList != dims.toList then
            pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")
          else
            pure <| Internal.tensorFromFlat "npy" dims data.values
  | .leadingPrefix =>
      let res ← readNpyLeadingAxisPrefix path dims.toArray
      match res with
      | .error e => pure (.error e)
      | .ok data => pure <| Internal.tensorFromFlat "npy" dims data.values

/-- Parse a float-encoded class label as a `Nat` in `[0, classes)`. -/
private def finLabelOfFloat (tag : String) (classes : Nat) (x : Float) :
    Except String (Fin classes) := do
  let n : Nat := x.toUInt64.toNat
  if (n : Float) != x then
    throw s!"{tag}: expected an integer class label, got {x}"
  else if h : n < classes then
    pure ⟨n, h⟩
  else
    throw s!"{tag}: class label {n} out of range (classes={classes})"

/--
Labeled dataset from a batched tensor `X : (n, σ)` and a label vector `y : (n,)`.

Labels are stored as floats (common when exporting from NumPy); we validate each label is an
integer in `[0, classes)` before returning the stream. Input conversion and one-hot encoding
happen only when a sample is requested.
-/
private def labeledFromLeadingAxis {α : Type} [TorchLean.Storage α]
    [Context α]
    [TorchLean.Runtime.FromFloat α]
    (tag : String) (classes : Nat)
    {n : Nat} {σ : Shape}
    (X : Tensor Float (σ.prependDim n))
    (y : Tensor Float [n]) :
    Except String
      (SampleStream (TorchLean.Sample.Supervised α σ [classes])) := do
  let labels ← Tensor.mapM (finLabelOfFloat tag classes) y
  pure <| SampleStream.fromFunction n fun i =>
    { input := Tensor.map Runtime.ofFloat (Spec.get X i)
      target := TorchLean.Tensor.oneHot classes labels[i] }

/--
Load a supervised dataset from a CSV with `inputWidth + targetWidth` columns per row:

`input_1, ..., input_n, target_1, ..., target_m`.
-/
private def readCsvSupervised {α : Type} [TorchLean.Storage α] [Context α]
    [TorchLean.Runtime.FromFloat α]
    (path : System.FilePath) (inputWidth targetWidth : Nat) (options : CsvOptions := {}) :
    IO (Except String (SampleStream
      (TorchLean.Sample.Supervised α [inputWidth] [targetWidth]))) := do
  let rowsResult ← readCsvFloatRows path options
  match rowsResult with
  | .error e => pure (.error e)
  | .ok rows =>
      let samplesResult :
          Except String
            (Array (TorchLean.Sample.Supervised Float [inputWidth] [targetWidth])) :=
        rows.mapM (fun row => do
          let inputValues := row.take inputWidth
          let targetValues := row.drop inputWidth
          let input ← Internal.tensorFromFlat "csv" [inputWidth] inputValues
          let target ← Internal.tensorFromFlat "csv" [targetWidth] targetValues
          pure { input, target })
      pure <| samplesResult.map (fun samples =>
        SampleStream.fromArray <| samples.map fun sample =>
          Sample.map
            (Tensor.map Runtime.ofFloat)
            (Tensor.map Runtime.ofFloat)
            sample)

/-!
## File Sources

The definitions below load tensors by path and expected dimensions, then build supervised or
labeled datasets by slicing the leading batch axis, just like PyTorch `TensorDataset`.

Policy for external ecosystems:
- NumPy `.npy` is the canonical interchange format for numeric tensors.
- CSV is supported for small tabular data.
- MATLAB `.mat`, PyTorch checkpoints, HDF5, Parquet, and image archives should be converted by a
  small preparation script into `.npy` tensors plus metadata. The Lean runtime loader intentionally
  handles a small deterministic interchange format rather than every external binary format.
-/

/-- File encodings inferred from source-path extensions. -/
inductive Internal.FileFormat where
  /-- NumPy `.npy`, supporting numeric C-order arrays decoded by TorchLean's NPY reader. -/
  | npy
  /-- Numeric CSV table. CSV sources are interpreted as 2D tensors `[rows, cols]`. -/
  | csv
deriving BEq, Repr

/-- Infer the supported file encoding from a source path. -/
def Internal.fileFormat (path : System.FilePath) : Except String Internal.FileFormat :=
  match path.extension.map String.toLower with
  | some "npy" => .ok .npy
  | some "csv" => .ok .csv
  | some ext =>
      .error s!"data source: unsupported extension '.{ext}' for {path}; expected .npy or .csv"
  | none =>
      .error s!"data source: {path} has no extension; expected .npy or .csv"

/--
Load a numeric CSV table as a tensor.

Supported shapes:
- `[rows, cols]`: ordinary numeric table,
- `[n]`: either one column with `n` rows or one row with `n` columns.
-/
def Internal.loadCsvTensor
    (path : System.FilePath) (dims : Shape) (options : CsvOptions := {}) :
    IO (Except String (Tensor Float dims)) := do
  match hDims : dims with
  | .dim rowsExpected (.dim colsExpected .scalar) =>
      let rowsRes ← readCsvFloatRows path options
      match rowsRes with
      | .error e => pure (.error e)
      | .ok rows =>
          if rows.size != rowsExpected then
            pure (.error s!"csv: expected {rowsExpected} rows, got {rows.size}")
          else
            let bad? := rows.zipIdx.find? (fun (row, _i) => row.size != colsExpected)
            match bad? with
            | some (row, i) =>
                pure (.error s!"csv: row {i + 1}: expected {colsExpected} columns, got {row.size}")
            | none =>
                let flat := rows.foldl (fun acc row => acc ++ row) #[]
                pure <|
                  (Internal.tensorFromFlat "csv" [rowsExpected, colsExpected] flat).map
                    (fun tensor => by simpa [hDims] using tensor)
  | .dim n .scalar =>
      let rowsRes ← readCsvFloatRows path options
      match rowsRes with
      | .error e => pure (.error e)
      | .ok rows =>
          let flat? : Except String (Array Float) :=
            if rows.size = n then
              rows.zipIdx.mapM fun (row, i) =>
                match row[0]? with
                | some value =>
                    if row.size = 1 then pure value
                    else throw s!"csv: row {i + 1}: expected one column, got {row.size}"
                | none => throw s!"csv: row {i + 1}: expected one column, got 0"
            else
              match rows[0]? with
              | some row =>
                  if rows.size = 1 then
                    if row.size = n then .ok row
                    else .error s!"csv: expected one row with {n} columns, got {row.size}"
                  else .error s!"csv: expected {n} values as one column or one row"
              | none => .error s!"csv: expected {n} values as one column or one row"
          match flat? with
          | .error e => pure (.error e)
          | .ok flat =>
              pure <| (Internal.tensorFromFlat "csv" [n] flat).map
                (fun tensor => by simpa [hDims] using tensor)
  | _ =>
      pure (.error s!"csv: expected shape=[rows, cols] or shape=[n], got {dims}")

/-- Load a Float tensor from a path and expected dimensions. -/
def Internal.loadFloatTensor (path : System.FilePath)
    (dims : Shape) (options : CsvOptions := {})
    (shapeMatch : Internal.TensorShapeMatch := .exact) :
    IO (Except String (Tensor Float dims)) := do
  match Internal.fileFormat path with
  | .error e => pure (.error e)
  | .ok .npy => Internal.readNpyTensor path dims shapeMatch
  | .ok .csv => Internal.loadCsvTensor path dims options

end Data

namespace Tensor

/--
Load a tensor from NPY or numeric CSV.

The expected result type supplies both the scalar type and shape. File metadata is checked before
the tensor is returned.
-/
opaque load {α : Type} [TorchLean.Storage α] [TorchLean.Runtime.FromFloat α]
    {shape : Shape} (path : System.FilePath) (csvOptions : Data.CsvOptions := {}) :
    IO (Tensor α shape) := do
  let loaded ← Data.Internal.loadFloatTensor path shape csvOptions
  match loaded with
  | .ok tensor => pure <| Tensor.map Runtime.ofFloat tensor
  | .error message =>
      throw <| IO.userError s!"Tensor.load: {message}"

end Tensor

namespace Data

/--
Two files representing supervised data:
- `inputPath` must contain shape `(sampleCount, input...)`,
- `targetPath` must contain shape `(sampleCount, target...)`.
-/
structure SupervisedSource where
  /-- Number of samples along the leading batch axis. -/
  sampleCount : Nat
  /-- Shape of one input sample. -/
  input : Shape
  /-- Shape of one target sample. -/
  target : Shape
  /-- Path to the batched input tensor. -/
  inputPath : System.FilePath
  /-- Path to the batched target tensor. -/
  targetPath : System.FilePath
  /-- CSV parsing options, ignored for NPY sources. -/
  csvOptions : CsvOptions := {}

namespace SupervisedSource

/--
Describe paired input and target tensor files. Encodings are inferred from their extensions.

Example:
```lean
-- Shapes are per sample; `sampleCount` says how many leading rows of the files this run uses.
def source : Data.SupervisedSource :=
  Data.SupervisedSource.fromFiles
    (inputPath := "data/inputs.npy") (targetPath := "data/targets.npy")
    (sampleCount := 64) (input := [16]) (target := [1])
```
-/
def fromFiles (inputPath targetPath : System.FilePath)
    (sampleCount : Nat) (input target : Shape)
    (csvOptions : CsvOptions := {}) : SupervisedSource :=
  { sampleCount, input, target, inputPath, targetPath, csvOptions }

/--
Load a supervised dataset by slicing the leading batch axis from the two tensors.

This is the preferred public loader for regression/operator-learning examples, regardless of
whether the backing files are `.npy` or small numeric CSV tables.
-/
opaque load {α : Type} [TorchLean.Storage α]
    [TorchLean.Runtime.FromFloat α] (source : SupervisedSource) :
    IO (SampleStream
      (TorchLean.Sample.Supervised α source.input source.target)) := do
  -- `sampleCount` is the number of rows used in this run. NPY files may contain more rows; CSV
  -- sources still require the requested shape exactly.
  let inputResult ← Internal.loadFloatTensor source.inputPath
    (source.input.prependDim source.sampleCount)
    source.csvOptions .leadingPrefix
  let targetResult ← Internal.loadFloatTensor source.targetPath
    (source.target.prependDim source.sampleCount)
    source.csvOptions .leadingPrefix
  match inputResult with
  | .error message =>
      throw <| IO.userError s!"SupervisedSource.load: {message}"
  | .ok inputs =>
      match targetResult with
      | .error message =>
          throw <| IO.userError s!"SupervisedSource.load: {message}"
      | .ok targets =>
          let typedInputs : Tensor Float (source.input.prependDim source.sampleCount) := inputs
          let typedTargets : Tensor Float (source.target.prependDim source.sampleCount) := targets
          pure <| SampleStream.fromFunction source.sampleCount fun sampleIndex =>
            { input := Tensor.map Runtime.ofFloat (Spec.get typedInputs sampleIndex)
              target := Tensor.map Runtime.ofFloat (Spec.get typedTargets sampleIndex) }

end SupervisedSource

/--
Two files representing labeled classification data:
- `inputPath` must contain shape `(sampleCount, input...)`,
- `labelPath` must contain shape `(sampleCount,)` and integer-valued class labels.
-/
structure LabeledSource where
  /-- Number of samples along the leading batch axis. -/
  sampleCount : Nat
  /-- Shape of one input sample. -/
  input : Shape
  /-- Number of classes for one-hot targets. -/
  classCount : Nat
  /-- Path to the batched input tensor. -/
  inputPath : System.FilePath
  /-- Path to the label vector. -/
  labelPath : System.FilePath
  /-- CSV parsing options, ignored for NPY sources. -/
  csvOptions : CsvOptions := {}

namespace LabeledSource

/-- Describe input and class-label tensor files. Encodings are inferred from their extensions. -/
def fromFiles (inputPath labelPath : System.FilePath)
    (sampleCount : Nat) (input : Shape) (classCount : Nat)
    (csvOptions : CsvOptions := {}) : LabeledSource :=
  { sampleCount, input, classCount, inputPath, labelPath, csvOptions }

/--
Load a labeled classification dataset by slicing the leading batch axis and one-hot encoding labels.

CSV label vectors may be stored as one column or one row; NPY label vectors have shape `[n]`.
-/
opaque load {α : Type} [TorchLean.Storage α]
    [Context α] [TorchLean.Runtime.FromFloat α]
    (source : LabeledSource) :
    IO (SampleStream
      (TorchLean.Sample.Supervised α source.input [source.classCount])) := do
  -- Labels use the same prefix-row convention as supervised tensors. This lets one full exported
  -- label vector back different bounded runs without making separate copies on disk.
  let inputResult ← Internal.loadFloatTensor source.inputPath
    (source.input.prependDim source.sampleCount)
    source.csvOptions .leadingPrefix
  let labelResult ← Internal.loadFloatTensor source.labelPath [source.sampleCount]
    source.csvOptions .leadingPrefix
  match inputResult with
  | .error message =>
      throw <| IO.userError s!"LabeledSource.load: {message}"
  | .ok inputs =>
      match labelResult with
      | .error message =>
          throw <| IO.userError s!"LabeledSource.load: {message}"
      | .ok labels =>
          let typedInputs : Tensor Float (source.input.prependDim source.sampleCount) := inputs
          let typedLabels : Tensor Float [source.sampleCount] := labels
          match labeledFromLeadingAxis (α := α) (σ := source.input)
              "LabeledSource.load" source.classCount typedInputs typedLabels with
          | .ok samples => pure samples
          | .error message =>
              throw <| IO.userError message

end LabeledSource

/--
Single-table supervised CSV source.

Use this when one CSV row contains both input and target columns:
`x1, ..., x_inputWidth, y1, ..., y_targetWidth`.
-/
structure TabularSupervisedSource where
  /-- CSV file path. -/
  path : System.FilePath
  /-- Number of input feature columns. -/
  inputWidth : Nat
  /-- Number of target columns. -/
  targetWidth : Nat
  /-- CSV parsing options. -/
  csvOptions : CsvOptions := {}

namespace TabularSupervisedSource

/-- Describe one CSV table whose rows contain input columns followed by target columns. -/
def fromCsv (path : System.FilePath) (inputWidth targetWidth : Nat)
    (csvOptions : CsvOptions := {}) :
    TabularSupervisedSource :=
  { path, inputWidth, targetWidth, csvOptions }

/-- Load a single-table supervised CSV source. -/
opaque load {α : Type} [TorchLean.Storage α]
    [Context α] [TorchLean.Runtime.FromFloat α]
    (source : TabularSupervisedSource) :
    IO (SampleStream
      (TorchLean.Sample.Supervised α [source.inputWidth] [source.targetWidth])) := do
  match ← readCsvSupervised (α := α) source.path source.inputWidth source.targetWidth
      source.csvOptions with
  | .ok samples => pure samples
  | .error message =>
      throw <| IO.userError s!"TabularSupervisedSource.load: {message}"

end TabularSupervisedSource

end Data
end TorchLean
