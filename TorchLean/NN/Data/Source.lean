/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.NN.Arithmetic -- shake: keep
public import TorchLean.NN.Sample -- shake: keep
public import TorchLean.NN.Data.SampleStream -- shake: keep
public import TorchLean.NN.Data.CSV -- shake: keep
public import TorchLean.NN.Data.NPY -- shake: keep
public import TorchLean.Tensor.Constructors
public import TorchLean.Tensor.Conversion

/-!
# File-Backed Dataset Sources

Typed NPY and CSV sources for supervised and labeled datasets. External files become TorchLean
tensors here, and `TorchLean.NN.Data.SampleStream` does not import these parsers.
`SupervisedSource`, `LabeledSource` and `TabularSupervisedSource` describe datasets on disk, and
`Tensor.load` reads one tensor.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace Data

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
    (hint := "run: python3 TorchLeanExamples/Data/generate_small_data.py")
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
Read the row count of an `.npy` file from its header, checking the trailing dimensions.

For a file of shape `(N, d₁, ..., dₖ)` this returns `N` when `(d₁, ..., dₖ)` is `tailShape` and
raises an error otherwise. The payload is not read, and the later full or prefix load checks the
byte range it needs.
-/
def NPY.availableRows
    (path : System.FilePath) (tailShape : Shape) (expectedDesc : String) :
    IO Nat := do
  match ← Data.NPY.readHeader path with
  | .error message =>
      throw <| IO.userError s!"Data.NPY.availableRows: {message}"
  | .ok data =>
      match data.shape[0]? with
      | some rows =>
          let trailingShape := data.shape.extract 1 data.shape.size
          if trailingShape.toList = tailShape then
            pure rows
          else
            throw <| IO.userError
              s!"Data.NPY.availableRows: expected {expectedDesc}, got {data.shape}"
      | none =>
          throw <| IO.userError
            s!"Data.NPY.availableRows: expected {expectedDesc}, got {data.shape}"

/--
Check an untrusted flat payload's length against `dims`, then reshape it into a tensor.

`Tensor.from` is total, so this length check is where bad external metadata is caught.
-/
def tensorFromFlat
    (tag : String) (dims : Shape) (values : Array Float) :
    Except String (Tensor Float dims) :=
  if h : values.size = dims.size then
    .ok <| (Tensor.from values).reshape dims (by
      simpa using h)
  else
    .error s!"{tag}: expected {dims.size} values for shape {dims}, got {values.size}"

/--
Check an unboxed flat payload's length against `dims`, then reshape it into a tensor.

`Tensor.from` on a `FloatArray` wraps the buffer in place, so a payload the NPY loader
decoded unboxed becomes a tensor without a copy or a boxed intermediate.
-/
def tensorFromFloatArray
    (tag : String) (dims : Shape) (values : FloatArray) :
    Except String (Tensor Float dims) :=
  -- The length is read through the storage instance, which is how the tensor states its shape.
  if h : @TorchLean.Storage.size Float TorchLean.instFloatStorage values = dims.size then
    .ok <| (Tensor.from values).reshape dims (by
      show Shape.size [@TorchLean.Storage.size Float TorchLean.instFloatStorage values]
        = dims.size
      simpa using h)
  else
    .error s!"{tag}: expected {dims.size} values for shape {dims}, got \
      {@TorchLean.Storage.size Float TorchLean.instFloatStorage values}"

/-- How a file's physical shape is matched against the requested tensor shape. -/
inductive TensorShapeMatch where
  /-- Require every physical dimension to equal the requested dimension. -/
  | exact
  /-- Permit a larger first dimension and read its requested prefix. -/
  | leadingPrefix
deriving BEq, Repr

/--
Load a tensor of any rank from a `.npy` file, matching the file's shape by `shapeMatch`.

With `exact`, every dimension must match. With `leadingPrefix`, the rank and trailing dimensions
must match and the file may hold more entries on its first axis. Prefix loading requires a C-order
NPY file because the requested values must form one contiguous prefix.
-/
def NPY.readTensor (path : System.FilePath) (dims : Shape)
    (shapeMatch : Data.TensorShapeMatch := .exact) :
    IO (Except String (Tensor Float dims)) := do
  match shapeMatch with
  | .exact =>
      let res ← read path
      match res with
      | .error e => pure (.error e)
      | .ok data =>
          if data.shape.toList != dims then
            pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")
          else
            pure <| Data.tensorFromFloatArray "npy" dims data.values
  | .leadingPrefix =>
      let res ← readPrefix path dims.toArray
      match res with
      | .error e => pure (.error e)
      | .ok data => pure <| Data.tensorFromFloatArray "npy" dims data.values

/-- Parse a float class label as a `Fin classes`, rejecting non-finite and non-integral floats and
labels outside `[0, classes)` or at or above `2 ^ 64`. Each error message prints the float. -/
private def finLabelOfFloat (tag : String) (classes : Nat) (x : Float) :
    Except String (Fin classes) := do
  if !x.isFinite || x.floor != x then
    throw s!"{tag}: expected an integer class label, got {x}"
  -- An integral float below `2 ^ 64` converts to `UInt64` exactly.
  let n : Nat := x.toUInt64.toNat
  if h : 0 ≤ x ∧ x < 18446744073709551616 ∧ n < classes then
    pure ⟨n, h.2.2⟩
  else
    throw s!"{tag}: class label {x} out of range (classes={classes})"

/--
Labeled dataset from a batched tensor `X : (n, σ)` and a label vector `y : (n,)`.

Labels arrive as floats (common in NumPy exports), and we check that each is an integer in
`[0, classes)` before returning the stream. Input conversion and one-hot encoding happen only when
a sample is requested.
-/
private def classificationSamples {α : Type} [TorchLean.Storage α]
    [Zero α] [One α]
    [FromFloat α]
    (tag : String) (classes : Nat)
    {n : Nat} {σ : Shape}
    (X : Tensor Float (Shape.dim n σ))
    (y : Tensor Float [n]) :
    Except String
      (SampleStream (TorchLean.Sample.Supervised α σ [classes])) := do
  let labels ← Tensor.mapM (finLabelOfFloat tag classes) y
  pure <| SampleStream.mk n fun i =>
    { input := Tensor.map ofFloat (Tensor.unstack X i)
      target := TorchLean.Tensor.oneHot classes labels[i] }

/--
Load a supervised dataset from a CSV whose rows hold `inputWidth` input columns followed by
`targetWidth` target columns. A row of any other width is an error.
-/
private def CSV.readSupervised {α : Type} [TorchLean.Storage α]
    [FromFloat α]
    (path : System.FilePath) (inputWidth targetWidth : Nat) (options : Options := {}) :
    IO (Except String (SampleStream
      (TorchLean.Sample.Supervised α [inputWidth] [targetWidth]))) := do
  let rowsResult ← read path options
  match rowsResult with
  | .error e => pure (.error e)
  | .ok rows =>
      let samplesResult :
          Except String
            (Array (TorchLean.Sample.Supervised Float [inputWidth] [targetWidth])) :=
        rows.mapM (fun row => do
          let inputValues := row.take inputWidth
          let targetValues := row.drop inputWidth
          let input ← tensorFromFlat "csv" [inputWidth] inputValues
          let target ← tensorFromFlat "csv" [targetWidth] targetValues
          pure { input, target })
      pure <| samplesResult.map (fun samples =>
        SampleStream.fromArray <| samples.map fun sample =>
          Sample.map
            (Tensor.map ofFloat)
            (Tensor.map ofFloat)
            sample)

/-!
## File Sources

These sources load tensors by path and expected dimensions, then build supervised or labeled
datasets by slicing the leading batch axis, like PyTorch's `TensorDataset`.

Formats:
- NumPy `.npy` is the canonical interchange format for numeric tensors.
- CSV serves small tabular data.
- MATLAB `.mat`, PyTorch checkpoints, HDF5, Parquet and image archives go through a small
  preparation script that writes `.npy` tensors plus metadata. The Lean loader reads one small,
  deterministic interchange format and leaves every other binary format to such scripts.
-/

/-- File encodings inferred from source-path extensions. -/
inductive FileFormat where
  /--
  NumPy `.npy` with a little-endian `float32` or `float64` payload. Full loads accept C or
  Fortran order, and prefix loads need C order.
  -/
  | npy
  /-- Numeric CSV table, read as a `[rows, cols]` matrix or an `[n]` vector. -/
  | csv
deriving BEq, Repr

/-- Infer the file encoding from the path's extension, `.npy` or `.csv` in any case. -/
def detectFormat (path : System.FilePath) : Except String Data.FileFormat :=
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
def CSV.readTensor
    (path : System.FilePath) (dims : Shape) (options : Options := {}) :
    IO (Except String (Tensor Float dims)) := do
  match hDims : dims with
  | .dim rowsExpected (.dim colsExpected .scalar) =>
      let rowsRes ← read path options
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
                  (Data.tensorFromFlat "csv" [rowsExpected, colsExpected] flat).map
                    (fun tensor => by simpa [hDims] using tensor)
  | .dim n .scalar =>
      let rowsRes ← read path options
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
              pure <| (Data.tensorFromFlat "csv" [n] flat).map
                (fun tensor => by simpa [hDims] using tensor)
  | _ =>
      pure (.error s!"csv: expected shape=[rows, cols] or shape=[n], got {dims}")

/--
Load a Float tensor from a path and expected dimensions, picking the format by extension.
`shapeMatch` applies to NPY files, and a CSV table must have shape `dims`.
-/
def loadFloatTensor (path : System.FilePath)
    (dims : Shape) (options : CSV.Options := {})
    (shapeMatch : Data.TensorShapeMatch := .exact) :
    IO (Except String (Tensor Float dims)) := do
  match Data.detectFormat path with
  | .error e => pure (.error e)
  | .ok .npy => NPY.readTensor path dims shapeMatch
  | .ok .csv => CSV.readTensor path dims options

end Data

namespace Tensor

/--
Load a tensor from an NPY or numeric CSV file. The expected result type supplies the scalar type
and the shape, and the file must have that shape.
-/
opaque load {α : Type} [TorchLean.Storage α] [FromFloat α]
    {shape : Shape} (path : System.FilePath) (csvOptions : Data.CSV.Options := {}) :
    IO (Tensor α shape) := do
  let loaded ← Data.loadFloatTensor path shape csvOptions
  let tensor ← IO.ofExcept <|
    loaded.mapError fun message => s!"Tensor.load: {message}"
  pure <| Tensor.map ofFloat tensor

end Tensor

namespace Data

/--
Supervised data in two files: `inputPath` holds shape `(sampleCount, input...)` and `targetPath`
holds `(sampleCount, target...)`. NPY files must be in C order and may hold more rows, of which
the first `sampleCount` are read.
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
  csvOptions : CSV.Options := {}

namespace SupervisedSource

/--
Describe paired input and target tensor files. Encodings are inferred from their extensions.

Example:
```lean
-- Shapes are per sample, and `sampleCount` is how many leading file rows this run reads.
def source : Data.SupervisedSource :=
  Data.SupervisedSource.fromFiles
    (inputPath := "data/inputs.npy") (targetPath := "data/targets.npy")
    (sampleCount := 64) (input := [16]) (target := [1])
```
-/
def fromFiles (inputPath targetPath : System.FilePath)
    (sampleCount : Nat) (input target : Shape)
    (csvOptions : CSV.Options := {}) : SupervisedSource :=
  { sampleCount, input, target, inputPath, targetPath, csvOptions }

/--
Load a supervised dataset by slicing the leading batch axis of the two tensors. Regression and
operator-learning examples load their data with it, whether the files are `.npy` or small numeric
CSV tables.
-/
opaque load {α : Type} [TorchLean.Storage α]
    [FromFloat α] (source : SupervisedSource) :
    IO (SampleStream
      (TorchLean.Sample.Supervised α source.input source.target)) := do
  -- An NPY file may hold more than `sampleCount` rows and only the first `sampleCount` are read,
  -- while a CSV source must have the requested shape.
  let inputResult ← loadFloatTensor source.inputPath
    (Shape.dim source.sampleCount source.input)
    source.csvOptions .leadingPrefix
  let targetResult ← loadFloatTensor source.targetPath
    (Shape.dim source.sampleCount source.target)
    source.csvOptions .leadingPrefix
  let inputs ← IO.ofExcept <|
    inputResult.mapError fun message => s!"SupervisedSource.load: {message}"
  let targets ← IO.ofExcept <|
    targetResult.mapError fun message => s!"SupervisedSource.load: {message}"
  pure <| SampleStream.mk source.sampleCount fun sampleIndex =>
    { input := Tensor.map ofFloat (Tensor.unstack inputs sampleIndex)
      target := Tensor.map ofFloat (Tensor.unstack targets sampleIndex) }

end SupervisedSource

/--
Labeled classification data in two files: `inputPath` holds shape `(sampleCount, input...)` and
`labelPath` holds `(sampleCount,)` integer class labels in `[0, classCount)`. NPY files follow
the row convention of `SupervisedSource`.
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
  csvOptions : CSV.Options := {}

namespace LabeledSource

/-- Describe input and class-label tensor files. Encodings are inferred from their extensions. -/
def fromFiles (inputPath labelPath : System.FilePath)
    (sampleCount : Nat) (input : Shape) (classCount : Nat)
    (csvOptions : CSV.Options := {}) : LabeledSource :=
  { sampleCount, input, classCount, inputPath, labelPath, csvOptions }

/--
Load a labeled classification dataset by slicing the leading batch axis and one-hot encoding the
labels. A CSV label vector may be one column or one row, and an NPY label vector is
one-dimensional.
-/
opaque load {α : Type} [TorchLean.Storage α]
    [Zero α] [One α] [FromFloat α]
    (source : LabeledSource) :
    IO (SampleStream
      (TorchLean.Sample.Supervised α source.input [source.classCount])) := do
  -- Labels use the same prefix-row convention as supervised tensors, so one exported label vector
  -- can back runs of different sample counts without separate copies on disk.
  let inputResult ← loadFloatTensor source.inputPath
    (Shape.dim source.sampleCount source.input)
    source.csvOptions .leadingPrefix
  let labelResult ← loadFloatTensor source.labelPath [source.sampleCount]
    source.csvOptions .leadingPrefix
  let inputs ← IO.ofExcept <|
    inputResult.mapError fun message => s!"LabeledSource.load: {message}"
  let labels ← IO.ofExcept <|
    labelResult.mapError fun message => s!"LabeledSource.load: {message}"
  IO.ofExcept <| classificationSamples
    "LabeledSource.load" source.classCount inputs labels

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
  csvOptions : CSV.Options := {}

namespace TabularSupervisedSource

/-- Load a single-table supervised CSV source. -/
opaque load {α : Type} [TorchLean.Storage α]
    [FromFloat α]
    (source : TabularSupervisedSource) :
    IO (SampleStream
      (TorchLean.Sample.Supervised α [source.inputWidth] [source.targetWidth])) := do
  let samples ← CSV.readSupervised (α := α) source.path source.inputWidth source.targetWidth
    source.csvOptions
  IO.ofExcept <| samples.mapError fun message => s!"TabularSupervisedSource.load: {message}"

end TabularSupervisedSource

end Data
end TorchLean
