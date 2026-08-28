/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.TensorDataset
public import NN.Data.IO

/-!
# File-Backed Dataset Sources

Typed NPY and CSV sources for supervised and labeled datasets. This module owns the boundary
between external files and TorchLean tensors; importing `Data.SampleStream` does not pull
in these parsers.
-/

@[expose] public section

namespace TorchLean
namespace Data

export _root_.TorchLean.Data.IO
  (CsvOptions readCsvFloatRows readNpy readNpyLeadingAxisPrefix)

/-- Require that every path exists, adding `hint` to a missing-file error when supplied. -/
def requireFiles (exeName : String) (paths : Array System.FilePath) (hint : String := "") :
    IO Unit := do
  for path in paths do
    unless (← path.pathExists) do
      let suffix := if hint.isEmpty then "" else "\n" ++ hint
      throw <| IO.userError s!"{exeName}: missing data file: {path}{suffix}"

/-- Require one named data file to exist. -/
def requireFile
    (exeName : String) (label : String) (path : System.FilePath) (hint : String := "") :
    IO Unit := do
  unless (← path.pathExists) do
    let suffix := if hint.isEmpty then "" else "\n" ++ hint
    throw <| IO.userError s!"{exeName}: missing {label}: {path}{suffix}"

/-- Require paired supervised input and target files to exist. -/
def requirePairedFiles
    (exeName : String)
    (xLabel : String) (xPath : System.FilePath)
    (yLabel : String) (yPath : System.FilePath)
    (hint : String := "") : IO Unit := do
  requireFile exeName xLabel xPath hint
  requireFile exeName yLabel yPath hint

/--
Read an `.npy` row count while checking all trailing dimensions.

For shape `(N, d₁, ..., dₖ)`, this returns `N` exactly when the trailing shape is `tailShape`.
-/
def availableNpyRows
    (path : System.FilePath) (tailShape : List Nat) (expectedDesc : String) :
    IO (Except String Nat) := do
  match ← readNpy path with
  | .error e => pure (.error e)
  | .ok data =>
      match data.shape[0]? with
      | some rows =>
          let trailingShape := data.shape.extract 1 data.shape.size
          if trailingShape.toList = tailShape then
            pure (.ok rows)
          else
            pure (.error s!"expected {expectedDesc}, got {data.shape}")
      | none => pure (.error s!"expected {expectedDesc}, got {data.shape}")

/-- How a file's physical shape is matched against the requested tensor shape. -/
private inductive TensorShapeMatch where
  /-- Require every physical dimension to equal the requested dimension. -/
  | exact
  /-- Permit a larger first dimension and read its requested prefix. -/
  | leadingPrefix
deriving BEq, Repr

/-- Box an unboxed NPY payload for the tensor constructors, which take an `Array Float`.

`NpyData.values` is a `FloatArray` so that decoding a large file does not pay a heap cell per
element. `Tensor.ofArray` takes an `Array α`, so the boxing happens once, here, at the point
where a payload becomes a tensor — rather than throughout the decode. -/
private def boxPayload (values : FloatArray) : Array Float := Id.run do
  let mut out := Array.emptyWithCapacity values.size
  for i in [0:values.size] do
    out := out.push (values.get! i)
  pure out

/--
Load an arbitrary-rank tensor from a `.npy` file under an explicit shape-matching policy.

With `exact`, every dimension must match. With `leadingPrefix`, rank and trailing dimensions must
still match, while the file may contain more entries on its first axis. Prefix loading requires a
C-order NPY file because the requested values must form one contiguous prefix.
-/
private def readNpyTensor (path : System.FilePath) (dims : List Nat)
    (shapeMatch : TensorShapeMatch := .exact) :
    IO (Except String (Tensor Float dims)) := do
  match shapeMatch with
  | .exact =>
      let res ← readNpy path
      match res with
      | .error e => pure (.error e)
      | .ok data =>
          if data.shape.toList != dims then
            pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")
          else
            pure <| TorchLean.Tensor.ofArray (α := Float) dims (boxPayload data.values)
  | .leadingPrefix =>
      let res ← readNpyLeadingAxisPrefix path dims.toArray
      match res with
      | .error e => pure (.error e)
      | .ok data => pure <| TorchLean.Tensor.ofArray (α := Float) dims (boxPayload data.values)

/-- Parse a float-encoded class label as a `Nat` in `[0, classes)`. -/
private def finLabelOfFloat (tag : String) (classes : Nat) (x : Float) : Except String (Fin classes) := do
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
integer in `[0, classes)`, then one-hot encode it.
-/
private def labeledFromLeadingAxis {α : Type} [_root_.Context α]
    [_root_.TorchLean.Runtime.FromFloat α]
    (tag : String) (classes : Nat)
    {n : Nat} {inputShape : List Nat}
    (X : Tensor Float (n :: inputShape))
    (y : Tensor Float [n]) :
    Except String
      (SampleStream (_root_.TorchLean.TensorPack α [inputShape, [classes]])) := do
  let samples : Array (Tensor Float inputShape × Fin classes) ←
    (Array.ofFn (fun i : Fin n => i)).mapM (fun i => do
      let x := Spec.get X i
      let labelF : Float := Tensor.item (Spec.get y i)
      let label ← finLabelOfFloat tag classes labelF
      pure (x, label))
  pure <| TensorDataset.ofLabeledFloatPairs
    (α := α) (inputShape := inputShape) classes samples

/--
Load a supervised dataset from a CSV with `inDim + outDim` columns per row:

`x1, ..., x_inDim, y1, ..., y_outDim`.
-/
def readCsvSupervised {α : Type} [_root_.Context α]
    [_root_.TorchLean.Runtime.FromFloat α]
    (path : System.FilePath) (inDim outDim : Nat) (opts : CsvOptions := {}) :
    IO (Except String (SampleStream (_root_.TorchLean.TensorPack α [[inDim],
      [outDim]]))) := do
  let rowsRes ← readCsvFloatRows path opts
  match rowsRes with
  | .error e => pure (.error e)
  | .ok rows =>
      let samplesRes :
          Except String (Array (Tensor Float [inDim] × Tensor Float
            [outDim])) :=
        rows.mapM (fun row => do
          let xs := row.take inDim
          let ys := row.drop inDim
          let xF ← (Tensor.ofArray [inDim] xs).mapError fun msg => s!"csv: {msg}"
          let yF ← (Tensor.ofArray [outDim] ys).mapError fun msg => s!"csv: {msg}"
          pure (xF, yF))
      pure <| samplesRes.map (fun samplesF =>
        TensorDataset.ofSupervisedFloatPairs (α := α) samplesF)

/-!
## File Sources

The definitions below describe tensors by path, format, and expected dimensions:

1. describe each tensor as a `TensorSource`;
2. load it as a typed TorchLean tensor;
3. build supervised/labeled datasets by slicing the leading batch axis, just like PyTorch `TensorDataset`.

Policy for external ecosystems:
- NumPy `.npy` is the canonical interchange format for numeric tensors.
- CSV is supported for small tabular data.
- MATLAB `.mat`, PyTorch checkpoints, HDF5, Parquet, and image archives should be converted by a
  small preparation script into `.npy` tensors plus metadata. The Lean runtime loader intentionally
  handles a small deterministic interchange format rather than every external binary format.
-/

/-- File formats supported directly by the Lean side unified data-source loader. -/
inductive TensorFormat where
  /-- NumPy `.npy`, supporting numeric C-order arrays decoded by TorchLean's NPY reader. -/
  | npy
  /-- Numeric CSV table. CSV sources are interpreted as 2D tensors `[rows, cols]`. -/
  | csv
deriving BEq, Repr

namespace TensorFormat

/-- Human-facing extension used by messages and examples. -/
def extension : TensorFormat → String
  | .npy => ".npy"
  | .csv => ".csv"

end TensorFormat

/--
Description of one tensor stored on disk.

`dims` is the expected tensor shape.  NPY can load any rank supported by `ofList`; CSV is treated
as a numeric table and therefore expects `dims = [rows, cols]`.
-/
structure TensorSource where
  /-- Path to the file. -/
  path : System.FilePath
  /-- Expected dimensions. -/
  dims : List Nat
  /-- Direct Lean side format. External formats should be preconverted to `.npy`. -/
  format : TensorFormat := .npy
  /-- CSV parsing options, used only when `format = .csv`. -/
  csvOptions : CsvOptions := {}

namespace TensorSource

/--
Load a numeric CSV table as a tensor.

Supported shapes:
- `[rows, cols]`: ordinary numeric table,
- `[n]`: either one column with `n` rows or one row with `n` columns.
-/
private def loadCsvTensor (path : System.FilePath) (dims : List Nat) (opts : CsvOptions := {}) :
    IO (Except String (Tensor Float dims)) := do
  match hDims : dims with
  | [rowsExpected, colsExpected] =>
      let rowsRes ← readCsvFloatRows path opts
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
                pure <| (TorchLean.Tensor.ofArray (α := Float) [rowsExpected, colsExpected] flat).map
                  (fun t => by
                    simpa [hDims] using t)
  | [n] =>
      let rowsRes ← readCsvFloatRows path opts
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
              pure <| (TorchLean.Tensor.ofArray (α := Float) [n] flat).map
                (fun t => by
                  simpa [hDims] using t)
  | _ =>
      pure (.error s!"csv: TensorSource expects dims=[rows, cols] or dims=[n], got {dims}")

/-- Load a Float tensor from a path/format/dimension tuple under one shape-matching policy. -/
private def loadFloatAs (format : TensorFormat) (path : System.FilePath)
    (dims : List Nat) (opts : CsvOptions := {})
    (shapeMatch : TensorShapeMatch := .exact) :
    IO (Except String (Tensor Float dims)) := do
  match format with
  | .npy => readNpyTensor path dims shapeMatch
  | .csv => loadCsvTensor path dims opts

/-- Load a `TensorSource` as a Float tensor with statically reflected dimensions. -/
opaque loadFloat (src : TensorSource) :
    IO (Except String (Tensor Float src.dims)) := do
  loadFloatAs src.format src.path src.dims src.csvOptions

end TensorSource

/--
Two tensor sources representing supervised data:
- `x` must have shape `(n, xDims...)`,
- `y` must have shape `(n, yDims...)`.
-/
structure SupervisedSource where
  /-- Number of samples along the leading batch axis. -/
  n : Nat
  /-- Per-sample input dimensions. -/
  xDims : List Nat
  /-- Per-sample target dimensions. -/
  yDims : List Nat
  /-- Source for the batched input tensor. -/
  x : TensorSource
  /-- Source for the batched target tensor. -/
  y : TensorSource

namespace SupervisedSource

/-- Construct a supervised source from paths using the same file format for `x` and `y`. -/
def ofPaths (format : TensorFormat) (xPath yPath : System.FilePath)
    (n : Nat) (xDims yDims : List Nat) (csvOptions : CsvOptions := {}) : SupervisedSource :=
  { n, xDims, yDims
    x := { path := xPath, dims := n :: xDims, format, csvOptions }
    y := { path := yPath, dims := n :: yDims, format, csvOptions } }

/--
Load a supervised dataset by slicing the leading batch axis from the two tensors.

This is the preferred public loader for regression/operator-learning examples, regardless of
whether the backing files are `.npy` or small numeric CSV tables.
-/
opaque load {α : Type} [_root_.TorchLean.Runtime.FromFloat α] (src : SupervisedSource) :
    IO (Except String
      (SampleStream (_root_.TorchLean.TensorPack α [src.xDims, src.yDims]))) := do
  -- Dataset sources interpret `src.n` as "number of rows to use in this run."  For NPY files, the
  -- physical file is allowed to contain more rows; for CSV files, the requested shape remains exact.
  let xRes ← TensorSource.loadFloatAs src.x.format src.x.path
    (src.n :: src.xDims)
    src.x.csvOptions .leadingPrefix
  let yRes ← TensorSource.loadFloatAs src.y.format src.y.path
    (src.n :: src.yDims)
    src.y.csvOptions .leadingPrefix
  match xRes with
  | .error e => pure (.error e)
  | .ok X =>
      match yRes with
      | .error e => pure (.error e)
      | .ok Y =>
          let X' : Tensor Float (src.n :: src.xDims) := X
          let Y' : Tensor Float (src.n :: src.yDims) := Y
          pure (.ok <| TensorDataset.ofBatchedFloat
            (α := α) (shapes := [src.xDims, src.yDims])
            (_root_.TorchLean.TensorPack! X', Y'))

end SupervisedSource

/--
Load paired `.npy` files as concrete `Float` supervised samples.

This is useful for reporting, custom evaluation loops, and native kernels that need concrete
`Float` tensors outside the high-level trainer API.
-/
def loadSupervisedNpy
    (xPath yPath : System.FilePath) (n : Nat)
    (xDims yDims : List Nat) :
  IO (Except String (Array (TorchLean.Sample.Supervised Float xDims yDims))) := do
  let ds ← SupervisedSource.load (α := Float)
    (SupervisedSource.ofPaths .npy xPath yPath n xDims yDims)
  pure <| ds.map (fun d => d.toArray)

/--
Two tensor sources representing labeled classification data:
- `x` must have shape `(n, xDims...)`,
- `y` must have shape `(n,)` and contain integer-valued labels.
-/
structure LabeledSource where
  /-- Number of samples along the leading batch axis. -/
  n : Nat
  /-- Per-sample input dimensions. -/
  xDims : List Nat
  /-- Number of classes for one-hot targets. -/
  classes : Nat
  /-- Source for the batched input tensor. -/
  x : TensorSource
  /-- Source for the label vector. -/
  y : TensorSource

namespace LabeledSource

/-- Construct a labeled source from paths using the same file format for `x` and `y`. -/
def ofPaths (format : TensorFormat) (xPath yPath : System.FilePath)
    (n : Nat) (xDims : List Nat) (classes : Nat) (csvOptions : CsvOptions := {}) : LabeledSource :=
  { n, xDims, classes
    x := { path := xPath, dims := n :: xDims, format, csvOptions }
    y := { path := yPath, dims := [n], format, csvOptions } }

/--
Load a labeled classification dataset by slicing the leading batch axis and one-hot encoding labels.

For CSV label vectors, store labels as a single-column table with `dims = [n, 1]` and use a custom
`TensorSource` if needed; the path constructor above is aimed at `.npy` label vectors.
-/
opaque load {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    (src : LabeledSource) :
    IO (Except String
      (SampleStream (_root_.TorchLean.TensorPack α [src.xDims, [src.classes]]))) := do
  -- Labels use the same prefix-row convention as supervised tensors. This lets one full exported
  -- label vector back different bounded runs without making separate copies on disk.
  let xRes ← TensorSource.loadFloatAs src.x.format src.x.path
    (src.n :: src.xDims)
    src.x.csvOptions .leadingPrefix
  let yRes ← TensorSource.loadFloatAs src.y.format src.y.path [src.n]
    src.y.csvOptions .leadingPrefix
  match xRes with
  | .error e => pure (.error e)
  | .ok X =>
      match yRes with
      | .error e => pure (.error e)
      | .ok y =>
          let X' : Tensor Float (src.n :: src.xDims) := X
          let y' : Tensor Float [src.n] := y
          pure <| by
            simpa using labeledFromLeadingAxis (α := α) (inputShape := src.xDims)
              "data-source" src.classes X' y'

end LabeledSource

/--
Single-table supervised CSV source.

Use this when one CSV row contains both input and target columns:
`x1, ..., x_inDim, y1, ..., y_outDim`.
-/
structure TabularSupervisedSource where
  /-- CSV file path. -/
  path : System.FilePath
  /-- Number of input feature columns. -/
  inDim : Nat
  /-- Number of target columns. -/
  outDim : Nat
  /-- CSV parsing options. -/
  csvOptions : CsvOptions := {}

namespace TabularSupervisedSource

/-- Load a single-table supervised CSV source. -/
opaque load {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    (src : TabularSupervisedSource) :
    IO (Except String (SampleStream (_root_.TorchLean.TensorPack α [[src.inDim],
      [src.outDim]]))) :=
  readCsvSupervised (α := α) src.path src.inDim src.outDim src.csvOptions

end TabularSupervisedSource

end Data
end TorchLean
