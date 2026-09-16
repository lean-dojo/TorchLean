/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Text
public import NN.API.Data.Image
public import NN.API.Trainer.Run

/-!
# Data API Tests

Regression checks for the public in-memory dataset constructors and tensor-backed sample streams.
-/

@[expose] public section

namespace NN.Tests.API.Data

open TorchLean

def fail {α : Type} (message : String) : IO α :=
  throw <| IO.userError s!"data API check failed: {message}"

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    fail label

def getOrFail {α : Type}
    (label : String) (stream : TorchLean.Data.SampleStream α) (index : Nat) : IO α :=
  match stream.get? index with
  | some value => pure value
  | none => fail s!"{label}: missing sample {index}"

def sample : Sample.Supervised Float [2] [1] :=
  { input := [1.25, -2.5], target := [3.75] }

def numberedSample (value : Float) : Sample.Supervised Float [1] [1] :=
  { input := [value], target := [value + 10.0] }

def generated : Trainer.Dataset [2] [1] :=
  TorchLean.Data.generate fun {_α} _ _ _ =>
    #[{ input := Tensor.map Runtime.ofFloat ([2.0, 4.0] : Tensor Float [2])
        target := Tensor.map Runtime.ofFloat ([6.0] : Tensor Float [1]) }]

/-- Exercise NPY metadata, prefix reads, and malformed file boundaries with small fixtures. -/
def runNpy (fixtureDirectory : System.FilePath) : IO Unit := do
  let path := fixtureDirectory / "nn_api_reader.npy"
  let encode (version : Nat) (dtype order shape : String) (payload : ByteArray) : ByteArray :=
    let dictionary := "{'descr': '" ++ dtype ++ "', 'fortran_order': " ++ order ++
      ", 'shape': " ++ shape ++ "}"
    let width := if version = 1 then 2 else 4
    let padding := (64 - (8 + width + dictionary.toUTF8.size + 1) % 64) % 64
    let header := (dictionary ++ String.ofList (List.replicate padding ' ') ++ "\n").toUTF8
    let magic := ByteArray.mk #[0x93, 78, 85, 77, 80, 89, version.toUInt8, 0]
    let preamble := (Array.range width).foldl
      (fun bytes shift => bytes.push (UInt8.ofNat (header.size / 256 ^ shift))) magic
    preamble ++ header ++ payload
  let payload (wide : Bool) (values : Array Float) : ByteArray :=
    values.foldl (fun bytes value =>
      let bits := if wide then value.toBits else value.toFloat32.toBits.toUInt64
      (Array.range (if wide then 8 else 4)).foldl
        (fun acc shift => acc.push ((bits >>> UInt64.ofNat (8 * shift)).toUInt8)) bytes)
      ByteArray.empty
  let requireData (result : Except String TorchLean.Data.IO.NpyData) :
      IO TorchLean.Data.IO.NpyData :=
    match result with
    | .ok data => pure data
    | .error message => fail message
  let rejected {α : Type} (result : Except String α) : Bool :=
    match result with
    | .error _ => true
    | .ok _ => false
  for version in #[1, 2] do
    for wide in #[false, true] do
      let dtype := if wide then "<f8" else "<f4"
      let bytes := encode version dtype "False" "(3, 2)" (payload wide #[1, 2, 3, 4, 5, 6])
      IO.FS.writeBinFile path bytes
      let full ← requireData (← TorchLean.Data.IO.readNpy path)
      expect "NPY full reads should support both header versions and float widths"
        (full.shape == #[3, 2] && full.values == #[1, 2, 3, 4, 5, 6] && !full.fortran)
      let selected ← requireData (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[2, 2])
      expect "NPY prefixes should preserve row boundaries"
        (selected.shape == #[2, 2] && selected.values == #[1, 2, 3, 4])
      expect "NPY row counting should validate trailing dimensions"
        ((← TorchLean.Data.availableNpyRows path [2] "(N, 2)") == 3)
      for shape in #[#[4, 2], #[1, 3], #[1, 1, 2], #[]] do
        expect "NPY prefixes should reject excess rows, wrong dimensions, and wrong rank"
          (rejected (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path shape))
      let missingTail := encode version dtype "False" "(1000000000, 2)" (payload wide #[7, 8])
      IO.FS.writeBinFile path missingTail
      let selected ← requireData (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[1, 2])
      let parsed ← requireData <|
        TorchLean.Data.IO.parseNpyLeadingAxisPrefix "test" #[1, 2] missingTail
      expect "NPY prefixes should require only the requested bytes"
        (selected.values == #[7, 8] && parsed.values == selected.values)
      expect "full NPY reads should still reject a missing tail"
        (rejected (← TorchLean.Data.IO.readNpy path) &&
          rejected (TorchLean.Data.IO.parseNpy "test" missingTail))
      expect "NPY prefixes should reject an incomplete requested range"
        (rejected (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[2, 2]))
      IO.FS.writeBinFile path (missingTail.extract 0 (missingTail.size - 1))
      expect "NPY prefixes should reject an incomplete final element"
        (rejected (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[1, 2]))
      IO.FS.writeBinFile path (encode version dtype "False" "(1000000000, 2)" ByteArray.empty)
      expect "NPY row counting must not require or read a numeric payload"
        ((← TorchLean.Data.availableNpyRows path [2] "(N, 2)") == 1000000000)
      let empty ← requireData (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[0, 2])
      expect "zero requested NPY rows should need no payload" empty.values.isEmpty
      IO.FS.writeBinFile path
        (encode version dtype "True" "(2, 3)" (payload wide #[1, 4, 2, 5, 3, 6]))
      let ordered ← requireData (← TorchLean.Data.IO.readNpy path)
      expect "full Fortran-order loads should retain C-order conversion"
        (ordered.values == #[1, 2, 3, 4, 5, 6] && !ordered.fortran)
      expect "Fortran-order prefixes should remain unsupported"
        (rejected (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[1, 3]))
      IO.FS.writeBinFile path (encode version dtype "False" "()" (payload wide #[9]))
      let scalar ← requireData (← TorchLean.Data.IO.readNpy path)
      expect "full NPY loads should support scalars" (scalar.shape.isEmpty && scalar.values == #[9])
      expect "NPY scalar prefixes should be rejected"
        (rejected (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[]))
      IO.FS.writeBinFile path (encode version dtype "False" "(2, 0)" ByteArray.empty)
      let zero ← requireData (← TorchLean.Data.IO.readNpy path)
      expect "NPY arrays with zero-sized axes should need no payload"
        (zero.shape == #[2, 0] && zero.values.isEmpty)
  let longShape := "(1,)" ++ String.ofList (List.replicate 65536 ' ')
  IO.FS.writeBinFile path (encode 2 "<f4" "False" longShape (payload false #[11]))
  let largeHeader ← requireData (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[1])
  expect "version-2 headers should support lengths beyond the version-1 limit"
    (largeHeader.shape == #[1] && largeHeader.values == #[11])
  let malformed := #[
    encode 3 "<f4" "False" "(1,)" ByteArray.empty,
    encode 1 "<i4" "False" "(1,)" ByteArray.empty,
    encode 1 "<f4" "Falsehood" "(1,)" ByteArray.empty,
    encode 1 "<f4" "False" "(-1,)" ByteArray.empty,
    encode 1 "<f4" "False" "(1.5,)" ByteArray.empty,
    encode 1 "<f4" "False" "(18446744073709551616,)" ByteArray.empty,
    encode 1 "<f4" "False" "(4294967296, 4294967296)" ByteArray.empty,
    ByteArray.mk #[0x93, 78, 85, 77, 80, 89, 2, 0, 255, 255, 255, 255],
    ByteArray.mk #[0x93, 78, 85, 77, 80, 89, 2, 0, 32],
    ByteArray.mk #[0x93, 78, 85, 77, 80, 89, 1, 0, 32, 0]]
  for bytes in malformed do
    IO.FS.writeBinFile path bytes
    expect "malformed or unaddressable NPY metadata should fail before payload allocation"
      (rejected (← TorchLean.Data.IO.readNpyHeader path) &&
        rejected (← TorchLean.Data.IO.readNpy path) &&
        rejected (← TorchLean.Data.IO.readNpyLeadingAxisPrefix path #[1]) &&
        rejected (TorchLean.Data.IO.parseNpy "test" bytes))
  IO.println "  NPY metadata and prefix IO: passed"

def run : IO Unit := do
  let fixtureDirectory : System.FilePath := ".lake/build/tmp"
  let tensorPath := fixtureDirectory / "nn_api_tensor_load.csv"
  IO.FS.createDirAll fixtureDirectory
  runNpy fixtureDirectory
  IO.FS.writeFile tensorPath "1.0,2.0\n3.0,4.0\n"
  let loaded : Tensor Float32 [2, 2] ← Tensor.load tensorPath
  expect "Tensor.load should infer scalar type and shape from the expected result"
    (loaded.to (Array Float32) ==
      #[Float.toFloat32 1.0, Float.toFloat32 2.0,
        Float.toFloat32 3.0, Float.toFloat32 4.0])
  let rejectedWrongShape ←
    try
      let _ : Tensor Float [3, 2] ← Tensor.load tensorPath
      pure false
    catch _ =>
      pure true
  expect "Tensor.load should reject file metadata that disagrees with the expected shape"
    rejectedWrongShape

  let labelPath := fixtureDirectory / "nn_api_labels.csv"
  IO.FS.writeFile labelPath "1\n2\n"
  let labeledSource := TorchLean.Data.LabeledSource.fromFiles
    tensorPath labelPath 2 [2] 3
  let labeled ← labeledSource.load (α := Float32)
  let secondLabeled ← getOrFail "LabeledSource.load" labeled 1
  expect "labeled sources should preserve inputs and encode the matching label"
    ((secondLabeled.input.to (Array Float32)).map Float32.toFloat == #[3.0, 4.0] &&
      (secondLabeled.target.to (Array Float32)).map Float32.toFloat == #[0.0, 0.0, 1.0])
  for invalidLabel in #["-1", "1.5", "3", "1e999"] do
    IO.FS.writeFile labelPath s!"1\n{invalidLabel}\n"
    let rejected ←
      try
        let _ ← labeledSource.load (α := Float32)
        pure false
      catch _ => pure true
    expect s!"labeled sources should reject {invalidLabel} before any sample is accessed" rejected

  let imagePath := fixtureDirectory / "nn_api_image.ppm"
  IO.FS.writeFile imagePath "unchanged"
  let rejectedEmptyImage ←
    try
      TorchLean.Data.Image.writeFirstRgbPpm imagePath (Tensor.zeros [1, 3, 0, 2])
      pure false
    catch _ => pure true
  expect "image export should reject empty dimensions before opening the output"
    (rejectedEmptyImage && (← IO.FS.readFile imagePath) == "unchanged")

  let samples ← (TorchLean.Data.fromSamples #[sample]).materialize (α := Float32)
  expect "Data.fromSamples should preserve sample count" (samples.size == 1)
  let converted ← getOrFail "Data.fromSamples" samples 0
  expect "Data.fromSamples should convert inputs to the selected scalar"
    (converted.input.to (Array Float32) ==
      #[Float.toFloat32 1.25, Float.toFloat32 (-2.5)])
  expect "Data.fromSamples should convert targets to the selected scalar"
    (converted.target.to (Array Float32) == #[Float.toFloat32 3.75])

  let singleton ← (TorchLean.Data.fromSample sample).materialize (α := Float)
  expect "Data.fromSample should contain exactly one sample" (singleton.size == 1)
  let singletonSample ← getOrFail "Data.fromSample" singleton 0
  expect "Data.fromSample should preserve the input"
    (singletonSample.input.to (Array Float) == #[1.25, -2.5])

  let generatedSamples ← generated.materialize (α := Float)
  let generatedSample ← getOrFail "Data.generate" generatedSamples 0
  expect "Data.generate should materialize its builder"
    (generatedSamples.size == 1 &&
      generatedSample.input.to (Array Float) == #[2.0, 4.0])

  let batchedInputs : Tensor Float [2, 2] := [[1.0, 2.0], [3.0, 4.0]]
  let batchedTargets : Tensor Float [2, 1] := [[5.0], [6.0]]
  let batched ←
    (TorchLean.Data.fromTensors batchedInputs batchedTargets).materialize (α := Float)
  expect "Data.fromTensors should expose every leading-axis slice" (batched.size == 2)
  let second ← getOrFail "Data.fromTensors" batched 1
  expect "Data.fromTensors should keep aligned input slices"
    (second.input.to (Array Float) == #[3.0, 4.0])
  expect "Data.fromTensors should keep aligned target slices"
    (second.target.to (Array Float) == #[6.0])

  let split := TorchLean.Data.randomSplit 1
    (TorchLean.Data.fromTensors batchedInputs batchedTargets) 17
  let train ← split.train.materialize (α := Float)
  let test ← split.test.materialize (α := Float)
  expect "Data.randomSplit should return named train/test views"
    (train.size == 1 && test.size == 1)

  let stream := TorchLean.Data.SampleStream.fromArray
    #[numberedSample 1.0, numberedSample 2.0, numberedSample 3.0]
  let loader := TorchLean.Data.Loader.fromStream stream 2
  let epoch ←
    match TorchLean.Data.Loader.nextEpoch "data API test" loader with
    | .ok result => pure result
    | .error message => fail message
  expect "Data.Loader should omit a final partial batch" (epoch.batches.size == 1)
  match epoch.batches[0]? with
  | some batch =>
      expect "Data.Loader should collate samples in order when shuffle is false"
        (batch.input.to (Array Float) == #[1.0, 2.0] &&
          batch.target.to (Array Float) == #[11.0, 12.0])
  | none => fail "Data.Loader should produce one full batch"
  expect "Data.Loader should preserve deterministic state when shuffle is false"
    (epoch.nextLoader.samples.size == 3 && epoch.nextLoader.seed == 0)
  let rejectedZeroBatch ←
    match TorchLean.Data.Loader.nextEpoch "data API zero batch"
        (TorchLean.Data.Loader.fromStream stream 0) with
    | .ok _ => pure false
    | .error _ => pure true
  expect "Data.Loader should reject batch size zero" rejectedZeroBatch

  let longer := stream.append <| TorchLean.Data.SampleStream.fromArray
    #[numberedSample 4.0, numberedSample 5.0]
  let collated ←
    match TorchLean.Data.collateStream 2 longer with
    | .ok batches => pure batches
    | .error message => fail message
  let collatedBatch ← getOrFail "Data.collateStream" collated 0
  expect "lazy collation should preserve alignment and omit partial batches"
    (collated.size == 2 &&
      collatedBatch.input.to (Array Float) == #[1.0, 2.0] &&
      collatedBatch.target.to (Array Float) == #[11.0, 12.0])
  let secondBatch ← getOrFail "Data.collateStream" collated 1
  expect "later lazy batches should advance by one batch width"
    (secondBatch.input.to (Array Float) == #[3.0, 4.0] &&
      secondBatch.target.to (Array Float) == #[13.0, 14.0])
  let first ←
    match TorchLean.Data.Loader.firstFullBatch "data API first batch" loader with
    | .ok batch => pure batch
    | .error message => fail message
  expect "firstFullBatch should agree with the first epoch batch"
    (first.input.to (Array Float) == collatedBatch.input.to (Array Float) &&
      first.target.to (Array Float) == collatedBatch.target.to (Array Float))
  let oversized ←
    match TorchLean.Data.collateStream 4 stream with
    | .ok batches => pure batches
    | .error message => fail message
  expect "collation larger than the source should contain no full batches" oversized.isEmpty

  let windows : Tensor (Fin 3) [2, 2, 3] :=
    Tensor.stackLeading fun batch : Fin 2 =>
      Tensor.stackLeading fun row : Fin 2 =>
        Tensor.ofFn fun position : Fin 3 => ⟨(batch.val + row.val + position.val) % 3, by omega⟩
  let tokenSample := TorchLean.Data.CausalLM.tokenSample [2, 2] 2 windows
  let oneHotSample :=
    TorchLean.Data.CausalLM.oneHotSample (α := Float32) [2, 2] 2 3 windows
  expect "causal inputs and targets should shift within each row"
    ((tokenSample.input.to (Array (Fin 3))).map Fin.val == #[0, 1, 1, 2, 1, 2, 2, 0] &&
      (tokenSample.target.to (Array (Fin 3))).map Fin.val == #[1, 2, 2, 0, 2, 0, 0, 1])
  expect "one-hot causal samples should use the same shift across multiple batch axes"
    ((oneHotSample.input.to (Array Float32)).map Float32.toFloat ==
      #[1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0] &&
      (oneHotSample.target.to (Array Float32)).map Float32.toFloat ==
      #[0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

  IO.println "  data API: passed"

end NN.Tests.API.Data
