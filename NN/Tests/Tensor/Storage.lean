/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# Tensor Storage Regression Tests

These tests pin down both sides of the public tensor contract:

* element-specific `Storage` instances select the intended physical buffer;
* mixed-element arithmetic promotes once into the inferred output storage.

The buffer witnesses below fail to typecheck if instance selection regresses.
-/

@[expose] public section

namespace NN.Tests.Tensor.Storage

open TorchLean

/-- `Float` tensors use Lean's native unboxed floating-point buffer. -/
def floatBufferWitness (tensor : Tensor Float [3]) : FloatArray :=
  tensor.buffer

/-- `UInt8` tensors use Lean's native packed byte buffer. -/
def byteBufferWitness (tensor : Tensor UInt8 [3]) : ByteArray :=
  tensor.buffer

/-- Element types without a specialized instance retain the generic boxed array. -/
def naturalBufferWitness (tensor : Tensor Nat [3]) : Array Nat :=
  tensor.buffer

/-- Composite element types also use the generic boxed array by default. -/
def complexBufferWitness
    (tensor : Tensor (TorchLean.Complex Float) [3]) :
    Array (TorchLean.Complex Float) :=
  tensor.buffer

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"tensor storage check failed: {label}"

def bytes : Tensor UInt8 [3] :=
  Tensor.ofFn fun
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 2
    | _ => 3

def floats : Tensor Float [3] :=
  Tensor.ofFn fun
    | ⟨0, _⟩ => 0.5
    | ⟨1, _⟩ => 1.5
    | _ => 2.5

/-- Ordinary bracket syntax constructs tensors of any positive rank. -/
def literalVector : Tensor Nat [3] :=
  [1, 2, 3]

def literalMatrix : Tensor Nat [2, 2] :=
  [[1, 2], [3, 4]]

def literalCube : Tensor Nat [2, 1, 2] :=
  [[[1, 2]], [[3, 4]]]

/-- Rank-zero tensors use the same shape notation and render as their sole value. -/
def rankZeroValue : Tensor Nat [] :=
  Tensor.full [] 7

/-- Flat sources are vectors until an equal-size reshape is requested explicitly. -/
def convertedMatrix : Tensor Nat [2, 2] :=
  (Tensor.from #[1, 2, 3, 4]).reshape [2, 2]

/-- Native byte slicing must clamp before allocating, even for heap-allocated naturals. -/
def checkByteSliceBounds : IO Unit := do
  let source := ByteArray.mk #[10, 20, 30, 40]
  let output := ByteArray.mk #[1, 2]
  let endpoints := [0, 1, 3, 4, 5, 2^64, 2^64 + 1, 2^200]
  for start in endpoints do
    for stop in endpoints do
      let actual := TorchLean.Storage.Internal.byteBufferAppendSlice source start stop output
      expect s!"byte slice clamp {start}:{stop}"
        (actual.data == output.data ++ source.data.extract start stop)

/-- Rational promotion rounds the exact quotient, including ties and extreme magnitudes. -/
def checkRationalPromotion : IO Unit := do
  let huge : Rat := (2^2048 : Nat)
  let cases32 : List (Rat × UInt32) := [
    (0, 0x0),
    (1 / 3, 0x3eaaaaab),
    ((huge + 1) / huge, 0x3f800000),
    (-(huge + 1) / huge, 0xbf800000),
    (huge, 0x7f800000),
    (1 / huge, 0x0),
    (-1 / huge, 0x80000000),
    (1 / (2^150 : Nat), 0x0),
    (3 / (2^150 : Nat), 0x2),
    (((2^24 : Nat) + 1) / (2^24 : Nat), 0x3f800000),
    (((2^24 : Nat) + 3) / (2^24 : Nat), 0x3f800002),
    (((2^24 : Nat) + 2) / ((2^24 : Nat) + 1), 0x3f800000)]
  for (value, expected) in cases32 do
    let tensor : Tensor Rat [1] := [value]
    let converted := tensor.cast Float32
    expect "rational Float32 rounding" ((converted.getScalar 0).toBits == expected)
  let cases64 : List (Rat × UInt64) := [
    (0, 0x0),
    (1 / 3, 0x3fd5555555555555),
    ((huge + 1) / huge, 0x3ff0000000000000),
    (-(huge + 1) / huge, 0xbff0000000000000),
    (huge, 0x7ff0000000000000),
    (1 / huge, 0x0),
    (-1 / huge, 0x8000000000000000),
    (1 / (2^1075 : Nat), 0x0),
    (3 / (2^1075 : Nat), 0x2),
    (((2^53 : Nat) + 1) / (2^53 : Nat), 0x3ff0000000000000),
    (((2^53 : Nat) + 3) / (2^53 : Nat), 0x3ff0000000000002),
    (((2^53 : Nat) + 2) / ((2^53 : Nat) + 1), 0x3ff0000000000000)]
  for (value, expected) in cases64 do
    let tensor : Tensor Rat [1] := [value]
    let converted := tensor.cast Float
    expect "rational Float rounding" ((converted.getScalar 0).toBits == expected)

/--
Exercise storage selection and automatic `UInt8`/`Float` promotion through
the ordinary `Tensor.add` API.
-/
def run : IO Unit := do
  checkRationalPromotion
  checkByteSliceBounds
  let emptyRow : Tensor Nat [0] := Tensor.from (#[] : Array Nat)
  let emptyStack : Tensor Nat [2^64, 0] := Tensor.dim fun _ => emptyRow
  expect "empty stacking does not allocate a cache for the leading extent"
    emptyStack.data.isEmpty

  expect "rank-one bracket literal"
    (literalVector.to (Array Nat) == #[1, 2, 3])
  expect "matrix bracket literal is row-major"
    (literalMatrix.to (Array Nat) == #[1, 2, 3, 4])
  expect "rank-three bracket literal is row-major"
    (literalCube.to (Array Nat) == #[1, 2, 3, 4])
  expect "array conversion and reshape"
    (convertedMatrix.to (Array Nat) == #[1, 2, 3, 4])
  expect "rank-zero tensor prints as its value"
    (reprStr rankZeroValue == "7")
  expect "rank-zero shape uses bracket notation"
    (reprStr ([] : Spec.Shape) == "[]")
  expect "higher-rank shapes use bracket notation"
    (reprStr ([2, 3] : Spec.Shape) == "[2, 3]")
  expect "tensor representation is shape-aware rather than array syntax"
    (reprStr literalVector == "[1, 2, 3]")
  expect "matrix representation preserves nested axes"
    (reprStr literalMatrix == "[[1, 2], [3, 4]]")
  let doubled : Tensor Nat [3] := literalVector + literalVector
  expect "computed tensors use the direct tensor representation"
    (reprStr doubled == "[2, 4, 6]")
  expect "general conversion accepts computed tensor expressions"
    (Tensor.to (literalVector + literalVector) (Array Nat) == #[2, 4, 6])
  let vectorView : Vector Nat (Spec.Shape.size [2, 2]) :=
    Tensor.to convertedMatrix (Vector Nat (Spec.Shape.size [2, 2]))
  expect "general conversion supports statically sized vectors"
    (vectorView.toArray == #[1, 2, 3, 4])
  let fromList : Tensor Nat [3] := Tensor.from ([4, 5, 6] : List Nat)
  expect "list conversion"
    (fromList.to (List Nat) == [4, 5, 6])

  let floatSource := FloatArray.mk #[0.25, 0.5, 0.75]
  let floatRoundtrip : FloatArray := (Tensor.from floatSource).to FloatArray
  expect "FloatArray conversion preserves packed values"
    (floatRoundtrip.data == floatSource.data)

  let floatStorage := (inferInstance : TorchLean.Storage Float)
  let wideFloatSource :=
    FloatArray.mk #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
      6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
  let wideFloatCopy :=
    floatStorage.appendSlice wideFloatSource 2 11
      (FloatArray.mk #[99.0])
  expect "FloatArray bulk slice preserves order"
    (wideFloatCopy.data ==
      #[99.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
  let aliasedFloatCopy :=
    floatStorage.appendSlice wideFloatSource 1 9 wideFloatSource
  expect "FloatArray bulk slice handles aliased source and output"
    (aliasedFloatCopy.data ==
      #[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
  let huge : Nat := 2 ^ 100
  let hugeStopCopy :=
    floatStorage.appendSlice wideFloatSource 0 huge
      (FloatArray.mk #[99.0])
  expect "FloatArray bulk slice clamps a big-Nat stop"
    (hugeStopCopy.data ==
      #[99.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
  let hugeStartCopy :=
    floatStorage.appendSlice wideFloatSource huge (huge + 20)
      (FloatArray.mk #[99.0])
  expect "FloatArray bulk slice clamps a big-Nat start"
    (hugeStartCopy.data == #[99.0])

  -- Buffer primitives are total even when callers bypass matching tensor shapes.
  let shortFloat := FloatArray.mk #[2.0]
  let longFloat := FloatArray.mk #[4.0, 8.0]
  let shortByte := ByteArray.mk #[2]
  let longByte := ByteArray.mk #[4, 8]
  let emptyFloat := FloatArray.mk #[]
  let emptyByte := ByteArray.mk #[]
  let kernels := #[
    ("add", TorchLean.Tensor.Internal.Elab.Impl.floatBufferAdd, #[6.0]),
    ("sub", TorchLean.Tensor.Internal.Elab.Impl.floatBufferSub, #[-2.0]),
    ("mul", TorchLean.Tensor.Internal.Elab.Impl.floatBufferMul, #[8.0]),
    ("div", TorchLean.Tensor.Internal.Elab.Impl.floatBufferDiv, #[0.5])]
  for (label, kernel, expected) in kernels do
    expect s!"native {label} truncates unequal buffer lengths"
      ((kernel shortFloat longFloat).data == expected)
    expect s!"native {label} accepts empty left buffers"
      (kernel emptyFloat longFloat).data.isEmpty
    expect s!"native {label} accepts empty right buffers"
      (kernel longFloat emptyFloat).data.isEmpty
  expect "byte/float addition truncates unequal buffers"
    ((TorchLean.Tensor.Internal.Elab.Impl.byteFloatBufferAdd shortByte longFloat).data ==
      #[6.0])
  expect "float/byte addition truncates unequal buffers"
    ((TorchLean.Tensor.Internal.Elab.Impl.floatByteBufferAdd shortFloat longByte).data ==
      #[6.0])
  expect "byte/float addition accepts empty buffers"
    (TorchLean.Tensor.Internal.Elab.Impl.byteFloatBufferAdd emptyByte longFloat).data.isEmpty
  expect "float/byte addition accepts empty buffers"
    (TorchLean.Tensor.Internal.Elab.Impl.floatByteBufferAdd longFloat emptyByte).data.isEmpty

  let byteSource := ByteArray.mk #[1, 2, 3]
  let byteRoundtrip : ByteArray := (Tensor.from byteSource).to ByteArray
  expect "ByteArray conversion preserves packed values"
    (byteRoundtrip.data == byteSource.data)

  let promoted : Tensor Float [3] := Tensor.add bytes floats
  expect "UInt8 + Float infers Float output"
    (promoted.to (Array Float) == #[1.5, 3.5, 5.5])

  let multiplied : Tensor Float [3] := Tensor.mul bytes floats
  expect "UInt8 * Float infers Float output"
    (multiplied.to (Array Float) == #[0.5, 3.0, 7.5])

  let subtracted : Tensor Float [3] := bytes - floats
  expect "UInt8 - Float infers Float output"
    (subtracted.to (Array Float) == #[0.5, 0.5, 0.5])

  let divided : Tensor Float [3] := bytes / floats
  expect "UInt8 / Float infers Float output"
    (divided.to (Array Float) == #[2.0, 2.0 / 1.5, 3.0 / 2.5])

  let row : Fin 2 := ⟨1, by decide⟩
  let column : Fin 2 := ⟨0, by decide⟩
  expect "chained static indexing returns a scalar"
    (convertedMatrix[row][column] == 3)

  let floatCoordinate : Spec.Shape.Coord [3] :=
    (⟨1, by decide⟩, PUnit.unit)
  expect "typed coordinate lookup"
    (floats.at floatCoordinate == 1.5)
  let replaced := floats.set floatCoordinate 9.5
  expect "packed Float update"
    (replaced.to (Array Float) == #[0.5, 9.5, 2.5])
  let modified := replaced.modify floatCoordinate (· + 0.5)
  expect "packed Float modify"
    (modified.to (Array Float) == #[0.5, 10.0, 2.5])
  expect "runtime coordinate lookup"
    (floats.at? #[2] == some 2.5)
  expect "runtime coordinate validation rejects wrong rank"
    (floats.at? #[0, 0] == none)
  expect "runtime coordinate replacement"
    ((floats.set? #[0] 7.0).map (Tensor.to · (Array Float)) ==
      some #[7.0, 1.5, 2.5])

  let casted : Tensor Float [3] := Tensor.cast bytes Float
  expect "casting to Float selects Float storage"
    (casted.to (Array Float) == #[1.0, 2.0, 3.0])

  IO.println "  tensor storage and mixed-element promotion: passed"

end NN.Tests.Tensor.Storage
