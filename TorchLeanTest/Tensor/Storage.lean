/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Reduce.Fold
public import TorchLean.Tensor.Tactic.Init

/-!
# Tensor storage through the C kernels

These checks run the C kernels of `csrc/cpu/torchlean_tensor.c` and the core `FloatArray` and
`ByteArray` primitives that compiled code calls. The packed `Float` arithmetic truncates unequal
buffers, `torchlean_float_array_append_slice` keeps order under aliasing and clamps big-`Nat`
bounds, byte slicing through `ByteArray.copySlice` clamps before it allocates, and mixed `UInt8`
Compiled stacking runs `dimFast`, which `Rep.dimFast_eq` proves equal to `dim`. Its packed `Float`
copies run native C code, so its values are compared with coordinate construction, including signed
zeros, a NaN, and empty tensors.
including signed zeros, NaN payloads, and empty tensors.
-/

@[expose] public section

namespace TorchLean.Test.Storage

/-- Fail with `label` unless `condition` holds. -/
def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"tensor storage check failed: {label}"

/-- The `UInt8` vector `[1, 2, 3]`, left operand of the mixed `UInt8` and `Float` arithmetic. -/
def bytes : Tensor UInt8 [3] :=
  Tensor.ofFn fun
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 2
    | _ => 3

/-- The `Float` vector `[0.5, 1.5, 2.5]`, right operand of the mixed arithmetic and input of the
coordinate reads and updates. -/
def floats : Tensor Float [3] :=
  Tensor.ofFn fun
    | ⟨0, _⟩ => 0.5
    | ⟨1, _⟩ => 1.5
    | _ => 2.5

/-- Native byte slicing must clamp before allocating, even for heap-allocated naturals. -/
def checkByteSliceBounds : IO Unit := do
  let source := ByteArray.mk #[10, 20, 30, 40]
  let output := ByteArray.mk #[1, 2]
  let endpoints := [0, 1, 3, 4, 5, 2^64, 2^64 + 1, 2^200]
  for start in endpoints do
    for stop in endpoints do
      let actual := Storage.byteBufferAppendSlice source start stop output
      expect s!"byte slice clamp {start}:{stop}"
        (actual.data == output.data ++ source.data.extract start stop)

/-- Compare compiled stacking with coordinate construction on packed and boxed storage. -/
def checkStack : IO Unit := do
  let components : Fin 5 → Tensor Float [2, 3] := fun i =>
    Tensor.generateFlat [2, 3] fun index =>
      if index % 3 = 0 then
        if index = 0 then -0.0 else Float.ofBits 0x7ff8000000000017
      else (i.val * 10 + index).toFloat
  let actual := Tensor.dim components
  let expected : Tensor Float [5, 2, 3] :=
    Tensor.Rep.ofFn fun coordinate => components coordinate.1 coordinate.2
  expect "Float stacking agrees bit for bit with coordinate construction"
    (Tensor.foldl (· && ·) true
      (Tensor.zipWith (fun x y => x.toBits == y.toBits) actual expected))
  let boxed : Fin 4 → Tensor Nat [3] := fun i =>
    Tensor.ofFn fun j => i.val * 100 + j.val
  let boxedActual := Tensor.dim boxed
  let boxedExpected : Tensor Nat [4, 3] :=
    Tensor.Rep.ofFn fun coordinate => boxed coordinate.1 coordinate.2
  expect "boxed stacking agrees with coordinate construction"
    (Tensor.foldl (· && ·) true
      (Tensor.zipWith (· == ·) boxedActual boxedExpected))
  let emptyTail := Tensor.dim
    (fun (_ : Fin 6) => Tensor.full [0] (0 : Float))
  let emptyHead := Tensor.dim
    (fun (i : Fin 0) => (nomatch i : Tensor Float [3]))
  expect "stacking preserves empty outputs"
    (Tensor.foldl (fun count _ => count + 1) (0 : Nat) emptyTail == 0 &&
      Tensor.foldl (fun count _ => count + 1) (0 : Nat) emptyHead == 0)

/--
Check native slice bounds, aliasing, buffer round trips, and mixed-element arithmetic.
-/
def run : IO Unit := do
  checkStack
  checkByteSliceBounds
  let floatSource := FloatArray.mk #[0.25, 0.5, 0.75]
  let floatRoundtrip : FloatArray := (Tensor.from floatSource).to FloatArray
  expect "FloatArray conversion preserves packed values"
    (floatRoundtrip.data == floatSource.data)

  let floatStorage := (inferInstance : Storage Float)
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
    ("add", Native.floatBufferBinary .add, #[6.0]),
    ("sub", Native.floatBufferBinary .sub, #[-2.0]),
    ("mul", Native.floatBufferBinary .mul, #[8.0]),
    ("div", Native.floatBufferBinary .div, #[0.5])]
  for (label, kernel, expected) in kernels do
    expect s!"native {label} truncates unequal buffer lengths"
      ((kernel shortFloat longFloat).data == expected)
    expect s!"native {label} accepts empty left buffers"
      (kernel emptyFloat longFloat).data.isEmpty
    expect s!"native {label} accepts empty right buffers"
      (kernel longFloat emptyFloat).data.isEmpty
  expect "byte/float addition truncates unequal buffers"
    ((Native.byteFloatBufferBinary .add shortByte longFloat).data ==
      #[6.0])
  expect "float/byte addition truncates unequal buffers"
    ((Native.floatByteBufferBinary .add shortFloat longByte).data ==
      #[6.0])
  expect "byte/float addition accepts empty buffers"
    (Native.byteFloatBufferBinary .add
      emptyByte longFloat).data.isEmpty
  expect "float/byte addition accepts empty buffers"
    (Native.floatByteBufferBinary .add
      longFloat emptyByte).data.isEmpty

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

  let floatCoordinate : Shape.Coord [3] :=
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

  IO.println "  tensor storage and mixed-element promotion: passed"

end TorchLean.Test.Storage
