/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.CheckpointIO
public import NN.Tensor.Conversion
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public import NN.Runtime.Autograd.Torch.Core.Trainer.Parameters
public import NN.Tensor.Internal.Elab.TensorLiteral

/-!
# Model State IO

This module saves and restores the complete shape-indexed state of a TorchLean runtime module.
State includes trainable parameters and persistent buffers; optimizer state has a separate format.

Two formats are provided for runtime state packs.

## Lean Float Format

We encode each `Float` value by its IEEE-754 bit pattern (`Float.toBits : Float → UInt64`) and
store those bits as JSON natural numbers. This format is specific to `Float` by definition: the
payload is the binary64 bit pattern, so the codec below is not generic over the scalar type. The
streamed float32 format further down is the scalar-generic one; it takes explicit `encode`/`decode`
functions between `α` and `Float32`.

This is:
- exact (round-trips every NaN payload and subnormal),
- stable across locales, and
- easy to validate (length = `Spec.Shape.size`).

The file layout is:

```json
{
  "format": "torchlean_state_bits_v1",
  "state": [
    { "shape": [d1, d2, ...], "values": [u64bits, u64bits, ...] },
    ...
  ]
}
```

## Runtime Float32 Format

Native `Float32` modules and device-backed modules use a versioned binary stream. Each state tensor
records its rank, dimensions, and element count before its little-endian binary32 payload. Loading
checks all metadata against the expected shape-indexed state and rejects unsupported versions,
truncated payloads, and trailing data. The stream is written one tensor at a time, so saving a large
CUDA model does not construct a second host-side copy of the whole checkpoint.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model
namespace StateIO

open Spec TorchLean

/-- Format tag stored in exact-bit model-state JSON files. -/
def formatTag : String := "torchlean_state_bits_v1"

/-- Versioned header for streamed runtime float32 state checkpoints. -/
def float32StreamFormat : Torch.Internal.CheckpointIO.Format where
  name := "TorchLean float32 state checkpoint"
  magic := "TLSF32B".toUTF8
  version := 1

/-- Encode a natural number as a JSON number. -/
def jsonNat (n : Nat) : Lean.Json :=
  Lean.Json.num (Lean.JsonNumber.fromInt (Int.ofNat n))

/-- Encode a Float by writing its exact IEEE bit pattern as a JSON natural number. -/
def floatToJsonBits (x : Float) : Lean.Json :=
  jsonNat x.toBits.toNat

/-- Decode a JSON natural number as the exact IEEE bit pattern of a Float. -/
def jsonBitsToFloat (j : Lean.Json) : Except String Float := do
  let n ← Lean.Json.getNat? j
  let limit : Nat := (2 : Nat) ^ 64
  if n >= limit then
    throw s!"StateIO: float bits out of range (expected < 2^64, got {n})"
  let bits : UInt64 := UInt64.ofNat n
  pure (Float.ofBits bits)

/-- Encode one Float tensor as shape metadata plus exact IEEE bit-pattern values. -/
def tensorToJsonBits (s : Shape) (t : Tensor Float s) : Lean.Json :=
  let dims : Lean.Json := Lean.Json.arr (Shape.toList s |>.toArray |>.map jsonNat)
  let values : Lean.Json :=
    Lean.Json.arr ((Tensor.to t (Array Float)).map floatToJsonBits)
  Lean.Json.mkObj [("shape", dims), ("values", values)]

/-- Decode one shape-checked Float tensor from the exact-bit state format. -/
def tensorFromJsonBits (tag : String) (s : Shape) (j : Lean.Json) :
    Except String (Tensor Float s) := do
  let o ← Lean.Json.getObj? j
  let shapeJ := (o.get? "shape").getD Lean.Json.null
  let valuesJ := (o.get? "values").getD (Lean.Json.arr #[])
  let dimsArr ← Lean.Json.getArr? shapeJ
  let dims : List Nat ←
    dimsArr.toList.mapM (fun d => Lean.Json.getNat? d)
  if Shape.ofList dims != s then
    throw s!"{tag}: shape mismatch (file={dims}, expected={Shape.toList s})"
  let valsArr ← Lean.Json.getArr? valuesJ
  let vals : Array Float ← valsArr.mapM jsonBitsToFloat
  if hSize : vals.size = Shape.size s then
    pure <| (Tensor.from vals).reshape s (by simpa [Shape.size] using hSize)
  else
    throw <|
      s!"{tag}: expected {Shape.size s} scalars for shape {Shape.toList s}, " ++
        s!"got {vals.size}"

/-- Encode shape-indexed model state as the JSON array stored under `state`. -/
def stateToJsonBits {ss : List Shape} : TorchLean.TensorPack Float ss → Lean.Json
  | .nil => Lean.Json.arr #[]
  | .cons (s := s) t ts =>
      match stateToJsonBits (ss := _) ts with
      | Lean.Json.arr xs =>
          Lean.Json.arr (#[tensorToJsonBits s t] ++ xs)
      | _ => Lean.Json.arr #[]

/-- Decode an expected state layout from a tensor array, starting at `offset`. -/
def stateFromJsonBitsArray (tag : String) (xs : Array Lean.Json) :
    {ss : List Shape} → (offset : Nat) → Except String (TorchLean.TensorPack Float ss)
  | .nil, offset => do
      unless offset = xs.size do
        throw s!"{tag}: unexpected {xs.size - offset} trailing state tensor(s)"
      pure .nil
  | .cons s ss, offset => do
      let headJson ← match xs[offset]? with
        | some value => pure value
        | none => throw s!"{tag}: missing state tensor {offset + 1}/{offset + 1 + ss.length}"
      let head ← tensorFromJsonBits (tag := tag) s headJson
      let tail ← stateFromJsonBitsArray (tag := tag) xs (ss := ss) (offset + 1)
      pure (.cons head tail)

/-- Decode the `state` JSON array into the expected shape-indexed state pack. -/
def stateFromJsonBits (tag : String) {ss : List Shape} (json : Lean.Json) :
    Except String (TorchLean.TensorPack Float ss) := do
  let xs ← Lean.Json.getArr? json
  stateFromJsonBitsArray (tag := tag) xs (ss := ss) 0

/-- Write Float model state using exact IEEE bit patterns rather than decimal floats. -/
def writeStateBits (path : System.FilePath) {ss : List Shape}
    (state : TorchLean.TensorPack Float ss) (pretty : Bool := true) : IO Unit := do
  let top : Lean.Json :=
    Lean.Json.mkObj [("format", Lean.Json.str formatTag), ("state", stateToJsonBits state)]
  let s := if pretty then top.pretty else top.compress
  Torch.Internal.CheckpointIO.writeAtomically path fun handle =>
    handle.write s.toUTF8

/-- Read Float model state previously written by `writeStateBits`. -/
def readStateBits (path : System.FilePath) {ss : List Shape} :
    IO (Except String (TorchLean.TensorPack Float ss)) := do
  let s ← IO.FS.readFile path
  match Lean.Json.parse s with
  | Except.error e =>
      pure (Except.error s!"StateIO: JSON parse error: {e}")
  | Except.ok j =>
      match Lean.Json.getObj? j with
      | Except.error e =>
          pure (Except.error s!"StateIO: expected object: {e}")
      | Except.ok o =>
          let fmt := (o.get? "format").getD (Lean.Json.str "")
          match Lean.Json.getStr? fmt with
          | Except.error _ =>
              pure (Except.error "StateIO: missing `format` string")
          | Except.ok t =>
              if t != formatTag then
                pure (Except.error s!"StateIO: unsupported format: {t}")
              else
                let stateJson := (o.get? "state").getD (Lean.Json.arr #[])
                pure (stateFromJsonBits (tag := "StateIO") (ss := ss) stateJson)

/-! ## Streaming float32 module checkpoints -/

/-- Append one binary32 bit pattern in little-endian byte order. -/
def pushUInt32LE (out : ByteArray) (bits : UInt32) : ByteArray := Id.run do
  let mut out := out
  let mut value := bits.toNat
  for _ in [0:4] do
    out := out.push (UInt8.ofNat (value % 256))
    value := value / 256
  return out

/-- Decode one little-endian 32-bit word at `offset`. -/
def uint32FromLE (bytes : ByteArray) (offset : Nat) : UInt32 := Id.run do
  let mut value := 0
  let mut place := 1
  for i in [0:4] do
    value := value + bytes[offset + i]!.toNat * place
    place := place * 256
  return UInt32.ofNat value

/-- Encode a tensor as exact little-endian binary32 values. -/
def tensorToFloat32Bytes {α : Type} [TorchLean.Storage α] (encode : α → Float32)
    {shape : Shape}
    (tensor : Tensor α shape) : ByteArray :=
  tensor.foldl
    (fun out value => pushUInt32LE out (encode value).toBits)
    ByteArray.empty

/-- Decode an exact binary32 payload into a shape-checked tensor. -/
def tensorFromFloat32Bytes {α : Type} [TorchLean.Storage α] (decode : Float32 → α)
    (shape : Shape)
    (bytes : ByteArray) : Except String (Tensor α shape) := do
  let expectedBytes := Shape.size shape * 4
  if bytes.size != expectedBytes then
    throw <|
      s!"StateIO: float32 payload size mismatch for {Shape.pretty shape} " ++
        s!"(got={bytes.size}, expected={expectedBytes})"
  let values := (Array.range (Shape.size shape)).map fun i =>
    decode (Float32.ofBits (uint32FromLE bytes (4 * i)))
  have hSize : values.size = Shape.size shape := by simp [values]
  pure <| (Tensor.from values).reshape shape (by simpa [Shape.size] using hSize)

/-- Write one expected tensor shape to a streaming checkpoint. -/
def writeShape (handle : IO.FS.Handle) (shape : Shape) : IO Unit := do
  let dims := Shape.toList shape
  Torch.Internal.CheckpointIO.writeNat64 "StateIO" handle dims.length
  for dim in dims do
    Torch.Internal.CheckpointIO.writeNat64 "StateIO" handle dim
  Torch.Internal.CheckpointIO.writeNat64 "StateIO" handle (Shape.size shape)

/-- Read and validate one tensor shape from a streaming checkpoint. -/
def readShape (handle : IO.FS.Handle) (expected : Shape) : IO Unit := do
  let rank ← Torch.Internal.CheckpointIO.readNat64 "StateIO" handle
  let mut reversedDims := []
  for _ in [0:rank] do
    reversedDims := (← Torch.Internal.CheckpointIO.readNat64 "StateIO" handle) :: reversedDims
  let dims := reversedDims.reverse
  let count ← Torch.Internal.CheckpointIO.readNat64 "StateIO" handle
  if Shape.ofList dims != expected then
    throw <| IO.userError
      s!"StateIO: checkpoint shape mismatch (file={dims}, expected={Shape.toList expected})"
  if count != Shape.size expected then
    throw <| IO.userError <|
      s!"StateIO: checkpoint element count mismatch for {Shape.pretty expected} "
        ++ s!"(file={count}, expected={Shape.size expected})"

/-- Obtain one state tensor as raw float32 bytes without materializing the whole state pack. -/
def tensorFloat32Bytes {α : Type} [TorchLean.Storage α] (encode : α → Float32)
    {shape : Shape}
    (tensorRef : Torch.Param α shape) : IO ByteArray := do
  match ← tensorRef.cudaValue.get with
  | some value =>
      if value.s != shape then
        throw <| IO.userError <|
          s!"StateIO: CUDA state-tensor shape mismatch "
            ++ s!"(buffer={Shape.pretty value.s}, expected={Shape.pretty shape})"
      Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO value.buf
  | none =>
      let tensor ← tensorRef.value.get
      pure <| tensorToFloat32Bytes encode tensor

/-- Stream shape-indexed runtime state to an open checkpoint handle. -/
def writeStateFloat32 {α : Type} [TorchLean.Storage α] (encode : α → Float32)
    (handle : IO.FS.Handle) :
    {shapes : List Shape} → Torch.ParamList α shapes → IO Unit
  | .nil, .nil => pure ()
  | .cons shape _, .cons tensorRef state => do
      writeShape handle shape
      let bytes ← tensorFloat32Bytes encode tensorRef
      let expectedBytes := Shape.size shape * 4
      if bytes.size != expectedBytes then
        throw <| IO.userError <|
          s!"StateIO: float32 payload size mismatch for {Shape.pretty shape} "
            ++ s!"(got={bytes.size}, expected={expectedBytes})"
      handle.write bytes
      writeStateFloat32 encode handle state

/--
Write runtime model state as a streamed float32 checkpoint.

Only one tensor payload is resident on the host at a time. This is the appropriate format for
large CUDA models; it records the exact values used by the float32 runtime instead of expanding
them into one in-memory JSON tree.
-/
def writeModuleStateFloat32 {α : Type} [TorchLean.Storage α] (encode : α → Float32)
    (path : System.FilePath) {shapes : List Shape}
    (state : Torch.ParamList α shapes) : IO Unit := do
  Torch.Internal.CheckpointIO.writeAtomically path fun handle => do
    Torch.Internal.CheckpointIO.writeFormat float32StreamFormat handle
    Torch.Internal.CheckpointIO.writeNat64 "StateIO" handle shapes.length
    writeStateFloat32 encode handle state

/-- Check whether a file begins with the streaming float32 checkpoint header. -/
def isModuleStateFloat32 (path : System.FilePath) : IO Bool := do
  Torch.Internal.CheckpointIO.hasFormatMagic float32StreamFormat path

/-- Read one float32 payload directly into an existing runtime state tensor. -/
def readTensorFloat32Into {α : Type} [TorchLean.Storage α] (decode : Float32 → α)
    (useCuda : Bool) (handle : IO.FS.Handle) {shape : Shape}
    (tensorRef : Torch.Param α shape) : IO Unit := do
  readShape handle shape
  let bytes ← Torch.Internal.CheckpointIO.readExact
    "StateIO" handle (Shape.size shape * 4)
  if useCuda then
    let buffer ← Runtime.Autograd.Cuda.Buffer.ofFloat32BytesIO bytes
    Torch.Internal.setParamCudaValue tensorRef { s := shape, buf := buffer }
  else
    let tensor ← match tensorFromFloat32Bytes decode shape bytes with
      | .ok tensor => pure tensor
      | .error message => throw <| IO.userError message
    Torch.Internal.setParamHostValue tensorRef tensor

/-- Stream a checkpoint into shape-indexed runtime state. -/
def readStateFloat32Into {α : Type} [TorchLean.Storage α] (decode : Float32 → α)
    (useCuda : Bool) (handle : IO.FS.Handle) :
    {shapes : List Shape} → Torch.ParamList α shapes → IO Unit
  | .nil, .nil => pure ()
  | .cons _ _, .cons tensorRef state => do
      readTensorFloat32Into decode useCuda handle tensorRef
      readStateFloat32Into decode useCuda handle state

/-- Validate every tensor record and payload without changing the destination parameters. -/
def validateStateFloat32 (handle : IO.FS.Handle) : (shapes : List Shape) → IO Unit
  | .nil => pure ()
  | .cons shape shapes => do
      readShape handle shape
      let _ ← Torch.Internal.CheckpointIO.readExact
        "StateIO" handle (Shape.size shape * 4)
      validateStateFloat32 handle shapes

/-- Read and validate the common header of a streamed float32 checkpoint. -/
def readAndCheckFloat32Header
    (handle : IO.FS.Handle) (expectedTensorCount : Nat) : IO Unit := do
  Torch.Internal.CheckpointIO.readFormat float32StreamFormat handle
  let tensorCount ← Torch.Internal.CheckpointIO.readNat64 "StateIO" handle
  if tensorCount != expectedTensorCount then
    throw <| IO.userError <|
      s!"StateIO: checkpoint state-tensor count mismatch "
        ++ s!"(file={tensorCount}, expected={expectedTensorCount})"

/-- Reject extra bytes after the last expected tensor payload. -/
def checkFloat32EndOfFile (handle : IO.FS.Handle) : IO Unit := do
  let trailing ← handle.read 1
  if trailing.size != 0 then
    throw <| IO.userError "StateIO: trailing bytes after final state tensor"

/-- Load a streamed float32 checkpoint into existing runtime state. -/
def readModuleStateFloat32Into {α : Type} [TorchLean.Storage α] (decode : Float32 → α)
    (path : System.FilePath) (useCuda : Bool) {shapes : List Shape}
    (state : Torch.ParamList α shapes) : IO Unit := do
  -- Check the complete stream first. Under ordinary file ownership this prevents a malformed or
  -- truncated checkpoint from leaving the live module with only a prefix of its state
  -- replaced, while retaining one-tensor-at-a-time memory use for large CUDA models.
  IO.FS.withFile path IO.FS.Mode.read fun handle => do
    readAndCheckFloat32Header handle shapes.length
    validateStateFloat32 handle shapes
    checkFloat32EndOfFile handle
  IO.FS.withFile path IO.FS.Mode.read fun handle => do
    readAndCheckFloat32Header handle shapes.length
    readStateFloat32Into decode useCuda handle state
    checkFloat32EndOfFile handle

end StateIO
end Model
end Autograd
end Runtime
