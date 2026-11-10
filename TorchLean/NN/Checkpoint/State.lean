/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Checkpoint.Format
public import TorchLean.Tensor.Constructors
public import TorchLean.Tensor.Pack
public import TorchLean.NN.Checkpoint.Encoding

/-!
# Model State IO

A state checkpoint holds the complete shape-indexed state of a TorchLean model: its trainable
parameters and its persistent buffers. Optimizer state has a separate format.

`Checkpoint.Encoding` gives each scalar type a format tag and maps every value to one word of a
fixed bit width. All scalar types share one versioned binary container:

1. the magic `TLSTATE` and the format version of `stateFormat`,
2. the scalar format tag, as its UTF-8 byte length followed by those bytes,
3. the number of state tensors, and
4. for each tensor, its rank, dimensions, and element count, then its row-major scalar words, each
   in `(width + 7) / 8` little-endian bytes.

Counts, dimensions, and lengths are unsigned 64-bit little-endian fields. The container is exact
for every exported scalar encoding, including subnormals and configured NaN payloads, and does not
depend on the locale. The writer and the reader stream one tensor record at a time. Loading checks
the format tag, the tensor count, every shape and element count, every scalar word, and the end of
the file before it returns a state pack.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace Checkpoint

/-- Versioned header of model state checkpoints. -/
def stateFormat : Format where
  name := "TorchLean state checkpoint"
  magic := "TLSTATE".toUTF8
  version := 1

/-- Write the shape fields of a tensor record: its rank, its dimensions and its element count. -/
def writeShape (handle : IO.FS.Handle) (shape : Shape) : IO Unit := do
  let dims : List Nat := shape
  writeNat64 "StateIO" handle dims.length
  for dim in dims do
    writeNat64 "StateIO" handle dim
  writeNat64 "StateIO" handle (Shape.size shape)

/--
Read the shape fields of a tensor record and fail unless its rank, dimensions and element count
match `expected`.
-/
def readShape (handle : IO.FS.Handle) (expected : Shape) : IO Unit := do
  let rank ← readNat64 "StateIO" handle
  -- The destination fixes the rank, so a wrong rank fails before any dimension is read.
  if rank != expected.rank then
    throw <| IO.userError
      s!"StateIO: checkpoint rank mismatch (file={rank}, expected={expected.rank})"
  let mut reversedDims := []
  for _ in [0:rank] do
    reversedDims := (← readNat64 "StateIO" handle) :: reversedDims
  let dims := reversedDims.reverse
  let count ← readNat64 "StateIO" handle
  if dims != expected then
    throw <| IO.userError
      s!"StateIO: checkpoint shape mismatch (file={dims}, expected={expected})"
  if count != Shape.size expected then
    throw <| IO.userError <|
      s!"StateIO: checkpoint element count mismatch for {expected} "
        ++ s!"(file={count}, expected={Shape.size expected})"

section Tensors

variable {α : Type} [Storage α] [Checkpoint.Encoding α]

/--
Write one tensor record: its shape fields, then its row-major scalar words. An encoded word that
does not fit in the encoding's width is an error.
-/
def writeTensor (handle : IO.FS.Handle) (shape : Shape) (tensor : Tensor α shape) :
    IO Unit := do
  writeShape handle shape
  let width := Checkpoint.Encoding.width (α := α)
  let bound := 2 ^ width
  let count := (width + 7) / 8
  let mut bytes := ByteArray.emptyWithCapacity (Shape.size shape * count)
  for value in Tensor.to tensor (Array α) do
    let word := Checkpoint.Encoding.encode value
    if word ≥ bound then
      throw <| IO.userError s!"StateIO: encoded scalar does not fit in {width} bits"
    bytes := Bytes.pushWordLE bytes count word
  handle.write bytes

/-- Read one tensor record of the expected shape and decode its row-major scalar words. -/
def readTensor (handle : IO.FS.Handle) (shape : Shape) : IO (Tensor α shape) := do
  readShape handle shape
  let count := (Checkpoint.Encoding.width (α := α) + 7) / 8
  let bytes ← readExact "StateIO" handle (Shape.size shape * count)
  IO.ofExcept <| Tensor.generateFlatM shape fun index =>
    Checkpoint.Encoding.decode (Bytes.wordFromLE bytes (count * index.val) count)

/-- Write the tensor records of a state pack in order. -/
def writeTensors (handle : IO.FS.Handle) :
    {shapes : List Shape} → TensorPack α shapes → IO Unit
  | .nil, .nil => pure ()
  | .cons shape _, .cons tensor tensors => do
      writeTensor handle shape tensor
      writeTensors handle tensors

/-- Read the tensor records of the expected state layout in order. -/
def readTensors (handle : IO.FS.Handle) :
    (shapes : List Shape) → IO (TensorPack α shapes)
  | .nil => pure .nil
  | .cons shape shapes => do
      let tensor ← readTensor handle shape
      pure (.cons tensor (← readTensors handle shapes))

/-- Write a state pack atomically, tagged with its scalar format. -/
def writeStateBits (path : System.FilePath) {ss : List Shape}
    (state : TensorPack α ss) : IO Unit :=
  writeAtomically path fun handle => do
    writeFormat stateFormat handle
    let tag := (Checkpoint.Encoding.format (α := α)).toUTF8
    writeNat64 "StateIO" handle tag.size
    handle.write tag
    writeNat64 "StateIO" handle ss.length
    writeTensors handle state

/--
Read a state pack of layout `ss` from `path`. Loading fails on a wrong header or format tag, a
different tensor count, a shape or element-count mismatch, a malformed scalar word, a truncated
payload or trailing bytes.
-/
def readStateBits (path : System.FilePath) {ss : List Shape} :
    IO (TensorPack α ss) :=
  IO.FS.withFile path IO.FS.Mode.read fun handle => do
    readFormat stateFormat handle
    let tag := Checkpoint.Encoding.format (α := α)
    let length ← readNat64 "StateIO" handle
    -- Read at most 1024 bytes past the expected tag length, so a corrupt length field cannot force
    -- a large allocation.
    let stored ← readExact "StateIO" handle
      (min length (tag.utf8ByteSize + 1024))
    unless length == tag.utf8ByteSize && stored == tag.toUTF8 do
      let shown := match String.fromUTF8? stored with
        | some text => text.quote
        | none => s!"<{stored.size} bytes>"
      throw <| IO.userError s!"StateIO: unsupported format {shown} (expected {tag.quote})"
    let tensorCount ← readNat64 "StateIO" handle
    if tensorCount != ss.length then
      throw <| IO.userError <|
        s!"StateIO: checkpoint state-tensor count mismatch "
          ++ s!"(file={tensorCount}, expected={ss.length})"
    let state ← readTensors (α := α) handle ss
    if (← handle.read 1).size != 0 then
      throw <| IO.userError "StateIO: trailing bytes after final state tensor"
    pure state

end Tensors

end Checkpoint
end TorchLean
