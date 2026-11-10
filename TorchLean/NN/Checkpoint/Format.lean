/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.System.IO
public import TorchLean.NN.Data.Parsing
public import TorchLean.Tensor.Bytes
import TorchLean.NN.Data.Monad

/-!
# Binary Checkpoint IO

Internal framing helpers shared by TorchLean's binary checkpoint formats. Integer fields are
little-endian. A replacement write goes to a fresh sibling file, closes its handle and renames the
file over the destination, so on filesystems with atomic same-directory replacement a reader never
sees a partially written checkpoint.

Every binary format starts with a stable family magic followed by an unsigned 64-bit version.
Because the two fields are separate, a reader can tell a foreign file from a known checkpoint
written in an unsupported format version.
-/

@[expose] public section

namespace TorchLean
namespace Checkpoint

/-- Stable identity and current version of one binary checkpoint family. -/
structure Format where
  /-- Name used in diagnostics. -/
  name : String
  /-- Stable byte prefix shared by every version of this format. -/
  magic : ByteArray
  /-- Version written immediately after `magic`. -/
  version : Nat

/-- Write one natural number after checking that it fits in an unsigned 64-bit field. -/
def writeNat64 (formatName : String) (handle : IO.FS.Handle) (n : Nat) : IO Unit := do
  if n ≥ (2 : Nat) ^ 64 then
    throw <| IO.userError s!"{formatName}: metadata value does not fit in 64 bits: {n}"
  handle.write (Bytes.pushWordLE .empty 8 n)

/-- Read exactly `count` bytes or report a truncated checkpoint. -/
def readExact (formatName : String) (handle : IO.FS.Handle) (count : Nat) : IO ByteArray := do
  let bytes ← Data.readUpTo handle count
  if bytes.size < count then
    throw <| IO.userError
      s!"{formatName}: truncated checkpoint (wanted {count} bytes, got {bytes.size})"
  pure bytes

/-- When `readExact formatName handle count` returns, it returns exactly `count` bytes. -/
theorem readExact_size {formatName : String} {handle : IO.FS.Handle} {count : Nat}
    {bytes : ByteArray} (h : MonadAttach.CanReturn (readExact formatName handle count) bytes) :
    bytes.size = count := by
  obtain ⟨read, hRead, h⟩ := LawfulMonadAttach.canReturn_bind_imp' h
  have hLe := Data.readUpTo_size_le hRead
  split at h
  · obtain ⟨_, hThrow, _⟩ := LawfulMonadAttach.canReturn_bind_imp' h
    exact absurd hThrow EIO.not_canReturn_throw
  · cases LawfulMonadAttach.eq_of_canReturn_pure h
    omega

/-- Read one unsigned 64-bit little-endian field. -/
def readNat64 (formatName : String) (handle : IO.FS.Handle) : IO Nat := do
  let bytes ← readExact formatName handle 8
  pure (Bytes.wordFromLE bytes 0 8)

/-- Write a checkpoint header: the family magic, then the version as an unsigned 64-bit field. -/
def writeFormat (format : Format) (handle : IO.FS.Handle) : IO Unit := do
  handle.write format.magic
  writeNat64 format.name handle format.version

/--
Read and validate a binary checkpoint header against `format`.

A wrong magic means a different file family. A matching magic with another version fails with a
version diagnostic, so the file is never misreported as malformed payload data.
-/
def readFormat (format : Format) (handle : IO.FS.Handle) : IO Unit := do
  let magic ← readExact format.name handle format.magic.size
  if magic != format.magic then
    throw <| IO.userError s!"{format.name}: unsupported checkpoint family"
  let version ← readNat64 format.name handle
  if version != format.version then
    throw <| IO.userError <|
      s!"{format.name}: unsupported format version {version}; expected {format.version}."

/--
Write a checkpoint through an exclusively reserved sibling and rename it after closing the handle.

The callback must not keep the handle past its call, because `withFile` closes it when its last
reference is dropped.

A failed reservation leaves the existing sibling untouched. After the reservation, a write or rename
failure removes only this writer's temporary file. On filesystems with atomic same-directory
replacement, the destination is never exposed as a partially written checkpoint.
-/
def writeAtomically
    (path : System.FilePath) (write : IO.FS.Handle → IO Unit) : IO Unit := do
  let timestamp ← IO.monoNanosNow
  let temporary := System.FilePath.mk s!"{path}.tmp-{timestamp}"
  -- Reserve before entering cleanup: a colliding name belongs to another writer.
  let writeError? ← IO.FS.withFile temporary IO.FS.Mode.writeNew fun handle => do
    try
      write handle
      handle.flush
      pure none
    catch error =>
      pure (some error)
  try
    if let some error := writeError? then
      throw error
    IO.FS.rename temporary path
  catch error =>
    try
      IO.FS.removeFile temporary
    catch _ =>
      pure ()
    throw error

end Checkpoint
end TorchLean
