/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Std.Do.Triple.SpecLemmas

/-!
# Binary Checkpoint IO

Internal framing utilities shared by TorchLean's binary checkpoint formats. Integer fields use an
explicit little-endian encoding. Replacement writes use a fresh sibling file and close its handle
before renaming it over the destination. On filesystems with atomic same-directory replacement,
this prevents readers from observing a partially written checkpoint.

Every binary format starts with a stable family magic followed by an explicit unsigned 64-bit
version. Keeping these fields separate lets readers distinguish a foreign file from a known
checkpoint written by an unsupported format version.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch
namespace Internal
namespace CheckpointIO

/-- Stable identity and current version of one binary checkpoint family. -/
structure Format where
  /-- Name used in diagnostics. -/
  name : String
  /-- Stable byte prefix shared by every version of this format. -/
  magic : ByteArray
  /-- Version written immediately after `magic`. -/
  version : Nat

/-- Encode a natural number as an unsigned 64-bit little-endian field. -/
def nat64LE (n : Nat) : ByteArray := Id.run do
  let mut out := ByteArray.empty
  let mut value := n
  for _ in [0:8] do
    out := out.push (UInt8.ofNat (value % 256))
    value := value / 256
  return out

/-- Write one natural number after checking that it fits in an unsigned 64-bit field. -/
def writeNat64 (formatName : String) (handle : IO.FS.Handle) (n : Nat) : IO Unit := do
  if n ≥ (2 : Nat) ^ 64 then
    throw <| IO.userError s!"{formatName}: metadata value does not fit in 64 bits: {n}"
  handle.write (nat64LE n)

/-- Decode an unsigned 64-bit little-endian field. -/
def nat64FromLE (formatName : String) (bytes : ByteArray) : Except String Nat := do
  if bytes.size != 8 then
    throw s!"{formatName}: expected 8 metadata bytes, got {bytes.size}"
  let mut value := 0
  let mut place := 1
  for i in [0:8] do
    value := value + bytes[i]!.toNat * place
    place := place * 256
  pure value

/-- Read exactly `count` bytes or report a truncated checkpoint. -/
partial def readExact
    (formatName : String) (handle : IO.FS.Handle) (count : Nat)
    (acc : ByteArray := ByteArray.empty) : IO ByteArray := do
  if acc.size = count then
    pure acc
  else
    let chunk ← handle.read (USize.ofNat (count - acc.size))
    if chunk.size = 0 then
      throw <| IO.userError
        s!"{formatName}: truncated checkpoint (wanted {count} bytes, got {acc.size})"
    readExact formatName handle count (acc.append chunk)

/-- Read one unsigned 64-bit little-endian field. -/
def readNat64 (formatName : String) (handle : IO.FS.Handle) : IO Nat := do
  match nat64FromLE formatName (← readExact formatName handle 8) with
  | .ok value => pure value
  | .error message => throw <| IO.userError message

/-- Write the stable family magic and explicit version of a binary checkpoint. -/
def writeFormat (format : Format) (handle : IO.FS.Handle) : IO Unit := do
  handle.write format.magic
  writeNat64 format.name handle format.version

/--
Read and validate a binary checkpoint header.

A wrong magic identifies a different file family. A matching magic with a different version gives
an explicit version diagnostic instead of being misreported as malformed payload data.
-/
def readFormat (format : Format) (handle : IO.FS.Handle) : IO Unit := do
  let magic ← readExact format.name handle format.magic.size
  if magic != format.magic then
    throw <| IO.userError s!"{format.name}: unsupported checkpoint family"
  let version ← readNat64 format.name handle
  if version != format.version then
    throw <| IO.userError <|
      s!"{format.name}: unsupported format version {version}; expected {format.version}"

/-- Test whether a file belongs to a binary checkpoint family, without checking its version. -/
def hasFormatMagic (format : Format) (path : System.FilePath) : IO Bool :=
  IO.FS.withFile path IO.FS.Mode.read fun handle => do
    let magic ← handle.read (USize.ofNat format.magic.size)
    pure (magic == format.magic)

/--
Write a checkpoint through a temporary sibling and rename it after the handle has been closed.

If writing or renaming fails, the temporary file is removed. The destination is never exposed as a
partially written checkpoint on filesystems with atomic same-directory replacement.
-/
def writeAtomically
    (path : System.FilePath) (write : IO.FS.Handle → IO Unit) : IO Unit := do
  let timestamp ← IO.monoNanosNow
  let temporary := System.FilePath.mk s!"{path}.tmp-{timestamp}"
  try
    IO.FS.withFile temporary IO.FS.Mode.write fun handle => do
      write handle
      handle.flush
    IO.FS.rename temporary path
  catch error =>
    try
      IO.FS.removeFile temporary
    catch _ =>
      pure ()
    throw error

end CheckpointIO
end Internal
end Torch
end Autograd
end Runtime
