/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.CPU
public import TorchLean.NN.Session.Config

/-!
# Eager Session State

The tapes and side tables shared by eager operations. Parameters own their CUDA mirrors, and the
session owns recorded intermediates. Handle generations separate consecutive forward passes.
-/

public section

namespace TorchLean

/-- The mutable cells owned by a parameter, without reading or comparing tensor elements.

Two registrations share optimizer storage only when all three cells coincide. Equal values and
equal names do not make separately allocated parameters aliases. The session holds this
descriptor, and the public `Param` record and its constructor do not mention it.
-/
structure Param.Cells (α : Type) [Storage α] where
  /-- Shape of the parameter. -/
  shape : Shape
  /-- The `Param.value` cell of the parameter. -/
  value : IO.Ref (Tensor α shape)
  /-- The `Param.cudaValue` cell of the parameter. -/
  cudaValue : IO.Ref (Option LibTorch.SomeBuffer)
  /-- The `Param.hostSynced` cell of the parameter. -/
  hostSynced : IO.Ref Bool

/-- Compare mutable-cell identity using Lean's safe stateful reference operation. -/
def Param.Cells.same {α : Type} [Storage α]
    (left right : Cells α) : IO Bool := do
  if h : left.shape = right.shape then
    let rightValue : IO.Ref (Tensor α left.shape) := h.symm ▸ right.value
    unless ← left.value.ptrEq rightValue do
      return false
    unless ← left.cudaValue.ptrEq right.cudaValue do
      return false
    left.hostSynced.ptrEq right.hostSynced
  else
    pure false

private structure Param.StorageKey where
  shape : Shape
  value : USize
  cudaValue : USize
  hostCurrent : USize
  deriving Ord

/--
The key of a storage: its shape and the addresses of its three reference cells, never tensor data
or CUDA allocations. `Param.Cells.canonicalIndices` retains each representative, so its cells
cannot be freed and their addresses reused while its key is present. Keys are local to one
grouping operation.
-/
private unsafe def Param.Storage.addressKey {α : Type} [Storage α]
    (storage : Cells α) : IO StorageKey :=
  pure ⟨storage.shape, ptrAddrUnsafe storage.value, ptrAddrUnsafe storage.cudaValue,
    ptrAddrUnsafe storage.hostSynced⟩

/--
A partition hint, not an identity decision: compiled code runs `addressKey`, which reads the
live addresses of the three reference cells. The logic has no model of addresses, so the key is
opaque, and `canonicalIndices` checks every candidate in a bucket with `Cells.same`.
-/
@[implemented_by addressKey]
private opaque Param.Storage.key {α : Type} [Storage α]
    (storage : Cells α) : IO StorageKey

/--
Map each present storage to the index of its first occurrence, with `none` at absent slots.

A balanced tree of full-width reference keys avoids pairwise searches among independent storages,
and reference equality decides even when keys collide. Representatives keep their cells alive
until the operation ends. Neither keys nor references are cached between calls, so a different
parameter list or recording can change its alias layout with no cache to invalidate.
-/
def Param.Cells.canonicalIndices {α : Type} [Storage α]
    (storages : Array (Option (Cells α))) : IO (Array (Option Nat)) := do
  let mut representatives :
      Std.TreeMap StorageKey (Array (Nat × Cells α)) := ∅
  let mut slots : Array (Option Nat) := Array.emptyWithCapacity storages.size
  for storage? in storages do
    let index := slots.size
    match storage? with
    | none => slots := slots.push none
    | some storage =>
        let key ← Storage.key storage
        let bucket := (representatives.get? key).getD #[]
        let mut canonical := index
        for (earlier, previous) in bucket do
          if ← storage.same previous then
            canonical := earlier
            break
        if canonical == index then
          representatives := representatives.insert key (bucket.push (index, storage))
        slots := slots.push (some canonical)
  pure slots

/--
The tape a session records on. Its values follow the session's device: shape-erased tensors on the
CPU, device buffers on CUDA. A session has exactly one, made empty for `options.device` when the
session opens and again at every reset.
-/
inductive Session.DeviceTape (α : Type) [Storage α] where
  /-- The tape of a CPU session. -/
  | cpu (tape : Tape (SomeTensor α))
  /-- The tape of a CUDA session. -/
  | cuda (tape : Tape LibTorch.SomeBuffer)

namespace Session.DeviceTape

/-- An empty tape for a session on `device`. -/
def empty {α : Type} [Storage α] (device : Backend.Device) : DeviceTape α :=
  if device == .cuda then .cuda Tape.empty else .cpu Tape.empty

/-- Number of recorded nodes. -/
def size {α : Type} [Storage α] : DeviceTape α → Nat
  | .cpu tape => tape.size
  | .cuda tape => tape.size

end Session.DeviceTape

/--
The tape and side tables of eager execution.

`paramsByLeaf` links parameter leaves to their persistent storage. A reset drops the recording and
advances `refGeneration`, but keeps the owner id, random counter, and backend selections.
-/
structure Session (α : Type) [Storage α] where
  /-- Session options controlling backend/device/kernel behavior. -/
  options : Session.Config
  /--
  Whether newly recorded leaves track gradients. Inference sessions set this to `false`: forward
  values are materialized so they can be read back, but parameter and input leaves are recorded
  with `requiresGrad = false`, and backend planning asks for no VJP.
  -/
  gradEnabled : Bool := true
  /-- The session's tape, on the device `options` selects. -/
  tape : IO.Ref (Session.DeviceTape α)
  /-- Map from tape leaf ids to parameter objects, including frozen snapshot owners. -/
  paramsByLeaf : IO.Ref (Std.HashMap Nat (SomeParam α))
  /-- Storage identities of this recording's parameter leaves: reference cells, never tensor
  snapshots. -/
  parameterStorageByLeaf : IO.Ref (Std.HashMap Nat (Param.Cells α))
  /-- Number of seeded random tensors generated by this session. -/
  rngCounter : IO.Ref Nat
  /-- The first accepted selection of each capsule name in this session, in selection order. -/
  selectedBackends : IO.Ref (Array Backend.AcceptedKernel)
  /-- Process-unique owner id for session references. -/
  refOwner : Nat
  /-- Current recording generation for session references. -/
  refGeneration : IO.Ref Nat


namespace Session

/--
Pair the result of a tape constructor with the tape to store afterwards.

On success this is the extended tape. Tape constructors fail before they append, so on failure
it is the tape the constructor started from. Call it inside the lambda passed to `recordCPU` or
`recordCUDA`, as `fun t0 => keepTapeOnError t0 (Tape.apply t0 ...)`. There the constructor is
inlined next to the match, the compiler sees that the failure branches are the only other uses of
`t0`, and the final `Array.push` in `addNode` updates the node array in place.
-/
@[inline] def keepTapeOnError {τ : Type} (t : τ)
    (result : Except String (τ × Nat)) : τ × Except String Nat :=
  match result with
  | .ok (t', id) => (t', .ok id)
  | .error e => (t, .error e)

/-- The session's tape, when the session runs on the CPU. -/
def cpuTape {α : Type} [Storage α] (s : Session α) :
    IO (Tape (SomeTensor α)) := do
  match ← s.tape.get with
  | .cpu tape => pure tape
  | .cuda _ => throw <| IO.userError "torch: this session records on CUDA, not on the CPU"

/-- The session's tape, when the session runs on CUDA. -/
def cudaTape {α : Type} [Storage α] (s : Session α) :
    IO (Tape LibTorch.SomeBuffer) := do
  match ← s.tape.get with
  | .cuda tape => pure tape
  | .cpu _ => throw <| IO.userError "torch: this session records on the CPU, not on CUDA"

/-- Replace the tape of a CUDA session. -/
def setCUDATape {α : Type} [Storage α] (s : Session α)
    (tape : Tape LibTorch.SomeBuffer) : IO Unit :=
  s.tape.set (.cuda tape)

/--
Run a CPU tape step on the session tape and return the new node id.

The tape leaves the reference while `op` runs, and `op` returns the tape to store back together
with the result (see `keepTapeOnError`). The tape is not used after `op`, so it stays unshared even
when this function is not inlined, which happens when the calling IO action is passed around as a
value, as in the backend dispatch.
-/
@[inline] def recordCPU {α : Type} [Storage α] (s : Session α)
    (op : Tape (SomeTensor α) → Tape (SomeTensor α)
        × Except String Nat) :
    IO Nat := do
  let result ← s.tape.modifyGet fun
    | .cpu t =>
        let (t', result) := op t
        (result, .cpu t')
    | .cuda t => (.error "torch: this session records on CUDA, not on the CPU", .cuda t)
  IO.ofExcept result

/-- `recordCPU` for constructors that cannot fail, such as leaves. -/
@[inline] def recordCPUPure {α : Type} [Storage α] (s : Session α)
    (op : Tape (SomeTensor α) → Tape (SomeTensor α)
        × Nat) : IO Nat :=
  s.recordCPU fun t =>
    let (t', id) := op t
    (t', .ok id)

/-- `recordCPU` for the tape of a CUDA session. -/
@[inline] def recordCUDA {α : Type} [Storage α] (s : Session α)
    (op : Tape LibTorch.SomeBuffer
        → Tape LibTorch.SomeBuffer × Except String Nat) :
    IO Nat := do
  let result ← s.tape.modifyGet fun
    | .cuda t =>
        let (t', result) := op t
        (result, .cuda t')
    | .cpu t => (.error "torch: this session records on the CPU, not on CUDA", .cpu t)
  IO.ofExcept result

/-- `recordCPUPure` for the tape of a CUDA session. -/
@[inline] def recordCUDAPure {α : Type} [Storage α] (s : Session α)
    (op : Tape LibTorch.SomeBuffer
        → Tape LibTorch.SomeBuffer × Nat) : IO Nat :=
  s.recordCUDA fun t =>
    let (t', id) := op t
    (t', .ok id)

end Session


end TorchLean
