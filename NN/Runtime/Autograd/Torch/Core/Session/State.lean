/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Runtime.Autograd.Torch.Core.Types

/-!
# Eager Session State

The tapes and side tables shared by eager operations. Parameters own their CUDA mirrors; the
session owns recorded intermediates. Handle generations separate consecutive forward passes.
-/

public section

namespace Runtime.Autograd.Torch.Internal

open Spec TorchLean

/-- The mutable cells owned by a parameter, without reading or comparing tensor elements.

Two registrations share optimizer storage only when all three cells coincide. Equal values and
equal names do not make separately allocated parameters aliases. Keeping this descriptor in the
session leaves the public `Param` record and its construction interface unchanged.
-/
structure ParameterStorage (α : Type) [Storage α] where
  shape : Shape
  value : IO.Ref (Tensor α shape)
  cudaValue : IO.Ref (Option Runtime.Autograd.Cuda.AnyBuffer)
  hostCurrent : IO.Ref Bool

/-- Compare mutable-cell identity using Lean's safe stateful reference operation. -/
def ParameterStorage.same {α : Type} [Storage α]
    (left right : ParameterStorage α) : IO Bool := do
  if h : left.shape = right.shape then
    let rightValue : IO.Ref (Tensor α left.shape) := h.symm ▸ right.value
    unless ← left.value.ptrEq rightValue do
      return false
    unless ← left.cudaValue.ptrEq right.cudaValue do
      return false
    left.hostCurrent.ptrEq right.hostCurrent
  else
    pure false

private structure StorageKey where
  shape : Shape
  value : USize
  cudaValue : USize
  hostCurrent : USize
  deriving Ord

/--
Use only addresses of the mutable reference cells, never tensor data or CUDA allocations.
The index below retains each representative, so these cells cannot be freed and their addresses
reused while its key is present. Keys are local to one grouping operation.
-/
private unsafe def storageKeyImpl {α : Type} [Storage α]
    (storage : ParameterStorage α) : IO StorageKey :=
  pure ⟨storage.shape, ptrAddrUnsafe storage.value, ptrAddrUnsafe storage.cudaValue,
    ptrAddrUnsafe storage.hostCurrent⟩

/--
A partition hint, not an identity decision. The logical implementation puts each shape in one
bucket; the native implementation refines that partition by live reference-cell addresses.
Every candidate in a bucket is still checked by `ParameterStorage.same`.
-/
@[implemented_by storageKeyImpl]
private def storageKey {α : Type} [Storage α]
    (storage : ParameterStorage α) : IO StorageKey :=
  pure ⟨storage.shape, 0, 0, 0⟩

/--
Map each present storage to its first occurrence, retaining absent slots.

A balanced tree of full-width reference keys avoids pairwise searches among independent storages.
Reference equality remains authoritative even if keys collide. Representatives retain their cells
until the operation ends; neither keys nor references are cached between calls. Consequently a
new parameter list or recording can change its alias layout without invalidating a cache.
-/
def ParameterStorage.canonicalIndices {α : Type} [Storage α]
    (storages : Array (Option (ParameterStorage α))) : IO (Array (Option Nat)) := do
  let mut representatives :
      Std.TreeMap StorageKey (Array (Nat × ParameterStorage α)) := ∅
  let mut slots : Array (Option Nat) := Array.emptyWithCapacity storages.size
  for storage? in storages do
    let index := slots.size
    match storage? with
    | none => slots := slots.push none
    | some storage =>
        let key ← storageKey storage
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
Mutable tapes and side tables for eager execution.

`paramsByLeaf` links parameter leaves to their persistent storage. A reset drops the recording and
advances `refGeneration`, but keeps the owner id, random counter, and backend selections.
-/
structure EagerSession (α : Type) [Storage α] where
  /-- Session options controlling backend/device/kernel behavior. -/
  options : Config
  /-- CPU eager tape used when `Config.device options = .cpu`. -/
  tape : IO.Ref (Runtime.Autograd.Tape α)
  /-- CUDA eager tape used when `Config.device options = .cuda`. -/
  cudaTape : IO.Ref (Runtime.Autograd.Cuda.Tape)
  /-- Map from tape leaf ids to trainable parameter objects. -/
  paramsByLeaf : IO.Ref (Std.HashMap Nat (AnyParam α))
  /-- Storage identities for this recording's parameter leaves; no tensor snapshots are cached. -/
  parameterStorageByLeaf : IO.Ref (Std.HashMap Nat (ParameterStorage α))
  /-- Non-differentiable integer inputs for dynamic indexing operations. -/
  nats : IO.Ref (Array Nat)
  /-- Number of seeded random tensors generated by this session. -/
  rngCounter : IO.Ref Nat
  /-- Accepted capsules already reported for this session, in first-use order. -/
  selectedBackends : IO.Ref (Array NN.Backend.AcceptedKernel)
  /-- Process-unique owner id for session references. -/
  refOwner : Nat
  /-- Current recording generation for session references. -/
  refGeneration : IO.Ref Nat


namespace EagerSession

/--
Pair the result of a tape constructor with the tape to store afterwards.

On success this is the extended tape. Tape constructors fail before they append, so on failure
it is the tape the constructor started from. Call it inside the lambda passed to `recordCpu` or
`recordCuda`, as `fun t0 => keepTapeOnError t0 (Tape.op (t := t0) ...)`. There the constructor is
inlined next to the match, the compiler sees that the failure branches are the only other uses of
`t0`, and the final `Array.push` in `addNode` updates the node array in place.
-/
@[inline] def keepTapeOnError {τ : Type} (t : τ)
    (result : Except String (τ × Nat)) : τ × Except String Nat :=
  match result with
  | .ok (t', id) => (t', .ok id)
  | .error e => (t, .error e)

/--
Run a CPU tape step on the session tape and return the new node id.

The tape leaves the reference while `op` runs, and `op` returns the tape to store back together
with the result (see `keepTapeOnError`). The tape is not used after `op`, so it stays unshared even
when this function is not inlined, which happens when the calling IO action is passed around as a
value, as in the backend dispatch.
-/
@[inline] def recordCpu {α : Type} [Storage α] (s : EagerSession α)
    (op : Runtime.Autograd.Tape α → Runtime.Autograd.Tape α × Except String Nat) :
    IO Nat := do
  let result ← s.tape.modifyGet fun t =>
    let (t', result) := op t
    (result, t')
  Runtime.Autograd.okOrThrow result

/-- `recordCpu` for constructors that cannot fail, such as leaves. -/
@[inline] def recordCpuPure {α : Type} [Storage α] (s : EagerSession α)
    (op : Runtime.Autograd.Tape α → Runtime.Autograd.Tape α × Nat) : IO Nat :=
  s.tape.modifyGet fun t =>
    let (t', id) := op t
    (id, t')

/-- `recordCpu` for the CUDA tape. -/
@[inline] def recordCuda {α : Type} [Storage α] (s : EagerSession α)
    (op : Runtime.Autograd.Cuda.Tape → Runtime.Autograd.Cuda.Tape × Except String Nat) :
    IO Nat := do
  let result ← s.cudaTape.modifyGet fun t =>
    let (t', result) := op t
    (result, t')
  Runtime.Autograd.okOrThrow result

/-- `recordCpuPure` for the CUDA tape. -/
@[inline] def recordCudaPure {α : Type} [Storage α] (s : EagerSession α)
    (op : Runtime.Autograd.Cuda.Tape → Runtime.Autograd.Cuda.Tape × Nat) : IO Nat :=
  s.cudaTape.modifyGet fun t =>
    let (t', id) := op t
    (id, t')

end EagerSession


end Runtime.Autograd.Torch.Internal
