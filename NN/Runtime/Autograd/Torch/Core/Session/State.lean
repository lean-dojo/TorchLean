/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Core.Base
public import NN.Runtime.Autograd.Torch.Core.Types
import Std.Data.TreeMap.Basic

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


end Runtime.Autograd.Torch.Internal
