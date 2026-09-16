/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session

/-!
# Shared Parameter Gradients

Each `use` records the value observed at that call. Optimizers combine the cotangents of leaves
whose parameter storage is shared, then update that storage once. Grouping at this boundary keeps
fresh reads after parameter mutation independent of earlier tape snapshots.
-/

@[expose] public section

namespace Runtime.Autograd.Torch.Internal.EagerSession

open Spec TorchLean

/-- Parameter leaves in recording order, with the first leaf retained as the optimizer key. -/
structure ParameterGroup (α : Type) [Storage α] where
  id : Nat
  parameter : AnyParam α
  leaves : Array Nat
  storage? : Option (ParameterStorage α)

/-- Group registrations by storage identity without touching tensor data.

The same reference index serves eager updates, generic optimizers, and checkpoint schemas.
A singleton group borrows its gradient unchanged; only an actual shared parameter requires
gradient addition. The first leaf remains the optimizer key, preserving ordered parameter slots.
Entries inserted directly into the internal parameter map without a storage descriptor stay
separate, because there is no sound alias test for an arbitrary `AnyParam` closure.
-/
def parameterGroups {α : Type} [Storage α] (session : EagerSession α) :
    IO (Array (ParameterGroup α)) := do
  let parameters ← session.paramsByLeaf.get
  let storages ← session.parameterStorageByLeaf.get
  let ordered := ((parameters.toList.filter fun entry => entry.2.requiresGrad).mergeSort
    (fun left right => left.1 ≤ right.1)).toArray
  let descriptors ← ordered.mapM fun (id, parameter) => do
    let storage? := storages.get? id
    if let some storage := storage? then
      unless storage.shape == parameter.s do
        throw <| IO.userError "torch: parameter storage descriptor has the wrong shape"
    pure storage?
  let slots ← ParameterStorage.canonicalIndices descriptors
  let mut groups : Array (ParameterGroup α) := #[]
  let mut groupBySlot : Array Nat := Array.emptyWithCapacity ordered.size
  for index in [:ordered.size] do
    let some (id, parameter) := ordered[index]?
      | throw <| IO.userError "torch: parameter registration index out of bounds"
    let some slot := slots[index]?
      | throw <| IO.userError "torch: canonical parameter slot missing"
    match slot with
    | some canonical =>
        if canonical < index then
          let some groupIndex := groupBySlot[canonical]?
            | throw <| IO.userError "torch: parameter group index out of bounds"
          groups := groups.modify groupIndex fun group =>
            { group with leaves := group.leaves.push id }
          groupBySlot := groupBySlot.push groupIndex
        else if canonical == index then
          groupBySlot := groupBySlot.push groups.size
          groups := groups.push { id, parameter, leaves := #[id], storage? := descriptors[index]! }
        else
          throw <| IO.userError "torch: invalid canonical parameter slot"
    | none =>
        groupBySlot := groupBySlot.push groups.size
        groups := groups.push { id, parameter, leaves := #[id], storage? := none }
  pure groups

/-- Recover the typed storage view used by the parameter synchronization helpers. -/
def parameterOfStorage {α : Type} [Storage α] (storage : ParameterStorage α) :
    Param α storage.shape :=
  { value := storage.value, cudaValue := storage.cudaValue, hostCurrent := storage.hostCurrent }

/-- Read current host storage, synchronizing a newer device mirror before a CPU update. -/
def ParameterGroup.currentHostValue {α : Type} [Storage α] [TensorTransfer α]
    (group : ParameterGroup α) : IO (Spec.SomeTensor α) := do
  match group.storage? with
  | none => group.parameter.get
  | some storage =>
      let parameter := parameterOfStorage storage
      syncParamCudaToHost parameter
      pure <| Spec.SomeTensor.ofTensor (← parameter.value.get)

/-- Read the current parameter value used as the starting point of a CUDA update.

Cotangents belong to the recorded computation, but an optimizer changes current parameter storage.
That storage can differ from every leaf snapshot if the parameter changed after its last `use`.
Ordinary registrations share their current mirror here; only an empty cache requires an upload.

An internal registration without a storage descriptor supplies its current host value through
`AnyParam.get`. Its custom getter must honor that contract; a tape leaf is not a substitute.
-/
def ParameterGroup.currentCudaValue {α : Type} [Storage α] [TensorTransfer α]
    (group : ParameterGroup α) : IO Runtime.Autograd.Cuda.AnyBuffer := do
  let value ← match group.storage? with
    | some storage => getParamCudaValue (parameterOfStorage storage)
    | none => do
        let host ← group.parameter.get
        if h : host.shape = group.parameter.s then
          CudaBridge.toAnyBuffer (α := α) (s := group.parameter.s) (host.cast h)
        else
          throw <| IO.userError "torch: current parameter getter returned the wrong shape"
  unless value.s == group.parameter.s do
    throw <| IO.userError "torch: current parameter storage has the wrong shape"
  checkCudaAnyBufferSize "current parameter storage" value
  pure value

/-- Sum host cotangents for one parameter, retaining the original tensor for a singleton. -/
def denseGroupGradient {α : Type} [Storage α] [Add α]
    (group : ParameterGroup α) (gradients : Array (Spec.SomeTensor α)) :
    IO (Spec.SomeTensor α) := do
  let first ← match gradients[group.id]? with
    | some gradient => pure gradient
    | none => throw <| IO.userError "torch: gradient array out of bounds during grouped update"
  unless first.shape == group.parameter.s do
    throw <| IO.userError "torch: parameter gradient shape mismatch during grouped update"
  let mut total := first
  for id in group.leaves.toList.drop 1 do
    let gradient ← match gradients[id]? with
      | some gradient => pure gradient
      | none => throw <| IO.userError "torch: gradient array out of bounds during grouped update"
    total ← okOrThrow <| Runtime.Autograd.SomeTensor.add total gradient
  pure total

/-- Borrow a singleton CUDA gradient or own the temporary sum of shared-leaf cotangents.

The input map remains borrowed throughout. Intermediate sums and the final sum are released on
both success and failure; the optimizer action must not retain that temporary gradient.
-/
def withCudaGroupGradient {α β : Type} [Storage α]
    (group : ParameterGroup α) (gradients : CudaGradMap)
    (action : Runtime.Autograd.Cuda.AnyBuffer → IO β) : IO β := do
  let first ← match gradients.get? group.id with
    | some gradient => pure gradient
    | none => throw <| IO.userError "torch: missing CUDA gradient during grouped update"
  if group.leaves.size == 1 then
    action first
  else
    let owned ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
    try
      let mut total := first
      for id in group.leaves.toList.drop 1 do
        let gradient ← match gradients.get? id with
          | some gradient => pure gradient
          | none => throw <| IO.userError "torch: missing shared CUDA gradient"
        let next ← okOrThrow <| Runtime.Autograd.Cuda.AnyBuffer.add total gradient
        let previous ← owned.swap (some next)
        if let some previous := previous then
          releaseCudaAnyBuffer previous
        checkCudaAnyBufferSize "shared parameter gradient sum" next
        total := next
      action total
    finally
      if let some value ← owned.get then
        releaseCudaAnyBuffer value

end Runtime.Autograd.Torch.Internal.EagerSession
