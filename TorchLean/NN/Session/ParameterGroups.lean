/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.Lifecycle
public import TorchLean.NN.Session.Parameters

/-!
# Shared Parameter Gradients

Each `use` records the value observed at that call. Optimizers combine the cotangents of leaves
whose parameter storage is shared, then update that storage once. Grouping at the optimizer keeps a
fresh read after a parameter change independent of the snapshots already on the tape.
-/

@[expose] public section

namespace TorchLean.Session

/-- Parameter leaves in recording order, with the first leaf retained as the optimizer key. -/
structure ParameterGroup (α : Type) [Storage α] where
  /-- Leaf id of the first registration, the optimizer key of the group. -/
  id : Nat
  /-- The parameter registered at `id`. -/
  parameter : SomeParam α
  /-- Leaf ids of all registrations in the group, in recording order. -/
  leaves : Array Nat
  /-- Storage descriptor of the group, or `none` when the registration at `id` has none. -/
  storage? : Option (Param.Cells α)

/-- Group the trainable registrations by storage identity without touching tensor data.

Eager updates, generic optimizers and checkpoint schemas use the same storage-identity index,
`Param.Cells.canonicalIndices`. A singleton group borrows its gradient unchanged, and only a
parameter that is actually shared needs gradient addition. The first leaf is the optimizer key, so
parameter slots keep their order. Entries inserted into the internal parameter map without a storage
descriptor form groups of their own, because an arbitrary `SomeParam` closure has no sound alias
test.
-/
def groupParameters {α : Type} [Storage α]
    (parameters : Std.HashMap Nat (SomeParam α))
    (storages : Std.HashMap Nat (Param.Cells α)) :
    IO (Array (ParameterGroup α)) := do
  let ordered := ((parameters.toList.filter fun entry => entry.2.requiresGrad).mergeSort
    (fun left right => left.1 ≤ right.1)).toArray
  let descriptors ← ordered.mapM fun (id, parameter) => do
    let storage? := storages.get? id
    if let some storage := storage? then
      unless storage.shape == parameter.s do
        throw <| IO.userError "torch: parameter storage descriptor has the wrong shape"
    pure storage?
  let slots ← Param.Cells.canonicalIndices descriptors
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

/-- Group the current eager recording using the shared storage-identity contract. -/
def parameterGroups {α : Type} [Storage α] (session : Session α) :
    IO (Array (ParameterGroup α)) := do
  groupParameters (← session.paramsByLeaf.get) (← session.parameterStorageByLeaf.get)

/-- Recover the typed storage view used by the parameter synchronization helpers. -/
def parameterOfStorage {α : Type} [Storage α] (storage : Param.Cells α) :
    Param α storage.shape :=
  { value := storage.value, cudaValue := storage.cudaValue, hostSynced := storage.hostSynced }

/-- Read the current parameter value that a CUDA update starts from.

Cotangents belong to the recorded computation, but an optimizer changes the current parameter
storage, which differs from every leaf snapshot if the parameter changed after its last `use`. A
registration with a storage descriptor shares its current mirror here and uploads only when the
cache is empty.

An internal registration without a storage descriptor supplies its current host value through
`SomeParam.get`. Its custom getter must honor that contract and return the current value, never the
value of a tape leaf.
-/
def ParameterGroup.cudaValue {α : Type} [Storage α] [TensorTransfer α]
    (group : ParameterGroup α) : IO LibTorch.SomeBuffer := do
  let value ← match group.storage? with
    | some storage => Param.getCUDAValue (parameterOfStorage storage)
    | none => do
        let host ← group.parameter.get
        if h : host.shape = group.parameter.s then
          TensorTransfer.toSomeBuffer (α := α) (s := group.parameter.s) (host.cast h)
        else
          throw <| IO.userError "torch: current parameter getter returned the wrong shape"
  unless value.s == group.parameter.s do
    throw <| IO.userError "torch: current parameter storage has the wrong shape"
  IO.ofExcept <| (LibTorch.SomeBuffer.validate value).mapError
    (s!"torch: current parameter storage: " ++ ·)

/-- Borrow a singleton CUDA gradient or own the temporary sum of shared-leaf cotangents.

The input map is borrowed throughout. Intermediate sums and the final sum are released on success
and on failure, so the optimizer action must not retain that temporary gradient.
-/
def withCUDAGroupGradient {α β : Type} [Storage α]
    (group : ParameterGroup α) (gradients : LibTorch.Tape.SparseGradMap)
    (action : LibTorch.SomeBuffer → IO β) : IO β := do
  let first ← match gradients.get? group.id with
    | some gradient => pure gradient
    | none => throw <| IO.userError "torch: missing CUDA gradient during grouped update"
  if group.leaves.size == 1 then
    action first
  else
    let owned ← IO.mkRef (none : Option LibTorch.SomeBuffer)
    try
      let mut total := first
      for id in group.leaves.toList.drop 1 do
        let gradient ← match gradients.get? id with
          | some gradient => pure gradient
          | none => throw <| IO.userError "torch: missing shared CUDA gradient"
        let next ← IO.ofExcept <| LibTorch.SomeBuffer.add total gradient
        let previous ← owned.swap (some next)
        if let some previous := previous then
          releaseCUDASomeBuffer previous
        discard <| IO.ofExcept <| (LibTorch.SomeBuffer.validate next).mapError
          (s!"torch: shared parameter gradient sum: " ++ ·)
        total := next
      action total
    finally
      if let some value ← owned.get then
        releaseCUDASomeBuffer value

end TorchLean.Session
