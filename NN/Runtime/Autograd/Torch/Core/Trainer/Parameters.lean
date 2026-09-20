/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.Parameters
public import NN.Runtime.Autograd.Torch.Core.Session.State
public import NN.Tensor.Pack

/-!
# Trainer Parameters

Mutable, shape-indexed parameter storage for the Torch-style trainer. Parameter values may retain a
CUDA mirror; reads and writes keep host and device ownership explicit through the underlying
`Param` operations.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

/-- A dependent pack of mutable parameters indexed by their tensor shapes. -/
inductive ParamList (α : Type) [TorchLean.Storage α] : List Shape → Type where
  | nil : ParamList α []
  | cons {s : Shape} {ss : List Shape} : Param α s → ParamList α ss → ParamList α (s :: ss)

namespace ParamList

/-- Parameter storage in list order, retaining `none` at every frozen slot. -/
def optimizerStorage {α : Type} [TorchLean.Storage α] {ss : List Shape}
    (parameters : ParamList α ss) : Array (Option (Internal.ParameterStorage α)) :=
  let rec collect : {shapes : List Shape} → ParamList α shapes →
      Array (Option (Internal.ParameterStorage α)) →
      Array (Option (Internal.ParameterStorage α))
    | [], .nil, storages => storages
    | shape :: _, .cons parameter rest, storages =>
        let storage : Option (Internal.ParameterStorage α) :=
          if parameter.requiresGrad then
            some
              { shape := shape
                value := parameter.value
                cudaValue := parameter.cudaValue
                hostCurrent := parameter.hostCurrent }
          else
            none
        collect rest (storages.push storage)
  collect parameters #[]

/--
Map each trainable slot to the first trainable slot sharing all its mutable storage cells.

Frozen slots remain `none`; list indices are never compressed. This is the shared alias map for
optimizer updates and checkpoint schemas. Storage identity does not inspect tensor elements.

Rebuild the reference index from this list on each call. This preserves changes to the alias
layout while avoiding pairwise identity checks among independent storages.
-/
def canonicalSlotIndices {α : Type} [TorchLean.Storage α] {ss : List Shape}
    (parameters : ParamList α ss) : IO (Array (Option Nat)) :=
  Internal.ParameterStorage.canonicalIndices (optimizerStorage parameters)

/-- Materialize `value - rate * gradient` in one traversal. -/
def subScaleMaterialize {α : Type} [TorchLean.Storage α] [Sub α] [Mul α] :
    {s : Shape} → Tensor α s → Tensor α s → α → Tensor α s
  | _, value, gradient, rate =>
      Tensor.map2Spec (fun x dx => x - rate * dx) value gradient

/-- Allocate mutable parameters from an ordered tensor pack. -/
def ofPack {α : Type} [TorchLean.Storage α] :
    {ss : List Shape} → TorchLean.TensorPack α ss → IO (ParamList α ss)
  | [], .nil => pure .nil
  | _ :: _, .cons value values => do
      let param ← Param.Internal.create value
      pure (.cons param (← ofPack (α := α) values))

/--
Allocate mutable parameters with an explicit trainability mask.

The mask follows parameter order and must have exactly the same size as the shape list.
-/
def ofPackWithRequiresGrad {α : Type} [TorchLean.Storage α] {ss : List Shape}
    (values : TorchLean.TensorPack α ss)
    (flags : Array Bool) : IO (ParamList α ss) := do
  let rec go : {shapes : List Shape} → TorchLean.TensorPack α shapes → Nat → IO (ParamList α shapes)
    | [], .nil, index =>
        if index = flags.size then
          pure .nil
        else
          throw <| IO.userError "torch: requiresGrad array longer than parameter pack"
    | _ :: shapes, .cons value rest, index =>
        match flags[index]? with
        | none => throw <| IO.userError "torch: requiresGrad array shorter than parameter pack"
        | some requiresGrad => do
            let param ← Param.Internal.create value (requiresGrad := requiresGrad)
            pure (.cons param (← go (shapes := shapes) rest (index + 1)))
  go values 0

/-- Read the trainability mask in parameter order. -/
def requiresGradArray {α : Type} [TorchLean.Storage α] :
    {ss : List Shape} → ParamList α ss → Array Bool := fun parameters =>
  let rec collect : {shapes : List Shape} → ParamList α shapes → Array Bool → Array Bool
    | [], .nil, flags => flags
    | _ :: _, .cons parameter rest, flags => collect rest (flags.push parameter.requiresGrad)
  collect parameters #[]

/-- Read current host parameter values without synchronizing stale CUDA mirrors. -/
def values {α : Type} [TorchLean.Storage α] :
    {ss : List Shape} → ParamList α ss → IO (TorchLean.TensorPack α ss)
  | [], .nil => pure .nil
  | _ :: shapes, .cons param params => do
      pure (.cons (← param.value.get) (← values (α := α) (ss := shapes) params))

/-- Read parameter values after synchronizing any current CUDA mirrors to the host. -/
def valuesSynced {α : Type} [TorchLean.Storage α] [TensorTransfer α] :
    {ss : List Shape} → ParamList α ss → IO (TorchLean.TensorPack α ss)
  | [], .nil => pure .nil
  | shape :: shapes, .cons param params => do
      Internal.syncParamCudaToHost (α := α) (sh := shape) param
      pure (.cons (← param.value.get) (← valuesSynced (α := α) (ss := shapes) params))

/-- Replace host parameter values from an ordered tensor pack. -/
def setValues {α : Type} [TorchLean.Storage α] :
    {ss : List Shape} → ParamList α ss → TorchLean.TensorPack α ss → IO Unit
  | [], .nil, .nil => pure ()
  | shape :: shapes, .cons param params, .cons value values => do
      Internal.setParamHostValue (α := α) (sh := shape) param value
      setValues (α := α) (ss := shapes) params values

/--
Sum occurrence gradients at their canonical parameter slots, in list order.

Frozen slots contribute nothing. The first gradient is retained directly so a parameter with no
aliases keeps the same arithmetic as an ordinary update. Complete shape and index validation
precedes any parameter mutation by the caller.
-/
def canonicalGradients {α : Type} [TorchLean.Storage α] [Add α] {ss : List Shape}
    (slots : Array (Option Nat)) (gradients : TorchLean.TensorPack α ss) :
    IO (Array (Option (Spec.SomeTensor α))) := do
  unless slots.size = ss.length do
    throw <| IO.userError "torch: canonical parameter map length mismatch"
  let rec collect : {shapes : List Shape} → TorchLean.TensorPack α shapes → Nat →
      Array (Option (Spec.SomeTensor α)) → IO (Array (Option (Spec.SomeTensor α)))
    | [], .nil, _, sums => pure sums
    | shape :: _, .cons gradient rest, index, sums => do
        let some slot := slots[index]?
          | throw <| IO.userError "torch: canonical parameter slot missing"
        let sums ← match slot with
          | none => pure sums
          | some canonical => do
              unless canonical ≤ index && slots[canonical]? == some (some canonical) do
                throw <| IO.userError "torch: invalid canonical parameter slot"
              let some previous := sums[canonical]?
                | throw <| IO.userError "torch: canonical gradient slot out of bounds"
              let combined ← match previous with
                | none => pure (Spec.SomeTensor.ofTensor gradient)
                | some previous =>
                    if h : previous.shape = shape then
                      pure <| Spec.SomeTensor.ofTensor <|
                        Tensor.map2Spec (· + ·) (previous.cast h) gradient
                    else
                      throw <| IO.userError "torch: canonical gradient shape mismatch"
              pure (sums.set! canonical (some combined))
        collect rest (index + 1) sums
  collect gradients 0 (Array.replicate ss.length none)

/-- Read one canonical gradient, checking its runtime shape. -/
def canonicalGradient {α : Type} [TorchLean.Storage α] (shape : Shape)
    (sums : Array (Option (Spec.SomeTensor α))) (index : Nat) : IO (Tensor α shape) := do
  let some (some gradient) := sums[index]?
    | throw <| IO.userError "torch: canonical gradient missing"
  if h : gradient.shape = shape then
    pure (gradient.cast h)
  else
    throw <| IO.userError "torch: canonical gradient shape mismatch"

/--
Apply one SGD update to each trainable storage, using the sum of its occurrence gradients.

Storage aliases use the same canonical-slot map as generic optimizers and checkpoints.
-/
def sgdStep {α : Type} [TorchLean.Storage α] [Context α] {ss : List Shape}
    (parameters : ParamList α ss) (rate : α) (gradients : TorchLean.TensorPack α ss) :
    IO Unit := do
  let slots ← canonicalSlotIndices parameters
  let sums ← canonicalGradients slots gradients
  let rec update : {shapes : List Shape} → ParamList α shapes → Nat → IO Unit
    | [], .nil, _ => pure ()
    | shape :: _, .cons parameter rest, index => do
        if slots[index]? == some (some index) then
          let gradient ← canonicalGradient shape sums index
          let value ← parameter.value.get
          let updated := subScaleMaterialize value gradient rate
          Internal.setParamHostValue parameter updated
        update rest (index + 1)
  update parameters 0

end ParamList
end Runtime.Autograd.Torch
