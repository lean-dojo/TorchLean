/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.Parameters
public import TorchLean.NN.Session
public import TorchLean.Tensor.Pack

/-!
# Trainer Parameters

Mutable, shape-indexed parameter storage for the Torch-style trainer. A parameter value may keep a
CUDA mirror, and every read or write goes through a `Param` operation that names the copy it uses,
host or device.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/-- A dependent pack of mutable parameters indexed by their tensor shapes. -/
abbrev ParamList (α : Type) [Storage α] : List Shape → Type := Pack (Param α)

namespace ParamList

/-- Parameter storage in list order, retaining `none` at every frozen slot. -/
def optimizerStorage {α : Type} [Storage α] {ss : List Shape}
    (parameters : ParamList α ss) : Array (Option (Param.Cells α)) :=
  let rec
    /-- Append the storage of each remaining parameter to the array, `none` for a frozen one. -/
    collect : {shapes : List Shape} → ParamList α shapes →
        Array (Option (Param.Cells α)) →
        Array (Option (Param.Cells α))
      | [], .nil, storages => storages
      | shape :: _, .cons parameter rest, storages =>
          let storage : Option (Param.Cells α) :=
            if parameter.requiresGrad then
              some
                { shape := shape
                  value := parameter.value
                  cudaValue := parameter.cudaValue
                  hostSynced := parameter.hostSynced }
            else
              none
          collect rest (storages.push storage)
  collect parameters #[]

/--
Map each trainable slot to the first trainable slot sharing all its mutable storage cells.

Frozen slots map to `none`, and indices count every slot of the list, frozen ones included.
Optimizer updates and checkpoint schemas share this alias map. Storage identity never inspects
tensor elements.

Each call rebuilds the reference index from the list, so a change in the alias layout shows at the
next call, and the index avoids pairwise identity checks among independent storages.
-/
def canonicalSlotIndices {α : Type} [Storage α] {ss : List Shape}
    (parameters : ParamList α ss) : IO (Array (Option Nat)) :=
  Param.Cells.canonicalIndices (optimizerStorage parameters)

/-- Allocate mutable parameters from an ordered tensor pack. -/
def ofPack {α : Type} [Storage α] :
    {ss : List Shape} → TensorPack α ss → IO (ParamList α ss)
  | [], .nil => pure .nil
  | _ :: _, .cons value values => do
      let param ← Param.create value
      pure (.cons param (← ofPack (α := α) values))

/--
Allocate mutable parameters with a trainability mask, one flag per parameter.

The mask follows parameter order and must have exactly the same size as the shape list.
-/
def ofPackWithRequiresGrad {α : Type} [Storage α] {ss : List Shape}
    (values : TensorPack α ss)
    (flags : Array Bool) : IO (ParamList α ss) := do
  let rec
    /-- Allocate the remaining parameters with the flags of `flags` from the given index on. -/
    go : {shapes : List Shape} → TensorPack α shapes → Nat → IO (ParamList α shapes)
      | [], .nil, index =>
          if index = flags.size then
            pure .nil
          else
            throw <| IO.userError "torch: requiresGrad array longer than parameter pack"
      | _ :: shapes, .cons value rest, index =>
          match flags[index]? with
          | none => throw <| IO.userError "torch: requiresGrad array shorter than parameter pack"
          | some requiresGrad => do
              let param ← Param.create value (requiresGrad := requiresGrad)
              pure (.cons param (← go (shapes := shapes) rest (index + 1)))
  go values 0

/-- Read the trainability mask in parameter order. -/
def requiresGradArray {α : Type} [Storage α] :
    {ss : List Shape} → ParamList α ss → Array Bool := fun parameters =>
  let rec
    /-- Append the flag of each remaining parameter to the array. -/
    collect : {shapes : List Shape} → ParamList α shapes → Array Bool → Array Bool
      | [], .nil, flags => flags
      | _ :: _, .cons parameter rest, flags => collect rest (flags.push parameter.requiresGrad)
  collect parameters #[]

/-- Read the host parameter values without downloading CUDA mirrors, so a value updated on the
device reads stale. -/
def values {α : Type} [Storage α] :
    {ss : List Shape} → ParamList α ss → IO (TensorPack α ss)
  | [], .nil => pure .nil
  | _ :: shapes, .cons param params => do
      pure (.cons (← param.value.get) (← values (α := α) (ss := shapes) params))

/-- Read parameter values after synchronizing any current CUDA mirrors to the host. -/
def valuesSynced {α : Type} [Storage α] [TensorTransfer α] :
    {ss : List Shape} → ParamList α ss → IO (TensorPack α ss)
  | [], .nil => pure .nil
  | shape :: shapes, .cons param params => do
      Param.syncToHost (α := α) (sh := shape) param
      pure (.cons (← param.value.get) (← valuesSynced (α := α) (ss := shapes) params))

/-- Replace host parameter values from an ordered tensor pack. -/
def setValues {α : Type} [Storage α] :
    {ss : List Shape} → ParamList α ss → TensorPack α ss → IO Unit
  | [], .nil, .nil => pure ()
  | shape :: shapes, .cons param params, .cons value values => do
      Param.setHostValue (α := α) (sh := shape) param value
      setValues (α := α) (ss := shapes) params values

/--
Sum occurrence gradients at their canonical parameter slots, in list order.

Frozen slots contribute nothing. The first gradient at a slot is kept as it is, so a parameter with
no aliases gets the same arithmetic as an ordinary update. Every shape and index check happens
here, before the caller mutates any parameter.
-/
def canonicalGradients {α : Type} [Storage α] [Add α] {ss : List Shape}
    (slots : Array (Option Nat)) (gradients : TensorPack α ss) :
    IO (Array (Option (SomeTensor α))) := do
  unless slots.size = ss.length do
    throw <| IO.userError "torch: canonical parameter map length mismatch"
  let rec
    /-- Add each remaining gradient into the sum at its canonical slot, given the index of the first
    remaining slot. -/
    collect : {shapes : List Shape} → TensorPack α shapes → Nat →
        Array (Option (SomeTensor α)) → IO (Array (Option (SomeTensor α)))
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
                  | none => pure (SomeTensor.ofTensor gradient)
                  | some previous =>
                      if h : previous.shape = shape then
                        pure <| SomeTensor.ofTensor <|
                          Tensor.zipWith (· + ·) (previous.cast h) gradient
                      else
                        throw <| IO.userError "torch: canonical gradient shape mismatch"
                pure (sums.set! canonical (some combined))
          collect rest (index + 1) sums
  collect gradients 0 (Array.replicate ss.length none)

/-- Read one canonical gradient, checking its runtime shape. -/
def canonicalGradient {α : Type} [Storage α] (shape : Shape)
    (sums : Array (Option (SomeTensor α))) (index : Nat) : IO (Tensor α shape) := do
  let some (some gradient) := sums[index]?
    | throw <| IO.userError "torch: canonical gradient missing"
  if h : gradient.shape = shape then
    pure (gradient.cast h)
  else
    throw <| IO.userError "torch: canonical gradient shape mismatch"

end ParamList
end TorchLean
