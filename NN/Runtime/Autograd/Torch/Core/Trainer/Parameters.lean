/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session.Parameters
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
    {ss : List Shape} → ParamList α ss → Array Bool
  | [], .nil => #[]
  | _ :: _, .cons param params => #[param.requiresGrad] ++ requiresGradArray params

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

/-- Apply `param := param - rate * gradient` to every trainable parameter. -/
def sgdStep {α : Type} [TorchLean.Storage α] [Context α] :
    {ss : List Shape} → ParamList α ss → α → TorchLean.TensorPack α ss → IO Unit
  | [], .nil, _, .nil => pure ()
  | shape :: shapes, .cons param params, rate, .cons gradient gradients => do
      if param.requiresGrad then
        let value ← param.value.get
        let updated := subScaleMaterialize (s := shape) value gradient rate
        Internal.setParamHostValue (α := α) (sh := shape) param updated
      sgdStep (α := α) (ss := shapes) params rate gradients

end ParamList
end Runtime.Autograd.Torch
