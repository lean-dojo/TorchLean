/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Parameters
public import NN.Runtime.Autograd.Torch.Core.Functional.Curried
public import NN.Runtime.Autograd.Torch.Core.Functional.Ops

/-!
# Scalar Trainer Interface

The trainer contract and optimizer options. Import this file when consuming a trainer;
backend construction is separate.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

/-- A scalar objective with differentiable state/inputs and a separate data pack. -/
abbrev ScalarLoss (α δ : Type) [TorchLean.Storage α] [TorchLean.Storage δ]
    [Context α]
    (paramShapes inputShapes dataInputShapes : List Shape) :=
  ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
    CurriedRef (fun s => Ops.Ref (m := m) (α := α) s) (paramShapes ++ inputShapes)
      (CurriedRef (fun s => Ops.DataRef (m := m) (α := α) δ s) dataInputShapes
        (m (Ops.Ref (m := m) (α := α) [])))

/--
Persistence hooks for optimizer state owned by a backend-specific trainer.

The payload is intentionally opaque to callers. CUDA eager training, for example, owns Adam's
moment buffers on the device and streams them without first constructing host tensors. A trainer
that has no hidden optimizer state leaves this hook absent.
-/
structure OptimizerStateCheckpoint where
  /-- Save the complete backend-owned optimizer state. -/
  save : System.FilePath → IO Unit
  /-- Replace the backend-owned optimizer state from a previously saved payload. -/
  load : System.FilePath → IO Unit

/-- Optimizer settings supported by device-native scalar training. -/
inductive NativeOptimizer (α : Type) where
  /-- SGD without momentum. -/
  | sgd (learningRate : α)
  /-- Adam with device-resident moments. -/
  | adam (learningRate beta1 beta2 epsilon : α)
  /-- Adam with decoupled weight decay and device-resident moments. -/
  | adamW (learningRate weightDecay beta1 beta2 epsilon : α)

/-- Where optimizer history is maintained for a trainer's updates. -/
inductive OptimizerUpdatePath where
  /-- Explicit tensor updates using the optimizer state returned to the caller. -/
  | generic
  /-- Backend-native updates, which may keep optimizer moments on the device. -/
  | native
  deriving BEq

/--
Bundle a scalar-loss training loop for fixed module state and an input signature.

This is the low-level trainer object used by module-backed execution:
- `loss` computes the scalar objective,
- `diff` computes that loss and its state-shaped gradients from one tape,
- `grad` exposes just the gradients when the loss is not needed,
- `stepWithLoss` applies an SGD update and returns the loss from the same tape,
- `step` applies the update without requiring callers to read the loss,
- `getState` reads the current parameters and persistent buffers.
-/
structure ScalarTrainer (α δ : Type) [TorchLean.Storage α] [TorchLean.Storage δ]
    (paramShapes inputShapes : List Shape)
    (dataInputShapes : List Shape := []) where
  /-- Mutable module state. Entries marked `requiresGrad = false` are persistent buffers. -/
  state : ParamList α paramShapes
  /-- Compute the scalar loss for a curried input pack. -/
  loss :
    Curried.Fn α inputShapes
      (Curried.Fn δ dataInputShapes (IO (Tensor α [])))
  /-- Compute the scalar loss and parameter gradients from one forward tape. -/
  diff :
    Curried.Fn α inputShapes
      (Curried.Fn δ dataInputShapes
        (IO (Tensor α [] × TorchLean.TensorPack α paramShapes)))
  /-- Compute gradients aligned with `paramShapes` for a curried input pack. -/
  grad :
    Curried.Fn α inputShapes
      (Curried.Fn δ dataInputShapes (IO (TorchLean.TensorPack α paramShapes)))
  /-- Apply one SGD-style update and return the loss used to compute that update. -/
  stepWithLoss : α →
    Curried.Fn α inputShapes
      (Curried.Fn δ dataInputShapes (IO (Tensor α [])))
  /-- Apply one SGD-style update for a curried input pack. -/
  step : α → Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit))
  /--
  Optional Adam update path.

  In eager CUDA mode this is a device-gradient/device-moment update path.  Other backends expose
  `none` and should use the generic optimizer wrappers.
  -/
  adamStep? : Option (α → α → α → α →
    Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit))) := none
  /-- CUDA-native Adam update that also returns the loss from its forward tape. -/
  adamStepWithLoss? :
    Option (α → α → α → α →
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (Tensor α [])))) := none
  /--
  Optional AdamW update path.

  In eager CUDA mode this is a device-gradient/device-moment update path with decoupled weight
  decay. Other backends expose `none` and should use the generic optimizer wrappers.
  -/
  adamWStep? : Option (α → α → α → α → α →
    Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit))) := none
  /-- CUDA-native AdamW update that also returns the loss from its forward tape. -/
  adamWStepWithLoss? :
    Option (α → α → α → α → α →
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (Tensor α [])))) := none
  /-- Save and restore optimizer state retained inside the selected runtime backend. -/
  optimizerStateCheckpoint? : Option OptimizerStateCheckpoint := none
  /--
  Release backend-owned optimizer buffers and clear their configuration and update path.

  `Objective.initOptimizer` calls this when starting a fresh history. Parameters are preserved;
  trainers without hidden optimizer state leave the default no-op.
  -/
  resetOptimizerState : IO Unit := pure ()
  /--
  Select the update path before the first update, rejecting a different path until reset.

  Device moments and explicit host optimizer state are separate histories. CUDA callers must
  call `initOptimizer` before switching between them. Backends without hidden optimizer state
  leave this hook a no-op.
  -/
  useOptimizerPath : OptimizerUpdatePath → IO Unit := fun _ => pure ()
  /--
  Apply one native update using the mean gradient of a nonempty batch.

  `readLoss` requests the mean loss from the same forward passes. A false flag returns `none`
  without reading loss values to the host. Singleton and batch updates share optimizer moments.
  -/
  nativeBatchStep? : Option (NativeOptimizer α →
    Array (TorchLean.TensorPack α inputShapes × TorchLean.TensorPack δ dataInputShapes) →
    (readLoss : Bool) → IO (Option (Tensor α []))) := none
  /-- Read current module state, synchronizing device mirrors if needed. -/
  getState : IO (TorchLean.TensorPack α paramShapes)

end Runtime.Autograd.Torch
