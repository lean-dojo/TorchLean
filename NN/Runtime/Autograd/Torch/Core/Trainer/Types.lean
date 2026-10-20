/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Parameters
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
  | sgd (lr : α)
  /-- Adam with device-resident moments. -/
  | adam (lr beta1 beta2 epsilon : α)
  /-- Adam with decoupled weight decay and device-resident moments. -/
  | adamW (lr weightDecay beta1 beta2 epsilon : α)

/-- Where optimizer history is maintained for a trainer's updates. -/
inductive OptimizerUpdatePath where
  /-- Explicit tensor updates using the optimizer state returned to the caller. -/
  | generic
  /-- Backend-native updates, which may keep optimizer moments on the device. -/
  | native
  deriving BEq

/--
Bundle a scalar-loss training loop for fixed module state and an input signature.

Backend constructors supply curried operations. The packed `ScalarTrainer.loss`, `grad`, and
`step` methods expose them to callers; `value := true` and `loss := true` request the objective
value with a gradient or update. `getState` reads the current parameters and persistent buffers.
-/
structure ScalarTrainer (α δ : Type) [TorchLean.Storage α] [TorchLean.Storage δ]
    (paramShapes inputShapes : List Shape)
    (dataInputShapes : List Shape := []) where
  /-- Mutable module state. Entries marked `requiresGrad = false` are persistent buffers. -/
  state : ParamList α paramShapes
  /-- Compute the scalar loss for a curried input pack. -/
  lossFn :
    Curried.Function α inputShapes
      (Curried.Function δ dataInputShapes (IO (Tensor α [])))
  /-- Curried backend operation returning `(loss, gradients)` from one forward tape. -/
  diffFn :
    Curried.Function α inputShapes
      (Curried.Function δ dataInputShapes
        (IO (Tensor α [] × TorchLean.TensorPack α paramShapes)))
  /-- Compute gradients aligned with `paramShapes` for a curried input pack. -/
  gradFn :
    Curried.Function α inputShapes
      (Curried.Function δ dataInputShapes (IO (TorchLean.TensorPack α paramShapes)))
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
