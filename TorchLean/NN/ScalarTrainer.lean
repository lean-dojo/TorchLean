/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.ScalarTrainer.ParamList
public import TorchLean.NN.Ops

/-!
# Scalar Trainer Interface

The trainer contract and optimizer options. Code that consumes a trainer needs only this module,
because backend construction lives in separate modules.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

/-- A scalar objective with differentiable state/inputs and a separate data pack. -/
abbrev ScalarLoss (α δ : Type) [Storage α] [Storage δ]
    [Context α]
    (paramShapes inputShapes dataInputShapes : List Shape) :=
  ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
    CurriedRef (fun s => Ops.Ref (m := m) (α := α) s) (paramShapes ++ inputShapes)
      (CurriedRef (fun s => Ops.DataRef (m := m) (α := α) δ s) dataInputShapes
        (m (Ops.Ref (m := m) (α := α) [])))

/--
Persistence hooks for optimizer state owned by a backend-specific trainer.

The payload is opaque to callers. CUDA eager training, for example, keeps Adam's moment buffers on
the device and streams them to the file without building host tensors. A trainer without hidden
optimizer state leaves this hook absent.
-/
structure OptimizerStateCheckpoint where
  /-- Save the complete backend-owned optimizer state. -/
  save : System.FilePath → IO Unit
  /-- Replace the backend-owned optimizer state with a payload that `save` wrote. -/
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
  /-- Updates on host tensors by an optimizer whose state is returned to the caller. -/
  | generic
  /-- Backend-native updates, which may keep optimizer moments on the device. -/
  | native
  deriving BEq

/--
Bundle a scalar-loss training loop for fixed module state and an input signature.

Backend constructors supply the curried operations. `ScalarTrainer.loss` and `ScalarTrainer.grad`
run them on packed inputs, and `Objective.step` uses them for one optimizer update. Passing
`value := true` to `grad` or `loss := true` to `step` also returns the objective value.
`getState` reads the current parameters and persistent buffers.
-/
structure ScalarTrainer (α δ : Type) [Storage α] [Storage δ]
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
        (IO (Tensor α [] × TensorPack α paramShapes)))
  /-- Compute gradients aligned with `paramShapes` for a curried input pack. -/
  gradFn :
    Curried.Function α inputShapes
      (Curried.Function δ dataInputShapes (IO (TensorPack α paramShapes)))
  /-- Save and restore optimizer state retained inside the selected runtime backend. -/
  optimizerStateCheckpoint? : Option OptimizerStateCheckpoint := none
  /--
  Release backend-owned optimizer buffers and clear their configuration and update path.

  `Objective.initOptimizer` calls this to start a fresh history. Parameters keep their values,
  and trainers without hidden optimizer state keep the default no-op.
  -/
  resetOptimizerState : IO Unit := pure ()
  /--
  Select the update path before the first update, rejecting a different path until reset.

  Device moments and the host optimizer state returned to the caller are separate histories.
  CUDA callers must call `initOptimizer` before switching between them. Backends without hidden
  optimizer state leave this hook a no-op.
  -/
  useOptimizerPath : OptimizerUpdatePath → IO Unit := fun _ => pure ()
  /--
  Apply one native update using the mean gradient of a nonempty batch.

  `readLoss` requests the mean loss from the same forward passes. A false flag returns `none`
  without reading loss values to the host. Singleton and batch updates share optimizer moments.
  -/
  nativeBatchStep? : Option (NativeOptimizer α →
    Array (TensorPack α inputShapes × TensorPack δ dataInputShapes) →
    (readLoss : Bool) → IO (Option (Tensor α []))) := none
  /-- Read current module state, synchronizing device mirrors if needed. -/
  getState : IO (TensorPack α paramShapes)

end TorchLean
