/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.Module
public import NN.API.Trainer.Memory
public import NN.API.Trainer.Reporting
public import NN.API.Seeded

/-!
# Fixed-Sample Training

Some runnable examples train repeatedly on one caller-supplied sample:

1. build a model with `TorchLean.nn.withModel`,
2. wrap it as an `ObjectiveDefinition` (model + supervised loss),
3. load or synthesize one supervised input and target,
4. run optimizer updates on that fixed sample, and
5. report the loss before and after training.

This module provides that loop without tying it to a particular model family.

Scope:
- it trains against one fixed sample supplied by the caller;
- it is model-agnostic: callers supply the loss wrapper and optimizer constructor;
- it is backend-agnostic: callers can use it on CPU or CUDA via `API.Runtime.Options`.

For dataset-backed training, use the `TorchLean.Trainer` API exported by `NN` or the shared model
loader helpers.
-/

@[expose] public section

namespace TorchLean

open Spec TorchLean TorchLean.Tensor

namespace Trainer
namespace FixedSample

/-- One fixed-sample run for an arbitrary scalar backend. -/
def train
    {α : Type} [TorchLean.Storage α] [Context α]
    [ToString α] [TorchLean.Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    {σ τ : Spec.Shape}
    (buildModel : TorchLean.nn.Builder
      (TorchLean.nn.Sequential σ τ))
    (buildObjective :
      (model : TorchLean.nn.Sequential σ τ) →
        TorchLean.Module.ObjectiveDefinition Unit (TorchLean.nn.stateShapes model)
          [σ, τ])
    (buildOptimizer :
      (cast : Float → α) → (stateShapes : List Shape) →
        Runtime.Autograd.Model.Optim.Optimizer α stateShapes)
    (cast : Float → α)
    (options : Runtime.Autograd.Torch.Config)
    (sample : TorchLean.Sample.Supervised α σ τ)
    (steps : Nat)
    (cudaMemorySampleEvery : Nat := 0) :
    IO (Training.LossProgress α) := do
  TorchLean.nn.withModel buildModel fun model => do
    let objectiveDefinition := buildObjective model
    let runtimeObjective ←
      TorchLean.Module.Internal.instantiate (α := α) objectiveDefinition cast options
    let arguments := TorchLean.Sample.Internal.arguments sample
    let lossBeforeTensor ← TorchLean.Module.Objective.loss (α := α) runtimeObjective
      arguments Arguments.empty
    let lossBefore := TorchLean.Tensor.item lossBeforeTensor
    let optimizer := buildOptimizer cast (TorchLean.nn.stateShapes model)
    let boundOptimizer ←
      TorchLean.Module.Internal.bindOptimizer (α := α) runtimeObjective optimizer
    let watchEvery := TorchLean.Trainer.Memory.cadence options steps cudaMemorySampleEvery
    let mut memorySample? ← TorchLean.Trainer.Memory.sample options watchEvery steps 0 none
    for step in [0:steps] do
      boundOptimizer.step (Arguments.Internal.toTensorPack arguments) TensorPack.empty
      memorySample? ←
        TorchLean.Trainer.Memory.sample
          options watchEvery steps (step + 1) memorySample?
    let lossAfterTensor ← TorchLean.Module.Objective.loss (α := α) runtimeObjective
      arguments Arguments.empty
    let lossAfter := TorchLean.Tensor.item lossAfterTensor
    pure { before := lossBefore, after := lossAfter }

end FixedSample
end Trainer
end TorchLean
