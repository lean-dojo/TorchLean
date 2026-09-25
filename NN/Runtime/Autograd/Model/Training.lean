/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Optim

/-!
# TorchLean training-loop helpers

Training loops over the runtime optimizer interface, with packed scalar-trainer gradients and
backend-native updates where the optimizer supports them.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
Train `steps` updates with an arbitrary TorchLean optimizer, cycling through `samples`.

PyTorch comparison: analogous to using a `torch.optim.Optimizer` and calling
`loss.backward(); opt.step()` in a loop, except here `opt.step` consumes an explicit gradient
`TorchLean.TensorPack` aligned with `paramShapes`.
-/
def trainCycleOptim
    {α : Type} [TorchLean.Storage α] [Context α] [ToString α]
    {paramShapes inputShapes : List Shape}
    (tr : Runtime.Autograd.Torch.ScalarTrainer α Unit paramShapes inputShapes)
    (opt : Optim.Optimizer α paramShapes)
    (st0 : opt.State)
    (steps : Nat) (samples : Array (TorchLean.TensorPack α inputShapes))
    (logEvery : Nat := 1) : IO opt.State := do
  match samples[0]? with
  | none =>
      throw <| IO.userError "trainCycleOptim: empty dataset"
  | some first =>
      let mut st := st0
      for step in [0:steps] do
        let xs := samples.getD (step % samples.size) first
        let result ←
          match ← opt.trainerStepWithLoss? tr st xs .nil with
          | some result =>
              pure result
          | none => do
              tr.useOptimizerPath .generic
              let (grads, lossTensor) ←
                Runtime.Autograd.Torch.ScalarTrainer.grad
                  (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) tr xs
                  .nil (value := true)
              let _ ← tr.getState
              let st' ← opt.step st tr.state grads
              pure { optimizerState := st', loss := lossTensor }
        st := result.optimizerState
        if logEvery != 0 && step % logEvery = 0 then
          IO.println s!"step {step}: loss={result.loss.item}"
      pure st

end Model
end Autograd
end Runtime
