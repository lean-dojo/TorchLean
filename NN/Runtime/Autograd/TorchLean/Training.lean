/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Utils
public import NN.Runtime.Autograd.TorchLean.Optim

/-!
# TorchLean training-loop helpers

This module contains training-loop utilities that need the high-level `TorchLean.Optim` optimizer
interface. The lower-level `Runtime.Autograd.Torch.Utils` file intentionally stops at
`trainCycleSGD`, because that helper only depends on the `ScalarTrainer` update bundled by the
low-level session layer.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

/--
Train `steps` updates with an arbitrary TorchLean optimizer, cycling through `samples`.

PyTorch comparison: analogous to using a `torch.optim.Optimizer` and calling
`loss.backward(); opt.step()` in a loop, except here `opt.step` consumes an explicit gradient
`TList` aligned with `paramShapes`.
-/
def trainCycleOptim
    {α : Type} [Context α] [ToString α]
    {paramShapes inputShapes : List Shape}
    (tr : _root_.Runtime.Autograd.Torch.ScalarTrainer α paramShapes inputShapes)
    (opt : Optim.Optimizer α paramShapes)
    (st0 : opt.State)
    (steps : Nat) (samples : List (_root_.Runtime.Autograd.Torch.TList α inputShapes))
    (logEvery : Nat := 1) : IO opt.State := do
  match samples with
  | [] =>
      throw <| IO.userError "trainCycleOptim: empty dataset"
  | hd :: _tl =>
      let mut st := st0
      for step in [0:steps] do
        let xs := samples.getD (step % samples.length) hd
        let lossT ← _root_.Runtime.Autograd.Torch.ScalarTrainer.forwardT
          (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) tr xs
        if logEvery != 0 && step % logEvery = 0 then
          IO.println s!"step {step}: loss={_root_.Runtime.Autograd.Torch.scalarOf lossT}"
        match ← opt.trainerStep? tr st xs with
        | some st' =>
            st := st'
        | none =>
            let grads ← _root_.Runtime.Autograd.Torch.ScalarTrainer.backwardT
              (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) tr xs
            st ← opt.step st tr.params grads
      pure st

end TorchLean
end Autograd
end Runtime
