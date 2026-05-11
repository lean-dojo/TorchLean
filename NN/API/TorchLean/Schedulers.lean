/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Learning-rate schedulers (public API)

This module contains the *pure schedule math* for learning-rate policies used by TorchLean
training loops and examples. The core entrypoint is `Config` together with `lrAt`.

## References

- PyTorch schedulers: `torch.optim.lr_scheduler.*`
  (`https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate`)
-/

@[expose] public section


namespace NN
namespace API
namespace TorchLean
namespace Schedulers

/--
Small learning-rate scheduler surface for higher-level training code.

This file keeps the interface deliberately small: a `Config` is just a description of a schedule,
and `lrAt cfg t` computes the learning rate at step/epoch index `t`.

### PyTorch mapping

`Config.step` and `Config.exponential` correspond to the schedule math of:
- `torch.optim.lr_scheduler.StepLR`
- `torch.optim.lr_scheduler.ExponentialLR`
-/
inductive Config where
  | constant (lr : Float)
  | step (base : Float) (stepSize : Nat) (gamma : Float := 0.1)
  | exponential (base : Float) (gamma : Float)
  deriving Repr

/-- Constant learning-rate schedule. -/
def constant (lr : Float) : Config := .constant lr

/-- Step decay learning-rate schedule. -/
def step (base : Float) (stepSize : Nat) (gamma : Float := 0.1) : Config :=
  .step base stepSize gamma

/-- Exponential learning-rate schedule. -/
def exponential (base : Float) (gamma : Float) : Config :=
  .exponential base gamma

/-- Learning rate at a given step or epoch index. -/
def lrAt : Config → Nat → Float
  | .constant lr, _ => lr
  | .step base stepSize gamma, t =>
      if stepSize = 0 then
        base
      else
        let k := t / stepSize
        base * (Float.pow gamma (Float.ofNat k))
  | .exponential base gamma, t =>
      base * (Float.pow gamma (Float.ofNat t))

end Schedulers
end TorchLean
end API
end NN
