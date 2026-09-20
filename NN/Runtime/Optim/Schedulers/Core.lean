/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Tensor.Internal.Representation.Storage -- shake: keep

/-!
# Scheduler Arithmetic

Learning-rate schedulers for TorchLean runtime training.

Schedulers are small deterministic state machines that answer:

- “what learning rate should we use *at this step*?”
- “how do we advance to the next step?”

TorchLean keeps schedulers explicit and pure so:
- runtime code can store scheduler state in a record (or serialize it),
- proofs and specs can refer to the exact schedule that was used.

Step counter convention:
- `currentStep` is **0-indexed**. The first call to `current` uses `currentStep = 0`.
- `advance` increments the counter by 1.

This module contains the shared scalar operations used by native and PyTorch-compatible schedules.

References (common schedules we implement):
- Cosine annealing / SGDR (Loshchilov–Hutter, 2017): https://arxiv.org/abs/1608.03983
- Cyclical learning rates (Smith, 2017): https://arxiv.org/abs/1506.01186
- 1cycle policy (Smith, 2018): https://arxiv.org/abs/1803.09820

PyTorch references:
- `torch.optim.lr_scheduler` overview:
  https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
-/

@[expose] public section


namespace Optim
namespace Scheduler

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

open MathFunctions

/-! ## Shared utilities -/
namespace Internal

/-- Clamp `value` to the closed interval `[lo, hi]`. -/
def clamp (value : α) (lo : α) (hi : α) : α :=
  if value < lo then lo
  else if value > hi then hi
  else value

/--
Safe division `num/denom`.

Returns `0` when `denom == 0` so schedulers stay total even when misconfigured.
This is used by the PyTorch-compatible schedulers, which mirror PyTorch's use of
floating `pct` values but avoid exceptions in pure code.
-/
def safeDiv (num denom : α) : α :=
  if denom == 0 then 0 else num / denom

/--
Safe ratio `num/denom` cast into the scalar type.

Returns `0` when `denom = 0` so schedulers stay total even when misconfigured.
-/
def ratioNat (num denom : Nat) : α :=
  if denom = 0 then 0 else (num : α) / (denom : α)

/-- Linear interpolation between `startValue` and `endValue` with `factor ∈ [0,1]`. -/
def linearInterpolation (startValue : α) (endValue : α) (factor : α) : α :=
  let factor := clamp factor 0 1
  startValue + factor * (endValue - startValue)

/--
Linear interpolation between `startValue` and `endValue` with no clamping.

This matches PyTorch's anneal helpers (`OneCycleLR._annealing_linear`), which permit
`factor` outside `[0,1]` and therefore extrapolate.
-/
def linearInterpolationUnclamped (startValue : α) (endValue : α) (factor : α) : α :=
  startValue + factor * (endValue - startValue)

/--
Cosine interpolation between `startValue` and `endValue` with `factor ∈ [0,1]`.

This is the usual smooth schedule: it starts and ends with zero slope.
-/
def cosineInterpolation (startValue : α) (endValue : α) (factor : α) : α :=
  let factor := clamp factor 0 1
  let cosVal := (1 + cos (pi * factor)) / (1 + 1)
  startValue + (1 - cosVal) * (endValue - startValue)

/--
Cosine anneal between `startValue` and `endValue` with no clamping.

This matches PyTorch's anneal helper (`OneCycleLR._annealing_cos`).
-/
def cosineInterpolationUnclamped (startValue : α) (endValue : α) (factor : α) : α :=
  let cosOut := cos (pi * factor) + 1
  endValue + (startValue - endValue) / (1 + 1) * cosOut

end Internal

end Scheduler
end Optim
