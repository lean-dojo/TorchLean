/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Schedulers.Core

/-!
# PyTorch-Compatible Learning-Rate Schedulers

Schedulers whose phase boundaries and step counters follow the corresponding
`torch.optim.lr_scheduler` behavior. They remain pure Lean state machines, so a training run can
store, inspect, and reason about the exact scheduler state without calling PyTorch.

`Schedulers.Core` documents the zero-indexed counter convention, shared scalar operations, and
literature. The `Native` module provides simpler total schedules when compatibility is not the
contract.

Only schedules whose semantics differ from the native ones live here. `StepLR` is not duplicated:
the native `Scheduler.StepDecay` already computes `base_lr * gamma ^ (step / step_size)` with the
same zero-indexed counter, so it is the PyTorch-compatible step schedule as well.
-/

@[expose] public section


namespace Optim
namespace Scheduler
namespace PyTorch

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

open MathFunctions

/-! ## PyTorch-compatible scheduler variants -/

/-!
The schedulers below use formulas and step-count conventions chosen to match PyTorch's
`torch.optim.lr_scheduler.*` semantics more directly.

Important convention note (PyTorch `last_epoch`):
- In modern PyTorch, schedulers effectively start at `last_epoch = 0` right after construction
  (fresh run with `last_epoch = -1` in the constructor triggers an initial internal step).
- We model that behavior by using a `currentStep : Nat := 0` counter.
  Think: `currentStep` corresponds to PyTorch's `last_epoch` after construction.

These schedulers are *LR-only* (they do not mutate optimizer momentum/betas). If you need the full
PyTorch OneCycle momentum behavior, consider adding a separate momentum schedule and stepping both
in lockstep.
-/

/-! ### CosineAnnealingLR -/

/--
PyTorch-compatible `CosineAnnealingLR`.

Key behavior difference from TorchLean's native `Scheduler.CosineAnnealing`:
- PyTorch's `CosineAnnealingLR` continues the cosine curve past `T_max` (it is periodic with period
  `2*T_max`), rather than clamping to `eta_min`.

PyTorch reference: `torch.optim.lr_scheduler.CosineAnnealingLR`.
-/
structure CosineAnnealing (α : Type) where
  /-- Base learning rate (`base_lrs[i]`). -/
  baseLearningRate : α
  /-- Maximum number of steps in a half-cycle (`T_max`). -/
  halfCycleSteps : Nat
  /-- Minimum learning rate (`eta_min`). -/
  minimumLearningRate : α
  /-- Step counter matching PyTorch `last_epoch` after construction (0-indexed). -/
  currentStep : Nat := 0

/-- Current learning rate for PyTorch-compatible cosine annealing. -/
def CosineAnnealing.current (scheduler : CosineAnnealing α) : α :=
  if scheduler.halfCycleSteps = 0 then
    scheduler.baseLearningRate
  else
    scheduler.minimumLearningRate
      + (scheduler.baseLearningRate - scheduler.minimumLearningRate)
          * (1 + cos ((pi : α) * (scheduler.currentStep : α) /
            (scheduler.halfCycleSteps : α)))
          / (1 + 1)

/-- Advance PyTorch-compatible cosine annealing by one step. -/
def CosineAnnealing.advance (scheduler : CosineAnnealing α) : CosineAnnealing α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/-- Create PyTorch-compatible cosine annealing at step zero. -/
def CosineAnnealing.create (baseLearningRate : α) (halfCycleSteps : Nat)
    (minimumLearningRate : α := 0) : CosineAnnealing α :=
  { baseLearningRate := baseLearningRate
    halfCycleSteps := halfCycleSteps
    minimumLearningRate := minimumLearningRate }

/-! ### OneCycleLR (LR-only) -/

/-- Anneal strategy used by `OneCycleLR` (matches PyTorch `"cos"` or `"linear"`). -/
inductive AnnealingStrategy
  | cosine
  | linear
  deriving Repr, DecidableEq

/--
PyTorch-compatible `OneCycleLR` (LR-only).

Notes:
- This mirrors PyTorch's `OneCycleLR` *learning-rate* schedule only. PyTorch can also cycle momentum
  (or Adam's `beta1`); TorchLean keeps this scheduler pure and LR-only.
- PyTorch defines:
  - `initial_lr = max_lr / div_factor`
  - `min_lr = initial_lr / final_div_factor`
  (note: `min_lr` is derived from `initial_lr`, not directly from `max_lr`).
- PyTorch uses "phase end steps" that are floats:
  - phase 1 ends at `pct_start * total_steps - 1`
  - phase 2 ends at `total_steps - 1` (and `three_phase` inserts a middle phase).
  This means the boundary can be fractional; the schedule uses interpolation ratios (`pct`)
  computed from these float endpoints. We match that behavior using `α` arithmetic.

PyTorch reference: `torch.optim.lr_scheduler.OneCycleLR`.
-/
structure OneCycle (α : Type) where
  /-- Peak learning rate (`max_lr`). -/
  maximumLearningRate : α
  /-- Total number of steps (`total_steps`). -/
  totalSteps : Nat
  /-- Fraction of steps spent increasing LR (`pct_start`). -/
  increasingFraction : α
  /-- `div_factor` used to derive `initial_lr = max_lr / div_factor`. -/
  divisionFactor : α
  /-- `final_div_factor` used to derive `min_lr = initial_lr / final_div_factor`. -/
  finalDivisionFactor : α
  /-- Anneal strategy (`cos` or `linear`). -/
  annealingStrategy : AnnealingStrategy := .cosine
  /-- Use PyTorch's `three_phase` variant when `true`. -/
  threePhase : Bool := false
  /-- Step counter matching PyTorch `last_epoch` after construction (0-indexed). -/
  currentStep : Nat := 0

namespace OneCycle

/-- Derived initial LR (`max_lr / div_factor`). -/
def initialLearningRate (scheduler : OneCycle α) : α :=
  scheduler.maximumLearningRate / scheduler.divisionFactor

/-- Derived minimum LR (`initial_lr / final_div_factor`). -/
def minimumLearningRate (scheduler : OneCycle α) : α :=
  scheduler.initialLearningRate / scheduler.finalDivisionFactor

/-- PyTorch-compatible anneal helper (no clamping). -/
def anneal (scheduler : OneCycle α) (startingLearningRate endingLearningRate fraction : α) : α :=
  match scheduler.annealingStrategy with
  | .cosine =>
      Internal.cosineInterpolationUnclamped startingLearningRate endingLearningRate fraction
  | .linear =>
      Internal.linearInterpolationUnclamped startingLearningRate endingLearningRate fraction

end OneCycle

/-- Current learning rate for PyTorch-compatible one-cycle scheduling (LR-only). -/
def OneCycle.current (scheduler : OneCycle α) : α :=
  let initialLearningRate := scheduler.initialLearningRate
  let minimumLearningRate := scheduler.minimumLearningRate
  if scheduler.totalSteps = 0 then
    initialLearningRate
  else
    -- PyTorch raises when `step_num > total_steps`. We clamp to keep the function total.
    let boundedStep :=
      if scheduler.currentStep ≤ scheduler.totalSteps then
        scheduler.currentStep
      else
        scheduler.totalSteps
    let step : α := boundedStep
    let total : α := scheduler.totalSteps
    if scheduler.threePhase then
      let increasingPhaseEnd : α := scheduler.increasingFraction * total - 1
      let decreasingPhaseEnd : α :=
        (2 : α) * scheduler.increasingFraction * total - (2 : α)
      let finalPhaseEnd : α := (scheduler.totalSteps - 1 : Nat)
      if step > increasingPhaseEnd then
        if step > decreasingPhaseEnd then
          let fraction :=
            Internal.safeDiv (step - decreasingPhaseEnd) (finalPhaseEnd - decreasingPhaseEnd)
          scheduler.anneal initialLearningRate minimumLearningRate fraction
        else
          let fraction :=
            Internal.safeDiv (step - increasingPhaseEnd) (decreasingPhaseEnd - increasingPhaseEnd)
          scheduler.anneal scheduler.maximumLearningRate initialLearningRate fraction
      else
        let fraction := Internal.safeDiv step (increasingPhaseEnd - 0)
        scheduler.anneal initialLearningRate scheduler.maximumLearningRate fraction
    else
      let increasingPhaseEnd : α := scheduler.increasingFraction * total - 1
      let finalPhaseEnd : α := (scheduler.totalSteps - 1 : Nat)
      if step > increasingPhaseEnd then
        let fraction := Internal.safeDiv (step - increasingPhaseEnd)
          (finalPhaseEnd - increasingPhaseEnd)
        scheduler.anneal scheduler.maximumLearningRate minimumLearningRate fraction
      else
        let fraction := Internal.safeDiv step (increasingPhaseEnd - 0)
        scheduler.anneal initialLearningRate scheduler.maximumLearningRate fraction

/-- Advance PyTorch-compatible one-cycle scheduling by one step. -/
def OneCycle.advance (scheduler : OneCycle α) : OneCycle α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Constructor for `OneCycleLR` starting at `current_step = 0` (LR-only).

This mirrors the PyTorch parameterization:
- `initial_lr = max_lr / div_factor`
- `min_lr = initial_lr / final_div_factor`
- phase endpoints computed as `pct_start * total_steps - 1` and `total_steps - 1` (with the optional
  `three_phase` middle phase).
-/
def OneCycle.create (maximumLearningRate : α) (totalSteps : Nat) (increasingFraction : α)
    (divisionFactor : α) (finalDivisionFactor : α)
    (annealingStrategy : AnnealingStrategy := .cosine)
    (threePhase : Bool := false) : OneCycle α :=
  { maximumLearningRate := maximumLearningRate
    totalSteps := totalSteps
    increasingFraction := increasingFraction
    divisionFactor := divisionFactor
    finalDivisionFactor := finalDivisionFactor
    annealingStrategy := annealingStrategy
    threePhase := threePhase }

end PyTorch
end Scheduler
end Optim
