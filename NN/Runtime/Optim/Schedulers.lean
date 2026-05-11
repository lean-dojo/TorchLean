/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# Schedulers

Learning-rate schedulers for TorchLean runtime training.

Schedulers are small deterministic state machines that answer:

- “what learning rate should we use *at this step*?”
- “how do we advance to the next step?”

TorchLean keeps schedulers explicit and pure so:
- runtime code can store scheduler state in a record (or serialize it),
- proofs and specs can refer to the exact schedule that was used.

Step counter convention:
- `currentStep` is **0-indexed**. The first call to `getLr` uses `currentStep = 0`.
- `step` increments the counter by 1.

PyTorch analogy: these mirror common `torch.optim.lr_scheduler.*` schedules, but expressed as
simple Lean structures with `getLr` and `step`.

Organization:
- the first section defines TorchLean-native schedules with total, easy-to-reason-about behavior;
- the later `*LR` section defines PyTorch-compatible variants when PyTorch's exact phase and
  step-count conventions matter;
- the `create*` names are constructor aliases, not duplicate formulas. They exist so config-heavy
  call sites can read naturally while the implementation remains centralized in the canonical
  constructors (`constantScheduler`, `stepLR`, `oneCycleLR`, and friends).

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

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

open MathFunctions

/-! ## Shared utilities -/
namespace SchedulerUtils

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
def linearInterpolationRaw (startValue : α) (endValue : α) (factor : α) : α :=
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
def cosineAnnealRaw (startValue : α) (endValue : α) (factor : α) : α :=
  let cosOut := cos (pi * factor) + 1
  endValue + (startValue - endValue) / (1 + 1) * cosOut

end SchedulerUtils

/-! ## Constant -/

/-- Constant scheduler (no learning rate changes). -/
structure ConstantScheduler (α : Type) where
  /-- Fixed learning rate. -/
  lr : α

/--
Get the learning rate for a constant schedule.

The step argument is ignored (the LR never changes).

PyTorch analogy: no scheduler (or a scheduler that keeps LR fixed).
-/
def ConstantScheduler.getLr (scheduler : ConstantScheduler α) (_ : Nat) : α :=
  scheduler.lr

/--
Advance a constant scheduler by one step.

This is the identity since there is no state to update.

PyTorch analogy: `scheduler.step()` for a scheduler that does nothing.
-/
def ConstantScheduler.step (scheduler : ConstantScheduler α) : ConstantScheduler α :=
  scheduler

omit [Context α] [DecidableRel ((· > ·) : α → α → Prop)] in
/-- Stepping a constant scheduler leaves the learning rate unchanged. -/
theorem ConstantScheduler.getLr_step (scheduler : ConstantScheduler α) (stepIdx : Nat) :
    (ConstantScheduler.step scheduler).getLr stepIdx = scheduler.getLr stepIdx := by
  rfl

/--
Create a constant learning-rate scheduler.

PyTorch analogy: constructing training code with a fixed `lr` and no `lr_scheduler`.
-/
def constantScheduler (lr : α) : ConstantScheduler α :=
  { lr := lr }

@[inherit_doc constantScheduler]
abbrev createConstantScheduler := @constantScheduler

/-! ## Exponential decay -/

/--
Exponential decay scheduler: `lr(step) = initial_lr * decay_rate^step`.

PyTorch analogy: similar spirit to `ExponentialLR`, but we keep state as a simple counter.
-/
structure ExponentialDecayScheduler (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Multiplicative decay factor per step (`gamma` in PyTorch terminology). -/
  decayRate : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for an exponential decay schedule at the current step.

Formula: `initial_lr * decay_rate ^ current_step`.

PyTorch analogy: `torch.optim.lr_scheduler.ExponentialLR` (but here kept as a pure counter-based
  record).
-/
def ExponentialDecayScheduler.getLr (scheduler : ExponentialDecayScheduler α) : α :=
  scheduler.initialLr * (scheduler.decayRate ^ (scheduler.currentStep : α))

/--
Advance the exponential decay scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def ExponentialDecayScheduler.step (scheduler : ExponentialDecayScheduler α) :
  ExponentialDecayScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create an exponential decay scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)`.
-/
def exponentialDecayScheduler (initialLr : α) (decay_rate : α) : ExponentialDecayScheduler α :=
  { initialLr := initialLr, decayRate := decay_rate }

@[inherit_doc exponentialDecayScheduler]
abbrev createExponentialDecayScheduler := @exponentialDecayScheduler

/-! ## Step decay -/

/--
Piecewise-constant decay: every `step_size` steps, multiply the learning rate by `decay_factor`.
-/
structure StepDecayScheduler (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Multiplicative decay factor applied every `step_size` steps. -/
  decayFactor : α
  /-- Number of steps between decays. -/
  stepSize : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for step decay at the current step.

Every `step_size` steps, the LR is multiplied by `decay_factor`. When `step_size = 0`, this falls
back to a constant LR.

PyTorch analogy: `torch.optim.lr_scheduler.StepLR`.
-/
def StepDecayScheduler.getLr (scheduler : StepDecayScheduler α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.initialLr
  else
    let decayCount := scheduler.currentStep / scheduler.stepSize
    scheduler.initialLr * (scheduler.decayFactor ^ (decayCount : α))

omit [DecidableRel ((· > ·) : α → α → Prop)] in
/--
The totalized `step_size = 0` case is constant.

PyTorch would reject this configuration; TorchLean keeps scheduler evaluation total so configs can
be validated separately from pure schedule semantics.
-/
theorem StepDecayScheduler.getLr_zero_stepSize
    (initialLr decayFactor : α) (currentStep : Nat) :
    StepDecayScheduler.getLr
      { initialLr := initialLr
        decayFactor := decayFactor
        stepSize := 0
        currentStep := currentStep } = initialLr := by
  simp [StepDecayScheduler.getLr]

/--
Advance the step-decay scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def StepDecayScheduler.step (scheduler : StepDecayScheduler α) : StepDecayScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a step-decay scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.StepLR(optimizer, step_size=..., gamma=decay_factor)`.
-/
def stepDecayScheduler (initialLr : α) (decay_factor : α) (stepSize : Nat) : StepDecayScheduler α
  :=
  { initialLr := initialLr, decayFactor := decay_factor, stepSize := stepSize }

@[inherit_doc stepDecayScheduler]
abbrev createStepDecayScheduler := @stepDecayScheduler

/-! ## Cosine annealing -/

/--
Cosine annealing down to `min_lr` over `max_steps` steps.

PyTorch analogy: `CosineAnnealingLR` (without restarts).
-/
structure CosineAnnealingScheduler (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Minimum learning rate after annealing completes. -/
  minLr : α
  /-- Number of steps over which to anneal. -/
  maxSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for cosine annealing at the current step.

We anneal from `initial_lr` to `min_lr` over `max_steps` steps (clamping once we pass `max_steps`).

PyTorch analogy: `torch.optim.lr_scheduler.CosineAnnealingLR` (without restarts).
-/
def CosineAnnealingScheduler.getLr (scheduler : CosineAnnealingScheduler α) : α :=
  if scheduler.maxSteps = 0 then
    scheduler.initialLr
  else
    let step := if scheduler.currentStep < scheduler.maxSteps then scheduler.currentStep else
      scheduler.maxSteps
    let factor := SchedulerUtils.ratioNat step scheduler.maxSteps
    SchedulerUtils.cosineInterpolation scheduler.initialLr scheduler.minLr factor

/--
Advance the cosine annealing scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def CosineAnnealingScheduler.step (scheduler : CosineAnnealingScheduler α) :
  CosineAnnealingScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a cosine annealing scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps,
  eta_min=min_lr)`.
-/
def cosineAnnealingScheduler (initialLr : α) (maxSteps : Nat) (minLr : α := 0) :
    CosineAnnealingScheduler α :=
  { initialLr := initialLr, minLr := minLr, maxSteps := maxSteps }

@[inherit_doc cosineAnnealingScheduler]
abbrev createCosineAnnealingScheduler := @cosineAnnealingScheduler

/-! ## Linear warmup -/

/--
Linear warmup from `start_lr` to `initial_lr` over `warmup_steps`, then constant.

Warmup is a practical trick commonly used when training large models (e.g. Transformers) to avoid
instability at the start of training.
-/
structure LinearWarmupScheduler (α : Type) where
  /-- Target learning rate after warmup. -/
  initialLr : α
  /-- Number of warmup steps. -/
  warmupSteps : Nat
  /-- Starting learning rate during warmup. -/
  startLr : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for linear warmup (then constant).

Before `warmup_steps`, linearly interpolate from `start_lr` to `initial_lr`. Afterwards, keep
`initial_lr` fixed.

PyTorch analogy: warmup logic commonly implemented in training scripts (and in some scheduler
  helpers).
-/
def LinearWarmupScheduler.getLr (scheduler : LinearWarmupScheduler α) : α :=
  if scheduler.warmupSteps = 0 then
    scheduler.initialLr
  else if scheduler.currentStep < scheduler.warmupSteps then
    let factor := SchedulerUtils.ratioNat scheduler.currentStep scheduler.warmupSteps
    SchedulerUtils.linearInterpolation scheduler.startLr scheduler.initialLr factor
  else
    scheduler.initialLr

/--
Advance the linear warmup scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def LinearWarmupScheduler.step (scheduler : LinearWarmupScheduler α) : LinearWarmupScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a linear warmup scheduler starting at step `0`.

PyTorch analogy: a warmup wrapper around an optimizer or a base scheduler.
-/
def linearWarmupScheduler (initialLr : α) (warmupSteps : Nat) (startLr : α := 0) :
    LinearWarmupScheduler α :=
  { initialLr := initialLr, warmupSteps := warmupSteps, startLr := startLr }

@[inherit_doc linearWarmupScheduler]
abbrev createLinearWarmupScheduler := @linearWarmupScheduler

/-! ## Warmup + cosine -/

/--
Warmup followed by cosine annealing.

This is a common “default” schedule for Transformer-style training: warm up for a few thousand
steps, then gradually anneal.
-/
structure WarmupCosineScheduler (α : Type) where
  /-- Peak learning rate (reached at the end of warmup). -/
  initialLr : α
  /-- Number of warmup steps. -/
  warmupSteps : Nat
  /-- Total number of steps for the whole schedule (warmup + anneal). -/
  totalSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the warmup-then-cosine schedule at the current step.

- During warmup, LR increases linearly from `0` to `initial_lr`.
- After warmup, LR follows a cosine anneal over the remaining steps.

PyTorch analogy: a common Transformer schedule, often implemented by composing warmup with cosine
  decay.
-/
def WarmupCosineScheduler.getLr (scheduler : WarmupCosineScheduler α) : α :=
  if scheduler.totalSteps = 0 then
    scheduler.initialLr
  else if scheduler.currentStep < scheduler.warmupSteps then
    if scheduler.warmupSteps = 0 then
      scheduler.initialLr
    else
      scheduler.initialLr * SchedulerUtils.ratioNat scheduler.currentStep scheduler.warmupSteps
  else
    let remaining_steps := scheduler.totalSteps - scheduler.warmupSteps
    if remaining_steps = 0 then
      scheduler.initialLr
    else
      let current_remaining := scheduler.currentStep - scheduler.warmupSteps
      let progress := SchedulerUtils.ratioNat current_remaining remaining_steps
      let cosine_factor := (1 + cos ((pi : α) * progress)) / (1 + 1)
      scheduler.initialLr * cosine_factor

/--
Advance the warmup+cosine scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def WarmupCosineScheduler.step (scheduler : WarmupCosineScheduler α) : WarmupCosineScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a warmup+cosine scheduler starting at step `0`.

PyTorch analogy: composing a warmup schedule with cosine annealing in a training script.
-/
def warmupCosineScheduler (initialLr : α) (warmupSteps : Nat) (totalSteps : Nat) :
    WarmupCosineScheduler α :=
  { initialLr := initialLr, warmupSteps := warmupSteps, totalSteps := totalSteps }

@[inherit_doc warmupCosineScheduler]
abbrev createWarmupCosineScheduler := @warmupCosineScheduler

/-! ## Cyclic LR -/

/--
Cyclic learning rate schedule.

This corresponds to the “triangular” family of schedules where the LR increases linearly from
`base_lr` to `max_lr` and then decreases back, repeating in cycles.

We keep `mode` as a `String` so this runtime layer can be configured from simple config files or
CLI arguments (mirroring how training scripts are usually written).
-/
structure CyclicScheduler (α : Type) where
  /-- Minimum learning rate within the cycle. -/
  baseLr : α
  /-- Maximum learning rate within the cycle (before any mode-specific adjustment). -/
  maxLr : α
  /-- Half-cycle size (in steps). -/
  stepSize : Nat
  /-- `"triangular"`, `"triangular2"`, or `"exp_range"`. -/
  mode : String
  /-- Decay factor used by `"exp_range"`. -/
  gamma : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the cyclic schedule at the current step.

Supports the common `"triangular"`, `"triangular2"`, and `"exp_range"` variants (matching the
flavor of PyTorch's `CyclicLR`).

PyTorch analogy: `torch.optim.lr_scheduler.CyclicLR`.
-/
def CyclicScheduler.getLr (scheduler : CyclicScheduler α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.baseLr
  else
    let cycleStep := scheduler.currentStep % (2 * scheduler.stepSize)
    let x := SchedulerUtils.ratioNat cycleStep scheduler.stepSize

    let cycle := scheduler.currentStep / (2 * scheduler.stepSize)
    let adjustedMaxLR :=
      if scheduler.mode == "triangular" then scheduler.maxLr
      else if scheduler.mode == "triangular2" then
        scheduler.maxLr - (scheduler.maxLr - scheduler.baseLr) * (1 - 1 / ((1 + 1) ^ (cycle :
          α)))
      else if scheduler.mode == "exp_range" then
        scheduler.baseLr + (scheduler.maxLr - scheduler.baseLr) * scheduler.gamma ^
          (scheduler.currentStep : α)
      else
        scheduler.maxLr

    if cycleStep < scheduler.stepSize then
      scheduler.baseLr + (adjustedMaxLR - scheduler.baseLr) * x
    else
      adjustedMaxLR - (adjustedMaxLR - scheduler.baseLr) * (x - 1)

/--
Advance the cyclic scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def CyclicScheduler.step (scheduler : CyclicScheduler α) : CyclicScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a cyclic learning-rate scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.CyclicLR(base_lr=..., max_lr=..., step_size_up=...)`.
-/
def cyclicScheduler (baseLr : α) (maxLr : α) (stepSize : Nat)
    (mode : String := "triangular") (gamma : α := 1) : CyclicScheduler α :=
  { baseLr := baseLr, maxLr := maxLr, stepSize := stepSize, mode := mode, gamma := gamma }

@[inherit_doc cyclicScheduler]
abbrev createCyclicScheduler := @cyclicScheduler

/-! ## Triangular cycle (special case) -/

/--
A specialized cyclic schedule with fixed amplitude.

This is essentially `CyclicScheduler` in `"triangular"` mode, but we provide it as a separate type
so callers don't have to thread mode strings around.
-/
structure TriangularCycleScheduler (α : Type) where
  /-- Minimum learning rate within the cycle. -/
  baseLr : α
  /-- Maximum learning rate within the cycle. -/
  maxLr : α
  /-- Half-cycle size (in steps). -/
  stepSize : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the triangular cycle schedule at the current step.

This is the canonical "triangle up then down" schedule with fixed amplitude.

PyTorch analogy: `CyclicLR` in `"triangular"` mode.
-/
def TriangularCycleScheduler.getLr (scheduler : TriangularCycleScheduler α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.baseLr
  else
    let cycle_step := scheduler.currentStep % (2 * scheduler.stepSize)
    if cycle_step < scheduler.stepSize then
      scheduler.baseLr + (scheduler.maxLr - scheduler.baseLr) * SchedulerUtils.ratioNat
        cycle_step scheduler.stepSize
    else
      let decStep := cycle_step - scheduler.stepSize
      scheduler.maxLr - (scheduler.maxLr - scheduler.baseLr) * SchedulerUtils.ratioNat decStep
        scheduler.stepSize

/--
Advance the triangular cycle scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def TriangularCycleScheduler.step (scheduler : TriangularCycleScheduler α) :
  TriangularCycleScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a triangular cycle scheduler starting at step `0`.

PyTorch analogy: `CyclicLR(base_lr=..., max_lr=..., mode=\"triangular\")`.
-/
def triangularCycleScheduler (baseLr : α) (maxLr : α) (stepSize : Nat) :
    TriangularCycleScheduler α :=
  { baseLr := baseLr, maxLr := maxLr, stepSize := stepSize }

@[inherit_doc triangularCycleScheduler]
abbrev createTriangularCycleScheduler := @triangularCycleScheduler

/-! ## 1cycle Learning Rate Schedule -/

/--
One-cycle learning-rate schedule.

- increase LR from `initial_lr` to `max_lr` over the first `pct_start` fraction of steps,
- then decrease to `final_lr` over the rest.

In the original 1cycle policy, momentum is also scheduled; we keep this runtime version LR-only.
-/
structure OneCycleScheduler (α : Type) where
  /-- Peak learning rate (reached at `pct_start` of the schedule). -/
  maxLr : α
  /-- Total number of steps in the schedule. -/
  totalSteps : Nat
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Learning rate after the full schedule finishes. -/
  finalLr : α
  /-- Divides `max_lr` to get `initial_lr` in the factory constructor. -/
  divFactor : α
  /-- Fraction of the schedule spent increasing LR (0..1). -/
  pctStart : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the one-cycle schedule at the current step.

This ramps up to `max_lr` over the `pct_start` fraction of the schedule, then anneals down to
`final_lr`.

PyTorch analogy: `torch.optim.lr_scheduler.OneCycleLR`, restricted here to the learning-rate curve.
-/
def OneCycleScheduler.getLr (scheduler : OneCycleScheduler α) : α :=
  if scheduler.totalSteps = 0 then
    scheduler.initialLr
  else if scheduler.currentStep >= scheduler.totalSteps then
    scheduler.finalLr
  else
    let stepInCycle := scheduler.currentStep
    let cycleStep := SchedulerUtils.ratioNat stepInCycle scheduler.totalSteps
    if cycleStep < scheduler.pctStart then
      let factor := cycleStep / scheduler.pctStart
      SchedulerUtils.linearInterpolation scheduler.initialLr scheduler.maxLr factor
    else
      let factor := (cycleStep - scheduler.pctStart) / (1 - scheduler.pctStart)
      SchedulerUtils.linearInterpolation scheduler.maxLr scheduler.finalLr factor

/--
Advance the 1cycle scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def OneCycleScheduler.step (scheduler : OneCycleScheduler α) : OneCycleScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a simplified 1cycle schedule starting at step `0`.

We derive `initial_lr := max_lr / div_factor` and `final_lr := max_lr / final_div_factor`.

PyTorch analogy: `torch.optim.lr_scheduler.OneCycleLR(max_lr=..., total_steps=...)`.
-/
def oneCycleScheduler (maxLr : α) (totalSteps : Nat) (divFactor : α) (pctStart : α)
    (finalDivFactor : α) :
    OneCycleScheduler α :=
  let initialLr := maxLr / divFactor
  let finalLr := maxLr / finalDivFactor
  { maxLr := maxLr, totalSteps := totalSteps, initialLr := initialLr, finalLr := finalLr
    divFactor := divFactor, pctStart := pctStart }

@[inherit_doc oneCycleScheduler]
abbrev createOneCycleScheduler := @oneCycleScheduler

/-! ## LR finder -/

/--
Learning-rate finder schedule: exponential sweep from `initial_lr` to `final_lr` over `num_steps`.
-/
structure LRFinder (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Target learning rate at the end of the sweep. -/
  finalLr : α
  /-- Number of steps in the sweep. -/
  numSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the LR-finder exponential sweep at the current step.

This increases LR exponentially from `initial_lr` toward `final_lr` across `num_steps`.

PyTorch analogy: LR finder utilities used by libraries like fastai, often implemented as a custom
  schedule.
-/
def LRFinder.getLr (finder : LRFinder α) : α :=
  if finder.numSteps = 0 then
    finder.initialLr
  else
    let progress := SchedulerUtils.ratioNat finder.currentStep finder.numSteps
    finder.initialLr * (finder.finalLr / finder.initialLr) ^ (progress : α)

/--
Advance the LR finder by one step.

PyTorch analogy: stepping a custom LR schedule inside a training loop.
-/
def LRFinder.step (finder : LRFinder α) : LRFinder α :=
  { finder with currentStep := finder.currentStep + 1 }

/--
Create an LR finder schedule starting at step `0`.

PyTorch analogy: setting up an LR finder run to sweep learning rates.
-/
def lrFinder (initialLr : α) (finalLr : α) (numSteps : Nat) : LRFinder α :=
  { initialLr := initialLr, finalLr := finalLr, numSteps := numSteps }

@[inherit_doc lrFinder]
abbrev createLrFinder := @lrFinder

/-! ## PyTorch-compatible scheduler variants -/

/-!
TorchLean already provides a set of small, pure schedulers above.

This section adds *additional* schedulers whose formulas and step-count conventions are chosen to
match PyTorch's `torch.optim.lr_scheduler.*` semantics more directly.

Important convention note (PyTorch `last_epoch`):
- In modern PyTorch, schedulers effectively start at `last_epoch = 0` right after construction
  (fresh run with `last_epoch = -1` in the constructor triggers an initial internal step).
- We model that behavior by using a `current_step : Nat := 0` counter.
  Think: `current_step` corresponds to PyTorch's `last_epoch` after construction.

These schedulers are *LR-only* (they do not mutate optimizer momentum/betas). If you need the full
PyTorch OneCycle momentum behavior, consider adding a separate momentum schedule and stepping both
in lockstep.
-/

/-! ### StepLR -/

/--
PyTorch-compatible `StepLR`.

Semantics:
- `current_step = 0` yields `base_lr`.
- Every `step_size` steps, multiply LR by `gamma`.
- When `step_size = 0`, this degenerates to a constant schedule (total, no exceptions).

PyTorch reference: `torch.optim.lr_scheduler.StepLR`.
-/
structure StepLR (α : Type) where
  /-- Base learning rate (what PyTorch calls `base_lrs[i]`). -/
  baseLr : α
  /-- Step interval (`step_size`). -/
  stepSize : Nat
  /-- Multiplicative decay factor (`gamma`). -/
  gamma : α
  /-- Step counter matching PyTorch `last_epoch` after construction (0-indexed). -/
  currentStep : Nat := 0

/-- Current learning rate for `StepLR` at `current_step`. -/
def StepLR.getLr (scheduler : StepLR α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.baseLr
  else
    let decayCount := scheduler.currentStep / scheduler.stepSize
    scheduler.baseLr * (scheduler.gamma ^ (decayCount : α))

omit [DecidableRel ((· > ·) : α → α → Prop)] in
/-- The PyTorch-compatible `StepLR` is also totalized to a constant when `step_size = 0`. -/
theorem StepLR.getLr_zero_stepSize
    (baseLr gamma : α) (currentStep : Nat) :
    StepLR.getLr
      { baseLr := baseLr
        stepSize := 0
        gamma := gamma
        currentStep := currentStep } = baseLr := by
  simp [StepLR.getLr]

/-- Advance `StepLR` by one step. -/
def StepLR.step (scheduler : StepLR α) : StepLR α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/-- Constructor for `StepLR` starting at `current_step = 0`. -/
def stepLR (baseLr : α) (stepSize : Nat) (gamma : α) : StepLR α :=
  { baseLr := baseLr, stepSize := stepSize, gamma := gamma }

@[inherit_doc stepLR]
abbrev createStepLr := @stepLR

/-! ### CosineAnnealingLR -/

/--
PyTorch-compatible `CosineAnnealingLR`.

Key behavior difference from TorchLean's `CosineAnnealingScheduler` above:
- PyTorch's `CosineAnnealingLR` continues the cosine curve past `T_max` (it is periodic with period
  `2*T_max`), rather than clamping to `eta_min`.

PyTorch reference: `torch.optim.lr_scheduler.CosineAnnealingLR`.
-/
structure CosineAnnealingLR (α : Type) where
  /-- Base learning rate (`base_lrs[i]`). -/
  baseLr : α
  /-- Maximum number of steps in a half-cycle (`T_max`). -/
  tMax : Nat
  /-- Minimum learning rate (`eta_min`). -/
  etaMin : α
  /-- Step counter matching PyTorch `last_epoch` after construction (0-indexed). -/
  currentStep : Nat := 0

/-- Current learning rate for `CosineAnnealingLR` at `current_step`. -/
def CosineAnnealingLR.getLr (scheduler : CosineAnnealingLR α) : α :=
  if scheduler.tMax = 0 then
    scheduler.baseLr
  else
    scheduler.etaMin
      + (scheduler.baseLr - scheduler.etaMin)
          * (1 + cos ((pi : α) * (scheduler.currentStep : α) / (scheduler.tMax : α)))
          / (1 + 1)

/-- Advance `CosineAnnealingLR` by one step. -/
def CosineAnnealingLR.step (scheduler : CosineAnnealingLR α) : CosineAnnealingLR α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/-- Constructor for `CosineAnnealingLR` starting at `current_step = 0`. -/
def cosineAnnealingLR (baseLr : α) (tMax : Nat) (etaMin : α := Numbers.zero) :
    CosineAnnealingLR α :=
  { baseLr := baseLr, tMax := tMax, etaMin := etaMin }

@[inherit_doc cosineAnnealingLR]
abbrev createCosineAnnealingLr := @cosineAnnealingLR

/-! ### OneCycleLR (LR-only) -/

/-- Anneal strategy used by `OneCycleLR` (matches PyTorch `"cos"` or `"linear"`). -/
inductive OneCycleAnnealStrategy
  | cos
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
structure OneCycleLR (α : Type) where
  /-- Peak learning rate (`max_lr`). -/
  maxLr : α
  /-- Total number of steps (`total_steps`). -/
  totalSteps : Nat
  /-- Fraction of steps spent increasing LR (`pct_start`). -/
  pctStart : α
  /-- `div_factor` used to derive `initial_lr = max_lr / div_factor`. -/
  divFactor : α
  /-- `final_div_factor` used to derive `min_lr = initial_lr / final_div_factor`. -/
  finalDivFactor : α
  /-- Anneal strategy (`cos` or `linear`). -/
  annealStrategy : OneCycleAnnealStrategy := .cos
  /-- Use PyTorch's `three_phase` variant when `true`. -/
  threePhase : Bool := false
  /-- Step counter matching PyTorch `last_epoch` after construction (0-indexed). -/
  currentStep : Nat := 0

namespace OneCycleLR

/-- Derived initial LR (`max_lr / div_factor`). -/
def initialLr (s : OneCycleLR α) : α :=
  s.maxLr / s.divFactor

/-- Derived minimum LR (`initial_lr / final_div_factor`). -/
def minLr (s : OneCycleLR α) : α :=
  initialLr s / s.finalDivFactor

/-- PyTorch-compatible anneal helper (no clamping). -/
def anneal (s : OneCycleLR α) (startLR endLR pct : α) : α :=
  match s.annealStrategy with
  | .cos => SchedulerUtils.cosineAnnealRaw startLR endLR pct
  | .linear => SchedulerUtils.linearInterpolationRaw startLR endLR pct

end OneCycleLR

/-- Current learning rate for `OneCycleLR` at `current_step` (LR-only). -/
def OneCycleLR.getLr (s : OneCycleLR α) : α :=
  let initLR := OneCycleLR.initialLr (α := α) s
  let minLR := OneCycleLR.minLr (α := α) s
  if s.totalSteps = 0 then
    initLR
  else
    -- PyTorch raises when `step_num > total_steps`. We clamp to keep the function total.
    let stepNat := if s.currentStep ≤ s.totalSteps then s.currentStep else s.totalSteps
    let stepNum : α := stepNat
    let total : α := s.totalSteps
    if s.threePhase then
      let end1 : α := s.pctStart * total - 1
      let end2 : α := (Numbers.two : α) * s.pctStart * total - (Numbers.two : α)
      let end3 : α := (s.totalSteps - 1 : Nat)
      if stepNum > end1 then
        if stepNum > end2 then
          let pct := SchedulerUtils.safeDiv (stepNum - end2) (end3 - end2)
          OneCycleLR.anneal (α := α) s initLR minLR pct
        else
          let pct := SchedulerUtils.safeDiv (stepNum - end1) (end2 - end1)
          OneCycleLR.anneal (α := α) s s.maxLr initLR pct
      else
        let pct := SchedulerUtils.safeDiv stepNum (end1 - 0)
        OneCycleLR.anneal (α := α) s initLR s.maxLr pct
    else
      let end1 : α := s.pctStart * total - 1
      let end2 : α := (s.totalSteps - 1 : Nat)
      if stepNum > end1 then
        let pct := SchedulerUtils.safeDiv (stepNum - end1) (end2 - end1)
        OneCycleLR.anneal (α := α) s s.maxLr minLR pct
      else
        let pct := SchedulerUtils.safeDiv stepNum (end1 - 0)
        OneCycleLR.anneal (α := α) s initLR s.maxLr pct

/-- Advance `OneCycleLR` by one step. -/
def OneCycleLR.step (s : OneCycleLR α) : OneCycleLR α :=
  { s with currentStep := s.currentStep + 1 }

/--
Constructor for `OneCycleLR` starting at `current_step = 0` (LR-only).

This mirrors the PyTorch parameterization:
- `initial_lr = max_lr / div_factor`
- `min_lr = initial_lr / final_div_factor`
- phase endpoints computed as `pct_start * total_steps - 1` and `total_steps - 1` (with the optional
  `three_phase` middle phase).
-/
def oneCycleLR (maxLr : α) (totalSteps : Nat) (pctStart : α) (divFactor : α)
    (finalDivFactor : α) (annealStrategy : OneCycleAnnealStrategy := .cos)
    (threePhase : Bool := false) : OneCycleLR α :=
  { maxLr := maxLr
    totalSteps := totalSteps
    pctStart := pctStart
    divFactor := divFactor
    finalDivFactor := finalDivFactor
    annealStrategy := annealStrategy
    threePhase := threePhase }

@[inherit_doc oneCycleLR]
abbrev createOneCycleLr := @oneCycleLR

end Optim
