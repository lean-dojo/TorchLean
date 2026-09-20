/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Schedulers.Core

/-!
# Native Learning-Rate Schedulers

TorchLean-native schedules with explicit state and total formulas. Zero-length warmup or cycle
phases have defined fallback behavior, which makes the schedules convenient for direct execution
and theorem statements. `currentStep` is zero-indexed and `advance` increments it once.

`Schedulers.Core` documents the shared arithmetic, state convention, and literature. Use the
separate `PyTorch` module when exact PyTorch phase and step-count behavior is required.
-/

@[expose] public section


namespace Optim
namespace Scheduler

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

open MathFunctions

/-! ## Native Schedulers -/

/-- Constant scheduler (no learning rate changes). -/
structure Constant (α : Type) where
  /-- Fixed learning rate. -/
  learningRate : α

/--
Get the learning rate for a constant schedule.

PyTorch analogy: no scheduler (or a scheduler that keeps LR fixed).
-/
def Constant.current (scheduler : Constant α) : α :=
  scheduler.learningRate

/--
Advance a constant scheduler by one step.

This is the identity since there is no state to update.

PyTorch analogy: `scheduler.advance()` for a scheduler that does nothing.
-/
def Constant.advance (scheduler : Constant α) : Constant α :=
  scheduler

/--
Create a constant learning-rate scheduler.

PyTorch analogy: constructing training code with a fixed `lr` and no `lr_scheduler`.
-/
def Constant.create (learningRate : α) : Constant α :=
  { learningRate := learningRate }

/-! ## Exponential decay -/

/--
Exponential decay scheduler: `lr(step) = initial_lr * decayRate^step`.

PyTorch analogy: similar spirit to `ExponentialLR`, but we keep state as a simple counter.
-/
structure ExponentialDecay (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLearningRate : α
  /-- Multiplicative decay factor per step (`gamma` in PyTorch terminology). -/
  decayRate : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for an exponential decay schedule at the current step.

Formula: `initial_lr * decayRate ^ current_step`.

PyTorch analogy: `torch.optim.lr_scheduler.ExponentialLR` (but here kept as a pure counter-based
  record).
-/
def ExponentialDecay.current (scheduler : ExponentialDecay α) : α :=
  scheduler.initialLearningRate * (scheduler.decayRate ^ (scheduler.currentStep : α))

/--
Advance the exponential decay scheduler by one step.

PyTorch analogy: `scheduler.advance()`.
-/
def ExponentialDecay.advance (scheduler : ExponentialDecay α) :
  ExponentialDecay α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create an exponential decay scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decayRate)`.
-/
def ExponentialDecay.create (initialLearningRate : α) (decayRate : α) : ExponentialDecay α :=
  { initialLearningRate := initialLearningRate, decayRate := decayRate }

/-! ## Step decay -/

/--
Piecewise-constant decay: every `stepSize` steps, multiply the learning rate by `decayFactor`.
-/
structure StepDecay (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLearningRate : α
  /-- Multiplicative decay factor applied every `stepSize` steps. -/
  decayFactor : α
  /-- Number of steps between decays. -/
  stepSize : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for step decay at the current step.

Every `stepSize` steps, the LR is multiplied by `decayFactor`. When `step_size = 0`, this falls
back to a constant LR.

PyTorch analogy: `torch.optim.lr_scheduler.StepLR`.
-/
def StepDecay.current (scheduler : StepDecay α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.initialLearningRate
  else
    let decayCount := scheduler.currentStep / scheduler.stepSize
    scheduler.initialLearningRate * (scheduler.decayFactor ^ (decayCount : α))

omit [TorchLean.Storage α] [DecidableRel ((· > ·) : α → α → Prop)] in
/--
The totalized `step_size = 0` case is constant.

PyTorch would reject this configuration; TorchLean keeps scheduler evaluation total so configs can
be validated separately from pure schedule semantics.
-/
theorem StepDecay.current_zero_stepSize
    (initialLearningRate decayFactor : α) (currentStep : Nat) :
    StepDecay.current
      { initialLearningRate := initialLearningRate
        decayFactor := decayFactor
        stepSize := 0
        currentStep := currentStep } = initialLearningRate := by
  simp [StepDecay.current]

/--
Advance the step-decay scheduler by one step.

PyTorch analogy: `scheduler.advance()`.
-/
def StepDecay.advance (scheduler : StepDecay α) : StepDecay α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a step-decay scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.StepLR(optimizer, step_size=..., gamma=decayFactor)`.
-/
def StepDecay.create (initialLearningRate : α) (decayFactor : α) (stepSize : Nat) : StepDecay α
  :=
  { initialLearningRate := initialLearningRate, decayFactor := decayFactor, stepSize := stepSize }

/-! ## Cosine annealing -/

/--
Cosine annealing down to `minimumLearningRate` over `maxSteps` steps.

PyTorch analogy: `CosineAnnealingLR` (without restarts).
-/
structure CosineAnnealing (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLearningRate : α
  /-- Minimum learning rate after annealing completes. -/
  minimumLearningRate : α
  /-- Number of steps over which to anneal. -/
  maxSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for cosine annealing at the current step.

We anneal from `initialLearningRate` to `minimumLearningRate` over `maxSteps` steps, clamping
once the step counter passes `maxSteps`.

PyTorch analogy: `torch.optim.lr_scheduler.CosineAnnealingLR` (without restarts).
-/
def CosineAnnealing.current (scheduler : CosineAnnealing α) : α :=
  if scheduler.maxSteps = 0 then
    scheduler.initialLearningRate
  else
    let step := if scheduler.currentStep < scheduler.maxSteps then scheduler.currentStep else
      scheduler.maxSteps
    let factor := Internal.ratioNat step scheduler.maxSteps
    Internal.cosineInterpolation
      scheduler.initialLearningRate scheduler.minimumLearningRate factor

/--
Advance the cosine annealing scheduler by one step.

PyTorch analogy: `scheduler.advance()`.
-/
def CosineAnnealing.advance (scheduler : CosineAnnealing α) :
  CosineAnnealing α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a cosine annealing scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps,
  eta_min=min_lr)`.
-/
def CosineAnnealing.create (initialLearningRate : α) (maxSteps : Nat)
    (minimumLearningRate : α := 0) : CosineAnnealing α :=
  { initialLearningRate := initialLearningRate
    minimumLearningRate := minimumLearningRate
    maxSteps := maxSteps }

/-! ## Linear warmup -/

/--
Linear warmup from `startingLearningRate` to `initialLearningRate` over `warmupSteps` steps,
then constant.

Warmup is a practical trick commonly used when training large models (e.g. Transformers) to avoid
instability at the start of training.
-/
structure LinearWarmup (α : Type) where
  /-- Target learning rate after warmup. -/
  initialLearningRate : α
  /-- Number of warmup steps. -/
  warmupSteps : Nat
  /-- Starting learning rate during warmup. -/
  startingLearningRate : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for linear warmup (then constant).

Before `warmupSteps`, linearly interpolate from `startingLearningRate` to `initialLearningRate`.
Afterwards, keep `initialLearningRate` fixed.

PyTorch analogy: warmup logic commonly implemented in training scripts (and in some scheduler
  helpers).
-/
def LinearWarmup.current (scheduler : LinearWarmup α) : α :=
  if scheduler.warmupSteps = 0 then
    scheduler.initialLearningRate
  else if scheduler.currentStep < scheduler.warmupSteps then
    let factor := Internal.ratioNat scheduler.currentStep scheduler.warmupSteps
    Internal.linearInterpolation
      scheduler.startingLearningRate scheduler.initialLearningRate factor
  else
    scheduler.initialLearningRate

/--
Advance the linear warmup scheduler by one step.

PyTorch analogy: `scheduler.advance()`.
-/
def LinearWarmup.advance (scheduler : LinearWarmup α) : LinearWarmup α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a linear warmup scheduler starting at step `0`.

PyTorch analogy: a warmup wrapper around an optimizer or a base scheduler.
-/
def LinearWarmup.create (initialLearningRate : α) (warmupSteps : Nat)
    (startingLearningRate : α := 0) : LinearWarmup α :=
  { initialLearningRate := initialLearningRate
    warmupSteps := warmupSteps
    startingLearningRate := startingLearningRate }

/-! ## Warmup + cosine -/

/--
Warmup followed by cosine annealing.

This is a common “default” schedule for Transformer-style training: warm up for a few thousand
steps, then gradually anneal.
-/
structure WarmupCosine (α : Type) where
  /-- Peak learning rate (reached at the end of warmup). -/
  initialLearningRate : α
  /-- Number of warmup steps. -/
  warmupSteps : Nat
  /-- Total number of steps for the whole schedule (warmup + anneal). -/
  totalSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the warmup-then-cosine schedule at the current step.

- During warmup, LR increases linearly from `0` to `initialLearningRate`.
- After warmup, LR follows a cosine anneal over the remaining steps.
- At and after `totalSteps`, LR remains at `0` instead of beginning another cosine period.

PyTorch analogy: a common Transformer schedule, often implemented by composing warmup with cosine
  decay.
-/
def WarmupCosine.current (scheduler : WarmupCosine α) : α :=
  if scheduler.totalSteps = 0 then
    scheduler.initialLearningRate
  else if scheduler.currentStep >= scheduler.totalSteps then
    0
  else if scheduler.currentStep < scheduler.warmupSteps then
    if scheduler.warmupSteps = 0 then
      scheduler.initialLearningRate
    else
      scheduler.initialLearningRate *
        Internal.ratioNat scheduler.currentStep scheduler.warmupSteps
  else
    let remainingSteps := scheduler.totalSteps - scheduler.warmupSteps
    if remainingSteps = 0 then
      scheduler.initialLearningRate
    else
      let currentRemaining := scheduler.currentStep - scheduler.warmupSteps
      let progress := Internal.ratioNat currentRemaining remainingSteps
      let cosineFactor := (1 + cos ((pi : α) * progress)) / (1 + 1)
      scheduler.initialLearningRate * cosineFactor

/--
Advance the warmup+cosine scheduler by one step.

PyTorch analogy: `scheduler.advance()`.
-/
def WarmupCosine.advance (scheduler : WarmupCosine α) : WarmupCosine α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a warmup+cosine scheduler starting at step `0`.

PyTorch analogy: composing a warmup schedule with cosine annealing in a training script.
-/
def WarmupCosine.create (initialLearningRate : α) (warmupSteps : Nat) (totalSteps : Nat) :
    WarmupCosine α :=
  { initialLearningRate := initialLearningRate
    warmupSteps := warmupSteps
    totalSteps := totalSteps }

/-! ## Cyclic LR -/

/--
Cyclic learning rate schedule.

This corresponds to the “triangular” family of schedules where the LR increases linearly from
`base_lr` to `maximumLearningRate` and then decreases back, repeating in cycles.

The mode is an enum, so unsupported schedule variants cannot enter the runtime state.
-/
inductive CyclicMode where
  /-- Fixed-amplitude triangular cycles. -/
  | triangular
  /-- Triangular cycles whose amplitude halves after each cycle. -/
  | shrinkingTriangular
  /-- Triangular cycles with an exponential amplitude factor. -/
  | exponentialRange
  deriving Repr, DecidableEq

/-- State of a cyclic learning-rate schedule (Smith, "Cyclical Learning Rates for Training Neural
Networks", WACV 2017), matching PyTorch `CyclicLR`.

The step counter lives in the structure rather than being passed in, so `advance` is a pure state
transition and a checkpoint can round-trip a schedule mid-cycle. -/
structure Cyclic (α : Type) where
  /-- Minimum learning rate within the cycle. -/
  baseLearningRate : α
  /-- Maximum learning rate within the cycle (before any mode-specific adjustment). -/
  maximumLearningRate : α
  /-- Half-cycle size (in steps). -/
  stepSize : Nat
  /-- Cycle amplitude policy. -/
  mode : CyclicMode := .triangular
  /-- Decay factor used by `exponentialRange`. -/
  decayFactor : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the cyclic schedule at the current step.

Supports the common `"triangular"`, `"triangular2"`, and `"exp_range"` variants (matching the
flavor of PyTorch's `CyclicLR`).

PyTorch analogy: `torch.optim.lr_scheduler.CyclicLR`.
-/
def Cyclic.current (scheduler : Cyclic α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.baseLearningRate
  else
    let cycleStep := scheduler.currentStep % (2 * scheduler.stepSize)
    let position := Internal.ratioNat cycleStep scheduler.stepSize

    let cycle := scheduler.currentStep / (2 * scheduler.stepSize)
    let adjustedMaximumLearningRate :=
      match scheduler.mode with
      | .triangular => scheduler.maximumLearningRate
      | .shrinkingTriangular =>
          scheduler.maximumLearningRate -
            (scheduler.maximumLearningRate - scheduler.baseLearningRate) *
              (1 - 1 / ((1 + 1) ^ (cycle : α)))
      | .exponentialRange =>
          scheduler.baseLearningRate +
            (scheduler.maximumLearningRate - scheduler.baseLearningRate) *
              scheduler.decayFactor ^ (scheduler.currentStep : α)

    if cycleStep < scheduler.stepSize then
      scheduler.baseLearningRate +
        (adjustedMaximumLearningRate - scheduler.baseLearningRate) * position
    else
      adjustedMaximumLearningRate -
        (adjustedMaximumLearningRate - scheduler.baseLearningRate) * (position - 1)

/--
Advance the cyclic scheduler by one step.

PyTorch analogy: `scheduler.advance()`.
-/
def Cyclic.advance (scheduler : Cyclic α) : Cyclic α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a cyclic learning-rate scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.CyclicLR(base_lr=..., max_lr=..., step_size_up=...)`.
-/
def Cyclic.create (baseLearningRate : α) (maximumLearningRate : α) (stepSize : Nat)
    (mode : CyclicMode := .triangular) (decayFactor : α := 1) : Cyclic α :=
  { baseLearningRate := baseLearningRate
    maximumLearningRate := maximumLearningRate
    stepSize := stepSize
    mode := mode
    decayFactor := decayFactor }

/-! ## Triangular cycle (special case) -/

/--
A specialized cyclic schedule with fixed amplitude.

This is essentially `Cyclic` in `"triangular"` mode, but we provide it as a separate type
so callers don't have to thread mode strings around.
-/
structure TriangularCycle (α : Type) where
  /-- Minimum learning rate within the cycle. -/
  baseLearningRate : α
  /-- Maximum learning rate within the cycle. -/
  maximumLearningRate : α
  /-- Half-cycle size (in steps). -/
  stepSize : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the triangular cycle schedule at the current step.

This is the canonical "triangle up then down" schedule with fixed amplitude.

PyTorch analogy: `CyclicLR` in `"triangular"` mode.
-/
def TriangularCycle.current (scheduler : TriangularCycle α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.baseLearningRate
  else
    let cycleStep := scheduler.currentStep % (2 * scheduler.stepSize)
    if cycleStep < scheduler.stepSize then
      scheduler.baseLearningRate +
        (scheduler.maximumLearningRate - scheduler.baseLearningRate) *
          Internal.ratioNat cycleStep scheduler.stepSize
    else
      let decreasingStep := cycleStep - scheduler.stepSize
      scheduler.maximumLearningRate -
        (scheduler.maximumLearningRate - scheduler.baseLearningRate) *
          Internal.ratioNat decreasingStep scheduler.stepSize

/--
Advance the triangular cycle scheduler by one step.

PyTorch analogy: `scheduler.advance()`.
-/
def TriangularCycle.advance (scheduler : TriangularCycle α) :
  TriangularCycle α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a triangular cycle scheduler starting at step `0`.

PyTorch analogy: `CyclicLR(base_lr=..., max_lr=..., mode=\"triangular\")`.
-/
def TriangularCycle.create (baseLearningRate : α) (maximumLearningRate : α) (stepSize : Nat) :
    TriangularCycle α :=
  { baseLearningRate := baseLearningRate
    maximumLearningRate := maximumLearningRate
    stepSize := stepSize }

/-! ## 1cycle Learning Rate Schedule -/

/--
One-cycle learning-rate schedule.

- increase LR from `initialLearningRate` to `maximumLearningRate` over the first
  `increasingFraction` of the steps,
- then decrease to `finalLearningRate` over the rest.

In the original 1cycle policy, momentum is also scheduled; we keep this runtime version LR-only.
-/
structure OneCycle (α : Type) where
  /-- Peak learning rate (reached at `increasingFraction` of the schedule). -/
  maximumLearningRate : α
  /-- Total number of steps in the schedule. -/
  totalSteps : Nat
  /-- Learning rate at step `0`. -/
  initialLearningRate : α
  /-- Learning rate after the full schedule finishes. -/
  finalLearningRate : α
  /-- Divides `maximumLearningRate` to get `initialLearningRate` in the factory constructor. -/
  divisionFactor : α
  /-- Fraction of the schedule spent increasing LR (0..1). -/
  increasingFraction : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the one-cycle schedule at the current step.

This ramps up to `maximumLearningRate` over the `increasingFraction` part of the schedule, then
anneals down to `finalLearningRate`.

PyTorch analogy: `torch.optim.lr_scheduler.OneCycleLR`, restricted here to the learning-rate curve.
-/
def OneCycle.current (scheduler : OneCycle α) : α :=
  if scheduler.totalSteps = 0 then
    scheduler.initialLearningRate
  else if scheduler.currentStep >= scheduler.totalSteps then
    scheduler.finalLearningRate
  else
    let stepInCycle := scheduler.currentStep
    let cycleStep := Internal.ratioNat stepInCycle scheduler.totalSteps
    if cycleStep < scheduler.increasingFraction then
      let factor := cycleStep / scheduler.increasingFraction
      Internal.linearInterpolation
        scheduler.initialLearningRate scheduler.maximumLearningRate factor
    else
      let factor := (cycleStep - scheduler.increasingFraction) / (1 - scheduler.increasingFraction)
      Internal.linearInterpolation
        scheduler.maximumLearningRate scheduler.finalLearningRate factor

/--
Advance the 1cycle scheduler by one step.

PyTorch analogy: `scheduler.advance()`.
-/
def OneCycle.advance (scheduler : OneCycle α) : OneCycle α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a simplified 1cycle schedule starting at step `0`.

We derive `initial_lr := max_lr / div_factor` and `final_lr := max_lr / final_div_factor`.

PyTorch analogy: `torch.optim.lr_scheduler.OneCycleLR(max_lr=..., total_steps=...)`.
-/
def OneCycle.create (maximumLearningRate : α) (totalSteps : Nat) (divisionFactor : α)
    (increasingFraction : α) (finalDivisionFactor : α) : OneCycle α :=
  let initialLearningRate := maximumLearningRate / divisionFactor
  let finalLearningRate := maximumLearningRate / finalDivisionFactor
  { maximumLearningRate := maximumLearningRate
    totalSteps := totalSteps
    initialLearningRate := initialLearningRate
    finalLearningRate := finalLearningRate
    divisionFactor := divisionFactor
    increasingFraction := increasingFraction }

/-! ## LR finder -/

/--
Learning-rate finder schedule: an exponential sweep from `initialLearningRate` to
`finalLearningRate` over `totalSteps` steps.
-/
structure RangeTest (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLearningRate : α
  /-- Target learning rate at the end of the sweep. -/
  finalLearningRate : α
  /-- Number of steps in the sweep. -/
  totalSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the LR-finder exponential sweep at the current step.

This increases LR exponentially from `initialLearningRate` toward `finalLearningRate` across
`totalSteps` steps.

PyTorch analogy: LR finder utilities used by libraries like fastai, often implemented as a custom
  schedule.
-/
def RangeTest.current (finder : RangeTest α) : α :=
  if finder.totalSteps = 0 then
    finder.initialLearningRate
  else
    let progress := Internal.ratioNat finder.currentStep finder.totalSteps
    finder.initialLearningRate *
      (finder.finalLearningRate / finder.initialLearningRate) ^ (progress : α)

/--
Advance the LR finder by one step.

PyTorch analogy: stepping a custom LR schedule inside a training loop.
-/
def RangeTest.advance (finder : RangeTest α) : RangeTest α :=
  { finder with currentStep := finder.currentStep + 1 }

/--
Create an LR finder schedule starting at step `0`.

PyTorch analogy: setting up an LR finder run to sweep learning rates.
-/
def RangeTest.create (initialLearningRate : α) (finalLearningRate : α) (totalSteps : Nat) :
    RangeTest α :=
  { initialLearningRate := initialLearningRate
    finalLearningRate := finalLearningRate
    totalSteps := totalSteps }

end Scheduler
end Optim
