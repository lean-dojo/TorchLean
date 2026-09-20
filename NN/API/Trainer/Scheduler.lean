/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Learning-Rate Schedules

Pure learning-rate schedules used by TorchLean training loops and examples. A `Config` describes a
schedule, and `learningRateAt` evaluates it at an optimizer step.

## References

- PyTorch schedulers: `torch.optim.lr_scheduler.*`
  (`https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate`)
-/

@[expose] public section


namespace TorchLean
namespace Trainer
namespace Scheduler

/--
Small learning-rate scheduler surface for higher-level training code.

This file keeps the interface compact: a `Config` is just a description of a schedule,
and `learningRateAt config stepIndex` computes the learning rate at that optimizer step or epoch.

### PyTorch mapping

`Config.step` and `Config.exponential` correspond to the schedule math of:
- `torch.optim.lr_scheduler.StepLR`
- `torch.optim.lr_scheduler.ExponentialLR`

`Config.warmupCosine` is the schedule commonly used for Transformer pretraining: a short linear
warm-up followed by cosine decay to a nonzero floor.
-/
inductive Config where
  | constant (learningRate : Float)
  | step (baseLearningRate : Float) (stepSize : Nat) (decayFactor : Float := 0.1)
  | exponential (baseLearningRate : Float) (decayFactor : Float)
  | warmupCosine
      (peakLearningRate minimumLearningRate : Float) (warmupSteps totalSteps : Nat)
  deriving Repr

/-- Constant learning-rate schedule. -/
def constant (learningRate : Float) : Config := .constant learningRate

/-- Step decay learning-rate schedule. -/
def step
    (baseLearningRate : Float) (stepSize : Nat) (decayFactor : Float := 0.1) : Config :=
  .step baseLearningRate stepSize decayFactor

/-- Exponential learning-rate schedule. -/
def exponential (baseLearningRate : Float) (decayFactor : Float) : Config :=
  .exponential baseLearningRate decayFactor

/--
Linearly warm up to `peakLearningRate`, then follow a cosine curve down to
`minimumLearningRate`.

`warmupSteps` counts optimizer updates. The first update uses
`peakLearningRate / warmupSteps`, and the last warm-up update reaches `peakLearningRate`. Once
`totalSteps` updates have been scheduled, the learning rate remains at `minimumLearningRate`.
A warm-up longer than the run is clamped to `totalSteps`. When `totalSteps = 0`, no update belongs
to the schedule and `learningRateAt` returns `minimumLearningRate`.
-/
def warmupCosine
    (peakLearningRate minimumLearningRate : Float)
    (warmupSteps totalSteps : Nat) : Config :=
  .warmupCosine peakLearningRate minimumLearningRate warmupSteps totalSteps

namespace Internal

/-- Reject a learning rate that is not a finite number at or above zero. -/
def requireRate (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 <= value do
    throw s!"scheduler: {name} must be finite and nonnegative"

/-- Reject a decay factor outside `[0, 1]`. A factor above one grows the learning rate instead of
decaying it, which in practice is a typo rather than an intention. -/
def requireDecay (name : String) (value : Float) : Except String Unit := do
  unless value.isFinite && 0.0 <= value && value <= 1.0 do
    throw s!"scheduler: {name} must be finite and satisfy 0 <= {name} <= 1"

end Internal

/--
Validate a learning-rate schedule before it is attached to optimizer state.

The checks keep every scheduled rate finite and nonnegative. Step and exponential schedules use
decay factors in `[0, 1]`; a zero step size is rejected instead of silently changing the schedule
to a constant rate.
-/
def validate : Config → Except String Unit
  | .constant learningRate =>
      Internal.requireRate "learning rate" learningRate
  | .step baseLearningRate stepSize decayFactor => do
      Internal.requireRate "base learning rate" baseLearningRate
      unless stepSize > 0 do
        throw "scheduler: step size must be positive"
      Internal.requireDecay "decay factor" decayFactor
  | .exponential baseLearningRate decayFactor => do
      Internal.requireRate "base learning rate" baseLearningRate
      Internal.requireDecay "decay factor" decayFactor
  | .warmupCosine peakLearningRate minimumLearningRate _ _ => do
      Internal.requireRate "peak learning rate" peakLearningRate
      Internal.requireRate "minimum learning rate" minimumLearningRate
      unless minimumLearningRate <= peakLearningRate do
        throw "scheduler: minimum learning rate must not exceed peak learning rate"

/-- Also check that the largest scheduled rate remains finite in binary32 training. -/
def validateFloat32 (config : Config) : Except String Unit := do
  validate config
  let maximumRate := match config with
    | .constant rate => rate
    | .step rate _ _ => rate
    | .exponential rate _ => rate
    | .warmupCosine peak _ _ _ => peak
  Internal.requireRate "learning rate after conversion to binary32" maximumRate.toFloat32.toFloat

/-- Learning rate at a given step or epoch index. -/
def learningRateAt : Config → Nat → Float
  | .constant learningRate, _ => learningRate
  | .step baseLearningRate stepSize decayFactor, stepIndex =>
      if stepSize = 0 then
        baseLearningRate
      else
        let decayCount := stepIndex / stepSize
        baseLearningRate * (Float.pow decayFactor (Float.ofNat decayCount))
  | .exponential baseLearningRate decayFactor, stepIndex =>
      baseLearningRate * (Float.pow decayFactor (Float.ofNat stepIndex))
  | .warmupCosine peakLearningRate minimumLearningRate warmupSteps totalSteps, stepIndex =>
      if totalSteps = 0 then
        minimumLearningRate
      else if stepIndex >= totalSteps then
        minimumLearningRate
      else
        let warmupSteps := Nat.min warmupSteps totalSteps
        if stepIndex < warmupSteps then
          peakLearningRate * (Float.ofNat (stepIndex + 1) / Float.ofNat warmupSteps)
        else
          let decaySteps := totalSteps - warmupSteps
          if decaySteps = 0 then
            minimumLearningRate
          else
            let progress := Float.ofNat (stepIndex - warmupSteps) / Float.ofNat decaySteps
            let cosine := (1.0 + Float.cos (3.141592653589793 * progress)) / 2.0
            minimumLearningRate + (peakLearningRate - minimumLearningRate) * cosine

end Scheduler
end Trainer
end TorchLean
