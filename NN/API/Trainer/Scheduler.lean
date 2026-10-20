/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Learning-Rate Schedules

Pure learning-rate schedules used by TorchLean training loops and examples. A `Config` describes a
schedule, and `Config.rate` evaluates it at an optimizer step.

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
and `config.rate stepIndex` computes the learning rate at that optimizer step or epoch.

### PyTorch mapping

`Config.step` and `Config.exponential` correspond to the schedule math of:
- `torch.optim.lr_scheduler.StepLR`
- `torch.optim.lr_scheduler.ExponentialLR`

`Config.warmupCosine` is the schedule commonly used for Transformer pretraining: a short linear
warm-up followed by cosine decay to a nonzero floor.
-/
inductive Config where
  | constant (lr : Float)
  | step (baseLr : Float) (stepSize : Nat) (decayFactor : Float := 0.1)
  | exponential (baseLr : Float) (decayFactor : Float)
  /--
  Linearly warm up to `peakLr`, then follow a cosine curve down to
  `minLr`.

  `warmupSteps` counts optimizer updates. The first update uses
  `peakLr / warmupSteps`, and the last warm-up update reaches `peakLr`. Once
  `totalSteps` updates have been scheduled, the learning rate remains at `minLr`.
  A warm-up longer than the run is clamped to `totalSteps`. When `totalSteps = 0`, no update belongs
  to the schedule and `Config.rate` returns `minLr`.
  -/
  | warmupCosine
      (peakLr minLr : Float) (warmupSteps totalSteps : Nat)
  deriving Repr

export Config (constant step exponential warmupCosine)

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
def Internal.validate : Config → Except String Unit
  | .constant lr =>
      Internal.requireRate "learning rate" lr
  | .step baseLr stepSize decayFactor => do
      Internal.requireRate "base learning rate" baseLr
      unless stepSize > 0 do
        throw "scheduler: step size must be positive"
      Internal.requireDecay "decay factor" decayFactor
  | .exponential baseLr decayFactor => do
      Internal.requireRate "base learning rate" baseLr
      Internal.requireDecay "decay factor" decayFactor
  | .warmupCosine peakLr minLr _ _ => do
      Internal.requireRate "peak learning rate" peakLr
      Internal.requireRate "minimum learning rate" minLr
      unless minLr <= peakLr do
        throw "scheduler: minimum learning rate must not exceed peak learning rate"

/--
Validate a schedule, optionally checking its largest rate after scalar conversion.

`round` reports the converted value in binary64. Its default leaves rates unchanged; a runtime
supplies its scalar conversion to catch overflow before training. Schedule arithmetic remains
binary64; each resulting rate is converted when the optimizer state is updated.
-/
def Config.validate (config : Config) (round : Float → Float := id) :
    Except String Unit := do
  Internal.validate config
  let maximumRate := match config with
    | .constant rate => rate
    | .step rate _ _ => rate
    | .exponential rate _ => rate
    | .warmupCosine peak _ _ _ => peak
  Internal.requireRate "learning rate after conversion to the runtime scalar"
    (round maximumRate)

/-- Learning rate at a given step or epoch index. -/
def Config.rate : Config → Nat → Float
  | .constant lr, _ => lr
  | .step baseLr stepSize decayFactor, stepIndex =>
      if stepSize = 0 then
        baseLr
      else
        let decayCount := stepIndex / stepSize
        baseLr * (Float.pow decayFactor (Float.ofNat decayCount))
  | .exponential baseLr decayFactor, stepIndex =>
      baseLr * (Float.pow decayFactor (Float.ofNat stepIndex))
  | .warmupCosine peakLr minLr warmupSteps totalSteps, stepIndex =>
      if totalSteps = 0 then
        minLr
      else if stepIndex >= totalSteps then
        minLr
      else
        let warmupSteps := Nat.min warmupSteps totalSteps
        if stepIndex < warmupSteps then
          peakLr * (Float.ofNat (stepIndex + 1) / Float.ofNat warmupSteps)
        else
          let decaySteps := totalSteps - warmupSteps
          if decaySteps = 0 then
            minLr
          else
            let progress := Float.ofNat (stepIndex - warmupSteps) / Float.ofNat decaySteps
            let cosine := (1.0 + Float.cos (3.141592653589793 * progress)) / 2.0
            minLr + (peakLr - minLr) * cosine

end Scheduler
end Trainer
end TorchLean
