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

/-- Interpolation between the endpoints of a one-cycle phase. -/
inductive Annealing where
  /-- Constant slope between the endpoints. -/
  | linear
  /-- A half cosine, with zero slope at both endpoints. -/
  | cosine
  deriving Repr

/--
Small learning-rate scheduler surface for higher-level training code.

This file keeps the interface compact: a `Config` is just a description of a schedule,
and `config.rate stepIndex` computes the learning rate after that many optimizer updates.

### PyTorch mapping

`Config.step` and `Config.exponential` correspond to the schedule math of:
- `torch.optim.lr_scheduler.StepLR`
- `torch.optim.lr_scheduler.ExponentialLR`

`Config.warmupCosine` is the schedule commonly used for Transformer pretraining: a short linear
warm-up followed by cosine decay to a nonzero floor.

All indices count completed optimizer updates, starting at zero. `oneCycle` uses explicit integer
phase lengths; it does not reproduce PyTorch's fractional `pct_start * total_steps - 1` boundaries.
-/
inductive Config where
  | constant (lr : Float)
  | step (baseLr : Float) (stepSize : Nat) (decayFactor : Float := 0.1)
  | exponential (baseLr : Float) (decayFactor : Float)
  /-- Increase from `startLr` at step zero to `peakLr` at `warmupSteps`, then hold the peak.
  Zero warm-up steps select the peak immediately. -/
  | linearWarmup (startLr peakLr : Float) (warmupSteps : Nat)
  /-- Decay from `baseLr` to `minLr` over `halfCycleSteps`, then hold the minimum. With
  `periodic := true`, continue back to the base rate and repeat every `2 * halfCycleSteps`.
  A zero half-cycle selects the minimum immediately. -/
  | cosine (baseLr minLr : Float) (halfCycleSteps : Nat) (periodic : Bool := false)
  /-- Repeated linear rises and falls, each lasting `halfCycleSteps`. The first cycle runs from
  `baseLr` to `peakLr` and back. `cycleDecay` scales the amplitude once per completed cycle:
  `1` gives fixed triangles and `0.5` halves each successive peak's distance above the base.
  A zero half-cycle is invalid. -/
  | cyclic (baseLr peakLr : Float) (halfCycleSteps : Nat) (cycleDecay : Float := 1.0)
  /-- Rise from `initialLr` to `peakLr` at `riseSteps`, then fall to `finalLr` at `totalSteps`
  and hold it. Both phases use `annealing`; only the learning rate changes. Validation requires
  `0 < riseSteps < totalSteps` and both endpoints at or below the peak. -/
  | oneCycle (initialLr peakLr finalLr : Float) (riseSteps totalSteps : Nat)
      (annealing : Annealing := .cosine)
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

export Config (constant step exponential linearWarmup cosine cyclic oneCycle warmupCosine)

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

/-- Check an interpolation's ordered, nonnegative endpoints. -/
def requireRange (lowerName upperName : String) (lower upper : Float) : Except String Unit := do
  requireRate lowerName lower
  requireRate upperName upper
  unless lower <= upper do
    throw s!"scheduler: {lowerName} must not exceed {upperName}"

/-- Reject a duration that would overflow the binary64 interpolation arithmetic. -/
def requireDuration (name : String) (steps : Nat) : Except String Unit := do
  unless (Float.ofNat steps).isFinite do
    throw s!"scheduler: {name} is too large for schedule arithmetic"

/-- Interpolate inside a phase, returning its exact endpoints at the boundaries. Roundoff near an
endpoint must not exceed the range checked by `Config.validate`, even near overflow. -/
def interpolate (annealing : Annealing) (start finish : Float) (elapsed duration : Nat) : Float :=
  if duration = 0 || elapsed >= duration then finish
  else if elapsed = 0 then start
  else
    let progress := Float.ofNat elapsed / Float.ofNat duration
    let value := match annealing with
      | .linear => start + (finish - start) * progress
      | .cosine =>
          let factor := (1.0 + Float.cos (3.141592653589793 * progress)) / 2.0
          finish + (start - finish) * factor
    min (max start finish) (max (min start finish) value)

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
  | .linearWarmup startLr peakLr warmupSteps => do
      Internal.requireRange "starting learning rate" "peak learning rate" startLr peakLr
      Internal.requireDuration "warm-up duration" warmupSteps
  | .cosine baseLr minLr halfCycleSteps _ => do
      Internal.requireRange "minimum learning rate" "base learning rate" minLr baseLr
      Internal.requireDuration "cosine half-cycle" halfCycleSteps
  | .cyclic baseLr peakLr halfCycleSteps cycleDecay => do
      Internal.requireRange "base learning rate" "peak learning rate" baseLr peakLr
      unless halfCycleSteps > 0 do
        throw "scheduler: cyclic half-cycle must be positive"
      Internal.requireDuration "cyclic half-cycle" halfCycleSteps
      Internal.requireDecay "cycle decay" cycleDecay
  | .oneCycle initialLr peakLr finalLr riseSteps totalSteps _ => do
      Internal.requireRange "initial learning rate" "peak learning rate" initialLr peakLr
      Internal.requireRange "final learning rate" "peak learning rate" finalLr peakLr
      unless 0 < riseSteps && riseSteps < totalSteps do
        throw "scheduler: one-cycle rise steps must satisfy 0 < riseSteps < totalSteps"
      Internal.requireDuration "one-cycle duration" totalSteps
  | .warmupCosine peakLr minLr _ totalSteps => do
      Internal.requireRate "peak learning rate" peakLr
      Internal.requireRate "minimum learning rate" minLr
      unless minLr <= peakLr do
        throw "scheduler: minimum learning rate must not exceed peak learning rate"
      Internal.requireDuration "warm-up/cosine duration" totalSteps

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
    | .linearWarmup _ peak _ => peak
    | .cosine base _ _ _ => base
    | .cyclic _ peak _ _ => peak
    | .oneCycle _ peak _ _ _ _ => peak
    | .warmupCosine peak _ _ _ => peak
  Internal.requireRate "learning rate after conversion to the runtime scalar"
    (round maximumRate)

/-- Learning rate after the given number of completed optimizer updates. -/
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
  | .linearWarmup startLr peakLr warmupSteps, stepIndex =>
      Internal.interpolate .linear startLr peakLr stepIndex warmupSteps
  | .cosine baseLr minLr halfCycleSteps periodic, stepIndex =>
      if halfCycleSteps = 0 then minLr
      else if periodic then
        let phase := stepIndex % (2 * halfCycleSteps)
        if phase <= halfCycleSteps then
          Internal.interpolate .cosine baseLr minLr phase halfCycleSteps
        else
          Internal.interpolate .cosine minLr baseLr (phase - halfCycleSteps) halfCycleSteps
      else
        Internal.interpolate .cosine baseLr minLr stepIndex halfCycleSteps
  | .cyclic baseLr peakLr halfCycleSteps cycleDecay, stepIndex =>
      if halfCycleSteps = 0 then baseLr
      else
        let cycle := stepIndex / (2 * halfCycleSteps)
        let scale := Float.pow cycleDecay (Float.ofNat cycle)
        let peak := min peakLr (max baseLr (baseLr + (peakLr - baseLr) * scale))
        let phase := stepIndex % (2 * halfCycleSteps)
        if phase <= halfCycleSteps then
          Internal.interpolate .linear baseLr peak phase halfCycleSteps
        else
          Internal.interpolate .linear peak baseLr (phase - halfCycleSteps) halfCycleSteps
  | .oneCycle initialLr peakLr finalLr riseSteps totalSteps annealing, stepIndex =>
      if stepIndex >= totalSteps then finalLr
      else if stepIndex <= riseSteps then
        Internal.interpolate annealing initialLr peakLr stepIndex riseSteps
      else
        Internal.interpolate annealing peakLr finalLr (stepIndex - riseSteps)
          (totalSteps - riseSteps)
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
            Internal.interpolate .cosine peakLr minLr (stepIndex - warmupSteps) decaySteps

end Scheduler
end Trainer
end TorchLean
