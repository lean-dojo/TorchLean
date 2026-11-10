/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise

/-!
# Learning-Rate Schedules

Pure learning-rate schedules used by TorchLean training loops and examples. A `Config` describes a
schedule, `Config.validate` checks it and `Config.rate` evaluates it at an optimizer step.

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
A learning-rate schedule, stored as data. `config.rate stepIndex` is the learning rate after
`stepIndex` completed optimizer updates, and every index in a schedule counts updates from zero.

### PyTorch mapping

`Config.step` and `Config.exponential` correspond to the schedule math of:
- `torch.optim.lr_scheduler.StepLR`
- `torch.optim.lr_scheduler.ExponentialLR`

`Config.warmupCosine` is the usual Transformer-pretraining schedule: a short linear warm-up, then
cosine decay to the floor `minLr`. `oneCycle` takes integer phase lengths, so it does not reproduce
PyTorch's fractional `pct_start * total_steps - 1` boundaries.
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
  and hold it. Both phases use `annealing`, and only the learning rate changes. Validation
  requires `0 < riseSteps < totalSteps` and both endpoints at or below the peak. -/
  | oneCycle (initialLr peakLr finalLr : Float) (riseSteps totalSteps : Nat)
      (annealing : Annealing := .cosine)
  /--
  Linearly warm up to `peakLr`, then follow a cosine curve down to `minLr`.

  `warmupSteps` counts optimizer updates. The first update uses `peakLr / warmupSteps`, rounded
  once, a later warm-up update `k` uses `peakLr * ((k + 1) / warmupSteps)`, and the last one
  reaches `peakLr`. Once
  `totalSteps` updates have been scheduled, the learning rate remains at `minLr`.
  A warm-up longer than the run is clamped to `totalSteps`. When `totalSteps = 0`, no update belongs
  to the schedule and `Config.rate` returns `minLr`.
  -/
  | warmupCosine
      (peakLr minLr : Float) (warmupSteps totalSteps : Nat)
  deriving Repr

/-- Require finite, nonnegative interpolation endpoints with `lower <= upper`. -/
def requireRange (lowerName upperName : String) (lower upper : Float) : Except String Unit := do
  Context.requireWithin s!"scheduler: {lowerName}" (some 0) none lower
  Context.requireWithin s!"scheduler: {upperName}" (some 0) none upper
  unless lower <= upper do
    throw s!"scheduler: {lowerName} must not exceed {upperName}"

/-- Reject a duration that would overflow the binary64 interpolation arithmetic. -/
def requireDuration (name : String) (steps : Nat) : Except String Unit := do
  unless (Float.ofNat steps).isFinite do
    throw s!"scheduler: {name} is too large for schedule arithmetic"

/-- Interpolate inside a phase, returning the endpoints exactly at its boundaries. Inner values are
clamped between the endpoints, so roundoff, even near overflow, cannot leave the range that
`Config.validate` checked. -/
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

/--
Validate a learning-rate schedule before it is attached to optimizer state.

The checks keep every scheduled rate finite and nonnegative. Step and exponential schedules need
decay factors in `[0, 1]`, and a step schedule with step size zero is rejected, because
`Config.rate` would silently hold it at the base rate.
-/
def Config.validateParameters : Config → Except String Unit
  | .constant lr =>
      Context.requireWithin "scheduler: learning rate" (some 0) none lr
  | .step baseLr stepSize decayFactor => do
      Context.requireWithin "scheduler: base learning rate" (some 0) none baseLr
      unless stepSize > 0 do
        throw "scheduler: step size must be positive"
      Context.requireWithin "scheduler: decay factor" (some 0) (some 1) decayFactor
  | .exponential baseLr decayFactor => do
      Context.requireWithin "scheduler: base learning rate" (some 0) none baseLr
      Context.requireWithin "scheduler: decay factor" (some 0) (some 1) decayFactor
  | .linearWarmup startLr peakLr warmupSteps => do
      Scheduler.requireRange "starting learning rate" "peak learning rate" startLr peakLr
      Scheduler.requireDuration "warm-up duration" warmupSteps
  | .cosine baseLr minLr halfCycleSteps _ => do
      Scheduler.requireRange "minimum learning rate" "base learning rate" minLr baseLr
      Scheduler.requireDuration "cosine half-cycle" halfCycleSteps
  | .cyclic baseLr peakLr halfCycleSteps cycleDecay => do
      Scheduler.requireRange "base learning rate" "peak learning rate" baseLr peakLr
      unless halfCycleSteps > 0 do
        throw "scheduler: cyclic half-cycle must be positive"
      Scheduler.requireDuration "cyclic half-cycle" halfCycleSteps
      Context.requireWithin "scheduler: cycle decay" (some 0) (some 1) cycleDecay
  | .oneCycle initialLr peakLr finalLr riseSteps totalSteps _ => do
      Scheduler.requireRange "initial learning rate" "peak learning rate" initialLr peakLr
      Scheduler.requireRange "final learning rate" "peak learning rate" finalLr peakLr
      unless 0 < riseSteps && riseSteps < totalSteps do
        throw "scheduler: one-cycle rise steps must satisfy 0 < riseSteps < totalSteps"
      Scheduler.requireDuration "one-cycle duration" totalSteps
  | .warmupCosine peakLr minLr _ totalSteps => do
      Context.requireWithin "scheduler: peak learning rate" (some 0) none peakLr
      Context.requireWithin "scheduler: minimum learning rate" (some 0) none minLr
      unless minLr <= peakLr do
        throw "scheduler: minimum learning rate must not exceed peak learning rate"
      Scheduler.requireDuration "warm-up/cosine duration" totalSteps

/--
Validate a schedule, then check its largest rate after the scalar conversion `round`.

`round` returns the converted rate as a binary64 value. The default `id` leaves rates unchanged,
and a runtime passes its own conversion to catch overflow before training. Schedule arithmetic
stays in binary64, and each rate is converted when the optimizer state is updated.
-/
def Config.validate (config : Config) (round : Float → Float := id) :
    Except String Unit := do
  validateParameters config
  let maximumRate := match config with
    | .constant rate => rate
    | .step rate _ _ => rate
    | .exponential rate _ => rate
    | .linearWarmup _ peak _ => peak
    | .cosine base _ _ _ => base
    | .cyclic _ peak _ _ => peak
    | .oneCycle _ peak _ _ _ _ => peak
    | .warmupCosine peak _ _ _ => peak
  Context.requireWithin "scheduler: learning rate after conversion to the runtime scalar"
    (some 0) none (round maximumRate)

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
      interpolate .linear startLr peakLr stepIndex warmupSteps
  | .cosine baseLr minLr halfCycleSteps periodic, stepIndex =>
      if halfCycleSteps = 0 then minLr
      else if periodic then
        let phase := stepIndex % (2 * halfCycleSteps)
        if phase <= halfCycleSteps then
          interpolate .cosine baseLr minLr phase halfCycleSteps
        else
          interpolate .cosine minLr baseLr (phase - halfCycleSteps) halfCycleSteps
      else
        interpolate .cosine baseLr minLr stepIndex halfCycleSteps
  | .cyclic baseLr peakLr halfCycleSteps cycleDecay, stepIndex =>
      if halfCycleSteps = 0 then baseLr
      else
        let cycle := stepIndex / (2 * halfCycleSteps)
        let scale := Float.pow cycleDecay (Float.ofNat cycle)
        let peak := min peakLr (max baseLr (baseLr + (peakLr - baseLr) * scale))
        let phase := stepIndex % (2 * halfCycleSteps)
        if phase <= halfCycleSteps then
          interpolate .linear baseLr peak phase halfCycleSteps
        else
          interpolate .linear peak baseLr (phase - halfCycleSteps) halfCycleSteps
  | .oneCycle initialLr peakLr finalLr riseSteps totalSteps annealing, stepIndex =>
      if stepIndex >= totalSteps then finalLr
      else if stepIndex <= riseSteps then
        interpolate annealing initialLr peakLr stepIndex riseSteps
      else
        interpolate annealing peakLr finalLr (stepIndex - riseSteps)
          (totalSteps - riseSteps)
  | .warmupCosine peakLr minLr warmupSteps totalSteps, stepIndex =>
      if totalSteps = 0 then
        minLr
      else if stepIndex >= totalSteps then
        minLr
      else
        let warmupSteps := Nat.min warmupSteps totalSteps
        if stepIndex < warmupSteps then
          -- The factor `1 / warmupSteps` would round before the product.
          if stepIndex = 0 then peakLr / Float.ofNat warmupSteps
          else peakLr * (Float.ofNat (stepIndex + 1) / Float.ofNat warmupSteps)
        else
          let decaySteps := totalSteps - warmupSteps
          if decaySteps = 0 then
            minLr
          else
            interpolate .cosine peakLr minLr (stepIndex - warmupSteps) decaySteps

end Scheduler
end Trainer
end TorchLean
