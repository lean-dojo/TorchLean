/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Constructor
public import NN.API.Trainer.Session

/-!
# Learning-rate schedules

A schedule reaches the optimizer through `Optimizer.setLr` before each update. Each
check compares the final loss of a scheduled run exactly with an unscheduled one: a zero rate
freezes the parameters, a constant rate equal to the configured one changes nothing, and a step
schedule that drops to zero after the first update matches a single unscheduled update. SGD and
Adam are both checked, which on CUDA covers their native updates.

The pure schedule checks cover warm-up endpoints, cosine termination and repetition, cyclic
amplitude decay, both one-cycle interpolations, and invalid configurations.
-/

@[expose] public section

namespace NN.Tests.Runtime.Scheduler

open TorchLean Trainer.Scheduler

/-- Check a schedule's rates at successive update indices against known phase boundaries. -/
def checkRates (name : String) (schedule : Trainer.Scheduler.Config)
    (expected : Array Float) : IO Unit := do
  IO.ofExcept schedule.validate
  for i in [0:expected.size] do
    let actual := schedule.rate i
    let target := expected[i]!
    unless actual.isFinite && Float.abs (actual - target) <= 1e-12 do
      throw <| IO.userError s!"scheduler ({name}): step {i}: expected {target}, got {actual}"

/-- Phase boundaries and validation for the schedules that do not have a constant decay factor. -/
def checkPhases : IO Unit := do
  checkRates "linear warm-up" (linearWarmup 0.0 1.0 4) #[0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
  checkRates "no warm-up" (linearWarmup 0.0 1.0 0) #[1.0, 1.0]
  checkRates "cosine" (cosine 1.0 0.0 2) #[1.0, 0.5, 0.0, 0.0, 0.0]
  checkRates "periodic cosine" (cosine 1.0 0.0 2 true)
    #[1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
  checkRates "zero cosine duration" (cosine 1.0 0.25 0) #[0.25, 0.25]
  checkRates "cyclic" (cyclic 0.0 1.0 2) #[0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0]
  checkRates "halving cycles" (cyclic 0.25 1.25 2 0.5)
    #[0.25, 0.75, 1.25, 0.75, 0.25, 0.5, 0.75, 0.5, 0.25]
  checkRates "linear one-cycle" (oneCycle 0.0 1.0 0.0 2 6 Annealing.linear)
    #[0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
  checkRates "cosine one-cycle" (oneCycle 0.0 1.0 0.0 4 8)
    #[0.0, 0.1464466094067262, 0.5, 0.8535533905932737, 1.0,
      0.8535533905932737, 0.5, 0.1464466094067262, 0.0, 0.0]
  let invalid : Array Config :=
    #[linearWarmup 1.0 0.0 4, cosine 0.0 1.0 4, cyclic 0.0 1.0 0,
      cyclic 0.0 1.0 2 1.5, oneCycle 0.0 1.0 0.0 0 4,
      oneCycle 0.0 1.0 0.0 4 4, oneCycle 2.0 1.0 0.0 1 4,
      oneCycle 0.0 1.0 2.0 1 4, cosine 1.0 (-1.0) 4,
      linearWarmup 0.0 1.0 (2 ^ 1100), warmupCosine 1.0 0.0 0 (2 ^ 1100)]
  for schedule in invalid do
    if schedule.validate.isOk then
      throw <| IO.userError s!"scheduler: accepted invalid configuration {repr schedule}"
  let converted := (oneCycle 0.0 1.0 0.0 1 4).validate
    (round := fun _ => Float.ofBits 0x7ff0000000000000)
  if converted.isOk then
    throw <| IO.userError "scheduler: accepted an overflowing runtime learning rate"
  -- Reconstructing an endpoint as base + (peak - base) can round above the validated peak.
  -- Check both binary64 overflow and a finite binary64 rate that would overflow binary32.
  for (base, peak) in #[(Float.ofBits 0x7fd60d63029fd2b7,
                         Float.ofBits 0x7fefffffffffffff),
                        (Float.ofBits 0x47d115ed452354f7,
                         Float.ofBits 0x47efffffefffffff)] do
    let duration := 2 ^ 60
    let cases := #[(cyclic base peak 1, 1),
                   (linearWarmup base peak duration, duration - 1),
                   (cosine peak base duration, 1),
                   (warmupCosine peak base 0 duration, 1)]
    for (schedule, index) in cases do
      IO.ofExcept schedule.validate
      let rate := schedule.rate index
      unless rate.isFinite && base <= rate && rate <= peak do
        throw <| IO.userError s!"scheduler: rate escaped validated range: {rate}"
      if peak.toFloat32.toFloat.isFinite then
        IO.ofExcept <| schedule.validate (round := fun x => x.toFloat32.toFloat)
        unless rate.toFloat32.toFloat.isFinite do
          throw <| IO.userError "scheduler: validated rate overflowed the runtime scalar"

/-- Loss on one sample after `steps` updates under an optional schedule. -/
def lossAfter (device : NN.Backend.Device) (optimizer : optim.Optimizer) (steps : Nat)
    (scheduler : Option Trainer.Scheduler.Config) : IO Float := do
  let trainer := TorchLean.Trainer.new (nn.linear 1 1) { device, optimizer, seed := 37 }
  let session ← trainer.open scheduler
  let sample : Sample.Supervised Float [1] [1] := { input := [1.0], target := [4.0] }
  for _ in [0:steps] do
    session.step sample
  session.loss sample

/-- The three schedule checks for one optimizer whose configured learning rate is `0.1`. -/
def check (device : NN.Backend.Device) (name : String) (optimizer : optim.Optimizer) :
    IO Unit := do
  let fail (message : String) : IO Unit :=
    throw <| IO.userError s!"scheduler ({device.cliName}, {name}): {message}"
  let lossAfter := lossAfter device optimizer
  let initial ← lossAfter 0 none
  let frozen ← lossAfter 5 (some (Trainer.Scheduler.constant 0.0))
  unless frozen == initial do fail s!"a zero rate moved the loss from {initial} to {frozen}"
  let plain ← lossAfter 5 none
  unless plain != initial do fail "five unscheduled updates left the loss unchanged"
  let constantRate ← lossAfter 5 (some (Trainer.Scheduler.constant 0.1))
  unless constantRate == plain do
    fail s!"a constant schedule at the configured rate gave {constantRate}, not {plain}"
  let oneStep ← lossAfter 1 none
  let dropped ← lossAfter 5 (some (Trainer.Scheduler.step 0.1 1 0.0))
  unless dropped == oneStep do
    fail s!"a rate that drops to zero after one update gave {dropped}, not {oneStep}"

def run (device : NN.Backend.Device := .cpu) : IO Unit := do
  checkPhases
  check device "sgd" (optim.sgd { lr := 0.1 })
  check device "adam" (optim.adam { lr := 0.1 })
  IO.println s!"learning-rate schedules ({device.cliName}): phases and optimizer updates: ok"

end NN.Tests.Runtime.Scheduler
