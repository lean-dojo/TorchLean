/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Session

/-!
# Learning-rate schedules

A schedule reaches the optimizer through `Optimizer.setLr` before each update. Each
check compares the final loss of a scheduled run exactly with an unscheduled one: a zero rate
freezes the parameters, a constant rate equal to the configured one changes nothing, and a step
schedule that drops to zero after the first update matches a single unscheduled update. SGD and
Adam are both checked on CUDA, where the rate reaches their native updates.
-/

@[expose] public section

namespace TorchLean.Test.Scheduler

open Trainer

/-- Loss on one sample after `steps` updates under an optional schedule. -/
def lossAfter (device : Backend.Device) (optimizer : optim.Optimizer) (steps : Nat)
    (scheduler : Option Trainer.Scheduler.Config) : IO Float := do
  let trainer := Trainer.create (nn.linear 1 1) { device, optimizer, seed := 37 }
  let session ← trainer.open scheduler
  let sample : Sample.Supervised Float [1] [1] := { input := [1.0], target := [4.0] }
  for _ in [0:steps] do
    session.step sample
  session.loss sample

/-- The three schedule checks for one optimizer whose configured learning rate is `0.1`. -/
def check (device : Backend.Device) (name : String) (optimizer : optim.Optimizer) :
    IO Unit := do
  let fail (message : String) : IO Unit :=
    throw <| IO.userError s!"scheduler ({device.cliName}, {name}): {message}"
  let lossAfter := lossAfter device optimizer
  let initial ← lossAfter 0 none
  let frozen ← lossAfter 5 (some (Trainer.Scheduler.Config.constant 0.0))
  unless frozen == initial do fail s!"a zero rate moved the loss from {initial} to {frozen}"
  let plain ← lossAfter 5 none
  unless plain != initial do fail "five unscheduled updates left the loss unchanged"
  let constantRate ← lossAfter 5 (some (Trainer.Scheduler.Config.constant 0.1))
  unless constantRate == plain do
    fail s!"a constant schedule at the configured rate gave {constantRate}, not {plain}"
  let oneStep ← lossAfter 1 none
  let dropped ← lossAfter 5 (some (Trainer.Scheduler.Config.step 0.1 1 0.0))
  unless dropped == oneStep do
    fail s!"a rate that drops to zero after one update gave {dropped}, not {oneStep}"

/-- Run `check` for SGD and Adam with learning rate `0.1` on `device`. -/
def run (device : Backend.Device) : IO Unit := do
  check device "sgd" (optim.sgd { lr := 0.1 })
  check device "adam" (optim.adam { lr := 0.1 })
  IO.println s!"learning-rate schedules ({device.cliName}): optimizer updates: ok"

end TorchLean.Test.Scheduler
