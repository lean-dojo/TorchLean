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

A schedule reaches the optimizer through `Optimizer.setLearningRate` before each update. Each
check compares the final loss of a scheduled run exactly with an unscheduled one: a zero rate
freezes the parameters, a constant rate equal to the configured one changes nothing, and a step
schedule that drops to zero after the first update matches a single unscheduled update. SGD and
Adam are both checked, which on CUDA covers their native updates.
-/

@[expose] public section

namespace NN.Tests.Runtime.Scheduler

open TorchLean

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
  check device "sgd" (optim.sgd { learningRate := 0.1 })
  check device "adam" (optim.adam { learningRate := 0.1 })
  IO.println s!"learning-rate schedules ({device.cliName}): zero, constant and step: ok"

end NN.Tests.Runtime.Scheduler
