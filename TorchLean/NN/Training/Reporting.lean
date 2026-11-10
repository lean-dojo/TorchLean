/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.NN.Training -- shake: keep

@[expose] public section

/-!
# Training Reports

Training curves, metric histories and logs for `Trainer` runs. The module publicly imports the log
types from `TorchLean.NN.Training` and adds `LossProgress`, `shouldReport` and the `writeLog`
family, which prints where each log went.
-/

namespace TorchLean.Training

/-- Loss measured before and after a training run. -/
structure LossProgress (α : Type) where
  /-- Loss measured before any updates in the run. -/
  before : α
  /-- Loss measured after all requested updates in the run. -/
  after : α
deriving Repr

/-- Whether step `completed` falls on the reporting period `every`, where `0` turns reports off. -/
def shouldReport (every completed : Nat) : Bool :=
  every != 0 && completed % every = 0

/-- Write a training log to `destination`, then print its JSON path, or a line saying the log is
disabled. -/
def writeLog (destination : LogDestination) (log : TrainLog) : IO Unit := do
  Training.LogDestination.writeTrainLog destination log
  match destination.path? with
  | some path => IO.println s!"  wrote TrainLog JSON: {path}"
  | none => IO.println "  TrainLog JSON disabled"

/-- Write a before/after loss log through `writeLog`. -/
def writeLossComparison (destination : LogDestination) (title : String) (steps : Nat)
    (lossBefore lossAfter : Float) (notes : Array String := #[]) : IO Unit :=
  writeLog destination (Training.TrainLog.lossComparison
    title steps lossBefore lossAfter notes)

/-- Write a scalar curve as a training log through `writeLog`. -/
def Curve.writeLog (curve : Curve) (destination : LogDestination) (title : String)
    (seriesName : String := "loss") (notes : Array String := #[])
    (color : String := "#4e79a7") : IO Unit :=
  TorchLean.Training.writeLog destination
    (Training.Curve.toTrainLog curve title seriesName color notes)

end TorchLean.Training

