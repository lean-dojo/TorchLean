/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.Runtime.Training.Log -- shake: keep

@[expose] public section

/-!
# Training Reports

Training curves, metric histories, and logs produced by `Trainer` runs.
-/

namespace TorchLean.Training

export Runtime.Training
  (Curve TrainLog ExperimentLog LogDestination MetricHistory)

namespace MetricHistory

export Runtime.Training.MetricHistory (empty)

end MetricHistory

/-- Loss measured before and after a training run. -/
structure LossProgress (α : Type) where
  /-- Loss measured before any updates in the run. -/
  before : α
  /-- Loss measured after all requested updates in the run. -/
  after : α
deriving Repr

/-- Return whether a completed step should emit a periodic report. -/
def shouldReport (every completed : Nat) : Bool :=
  every != 0 && completed % every = 0

/-- Write a training log to an enabled or disabled destination. -/
def writeLog (destination : LogDestination) (log : TrainLog) : IO Unit := do
  Runtime.Training.LogDestination.writeTrainLog destination log
  match destination.path? with
  | some path => IO.println s!"  wrote TrainLog JSON: {path}"
  | none => IO.println "  TrainLog JSON disabled"

/-- Write a before/after loss log to an enabled or disabled destination. -/
def writeLossComparison (destination : LogDestination) (title : String) (steps : Nat)
    (lossBefore lossAfter : Float) (notes : Array String := #[]) : IO Unit :=
  writeLog destination (Runtime.Training.TrainLog.lossComparison
    title steps lossBefore lossAfter notes)

/-- First and last values of a nonempty scalar training curve. -/
structure CurveEndpoints where
  /-- Step associated with the last value. -/
  lastStep : Nat
  /-- First recorded value. -/
  firstValue : Float
  /-- Last recorded value. -/
  lastValue : Float
deriving Repr

/-- Return the endpoints of a scalar curve, or `none` when it contains no values. -/
def Curve.endpoints? (curve : Curve) : Option CurveEndpoints := do
  let firstValue ← curve.values[0]?
  let lastValue ← curve.values.back?
  let lastStep := curve.steps.back?.getD (curve.values.size - 1)
  pure { lastStep, firstValue, lastValue }

/-- Return the endpoints of a scalar curve or raise a contextual error when it is empty. -/
def Curve.endpoints (curve : Curve) (context : String := "training curve") : IO CurveEndpoints :=
  match TorchLean.Training.Curve.endpoints? curve with
  | some endpoints => pure endpoints
  | none => throw <| IO.userError s!"{context}: empty training curve"

/-- Print loss before and after the updates represented by a scalar training curve. -/
def Curve.printLossSummary (curve : Curve) (steps : Nat) : IO Unit := do
  let endpoints ← TorchLean.Training.Curve.endpoints curve "Curve.printLossSummary"
  let lastStep := if curve.steps.isEmpty then steps else endpoints.lastStep
  IO.println
    s!"  steps={lastStep} loss={endpoints.firstValue} -> {endpoints.lastValue}"

/-- Write a scalar curve to an enabled or disabled training-log destination. -/
def Curve.writeLog (curve : Curve) (destination : LogDestination) (title : String)
    (seriesName : String := "loss") (notes : Array String := #[])
    (color : String := "#4e79a7") : IO Unit :=
  TorchLean.Training.writeLog destination
    (Runtime.Training.Curve.toTrainLog curve title seriesName color notes)

/-- Write named metric series to an enabled or disabled training-log destination. -/
def MetricHistory.writeLog (history : MetricHistory) (destination : LogDestination)
    (title : String) (notes : Array String := #[]) : IO Unit :=
  TorchLean.Training.writeLog destination (history.toTrainLog (title := title) (notes := notes))

end TorchLean.Training
