/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Arithmetic
public import NN.API.Trainer.Reporting

/-!
# Training Reports

Small report types returned by the high-level trainer API.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
Backend-independent training report.

Executable scalar backends are read back once at the result boundary, so callers receive ordinary
host `Float` losses without parsing rendered values.
-/
structure Report where
  /-- Number of optimizer steps requested by the configuration. -/
  steps : Nat
  /-- Host-readable loss measured before and after training. -/
  loss : Training.LossProgress Float
  /-- Arithmetic the run executed under; it fixes the binary32 scalar named by `runtimeScalar`. -/
  arithmetic : Runtime.Arithmetic
deriving Repr

namespace Report

/-- Name of the runtime scalar type that produced this report. -/
def runtimeScalar (report : Report) : String :=
  match report.arithmetic with
  | .native => "Float32"
  | .ieee => "ExecFloat.Binary 8 23"
  | .complex => "Complex (ExecFloat.Binary 8 23)"

/-- One-line summary suitable for quickstarts and scripts. -/
def summary (report : Report) : String :=
  s!"steps={report.steps} arithmetic={report.arithmetic} scalar={report.runtimeScalar} " ++
    s!"loss={report.loss.before} -> {report.loss.after}"

/-- Print the one-line training summary. -/
def printSummary (report : Report) : IO Unit :=
  IO.println (summary report)

instance : ToString Report where
  toString := summary

/-- Convert the report into the standard two-point training log. -/
def toTrainLog (title : String) (notes : Array String) (report : Report) :
    Training.TrainLog :=
  Runtime.Training.TrainLog.lossComparison
    title report.steps report.loss.before report.loss.after notes

/-- Write this report to a log destination when logging is enabled. -/
def writeLog (destination : Training.LogDestination) (title : String) (notes : Array String)
    (report : Report) : IO Unit := do
  if destination.isEnabled then
    Training.writeLog destination (report.toTrainLog title notes)

end Report

end Trainer

end TorchLean
