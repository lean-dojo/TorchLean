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

The default trainer API reads binary32 losses back to `Float` exactly. Typed sessions retain their
selected scalar in the structured loss values.
-/
structure Report (α : Type := Float) where
  /-- Number of optimizer updates applied by the session. -/
  steps : Nat
  /-- Scalar-valued loss measured before and after training. -/
  loss : Training.LossProgress α
  /-- Runtime selector used by `open`; an explicit scalar in `openTyped` takes precedence. -/
  arithmetic : Runtime.Arithmetic
  /-- Exact encoding descriptor when a typed session explicitly selects its scalar. -/
  scalarFormat? : Option String := none
deriving Repr

namespace Report

/-- Name of the runtime scalar type that produced this report. -/
def runtimeScalar {α : Type} (report : Report α) : String :=
  match report.scalarFormat? with
  | some format => format
  | none =>
      match report.arithmetic with
      | .native => "Float32"
      | .ieee => "ExecFloat.Binary 8 23"
      | .complex => "Complex (ExecFloat.Binary 8 23)"

/-- One-line summary suitable for quickstarts and scripts. -/
def summary {α : Type} [ToString α] (report : Report α) : String :=
  let arithmetic :=
    if report.scalarFormat?.isSome then "" else s!"arithmetic={report.arithmetic} "
  s!"steps={report.steps} {arithmetic}scalar={report.runtimeScalar} " ++
    s!"loss={report.loss.before} -> {report.loss.after}"

/-- Print the one-line training summary. -/
def printSummary {α : Type} [ToString α] (report : Report α) : IO Unit :=
  IO.println (summary report)

instance {α : Type} [ToString α] : ToString (Report α) where
  toString := summary

/-- Convert a `Float` report into the standard two-point training log. -/
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
