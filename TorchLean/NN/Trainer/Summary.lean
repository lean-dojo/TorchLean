/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Arithmetic
public import TorchLean.NN.Training.Reporting

/-!
# Training Reports

`Trainer.Report` records the step count, the loss before and after training and the runtime scalar
of a finished run. `Report.summary` formats it as one line, which `Report.printSummary` prints.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
Backend-independent training report.

The default trainer API reads binary32 losses back to `Float` exactly, while typed sessions keep
their own scalar in the loss values.
-/
structure Report (α : Type := Float) where
  /-- Number of optimizer updates applied by the session. -/
  steps : Nat
  /-- Scalar-valued loss measured before and after training. -/
  loss : Training.LossProgress α
  /-- Runtime selector used by `open`. For an `openTyped` session, `scalarFormat?` names the scalar
  and takes precedence. -/
  arithmetic : Arithmetic
  /-- Encoding descriptor of a typed session's scalar, `none` for a session from `open`. -/
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

/-- One-line summary `steps=... arithmetic=... scalar=... loss=before -> after`, which leaves out
the arithmetic field for a typed session. -/
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

end Report

end Trainer

end TorchLean
