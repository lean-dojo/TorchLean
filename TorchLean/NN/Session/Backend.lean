/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Report
public import TorchLean.NN.Session

/-!
# Eager Backend Dispatch

Select the accepted backend capsule of each registered operator a session records.
`Session.apply` runs the selected capsule only with an implementation of that exact operator,
provider and device. Reports describe selections made before execution, so a reported capsule may
never have run.
-/

public section

namespace TorchLean

namespace Session

/--
Select the accepted capsule for one registered operator, attributes included.

Every operator is selected on its own: a capsule names its operator's wire record, so a capsule
accepted for one operator is never reused for another.
`Session.selectedCapsule_sound` states that a returned capsule
is the capsule of the kernel `planOp` accepts, with the operator's wire record and the session's
device. The first acceptance of each capsule name is kept for `backendSelections` and reported when
`showBackend` is set.
-/
def selectedCapsule {α : Type} [Storage α]
    (s : Session α) {ins outs : List Port} (op : Op ins outs) :
    IO Backend.KernelCapsule := do
  let wire := Op.encode ⟨ins, outs, op⟩
  let planned ←
    match s.options.planOp op s.gradEnabled with
    | .ok planned => pure planned
    | .error msg =>
        throw <| IO.userError s!"torch: kernel selection failed for `{wire.tag}`: {msg}"
  let capsule := planned.capsule
  match capsule.validateEagerRequest wire s.options.device with
  | .ok () => pure ()
  | .error msg =>
    throw <| IO.userError s!"torch: invalid backend selection during eager execution: {msg}"
  let selected ← s.selectedBackends.get
  unless selected.any (fun accepted => accepted.capsule.name == capsule.name) do
    s.selectedBackends.set (selected.push planned)
    if s.options.showBackend then
      let kernel : Backend.PlannedKernel := { op := planned.op, capsule := planned.capsule }
      for line in kernel.detailedReportLines do
        IO.println line
  pure capsule

/-- A representative report of this eager session's selections: the first acceptance of each
capsule name. A later selection under the same name may be for another operator, and execution
never reads this array. -/
def backendSelections {α : Type} [Storage α]
    (s : Session α) : IO (Array Backend.AcceptedKernel) :=
  s.selectedBackends.get

end Session

end TorchLean
