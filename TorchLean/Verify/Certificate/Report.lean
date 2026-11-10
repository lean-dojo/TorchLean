/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Report

The value a certificate checker returns: the lines that report the check and the reason for a
rejection. The checkers do not print. A command-line entry prints the lines and fails with the
reason.

- `Report`: the lines and the rejection reason, `none` when the checker accepted the artifact.
- `Report.ok`: whether the checker accepted the artifact (`Report.ok_iff`).
-/

@[expose] public section

namespace TorchLean.Certificate

/-- What a certificate check found: the lines that report it, in the order the checker produced
them, and the reason for rejecting the artifact, `none` when the checker accepted it. -/
structure Report where
  /-- The lines that report the check, one entry per finding. -/
  lines : Array String := #[]
  /-- The reason for rejecting the artifact, `none` when the checker accepted it. -/
  rejection : Option String := none

namespace Report

/-- Whether the checker accepted the artifact. -/
def ok (report : Report) : Bool :=
  report.rejection.isNone

/-- A report accepts exactly when it gives no reason for a rejection. -/
theorem ok_iff (report : Report) : report.ok = true ↔ report.rejection = none := by
  simp [ok]

end Report

end TorchLean.Certificate
