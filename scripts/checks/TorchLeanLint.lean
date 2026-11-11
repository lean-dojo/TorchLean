/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# TorchLeanLint

TorchLean's lint driver, wired into Lake as `lintDriver := "torchlean_lint"` and run with
`scripts/lake.sh lint` locally. CI can run the same command. Instead of Mathlib's `#lint` linters
it runs the four checks listed in `checks`, and `main` stops at the first check that fails.

Its first check is repository policy: forbidden constructs (unverified proof stubs, unexpected
axioms, native decision procedures, etc.), whitespace hygiene, line length and consistent file
headers. Every finding of `scripts/checks/repo_lint.py` is an error, so any finding fails the lint.
The second check, the import graph's layering gate in `scripts/checks/dependency_audit.py`, fails
the lint only on errors. The third, `scripts/checks/libtorch_abi.py`, checks the ATen operators and
torch C functions that the sources of the LibTorch bridge call against the lists of
`docs/TRUST_BOUNDARIES.md`. The fourth check, the library check `scripts/checks/Library.lean`, runs
the axiom check on every theorem of the libraries and the coverage check on the `#roundoff_coverage`
registrations. It imports the libraries, so the `needs` of `torchlean_lint` in `lakefile.lean` makes
Lake build them before it runs the driver.
-/

@[expose] public section

/-- The checks `scripts/lake.sh lint` runs, in order: the repository policy, the import graph's
layering gate, the LibTorch bridge's ABI lists, then the library check
`scripts/checks/Library.lean`. -/
def checks : List (String × Array String) := [
  ("python3", #["scripts/checks/repo_lint.py"]),
  ("python3", #["-B", "scripts/checks/dependency_audit.py", "--fail-on-error"]),
  ("python3", #["-B", "scripts/checks/libtorch_abi.py"]),
  ("lean", #["scripts/checks/Library.lean"])
]

/-- Entry point of `scripts/lake.sh lint`: runs each check in order, forwards its output and throws
at the first check that exits with a nonzero status. -/
def main (_args : List String) : IO Unit := do
  for (cmd, args) in checks do
    let out ← IO.Process.output { cmd := cmd, args := args }
    if !out.stdout.isEmpty then
      IO.print out.stdout
    if !out.stderr.isEmpty then
      IO.eprint out.stderr
    if out.exitCode != 0 then
      throw <| IO.userError
        s!"command failed ({out.exitCode}): {cmd} {String.intercalate " " args.toList}"
