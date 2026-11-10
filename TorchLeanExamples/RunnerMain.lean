/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Runner

/-!
# TorchLean Example Runner Entry Point

The reusable command dispatcher lives in `TorchLeanExamples.Runner`. Keeping the global executable
entry point here lets profiling tools and other Lean programs import that dispatcher without also
importing a competing declaration named `main`.
-/

@[expose] public section

/-- Run the selected example. An uncaught error is printed as `error: ...` on stderr and gives
exit status 1. -/
def main (args : List String) : IO UInt32 :=
  TorchLean.CLI.exitOnError (TorchLean.Examples.Runner.main args)
