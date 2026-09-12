/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Runner

/-!
# TorchLean Example Runner Entry Point

The reusable command dispatcher lives in `NN.Examples.Runner`. Keeping the global executable
entry point here lets profiling tools and other Lean programs import that dispatcher without also
importing a competing declaration named `main`.
-/

@[expose] public section

def main (args : List String) : IO UInt32 :=
  TorchLean.CLI.exitOnError (NN.Examples.Runner.main args)
