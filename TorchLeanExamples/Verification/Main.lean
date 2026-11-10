/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.CLI

/-!
# Verification executable

The reusable registry and dispatcher live in `TorchLeanExamples.Verification.CLI`, and this module
holds only the global `main` that `lake exe verify` needs. Keeping `main` here means that importing
the verification API cannot collide with another executable's entry point.
-/

@[expose] public section

/-- `lake exe verify` entry point. A tool that throws exits with `error: …` and status 1. -/
def main (args : List String) : IO UInt32 :=
  TorchLean.CLI.exitOnError do
    TorchLean.Examples.Verify.dispatch args
    pure 0
