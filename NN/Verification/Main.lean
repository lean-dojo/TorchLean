/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.CLI

/-!
# Verification executable

The reusable registry and dispatcher live in `NN.Verification.CLI`. This module contains only the
global `main` wrapper required by `lake exe verify`, so importing the verification API cannot
collide with another executable entry point.
-/

@[expose] public section

/-- `lake exe verify` entry point. A tool that throws exits with `error: …` and status 1. -/
def main (args : List String) : IO UInt32 :=
  TorchLean.CLI.exitOnError do
    NN.Verification.CLI.dispatch args
    pure 0

/-- C-exported wrapper used by native executable startup. -/
@[export lean_main]
def exportedMain (args : List String) : IO UInt32 :=
  main args
