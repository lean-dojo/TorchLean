/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import TorchLeanTest.Floats.NativePrimitives

/-!
Entry point for `scripts/lake.sh exe native_float32_parity`, including CUDA reference-case emission.
-/

/-- Run native binary32 conformance checks or emit reference cases for the CUDA harness. -/
public def main (args : List String) : IO Unit :=
  TorchLean.Test.NativePrimitives.main args
