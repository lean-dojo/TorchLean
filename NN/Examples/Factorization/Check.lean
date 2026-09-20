/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.Examples.Factorization.Cholesky
public import NN.Examples.Factorization.QR

/-!
# Compiled Factorization Checks

The packed tensor runtime uses native externs, so executable examples run through a compiled
`lean_exe` rather than the Lean interpreter.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Factorization

/-- Run the Cholesky and QR positive checks and their negative controls. -/
def checkAll : IO Unit := do
  -- An invalid factor must not pass because IEEE maximum discards its NaN entries.
  let invalid : Tensor Float [1] := [0.0 / 0.0]
  let zero : Tensor Float [1] := [0.0]
  assertNotBelow "NaN reconstruction error is rejected" (Tensor.maxAbsDiff invalid zero)
  Cholesky.check
  QR.check

/-- Command-line help for the compiled factorization checks. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean factorization checks"
    , ""
    , "Usage:"
    , "  lake exe torchlean factorizations"
    , ""
    , "Checks A = L Lᵀ for Cholesky and A = Q R, Qᵀ Q = I for reduced QR."
    , "Also checks that indefinite and rank-deficient inputs fail the relevant property."
    , "Every line should say OK; these are Float tests with tolerance 1e-6, not proofs."
    ]

/-- Entry point: run every factorization check, positive cases and negative controls alike. -/
def main (args : List String) : IO Unit := do
  let args := TorchLean.CLI.dropDashDash args
  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return
  TorchLean.CLI.requireNoArgs "factorizations" args
  checkAll

end NN.Examples.Factorization
