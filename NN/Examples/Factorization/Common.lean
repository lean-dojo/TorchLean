/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# Factorization Example Checks

Shared IO assertions used by the Cholesky and QR examples. Numerical error reductions use the
public tensor operations directly.
-/

@[expose] public section

namespace NN.Examples.Factorization

open TorchLean

/-- Shared tolerance for reconstruction-error assertions. -/
def errorTolerance : Float := 1e-6

/-- Fail unless `err` is below `tolerance`. -/
def assertBelow (name : String) (error : Float)
    (tolerance : Float := errorTolerance) : IO Unit :=
  if error < tolerance then
    IO.println s!"{name}: OK (error = {error})"
  else
    throw (IO.userError s!"{name}: FAIL (error = {error} ≥ tolerance = {tolerance})")

/-- Fail unless `err` is at least `threshold`. -/
def assertAtLeast (name : String) (error : Float) (threshold : Float := 0.5) : IO Unit :=
  if error ≥ threshold then
    IO.println s!"{name}: OK (correctly rejected, error = {error} ≥ {threshold})"
  else
    throw (IO.userError
      s!"{name}: FAIL (error = {error} < {threshold}; expected the property to fail)")

/-- Fail when a reconstruction unexpectedly succeeds. -/
def assertNotBelow (name : String) (error : Float)
    (tolerance : Float := errorTolerance) : IO Unit :=
  if error < tolerance then
    throw (IO.userError
      s!"{name}: FAIL (unexpectedly below tolerance, error = {error} < {tolerance})")
  else if error.isNaN then
    IO.println s!"{name}: OK (failure detected)"
  else
    IO.println s!"{name}: OK (correctly failed, error = {error})"

end NN.Examples.Factorization
