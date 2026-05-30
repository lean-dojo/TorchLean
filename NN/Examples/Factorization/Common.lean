/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations

/-!
# Factorization examples — shared helpers

Small `Float`-valued helpers used by the matrix-factorization examples
(`Cholesky`, `QR`, `SymEig`, `SVD`). These examples are *executable sanity checks*: each one
reconstructs the original matrix from its factors and asserts (via `#guard`) that the maximum
entrywise reconstruction error is below a tolerance, so the build fails if a factorization is wrong.

These run over `Float` (the executable 64-bit runtime scalar), which is the precision the
factorizations target for Gaussian-process / kernel-method use.
-/

@[expose] public section


namespace NN.Examples.Factorization

/-- Build an `m × n` `Float` matrix tensor from a row-major nested list. Missing entries are `0`. -/
def mkMat {m n : Nat} (rows : List (List Float)) : Spec.Tensor Float (.dim m (.dim n .scalar)) :=
  Spec.ofMatFn (fun i j => (rows.getD i.val []).getD j.val 0.0)

/-- Maximum entrywise absolute difference between two `m × n` matrices. -/
def maxMatErr {m n : Nat} (A B : Spec.Tensor Float (.dim m (.dim n .scalar))) : Float :=
  (List.finRange m).foldl (fun acc i =>
    (List.finRange n).foldl
      (fun a j => max a (Float.abs (Spec.get2 A i j - Spec.get2 B i j))) acc) 0.0

/-- Matrix product `A · B` (thin wrapper over `matMulSpec`). -/
def mm {m n p : Nat} (A : Spec.Tensor Float (.dim m (.dim n .scalar)))
    (B : Spec.Tensor Float (.dim n (.dim p .scalar))) : Spec.Tensor Float (.dim m (.dim p .scalar)) :=
  Spec.matMulSpec A B

/-- Matrix transpose. -/
def tr {m n : Nat} (A : Spec.Tensor Float (.dim m (.dim n .scalar))) :
    Spec.Tensor Float (.dim n (.dim m .scalar)) :=
  Spec.Tensor.matrixTransposeSpec A

/-- Turn a length-`n` vector into an `n × n` diagonal matrix. -/
def diagFromVec {n : Nat} (v : Spec.Tensor Float (.dim n .scalar)) :
    Spec.Tensor Float (.dim n (.dim n .scalar)) :=
  Spec.ofMatFn (fun i j => if i.val == j.val then Spec.Tensor.toScalar (Spec.get v i) else 0.0)

/-- Read a vector tensor back out as a `List Float` (for display). -/
def vecToList {n : Nat} (v : Spec.Tensor Float (.dim n .scalar)) : List Float :=
  (List.finRange n).map (fun i => Spec.Tensor.toScalar (Spec.get v i))

/-- Shared tolerance for reconstruction-error assertions. -/
def tol : Float := 1e-6

/--
Compiled assertion used by the examples: print `name: OK (err)` when `err < tol`, otherwise raise an
`IO` error so the build/`#eval` fails. Running this through `#eval` evaluates with the compiler
(fast), unlike `#guard`, which forces slow kernel reduction of the whole factorization.
-/
def assertLt (name : String) (err : Float) (tolerance : Float := tol) : IO Unit :=
  if err < tolerance then
    IO.println s!"{name}: OK (err = {err})"
  else
    throw (IO.userError s!"{name}: FAIL (err = {err} ≥ tol = {tolerance})")

end NN.Examples.Factorization
