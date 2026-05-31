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

/-- Extract the diagonal of a square matrix as a length-`n` vector. -/
def diagOf {n : Nat} (M : Spec.Tensor Float (.dim n (.dim n .scalar))) :
    Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => Spec.get2 M i i)

/-- Read a vector tensor back out as a `List Float` (for display). -/
def vecToList {n : Nat} (v : Spec.Tensor Float (.dim n .scalar)) : List Float :=
  (List.finRange n).map (fun i => Spec.Tensor.toScalar (Spec.get v i))

/-- Squared Frobenius distance `Σ_{i,j} (A_ij - B_ij)²` between two `m × n` matrices. -/
def frobSqErr {m n : Nat} (A B : Spec.Tensor Float (.dim m (.dim n .scalar))) : Float :=
  (List.finRange m).foldl (fun acc i =>
    (List.finRange n).foldl
      (fun a j => let d := Spec.get2 A i j - Spec.get2 B i j; a + d * d) acc) 0.0

/-- Squared Frobenius off-diagonal mass `Σ_{i≠j} M_ij²` of a square matrix. -/
def offDiagFrobSq {n : Nat} (M : Spec.Tensor Float (.dim n (.dim n .scalar))) : Float :=
  (List.finRange n).foldl (fun acc i =>
    (List.finRange n).foldl
      (fun a j => if i.val == j.val then a else let x := Spec.get2 M i j; a + x * x) acc) 0.0

/-- Total squared Frobenius mass `Σ_{i,j} M_ij²` of a square matrix (off-diagonal + diagonal mass). -/
def totalFrobSq {n : Nat} (M : Spec.Tensor Float (.dim n (.dim n .scalar))) : Float :=
  (List.finRange n).foldl (fun acc i =>
    (List.finRange n).foldl
      (fun a j => let x := Spec.get2 M i j; a + x * x) acc) 0.0

/-- View a square `Float` matrix tensor as a strict array matrix (the representation the Jacobi
iteration runs over). -/
def arrOfMat {n : Nat} (A : Spec.Tensor Float (.dim n (.dim n .scalar))) : Array (Array Float) :=
  Spec.matToArr (Spec.toMatFn A)

/-- Read a strict array matrix back as a square `Float` matrix tensor. -/
def matOfArr {n : Nat} (M : Array (Array Float)) : Spec.Tensor Float (.dim n (.dim n .scalar)) :=
  Spec.ofMatFn (fun i j => Spec.arrGet M i.val j.val)

/-- Apply the **annihilating** Jacobi rotation at pivot `(p, q)`: returns `A' = Jᵀ A J` for the
Givens rotation whose angle zeroes `A'[p,q]` (the rotation the solver actually performs). -/
def jacobiRotateAt {n : Nat} (A : Spec.Tensor Float (.dim n (.dim n .scalar))) (p q : Nat) :
    Spec.Tensor Float (.dim n (.dim n .scalar)) :=
  matOfArr (Spec.arrJacobiRotate n (arrOfMat A) (Spec.arrId n) p q).1

/-- Apply an **arbitrary** Givens conjugation `A' = Jᵀ A J` with caller-chosen `(c, s)` at `(p, q)`
(not necessarily the annihilating angle, nor even orthogonal). Used for negative controls. -/
def givensConjAt {n : Nat} (A : Spec.Tensor Float (.dim n (.dim n .scalar))) (p q : Nat)
    (c s : Float) : Spec.Tensor Float (.dim n (.dim n .scalar)) :=
  let J := Spec.arrGivens n p q c s
  matOfArr (Spec.arrMatMul n (Spec.arrTr n J) (Spec.arrMatMul n (arrOfMat A) J))

/-- Shared tolerance for reconstruction-error assertions. -/
def tol : Float := 1e-6

/--
Compiled **positive** assertion: print `name: OK (err)` when `err < tol`, otherwise raise an
`IO` error so the build/`#eval` fails. Running this through `#eval` evaluates with the compiler
(fast), unlike `#guard`, which forces slow kernel reduction of the whole factorization.
-/
def assertLt (name : String) (err : Float) (tolerance : Float := tol) : IO Unit :=
  if err < tolerance then
    IO.println s!"{name}: OK (err = {err})"
  else
    throw (IO.userError s!"{name}: FAIL (err = {err} ≥ tol = {tolerance})")

/--
Compiled **negative-control** assertion: succeeds only when `err ≥ threshold`, i.e. when a property
that *should not* hold is correctly detected as violated. Gives the metric teeth — a reviewer can see
the same `maxMatErr`/residual that reports `0` on a valid factorization reports a large value on an
invalid one, so the positive checks are not vacuous.
-/
def assertGe (name : String) (err : Float) (threshold : Float := 0.5) : IO Unit :=
  if err ≥ threshold then
    IO.println s!"{name}: OK (correctly rejected, err = {err} ≥ {threshold})"
  else
    throw (IO.userError s!"{name}: FAIL (err = {err} < {threshold}; expected the property to fail)")

/--
Compiled **negative-control** assertion that a reconstruction *fails*: succeeds when the error is not
below `tol` — including the `NaN` produced when a hypothesis is violated (e.g. Cholesky of a
non-positive-definite matrix takes `√(negative)`). Documents that the success hypotheses (SPD pivots,
full column rank) are genuinely necessary.
-/
def assertReconFails (name : String) (err : Float) (tolerance : Float := tol) : IO Unit :=
  if err < tolerance then
    throw (IO.userError s!"{name}: FAIL (unexpectedly reconstructed, err = {err} < {tolerance})")
  else
    IO.println s!"{name}: OK (correctly failed, err = {err})"

/--
Compiled assertion that two scalars agree to `tolerance`. Used to verify the *exact* residual
identity numerically: the reconstruction error and the off-diagonal mass it equals are computed by
independent routines and shown to match, so the identity `symEigJacobi_frobenius_residual` is not a
tautology of the code.
-/
def assertApproxEq (name : String) (a b : Float) (tolerance : Float := tol) : IO Unit :=
  if Float.abs (a - b) < tolerance then
    IO.println s!"{name}: OK (lhs = {a}, rhs = {b}, |Δ| = {Float.abs (a - b)})"
  else
    throw (IO.userError s!"{name}: FAIL (lhs = {a}, rhs = {b}, |Δ| = {Float.abs (a - b)} ≥ {tolerance})")

end NN.Examples.Factorization
