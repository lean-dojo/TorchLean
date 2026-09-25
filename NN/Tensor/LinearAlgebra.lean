/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations
public import NN.Tensor.Reductions

/-!
# Public Tensor Linear Algebra

User-facing matrix factorizations and solves on the canonical tensor type.
-/

@[expose] public section

namespace TorchLean.Tensor

/-- Construct the `n x n` identity matrix. -/
def identity {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (n : Nat) : Tensor α [n, n] :=
  Spec.identityTensorSpec n

/-- The reduced QR factors of an `m x n` matrix, with reduced width `min m n`. -/
structure QRFactors (α : Type) [TorchLean.Storage α] (m n : Nat) where
  /-- Matrix containing the reduced basis-column candidates. -/
  q : Tensor α [m, min m n]
  /-- Upper-trapezoidal factor. -/
  r : Tensor α [min m n, n]
deriving Repr

namespace Internal

/-- Reduced QR by classical Gram-Schmidt, sweeping source columns left to right.

The sweep keeps the populated basis columns and the finished coefficient columns in arrays and
builds the two factor matrices once at the end. A dependent column does not consume a basis slot,
so later independent columns can still enter. Each dot product and projection sums only active
entries in their original order, starting from zero. Unused basis columns are zero and never
introduce spurious `0 * NaN` terms.
-/
def wideQR {α : Type} [TorchLean.Storage α] [Context α] {m n : Nat}
    (matrix : Tensor α [m, n]) : QRFactors α m n := Id.run do
  let width := min m n
  let mut basisColumns : Array (Tensor α [m]) := Array.emptyWithCapacity width
  let mut coefficientColumns : Array (Tensor α [width]) := Array.emptyWithCapacity n
  for column in List.finRange n do
    let count := basisColumns.size
    let basisEntry (index : Fin count) (row : Fin m) : α := basisColumns[index][row]
    let source : Tensor α [m] := Tensor.ofFn fun row => matrix[(row, column)]
    let coefficients : Tensor α [count] := Tensor.ofFn fun index =>
      Tensor.sum <| Tensor.ofFn fun (row : Fin m) => basisEntry index row * source[row]
    let residual : Tensor α [m] := Tensor.ofFn fun row =>
      source[row] - Tensor.sum (Tensor.ofFn fun (index : Fin count) =>
        coefficients[index] * basisEntry index row)
    let diagonal := MathFunctions.sqrt (Tensor.sum (Tensor.square residual))
    let positive := Context.gtBool diagonal 0
    let appendBasis := positive && decide (count < width)
    coefficientColumns := coefficientColumns.push <| Tensor.ofFn fun (row : Fin width) =>
      if h : row.val < count then coefficients[(⟨row.val, h⟩ : Fin count)]
      else if row.val == count && appendBasis then diagonal else 0
    if appendBasis then
      basisColumns := basisColumns.push <| Tensor.ofFn fun row => residual[row] / diagonal
  let q : Tensor α [m, width] := Tensor.matrix fun row basisColumn =>
    if h : basisColumn.val < basisColumns.size then basisColumns[basisColumn.val][row] else 0
  let r : Tensor α [width, n] := Tensor.matrix fun row sourceColumn =>
    if h : sourceColumn.val < coefficientColumns.size then
      coefficientColumns[sourceColumn.val][row]
    else 0
  return { q, r }

end Internal

/--
Compute a reduced QR factorization with classical Gram-Schmidt.

The result uses the NumPy/PyTorch reduced shapes
`q : Tensor α [m, min m n]` and `r : Tensor α [min m n, n]`.
For tall and square inputs, this is the theorem-backed specification computation with its dimensions
presented in reduced form. For wide inputs, linearly dependent source columns do not consume a
basis column, so a later independent column can still enter the reduced basis. Any unused trailing
basis columns are zero.
-/
def qr {α : Type} [TorchLean.Storage α] [Context α] {m n : Nat}
    (matrix : Tensor α [m, n]) : QRFactors α m n :=
  if hTall : n ≤ m then
    let factors := Spec.qrSpec matrix
    let q : Tensor α [m, min m n] := by
      simpa [Nat.min_eq_right hTall] using factors.1
    let r : Tensor α [min m n, n] := by
      simpa [Nat.min_eq_right hTall] using factors.2
    { q, r }
  else
    Internal.wideQR matrix

/--
Compute the lower-triangular Cholesky factor candidate of a square matrix.

For symmetric inputs with positive executable pivots, the specification layer proves
`A = L @ L.transpose`. Inputs outside that domain follow the scalar backend's arithmetic behavior;
for example, `Float` produces `NaN` after a negative square root.
-/
def cholesky {α : Type} [TorchLean.Storage α] [Context α] {n : Nat}
    (matrix : Tensor α [n, n]) : Tensor α [n, n] :=
  Spec.choleskySpec matrix

/--
Solve `(kernel + regularization * I) x = target` through the Cholesky path.

The operation is intended for symmetric positive-semidefinite kernels and positive regularization.
Its executable implementation is available across scalar backends supporting `Context`.
-/
def solveRidge {α : Type} [TorchLean.Storage α] [Context α] {n : Nat}
    (kernel : Tensor α [n, n]) (regularization : α)
    (target : Tensor α [n]) : Tensor α [n] :=
  Spec.solveRidgeSpec kernel regularization target

end TorchLean.Tensor
