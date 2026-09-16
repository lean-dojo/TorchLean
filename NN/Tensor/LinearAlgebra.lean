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

/-- Tensor buffers and the active basis width of the Gram-Schmidt sweep. -/
structure WideQRState (α : Type) [TorchLean.Storage α] (rows columns : Nat) where
  /-- Number of populated basis columns; remaining columns stay zero. -/
  count : Fin (min rows columns + 1)
  /-- Reduced basis matrix. -/
  basis : Tensor α [rows, min rows columns]
  /-- Coefficients for the source columns processed so far. -/
  coefficients : Tensor α [min rows columns, columns]

/-- Reduced QR by classical Gram-Schmidt, sweeping source columns left to right.

The basis and coefficient buffers keep their final tensor shapes; `count` selects the populated
basis columns. A dependent column does not consume a basis slot, so later independent columns can
still enter. Each dot product and projection sums only active entries in their original order,
starting from zero. Unused basis columns therefore never introduce spurious `0 * NaN` terms.
-/
def wideQR {α : Type} [TorchLean.Storage α] [Context α] {m n : Nat}
    (matrix : Tensor α [m, n]) : QRFactors α m n := Id.run do
  let width := min m n
  let mut state : WideQRState α m n :=
    { count := ⟨0, by omega⟩, basis := Tensor.zeros [m, width]
      coefficients := Tensor.zeros [width, n] }
  for column in List.finRange n do
    let basisIndex : Fin state.count.val → Fin width :=
      fun index => ⟨index.val, Nat.lt_of_lt_of_le index.isLt (Nat.le_of_lt_succ state.count.isLt)⟩
    let source : Tensor α [m] := Tensor.ofFn fun row => matrix[(row, column)]
    let coefficients : Tensor α [state.count.val] := Tensor.ofFn fun index =>
      Tensor.sum <| Tensor.ofFn fun (row : Fin m) =>
        state.basis[(row, basisIndex index)] * source[row]
    let residual : Tensor α [m] := Tensor.ofFn fun row =>
      source[row] - Tensor.sum (Tensor.ofFn fun (index : Fin state.count.val) =>
        coefficients[index] * state.basis[(row, basisIndex index)])
    let diagonal := MathFunctions.sqrt (Tensor.sum (Tensor.square residual))
    let positive := Context.gtBool diagonal 0
    let appendBasis := positive && decide (state.count.val < width)
    let nextCoefficients : Tensor α [width, n] := Tensor.matrix fun row sourceColumn =>
      if sourceColumn == column then
        if h : row.val < state.count.val then coefficients[(⟨row.val, h⟩ : Fin state.count.val)]
        else if row.val == state.count.val && appendBasis then diagonal else 0
      else state.coefficients[(row, sourceColumn)]
    if h : state.count.val < width then
      if positive then
        let nextBasis : Tensor α [m, width] := Tensor.matrix fun row basisColumn =>
          if basisColumn.val == state.count.val then residual[row] / diagonal
          else state.basis[(row, basisColumn)]
        state :=
          { count := ⟨state.count.val + 1, Nat.succ_lt_succ h⟩
            basis := nextBasis, coefficients := nextCoefficients }
      else state := { state with coefficients := nextCoefficients }
    else state := { state with coefficients := nextCoefficients }
  return { q := state.basis, r := state.coefficients }

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
