/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Factorization
public import TorchLean.Tensor.LinearAlgebra.Identity
public import TorchLean.Tensor.Reduce.Fold

/-!
# Public Tensor Linear Algebra

Reduced QR on the canonical tensor type (`Tensor.qr`). The identity matrix `Tensor.identity` is
declared in `TorchLean.Tensor.LinearAlgebra.Identity`, and `Tensor.cholesky` and
`Tensor.solveRidge` sit with their recurrences in
`TorchLean.Tensor.Factorization`.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Tensor

/-- The reduced QR factors of an `m x n` matrix, with reduced width `min m n`. -/
structure QRFactors (α : Type) [TorchLean.Storage α] (m n : Nat) where
  /-- Matrix containing the reduced basis-column candidates. -/
  q : Tensor α [m, min m n]
  /-- Upper-trapezoidal factor. -/
  r : Tensor α [min m n, n]
deriving Repr

/--
Compute a reduced QR factorization with classical Gram-Schmidt, by the recurrence
`Factorization.gramSchmidtFn`.

The result has the NumPy/PyTorch reduced shapes `q : Tensor α [m, min m n]` and
`r : Tensor α [min m n, n]`. A column whose residual norm is not positive, a NaN norm included,
takes no basis column, and a later independent column can enter the reduced basis. Any unused
trailing basis columns are zero. The entries are those of `Factorization.qr` cut to the reduced
shapes, and the cut columns and rows are zero. The reconstruction and orthonormality theorems of
`Factorization.qr` are stated over real scalars and assume positive diagonal pivots.
-/
def qr {α : Type} [TorchLean.Storage α] [Context α] {m n : Nat}
    (matrix : Tensor α [m, n]) : QRFactors α m n :=
  let factors := Factorization.qr matrix
  { q := Tensor.matrix fun row column =>
      Tensor.get2 factors.1 row ⟨column.val, lt_of_lt_of_le column.isLt (Nat.min_le_right m n)⟩
    r := Tensor.matrix fun row column =>
      Tensor.get2 factors.2 ⟨row.val, lt_of_lt_of_le row.isLt (Nat.min_le_right m n)⟩ column }

end TorchLean.Tensor
