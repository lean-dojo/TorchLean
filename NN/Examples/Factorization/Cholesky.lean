/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: Cholesky factorization

`choleskySpec A` returns the lower-triangular `L` with `A = L · Lᵀ` for a symmetric
positive-definite `A`. Here we factor a 3×3 SPD matrix and check the reconstruction error.
-/

@[expose] public section


namespace NN.Examples.Factorization.Cholesky

/-- A symmetric positive-definite test matrix. -/
def A : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[4, 2, 2],
         [2, 5, 3],
         [2, 3, 6]]

/-- The Cholesky factor `L` (lower-triangular). -/
def L : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.choleskySpec A

/-- Reconstruction error `‖A - L·Lᵀ‖_max`. -/
def reconErr : Float := maxMatErr A (mm L (tr L))

-- Inspect the diagonal of the factor.
#eval vecToList (Spec.ofVecFn (fun i : Fin 3 => Spec.get2 L i i))

-- Compiled assertion: the factorization reconstructs A (fails the build otherwise).
#eval assertLt "Cholesky A = L·Lᵀ" reconErr

/-! ## Negative control: the SPD hypothesis is necessary

`isCholesky_of_pos` requires the executable pivots `L[j,j]` to be positive (`0 < choleskyFn A j j`),
which is exactly the success condition over the reals. The matrix below is symmetric but *not*
positive-definite (eigenvalues `3` and `-1`), so the diagonal step takes `√(negative)` and the
reconstruction is `NaN` — never a small error. This documents that the hypothesis genuinely bites. -/

/-- A symmetric but **indefinite** matrix (eigenvalues `{3, -1}`), outside Cholesky's domain. -/
def Abad : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) :=
  mkMat [[1, 2],
         [2, 1]]

def Lbad : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) := Spec.choleskySpec Abad
-- Use the *summed* Frobenius error here, not `maxMatErr`: IEEE `max` ignores `NaN`, whereas the sum
-- propagates the `NaN` produced by `√(negative)`, faithfully reporting that no factor exists.
def reconErrBad : Float := frobSqErr Abad (mm Lbad (tr Lbad))

#eval assertReconFails "Cholesky on indefinite A correctly fails (no SPD ⇒ no factor)" reconErrBad

end NN.Examples.Factorization.Cholesky
