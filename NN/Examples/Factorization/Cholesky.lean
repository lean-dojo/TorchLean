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

end NN.Examples.Factorization.Cholesky
