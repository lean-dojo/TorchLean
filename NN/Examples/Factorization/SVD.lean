/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: singular value decomposition

`svdSpec A sweeps` returns `(U, σ, V)` with `A = U · diag(σ) · Vᵀ`. The singular values come from
the symmetric eigendecomposition of `Aᵀ·A`. We check the reconstruction of a 2×3 matrix whose
singular values are `{5, 3, 0}`.
-/

@[expose] public section


namespace NN.Examples.Factorization.SVD

/-- A 2×3 test matrix with singular values `{5, 3}` (third is `0` since rank 2 < 3). -/
def A : Spec.Tensor Float (.dim 2 (.dim 3 .scalar)) :=
  mkMat [[3, 2, 2],
         [2, 3, -2]]

/-- `(U, σ, V)` from the SVD. -/
def svd : Spec.Tensor Float (.dim 2 (.dim 3 .scalar)) × Spec.Tensor Float (.dim 3 .scalar) ×
    Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  Spec.svdSpec A 12

/-- Left singular vectors `U` (2×3). -/
def U : Spec.Tensor Float (.dim 2 (.dim 3 .scalar)) := svd.1
/-- Singular values `σ`. -/
def σ : Spec.Tensor Float (.dim 3 .scalar) := svd.2.1
/-- Right singular vectors `V` (3×3). -/
def V : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := svd.2.2

/-- Reconstruction error `‖A - U·diag(σ)·Vᵀ‖_max`. -/
def reconErr : Float := maxMatErr A (mm (mm U (diagFromVec σ)) (tr V))

#eval vecToList σ

-- Compiled assertion (fails the build otherwise).
#eval assertLt "SVD A = U·diag(σ)·Vᵀ" reconErr

end NN.Examples.Factorization.SVD
