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
/-- Orthogonality error `‖Vᵀ·V - I‖_max` for the right singular vectors. -/
def orthoErrV : Float := maxMatErr (mm (tr V) V) (Spec.identityTensorSpec 3)

#eval vecToList σ

-- Compiled assertions (fail the build otherwise).
#eval assertLt "SVD A = U·diag(σ)·Vᵀ" reconErr
-- `V` are the eigenvectors of `Aᵀ A` (see `IsSVD.gram_isSymEig`), hence orthogonal a-priori — the
-- numeric witness of `jacobi_orthogonal` applied to the Gram matrix, even though `σ₃ = 0` (rank 2).
#eval assertLt "SVD Vᵀ·V = I" orthoErrV

/-! ## Negative control: a wrong factor is rejected

Permuting the singular values (so they no longer pair with their vectors) must break the
reconstruction — otherwise the `maxMatErr` reconstruction check would be vacuous. -/

/-- A deliberately mismatched singular-value vector (permuted, and nonzero where the true `σ₃ = 0`). -/
def σbad : Spec.Tensor Float (.dim 3 .scalar) :=
  Spec.ofVecFn (fun i => ([3.0, 5.0, 1.0] : List Float).getD i.val 0.0)
/-- Reconstruction with the mismatched `σ` (should be far from `A`). -/
def reconErrBad : Float := maxMatErr A (mm (mm U (diagFromVec σbad)) (tr V))

#eval assertGe "SVD with permuted σ correctly fails to reconstruct" reconErrBad

end NN.Examples.Factorization.SVD
