/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: QR factorization

`qrSpec A` returns `(Q, R)` with `A = Q · R`, `Q` having orthonormal columns and `R`
upper-triangular (modified Gram–Schmidt). We check both `A = Q·R` and `Qᵀ·Q = I`.
-/

@[expose] public section


namespace NN.Examples.Factorization.QR

/-- A 3×3 test matrix (the classic Householder/QR example). -/
def A : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[12, -51, 4],
         [6, 167, -68],
         [-4, 24, -41]]

/-- Orthonormal `Q` factor. -/
def Q : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.qrQSpec A
/-- Upper-triangular `R` factor. -/
def R : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.qrRSpec A

/-- Reconstruction error `‖A - Q·R‖_max`. -/
def reconErr : Float := maxMatErr A (mm Q R)
/-- Orthonormality error `‖Qᵀ·Q - I‖_max`. -/
def orthoErr : Float := maxMatErr (mm (tr Q) Q) (Spec.identityTensorSpec 3)

-- Compiled assertions (fail the build otherwise).
#eval assertLt "QR A = Q·R" reconErr
#eval assertLt "QR Qᵀ·Q = I" orthoErr

end NN.Examples.Factorization.QR
