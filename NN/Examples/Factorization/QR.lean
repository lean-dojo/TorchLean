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

/-! ## Negative control: full column rank is necessary for orthonormality

`qrSpec_orthonormal` (`Qᵀ Q = 1`) requires full column rank — positive `R`-pivots
(`0 < R[j,j]`). The matrix below has a dependent column (`col₂ = 2·col₁`), so Gram–Schmidt produces a
**zero** `Q` column where the pivot vanishes: `A = Q·R` still holds, but `Qᵀ Q` has a `0` on the
diagonal, so orthonormality fails. This separates the two guarantees and shows the rank hypothesis
genuinely bites. -/

/-- A rank-2 matrix (`col₂ = 2·col₁`): reconstructs, but `Q` cannot be orthonormal. -/
def Adef : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[1, 2, 0],
         [2, 4, 1],
         [1, 2, 0]]

def Qdef : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.qrQSpec Adef
def Rdef : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := Spec.qrRSpec Adef

/-- Reconstruction still holds even without full rank. -/
def reconErrDef : Float := maxMatErr Adef (mm Qdef Rdef)
/-- Orthonormality fails: `Qᵀ·Q` has a zero diagonal entry, so it is far from `I`. -/
def orthoErrDef : Float := maxMatErr (mm (tr Qdef) Qdef) (Spec.identityTensorSpec 3)

#eval assertLt "QR(rank-deficient) A = Q·R still reconstructs" reconErrDef
#eval assertGe "QR(rank-deficient) Qᵀ·Q = I correctly fails (needs full column rank)" orthoErrDef

end NN.Examples.Factorization.QR
