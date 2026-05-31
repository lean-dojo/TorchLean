/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: the kernel-ridge (Tikhonov) linear solve

These checks corroborate the development in `NN.Proofs.Tensor.Basic.FactorizationsSolve`: the
Cholesky-based solve of `(K + γ·I)·x = b`, the linear solve at the heart of CHD `solve_variationnal`.

The verified pipeline is:

* `triSolveLowerFn` / `triSolveUpperFn` solve triangular systems by forward/back substitution
  (`triSolveLowerFn_mulVec`, `triSolveUpperFn_mulVec` — exact);
* `cholSolveFn` composes them through a Cholesky factor `L` to solve `(L·Lᵀ)·x = b`
  (`cholSolveFn_mulVec` — exact);
* `solveRidgeFn` factors `K + γ·I` and solves, giving `(K + γ·I)·x = b`
  (`solveRidgeFn_mulVec`, under the SPD success condition `posDef_addScaledIdFn` provides).

The kernel `K = G · Gᵀ` here is a rank-deficient (singular) Gram matrix — exactly the GP/kernel
setting CHD targets — so it is *not* invertible on its own. The checks exhibit:

* **Positive — regularization makes it solvable.** With `γ = 0.5 > 0`, `K + γ·I` is SPD, the Cholesky
  succeeds, and `solveRidgeFn` returns `x` with `(K + γ·I)·x = b` to machine precision (the exact
  `solveRidgeFn_mulVec`).
* **Negative — regularization is necessary.** With `γ = 0` the singular `K` has a zero Cholesky pivot:
  forward/back substitution divides by zero and the residual blows up (`NaN`/large). This is why CHD
  regularizes; it is also exactly the `γ > 0` hypothesis of `posDef_addScaledIdFn`.

It then exercises the two capstone theorems that close the solve story:

* `cholesky_posDef` — for the SPD `K + γ·I` the executable Cholesky reconstructs *exactly*
  (`L · Lᵀ = K + γ·I`); an *indefinite* matrix instead gets a `√(negative) = NaN` pivot and fails, so
  positive-definiteness is what the capstone needs.
* `solveRidgeFn_eq_inv_mulVec` — `solveRidgeFn K γ b = (K + γ·I)⁻¹ b`, the closed form CHD
  `solve_variationnal` specifies. Solving against each basis vector builds the columns of the inverse,
  and the assembled matrix satisfies `(K + γ·I) · (K + γ·I)⁻¹ = I` — no inverse is ever formed by the
  algorithm; every column is a verified Cholesky solve.
-/

@[expose] public section


namespace NN.Examples.Factorization.RidgeSolve

/-- Build a length-`n` `Float` vector from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- The regularized matrix `K + γ·I` as a tensor. -/
def addGammaI {n : Nat} (K : Spec.Tensor Float (.dim n (.dim n .scalar))) (γ : Float) :
    Spec.Tensor Float (.dim n (.dim n .scalar)) :=
  Spec.ofMatFn (fun i j => Spec.get2 K i j + (if i.val == j.val then γ else 0.0))

/-- `ℓ¹` magnitude `Σᵢ |vᵢ|` of a vector (residual size). A *sum* rather than a `max` so that a `NaN`
entry — produced when an unregularized singular solve divides by a zero pivot — propagates to the
result instead of being silently dropped by `Float`'s `max`. -/
def vecAbsErr {n : Nat} (v : Spec.Tensor Float (.dim n .scalar)) : Float :=
  (List.finRange n).foldl (fun a i => a + Float.abs (Spec.Tensor.toScalar (Spec.get v i))) 0.0

/-- Residual `(K + γ·I)·x − b` of a proposed solution `x`. -/
def ridgeResidual {n : Nat} (K : Spec.Tensor Float (.dim n (.dim n .scalar))) (γ : Float)
    (b x : Spec.Tensor Float (.dim n .scalar)) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i =>
    Spec.Tensor.toScalar (Spec.get (Spec.matVecMulSpec (addGammaI K γ) x) i)
      - Spec.Tensor.toScalar (Spec.get b i))

/-- A `3 × 2` factor; its Gram `K = G · Gᵀ` is a rank-2 (hence singular) `3 × 3` kernel matrix. -/
def G : Spec.Tensor Float (.dim 3 (.dim 2 .scalar)) :=
  mkMat [[1, 2],
         [3, 1],
         [0, 1]]

/-- The (symmetric, PSD, singular) kernel `K = G · Gᵀ`. -/
def K : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := mm G (tr G)

def γ : Float := 0.5
def b : Spec.Tensor Float (.dim 3 .scalar) := mkVec [1, 2, 3]

/-- The ridge solution `x = (K + γ·I)⁻¹ b`, via the verified Cholesky solve. -/
def x : Spec.Tensor Float (.dim 3 .scalar) := Spec.solveRidgeSpec K γ b

#eval IO.println s!"K = G·Gᵀ (rank-2, singular); γ = {γ}; b = {vecToList b}"
#eval IO.println s!"ridge solution x = {vecToList x}"
#eval IO.println s!"residual (K+γI)·x − b = {vecToList (ridgeResidual K γ b x)}"

-- Positive — the verified solve reconstructs `b`: `(K + γ·I)·x = b` (instance of `solveRidgeFn_mulVec`).
#eval assertLt "kernel-ridge solve: (K + γ·I)·x = b to machine precision"
  (vecAbsErr (ridgeResidual K γ b x))

/-! ## Negative control: regularization is necessary

The kernel `K` is singular, so with `γ = 0` its Cholesky has a zero pivot and the substitution
divides by zero — the "solution" does not satisfy the (singular) system. -/

def x0 : Spec.Tensor Float (.dim 3 .scalar) := Spec.solveRidgeSpec K 0.0 b

#eval IO.println s!"unregularized (γ = 0) on singular K: x0 = {vecToList x0}, \
  residual = {vecToList (ridgeResidual K 0.0 b x0)}"

-- Negative — without regularization the singular system is not solved (zero pivot → NaN/blow-up).
#eval assertReconFails "unregularized solve of singular K fails (γ = 0 → zero Cholesky pivot)"
  (vecAbsErr (ridgeResidual K 0.0 b x0))

/-! ## Keystone: positive-definite ⟹ strictly positive Cholesky pivots

`Spec.Factorization.Reconstruction.choleskyFn_diag_pos_of_posDef` proves that an SPD matrix has *all*
Cholesky pivots `> 0` — exactly the success condition the solve needs — and
`solveRidgeFn_mulVec_of_posSemidef` uses it to make the ridge solve unconditional for PSD `K`, `γ > 0`.
These checks exhibit the dichotomy the keystone formalizes. -/

/-- Count of non-positive Cholesky pivots of a square matrix. A `NaN` pivot (from `√(negative)` on a
non-SPD matrix) also counts, since `NaN > 0` is `false`. The keystone guarantees this is `0` for an
SPD matrix. -/
def numNonPosPivots {k : Nat} (M : Spec.Tensor Float (.dim k (.dim k .scalar))) : Float :=
  let L := Spec.choleskySpec M
  (List.finRange k).foldl (fun acc j => acc + (if Spec.get2 L j j > 0 then 0.0 else 1.0)) 0.0

/-- The SPD regularized matrix `K + γ·I` (`γ = 0.5 > 0`, `K` PSD). -/
def Kγ : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := addGammaI K γ

#eval IO.println s!"Cholesky pivots of K + γ·I (SPD): {vecToList (diagOf (Spec.choleskySpec Kγ))}"
#eval IO.println s!"Cholesky pivots of K (singular, γ = 0): {vecToList (diagOf (Spec.choleskySpec K))}"

-- Positive — SPD ⟹ every Cholesky pivot is > 0 (an instance of `choleskyFn_diag_pos_of_posDef`).
#eval assertLt "SPD K + γ·I has all-positive Cholesky pivots (keystone)" (numNonPosPivots Kγ)

-- Negative — the singular kernel `K` (PSD but not PD) has a non-positive pivot, so PosDef is needed.
#eval assertGe "singular K has a non-positive Cholesky pivot (PosDef necessary)"
  (numNonPosPivots K) 0.5

/-! ## Capstone: the SPD Cholesky reconstructs exactly

`Spec.Factorization.Reconstruction.cholesky_posDef` bundles the keystone with the reconstruction
theorem: for the *positive-definite* `K + γ·I`, the executable Cholesky factor is a genuine factor —
`L · Lᵀ = K + γ·I` exactly — with no pivot or symmetry hypothesis. The negative control is an
*indefinite* symmetric matrix: there a radicand goes negative, the pivot is `√(negative) = NaN`, and
reconstruction fails — so positive-definiteness (not mere symmetry) is what the capstone needs. (Note
the singular `K` itself, being PSD, *does* reconstruct with a zero pivot; the zero pivot breaks only
the *solve*, which is the dichotomy the keystone above isolates.) -/

/-- An indefinite symmetric matrix (top-left block has eigenvalues `3, −1`): not PosDef, so its
Cholesky hits `√(negative)`. -/
def Aindef : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[1, 2, 0],
         [2, 1, 0],
         [0, 0, 1]]

-- Positive — the SPD `K + γ·I` Cholesky reconstructs exactly: `L · Lᵀ = K + γ·I` (`cholesky_posDef`).
#eval assertLt "SPD Cholesky reconstructs: L·Lᵀ = K + γ·I (capstone)"
  (frobSqErr (let L := Spec.choleskySpec Kγ; mm L (tr L)) Kγ)

-- Negative — an indefinite matrix gets a `√(negative) = NaN` pivot, so it does not reconstruct.
#eval assertReconFails "indefinite matrix Cholesky does not reconstruct (PosDef necessary)"
  (frobSqErr (let L := Spec.choleskySpec Aindef; mm L (tr L)) Aindef)

/-! ## Closing the loop: the ridge solve *is* the regularized inverse

`Spec.Factorization.Reconstruction.solveRidgeFn_eq_inv_mulVec` proves `solveRidgeFn K γ b
= (K + γ·I)⁻¹ b` — the closed form CHD `solve_variationnal` specifies. Solving against each standard
basis vector `eⱼ` therefore produces column `j` of `(K + γ·I)⁻¹`; assembling the columns gives a
genuine inverse, witnessed by `(K + γ·I) · (K + γ·I)⁻¹ = I`. No matrix inverse is formed by the
algorithm — every column comes from the verified Cholesky solve. -/

/-- The `j`-th standard basis vector. -/
def unitVec {k : Nat} (j : Fin k) : Spec.Tensor Float (.dim k .scalar) :=
  Spec.ofVecFn (fun i => if i = j then 1.0 else 0.0)

/-- The `k × k` identity matrix. -/
def idMat {k : Nat} : Spec.Tensor Float (.dim k (.dim k .scalar)) :=
  Spec.ofMatFn (fun i j => if i = j then 1.0 else 0.0)

/-- The regularized inverse `(K + γ·I)⁻¹`, built column-by-column by the verified ridge solve: column
`j` is `solveRidgeSpec K γ eⱼ` (an instance of `solveRidgeFn_eq_inv_mulVec`). -/
def ridgeInv {k : Nat} (K : Spec.Tensor Float (.dim k (.dim k .scalar))) (γ : Float) :
    Spec.Tensor Float (.dim k (.dim k .scalar)) :=
  Spec.ofMatFn (fun i j => Spec.Tensor.toScalar (Spec.get (Spec.solveRidgeSpec K γ (unitVec j)) i))

#eval IO.println s!"(K+γI)⁻¹ diagonal (assembled from ridge solves): \
  {vecToList (diagOf (ridgeInv K γ))}"

-- Positive — the assembled inverse really inverts: `(K + γ·I) · (K + γ·I)⁻¹ = I`.
#eval assertLt "ridge solve builds the regularized inverse: (K+γI)·(K+γI)⁻¹ = I"
  (frobSqErr (mm Kγ (ridgeInv K γ)) idMat)

-- Negative — with `γ = 0` the singular `K` has no inverse: the column solves diverge (NaN).
#eval assertReconFails "unregularized singular K has no inverse (γ = 0 → solve diverges)"
  (frobSqErr (mm K (ridgeInv K 0.0)) idMat)

end NN.Examples.Factorization.RidgeSolve
