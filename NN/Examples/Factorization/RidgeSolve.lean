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

end NN.Examples.Factorization.RidgeSolve
