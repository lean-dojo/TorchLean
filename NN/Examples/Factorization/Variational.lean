/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: the CHD variational solve, noise, and `Z_test` statistic (eigendecomposition form)

These checks corroborate `NN.Proofs.Tensor.Basic.FactorizationsVariational`, the eigendecomposition
route CHD's `perform_regression_and_find_gamma` actually takes (`interpolatory.py`). From `eigh(K)` it
forms the projected data `Pga = Vᵀ·ga` and shrinkage coefficients `rᵢ = γ/(λᵢ+γ)`, then runs three
routines off that shared core. We exercise each:

* **The variational solve is the regularized inverse.** `variationalSolveSpec` returns
  `yb = -(K+γ·I)⁻¹·ga`, so `(K+γ·I)·yb = -ga` to machine precision (`variationalSolveFn_eq_inv_mulVec`).
* **Eig route = Cholesky route.** The same `yb` equals `-solveRidgeSpec K γ ga` — the verified Cholesky
  solve from `FactorizationsSolve` — to machine precision (`variationalSolveFn_eq_neg_solveRidgeFn`):
  two independent implementations, one closed form.
* **The noise is a fraction.** `varNoiseSpec` (the `noise`, the `find_gamma` loss, the `Z_test`
  statistic) lies in `[0,1]` (`varNoiseFn_nonneg`, `varNoiseFn_le_one`).
* **`Z_test` spectral invariance.** Feeding `ga = V·z` makes `V` drop out: the noise of `V·z` under `V`
  equals the noise of `z` under the identity (`varNoiseFn_projFn_mulVec`) — the statistic depends on
  the kernel only through its eigenvalues.

Negative controls give the metrics teeth:

* feeding the **wrong** eigenvectors (the identity instead of the true `V`) breaks the solve — the
  residual `(K+γ·I)·yb + ga` is large, so the *actual* eigendecomposition is needed;
* with **`γ < 0`** the shrinkage coefficients leave `(0,1]` and the noise falls outside `[0,1]`, so
  `γ > 0` is necessary for the bound.
-/

@[expose] public section


namespace NN.Examples.Factorization.Variational

/-- Build a length-`n` `Float` vector from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- The regularized matrix `K + γ·I` as a tensor. -/
def addGammaI {n : Nat} (K : Spec.Tensor Float (.dim n (.dim n .scalar))) (γ : Float) :
    Spec.Tensor Float (.dim n (.dim n .scalar)) :=
  Spec.ofMatFn (fun i j => Spec.get2 K i j + (if i.val == j.val then γ else 0.0))

/-- Matrix–vector product `M · v`. -/
def mv {n : Nat} (M : Spec.Tensor Float (.dim n (.dim n .scalar)))
    (v : Spec.Tensor Float (.dim n .scalar)) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.matVecMulSpec M v

/-- Entrywise negation of a vector. -/
def negVec {n : Nat} (v : Spec.Tensor Float (.dim n .scalar)) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => 0.0 - Spec.Tensor.toScalar (Spec.get v i))

/-- `ℓ¹` magnitude `Σᵢ |vᵢ|` (a sum, so a `NaN` entry propagates instead of being dropped). -/
def vecAbsErr {n : Nat} (v : Spec.Tensor Float (.dim n .scalar)) : Float :=
  (List.finRange n).foldl (fun a i => a + Float.abs (Spec.Tensor.toScalar (Spec.get v i))) 0.0

/-- `ℓ¹` distance `Σᵢ |uᵢ − vᵢ|` between two vectors. -/
def vecDist {n : Nat} (u v : Spec.Tensor Float (.dim n .scalar)) : Float :=
  (List.finRange n).foldl
    (fun a i => a + Float.abs (Spec.Tensor.toScalar (Spec.get u i)
      - Spec.Tensor.toScalar (Spec.get v i))) 0.0

/-- The `k × k` identity matrix. -/
def idMat {k : Nat} : Spec.Tensor Float (.dim k (.dim k .scalar)) :=
  Spec.ofMatFn (fun i j => if i = j then 1.0 else 0.0)

/-- A symmetric positive-definite kernel (eigenvalues ≈ {0.5858, 2, 3.4142}). -/
def K : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[2, 1, 0],
         [1, 2, 1],
         [0, 1, 2]]

def γ : Float := 0.5
def ga : Spec.Tensor Float (.dim 3 .scalar) := mkVec [1, 2, 3]

/-- Eigendecomposition `K = V·diag(λ)·Vᵀ` via cyclic Jacobi (12 sweeps). -/
def eig : Spec.Tensor Float (.dim 3 .scalar) × Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  Spec.symEigJacobiSpec K 12
def evals : Spec.Tensor Float (.dim 3 .scalar) := eig.1
def V : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := eig.2

/-- The variational solution `yb = -(K + γ·I)⁻¹·ga` (eigendecomposition form). -/
def yb : Spec.Tensor Float (.dim 3 .scalar) := Spec.variationalSolveSpec evals V γ ga

#eval IO.println s!"eigenvalues λ = {vecToList evals}; γ = {γ}; ga = {vecToList ga}"
#eval IO.println s!"variational solution yb = {vecToList yb}"
#eval IO.println s!"(K+γI)·yb + ga = {vecToList (Spec.ofVecFn (fun i =>
  Spec.Tensor.toScalar (Spec.get (mv (addGammaI K γ) yb) i)
    + Spec.Tensor.toScalar (Spec.get ga i)))}"

/-! ## The variational solve is the regularized-inverse solve -/

-- Positive — `yb = -(K+γI)⁻¹·ga`, so `(K+γI)·yb = -ga`, i.e. `(K+γI)·yb + ga ≈ 0`.
#eval assertLt "variational solve: (K+γI)·yb = -ga to machine precision"
  (vecAbsErr (Spec.ofVecFn (fun i =>
    Spec.Tensor.toScalar (Spec.get (mv (addGammaI K γ) yb) i)
      + Spec.Tensor.toScalar (Spec.get ga i))))

-- Positive — eig route = Cholesky route: `yb = -solveRidgeSpec K γ ga` to machine precision.
#eval assertLt "eig-form solve = -(Cholesky ridge solve) (two implementations agree)"
  (vecDist yb (negVec (Spec.solveRidgeSpec K γ ga)))

/-! ## The noise level is a fraction in `[0,1]` -/

/-- The CHD `noise` / `find_gamma` loss / `Z_test` statistic at this `(K, γ, ga)`. -/
def noise : Float := Spec.varNoiseSpec evals V γ ga

#eval IO.println s!"noise level = {noise}"

-- Positive — `noise ≤ 1` (err = noise − 1 < tol ⟺ noise < 1 + tol).
#eval assertLt "noise ≤ 1 (find_gamma loss is a fraction)" (noise - 1.0)
-- Positive — `0 ≤ noise` (err = −noise < tol ⟺ noise > −tol).
#eval assertLt "0 ≤ noise" (0.0 - noise)

/-! ## `Z_test` spectral invariance: feeding `ga = V·z` drops `V` -/

def z : Spec.Tensor Float (.dim 3 .scalar) := mkVec [0.7, -1.3, 2.1]
/-- Data expressed in eigencoordinates: `ga = V·z`. -/
def gaVz : Spec.Tensor Float (.dim 3 .scalar) := mv V z

#eval IO.println s!"noise(V·z under V) = {Spec.varNoiseSpec evals V γ gaVz}; \
  noise(z under I) = {Spec.varNoiseSpec evals idMat γ z}"

-- Positive — `noise` of `V·z` under `V` equals `noise` of `z` under the identity (spectral only).
#eval assertApproxEq "Z_test statistic depends only on the spectrum (ga = V·z ⟹ V drops out)"
  (Spec.varNoiseSpec evals V γ gaVz) (Spec.varNoiseSpec evals idMat γ z)

/-! ## Negative controls -/

/-- The solve fed the **wrong** eigenvectors (identity instead of the true `V`). -/
def ybWrong : Spec.Tensor Float (.dim 3 .scalar) := Spec.variationalSolveSpec evals idMat γ ga

#eval IO.println s!"wrong-V residual (K+γI)·ybWrong + ga = {vecToList (Spec.ofVecFn (fun i =>
  Spec.Tensor.toScalar (Spec.get (mv (addGammaI K γ) ybWrong) i)
    + Spec.Tensor.toScalar (Spec.get ga i)))}"

-- Negative — with the wrong eigenvectors the solve no longer inverts: the residual is large.
#eval assertGe "wrong eigenvectors break the solve (true eigendecomposition needed)"
  (vecAbsErr (Spec.ofVecFn (fun i =>
    Spec.Tensor.toScalar (Spec.get (mv (addGammaI K γ) ybWrong) i)
      + Spec.Tensor.toScalar (Spec.get ga i)))) 0.5

/-- The noise level computed with `γ < 0` (here `γ = -0.7`, below the smallest eigenvalue ≈ 0.586, so a
shrinkage coefficient `rᵢ = γ/(λᵢ+γ)` leaves `(0,1]`). -/
def noiseNeg : Float := Spec.varNoiseSpec evals V (-0.7) ga

#eval IO.println s!"noise with γ = -0.7 (outside [0,1]) = {noiseNeg}"

-- Negative — with `γ < 0` the noise falls outside `[0,1]`, so `γ > 0` is necessary for the bound.
#eval assertGe "γ < 0 pushes noise outside [0,1] (γ > 0 necessary)"
  (max (0.0 - noiseNeg) (noiseNeg - 1.0)) 0.01

end NN.Examples.Factorization.Variational
