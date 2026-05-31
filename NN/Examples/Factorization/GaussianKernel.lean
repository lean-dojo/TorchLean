/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: the CHD Gaussian-mode product kernel is symmetric positive-semidefinite

These checks corroborate `NN.Proofs.Tensor.Basic.FactorizationsKernels`. As with the linear and
quadratic modes, the whole verified CHD solve / `find_gamma` / `Z_test` development assumes the kernel
`K` is positive-semidefinite, and CHD *builds* `K` from data (`Modes/kernels.py`). The Gaussian
(fully-nonlinear) mode introduces the per-feature Gaussian `exp(−Δ²/2l²)`, whose product contribution is

`K[i,j] = scale · ∏_dim (1 + w[dim] · exp(−(X[i,dim]−X[j,dim])²/2l²))`   (`scale · jnp.prod(1 + w·exps)`).

`gaussianKernelFn_posSemidef` proves `K` PSD for `scale ≥ 0` and a nonnegative mask `w ≥ 0`, *without*
Bochner/Schoenberg (absent from Mathlib): the entrywise exponential of a PSD matrix is PSD (a
Hadamard-power series, the PSD cone closed under limits), each feature factor `𝟙𝟙ᵀ + w·Gaussian` is PSD,
and the product over features is PSD by the **Schur product theorem**. We exhibit:

* **symmetric** — `K = Kᵀ` to machine precision (`gaussianKernelFn_symm`);
* **matches CHD** — `K[i,j]` agrees with the direct `scale · ∏_dim (1 + w·exp(−Δ²/2l²))` formula;
* **positive-semidefinite** — every Jacobi eigenvalue is `≥ 0` (the numeric witness of
  `gaussianKernelFn_posSemidef`), and masking a feature (`w = [1,0]`) keeps it PSD;
* **feeds the verified solve** — because `K` is PSD, `solveRidgeSpec K γ b` is the exact regularized
  solve for `γ > 0` (`(K+γ·I)·x = b` to machine precision).

**Negative controls**: with `scale < 0` the whole kernel flips sign, and with a *negative* mask weight
(`w = [−2,0]`) a feature factor `1 − 2·exp(−Δ²/2l²)` drives the diagonal below zero — in both cases a
Jacobi eigenvalue goes negative, so `scale ≥ 0` *and* `w ≥ 0` are both necessary.
-/

@[expose] public section


namespace NN.Examples.Factorization.GaussianKernel

/-- Build a length-`n` `Float` vector from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- Count Jacobi eigenvalues that are negative (below `−10⁻⁹`). `0` certifies positive-semidefiniteness
numerically; `≥ 1` certifies an indefinite matrix. -/
def numNegEigs {k : Nat} (M : Spec.Tensor Float (.dim k (.dim k .scalar))) : Float :=
  let evals := (Spec.symEigJacobiSpec M 12).1
  (List.finRange k).foldl
    (fun a i => a + (if Spec.Tensor.toScalar (Spec.get evals i) < -1e-9 then 1.0 else 0.0)) 0.0

/-- The regularized matrix `K + γ·I` as a tensor. -/
def addGammaI {n : Nat} (K : Spec.Tensor Float (.dim n (.dim n .scalar))) (γ : Float) :
    Spec.Tensor Float (.dim n (.dim n .scalar)) :=
  Spec.ofMatFn (fun i j => Spec.get2 K i j + (if i.val == j.val then γ else 0.0))

/-- `ℓ¹` magnitude `Σᵢ |vᵢ|` (a sum, so a `NaN` propagates). -/
def vecAbsErr {n : Nat} (v : Spec.Tensor Float (.dim n .scalar)) : Float :=
  (List.finRange n).foldl (fun a i => a + Float.abs (Spec.Tensor.toScalar (Spec.get v i))) 0.0

/-- Residual `(K + γ·I)·x − b`. -/
def ridgeResidual {n : Nat} (K : Spec.Tensor Float (.dim n (.dim n .scalar))) (γ : Float)
    (b x : Spec.Tensor Float (.dim n .scalar)) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i =>
    Spec.Tensor.toScalar (Spec.get (Spec.matVecMulSpec (addGammaI K γ) x) i)
      - Spec.Tensor.toScalar (Spec.get b i))

/-- A `4 × 2` data matrix (4 samples, 2 features). -/
def X : Spec.Tensor Float (.dim 4 (.dim 2 .scalar)) :=
  mkMat [[1, 0],
         [0, 1],
         [1, 1],
         [2, 1]]

/-- Selection mask `which_dim = [1,1]` (both features active). -/
def wAll : Spec.Tensor Float (.dim 2 .scalar) := mkVec [1, 1]
/-- Kernel scale (CHD `GaussianMode._scale`). -/
def scale : Float := 1.0
/-- Gaussian length scale (CHD `GaussianMode.l`). -/
def l : Float := 1.0

/-- The Gaussian-mode product kernel `K = scale · ∏_dim (1 + w·exp(−Δ²/2l²))` (4×4). -/
def K : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.gaussianKernelSpec X wAll scale l

/-- Direct CHD `GaussianMode` product formula (mask all-ones):
`Kref[i,j] = scale · ∏_k (1 + w[k]·exp(−(X[i,k]−X[j,k])²/2l²))`. -/
def Kref : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  Spec.ofMatFn (fun i j =>
    scale * (List.finRange 2).foldl
      (fun acc k =>
        let dx := Spec.get2 X i k - Spec.get2 X j k
        acc * (1.0 + Spec.Tensor.toScalar (Spec.get wAll k) * Float.exp (-(dx * dx) / (2.0 * l * l))))
      1.0)

#eval IO.println s!"Gaussian kernel K =\n{(List.finRange 4).map (fun i =>
  (List.finRange 4).map (fun j => Spec.get2 K i j))}"
#eval IO.println s!"eigenvalues of K = {vecToList (Spec.symEigJacobiSpec K 12).1}"

-- Positive — `K` is symmetric (`gaussianKernelFn_symm`).
#eval assertLt "Gaussian kernel is symmetric: K = Kᵀ" (maxMatErr K (tr K))

-- Positive — `K` matches the direct CHD `GaussianMode` product formula.
#eval assertLt "Gaussian kernel matches CHD GaussianMode formula" (maxMatErr K Kref)

-- Positive — `K` is PSD: no negative Jacobi eigenvalue (`gaussianKernelFn_posSemidef`).
#eval assertLt "Gaussian kernel is PSD: no negative eigenvalue" (numNegEigs K)

/-! ## Masking a feature preserves PSD -/

/-- Mask out feature 1: `which_dim = [1,0]` (still `w ≥ 0`). -/
def wMask : Spec.Tensor Float (.dim 2 .scalar) := mkVec [1, 0]
def Kmask : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.gaussianKernelSpec X wMask scale l

#eval IO.println s!"masked-feature kernel eigenvalues = {vecToList (Spec.symEigJacobiSpec Kmask 12).1}"

-- Positive — masking a feature keeps the kernel PSD (PSD holds for any nonnegative mask).
#eval assertLt "masked Gaussian kernel is still PSD" (numNegEigs Kmask)

/-! ## The PSD kernel feeds the verified ridge solve -/

def γ : Float := 0.5
def b : Spec.Tensor Float (.dim 4 .scalar) := mkVec [1, 2, 3, 4]
/-- The ridge solution against the Gaussian kernel `K`. -/
def x : Spec.Tensor Float (.dim 4 .scalar) := Spec.solveRidgeSpec K γ b

#eval IO.println s!"ridge solve on the Gaussian kernel: residual = {vecToList (ridgeResidual K γ b x)}"

-- Positive — `K` PSD ⟹ `solveRidgeSpec K γ b` is the exact solve of `(K+γI)·x = b` (γ > 0).
#eval assertLt "PSD Gaussian kernel ⟹ exact ridge solve (K+γI)·x = b"
  (vecAbsErr (ridgeResidual K γ b x))

/-! ## Negative controls: `scale < 0` and a negative mask weight break positive-semidefiniteness -/

/-- The same kernel with `scale = −1`: the whole product is negated. -/
def Kscale : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.gaussianKernelSpec X wAll (-1.0) l

#eval IO.println s!"scale = -1 kernel eigenvalues = {vecToList (Spec.symEigJacobiSpec Kscale 12).1}"

-- Negative — with `scale < 0` the kernel is indefinite: at least one eigenvalue is negative.
#eval assertGe "scale < 0 breaks PSD (indefinite kernel)" (numNegEigs Kscale) 1.0

/-- A negative mask weight `w = [−2,0]`: the feature factor `1 − 2·exp(−Δ²/2l²)` makes the diagonal
(`Δ = 0 ⟹ 1 − 2 = −1`) negative, so the kernel cannot be PSD. -/
def wNeg : Spec.Tensor Float (.dim 2 .scalar) := mkVec [-2, 0]
def Kw : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.gaussianKernelSpec X wNeg scale l

#eval IO.println s!"w = [-2,0] kernel eigenvalues = {vecToList (Spec.symEigJacobiSpec Kw 12).1}"

-- Negative — a negative mask weight makes the kernel indefinite: at least one eigenvalue is negative.
#eval assertGe "negative mask weight breaks PSD (indefinite kernel)" (numNegEigs Kw) 1.0

end NN.Examples.Factorization.GaussianKernel
