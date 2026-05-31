/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: the CHD quadratic-mode kernel is symmetric positive-semidefinite

These checks corroborate `NN.Proofs.Tensor.Basic.FactorizationsKernels`. As with the linear mode, the
whole verified CHD solve / `find_gamma` / `Z_test` development assumes the kernel `K` is
positive-semidefinite, and CHD *builds* `K` from data (`Modes/kernels.py`). For the quadratic mode,

`K[i,j] = scale · (alpha + ⟨Φ i, Φ j⟩)² + (1 − alpha²·scale)`   (`Φ` = column-masked data),

which expands to `K = 𝟙𝟙ᵀ + (2·scale·alpha)·Φ·Φᵀ + scale·(Φ·Φᵀ ⊙ Φ·Φᵀ)` — the last term a **Hadamard
square**, PSD by the Schur product theorem. `quadraticKernelFn_posSemidef` proves `K` PSD for
`scale ≥ 0` and `alpha ≥ 0`, discharging that standing hypothesis for the real quadratic kernel. We
exhibit:

* **symmetric** — `K = Kᵀ` to machine precision (`quadraticKernelFn_symm`);
* **matches CHD** — `K[i,j] = scale·(alpha + ⟨xᵢ, xⱼ⟩)² + (1 − alpha²·scale)` agrees with the direct
  `QuadraticMode.vectorized_kernel` formula;
* **positive-semidefinite** — every Jacobi eigenvalue is `≥ 0` (the numeric witness of
  `quadraticKernelFn_posSemidef`), and masking a feature (`w = [1,0]`) keeps it PSD;
* **feeds the verified solve** — because `K` is PSD, `solveRidgeSpec K γ b` is the exact regularized
  solve for `γ > 0` (`(K+γ·I)·x = b` to machine precision).

**Negative controls**: with `alpha < 0` the middle term `2·scale·alpha·Φ·Φᵀ` goes negative (the
diagonal `scale·alpha² + … ` drops below zero) and with `scale < 0` the whole quadratic part flips sign
— in both cases a Jacobi eigenvalue goes negative, so `scale ≥ 0` *and* `alpha ≥ 0` are both necessary.
-/

@[expose] public section


namespace NN.Examples.Factorization.QuadraticKernel

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
/-- Kernel scale (CHD `QuadraticMode._scale`). -/
def scale : Float := 2.0
/-- Quadratic offset (CHD `alpha = 0.5·scales["linear"]/scale`; here `0.5·2.0/2.0 = 0.5`). -/
def alpha : Float := 0.5

/-- The quadratic-mode kernel `K = scale·(alpha + Φ·Φᵀ)² + (1 − alpha²·scale)` (4×4). -/
def K : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.quadraticKernelSpec X wAll scale alpha

/-- Direct CHD `QuadraticMode.vectorized_kernel` formula (mask all-ones):
`Kref[i,j] = scale·(alpha + Σ_k X[i,k]·X[j,k])² + (1 − alpha²·scale)`. -/
def Kref : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  Spec.ofMatFn (fun i j =>
    let m := (List.finRange 2).foldl (fun a k => a + Spec.get2 X i k * Spec.get2 X j k) 0.0
    scale * (alpha + m) ^ 2 + (1.0 - alpha ^ 2 * scale))

#eval IO.println s!"quadratic kernel K =\n{(List.finRange 4).map (fun i =>
  (List.finRange 4).map (fun j => Spec.get2 K i j))}"
#eval IO.println s!"eigenvalues of K = {vecToList (Spec.symEigJacobiSpec K 12).1}"

-- Positive — `K` is symmetric (`quadraticKernelFn_symm`).
#eval assertLt "quadratic kernel is symmetric: K = Kᵀ" (maxMatErr K (tr K))

-- Positive — `K` matches the direct CHD `QuadraticMode` formula.
#eval assertLt "quadratic kernel matches CHD QuadraticMode formula" (maxMatErr K Kref)

-- Positive — `K` is PSD: no negative Jacobi eigenvalue (`quadraticKernelFn_posSemidef`).
#eval assertLt "quadratic kernel is PSD: no negative eigenvalue" (numNegEigs K)

/-! ## Masking a feature preserves PSD -/

/-- Mask out feature 1: `which_dim = [1,0]`. -/
def wMask : Spec.Tensor Float (.dim 2 .scalar) := mkVec [1, 0]
def Kmask : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.quadraticKernelSpec X wMask scale alpha

#eval IO.println s!"masked-feature kernel eigenvalues = {vecToList (Spec.symEigJacobiSpec Kmask 12).1}"

-- Positive — masking a feature keeps the kernel PSD (PSD holds for any mask).
#eval assertLt "masked quadratic kernel is still PSD" (numNegEigs Kmask)

/-! ## The PSD kernel feeds the verified ridge solve -/

def γ : Float := 0.5
def b : Spec.Tensor Float (.dim 4 .scalar) := mkVec [1, 2, 3, 4]
/-- The ridge solution against the quadratic kernel `K`. -/
def x : Spec.Tensor Float (.dim 4 .scalar) := Spec.solveRidgeSpec K γ b

#eval IO.println s!"ridge solve on the quadratic kernel: residual = {vecToList (ridgeResidual K γ b x)}"

-- Positive — `K` PSD ⟹ `solveRidgeSpec K γ b` is the exact solve of `(K+γI)·x = b` (γ > 0).
#eval assertLt "PSD quadratic kernel ⟹ exact ridge solve (K+γI)·x = b"
  (vecAbsErr (ridgeResidual K γ b x))

/-! ## Negative controls: `alpha < 0` and `scale < 0` break positive-semidefiniteness -/

/-- The same kernel with `alpha = −1`: the linear term `2·scale·alpha·Φ·Φᵀ` is negative. -/
def Kalpha : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.quadraticKernelSpec X wAll scale (-1.0)

#eval IO.println s!"alpha = -1 kernel eigenvalues = {vecToList (Spec.symEigJacobiSpec Kalpha 12).1}"

-- Negative — with `alpha < 0` the kernel is indefinite: at least one eigenvalue is negative.
#eval assertGe "alpha < 0 breaks PSD (indefinite kernel)" (numNegEigs Kalpha) 1.0

/-- The same kernel with `scale = −1`: the whole quadratic part flips sign. -/
def Kscale : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.quadraticKernelSpec X wAll (-1.0) alpha

#eval IO.println s!"scale = -1 kernel eigenvalues = {vecToList (Spec.symEigJacobiSpec Kscale 12).1}"

-- Negative — with `scale < 0` the kernel is indefinite: at least one eigenvalue is negative.
#eval assertGe "scale < 0 breaks PSD (indefinite kernel)" (numNegEigs Kscale) 1.0

end NN.Examples.Factorization.QuadraticKernel
