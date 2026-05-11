/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Core

public import Mathlib.Analysis.Normed.Module.FiniteDimension

/-!
# Params

Analytic (`HasFDerivAt`) building blocks for **parameter gradients**.

The key fact is the Frobenius/outer-product identity: for fixed `x`,
the linear map `W ↦ W x` has adjoint `δ ↦ δ ⊗ x`.

This is used to connect weight gradients produced by backprop to adjoints of `fderiv`.

## PyTorch correspondence / citations
For a linear layer `y = W x + b`, PyTorch’s backward returns:
- `∂L/∂W = δ ⊗ x` (outer product of upstream gradient and input), and
- `∂L/∂x = Wᵀ δ`.
See `torch.nn.Linear` documentation for the forward definition and standard gradients:
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open scoped BigOperators

noncomputable section

/-- Weight matrices as a real Hilbert space (Frobenius / L2 inner product). -/
abbrev Mat (m n : Nat) := PiLp 2 (fun _ : Fin m => PiLp 2 (fun _ : Fin n => ℝ))

/-- Convert a coordinate function `Fin n → ℝ` into the bundled vector type `Vec n`. -/
def vecOfFunMat {n : Nat} (f : Fin n → ℝ) : Vec n :=
  (euclideanEquiv n).symm f

@[simp] lemma vecOfFunMat_ofLp {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    (vecOfFunMat (n := n) f).ofLp i = f i := by
  simp [vecOfFunMat, euclideanEquiv]

/-- View `Mat m n` as a Mathlib `Matrix` with the same coordinate function. -/
def toMatrix {m n : Nat} (W : Mat m n) : Matrix (Fin m) (Fin n) ℝ := fun i j => W i j

/-- `toMatrix` preserves addition. -/
lemma toMatrix_add {m n : Nat} (W1 W2 : Mat m n) :
    toMatrix (W1 + W2) = toMatrix W1 + toMatrix W2 := by
  funext i j
  rfl

/-- `toMatrix` preserves scalar multiplication. -/
lemma toMatrix_smul {m n : Nat} (a : ℝ) (W : Mat m n) :
    toMatrix (a • W) = a • toMatrix W := by
  funext i j
  rfl

/-- Linear map `W ↦ W.mulVec x` (matrix-vector product, linear in `W`). -/
def matApplyLM {m n : Nat} (x : Vec n) : Mat m n →ₗ[ℝ] Vec m :=
{ toFun := fun W => vecOfFunMat (n := m) ((toMatrix W).mulVec x.ofLp)
  map_add' := by
    intro W1 W2
    ext i
    simp [vecOfFunMat_ofLp, toMatrix_add, Matrix.add_mulVec]
  map_smul' := by
    intro a W
    ext i
    simp [vecOfFunMat_ofLp, toMatrix_smul, Matrix.smul_mulVec, smul_eq_mul] }

/-- Continuous version of `matApplyLM`. -/
def matApplyLin {m n : Nat} (x : Vec n) : Mat m n →L[ℝ] Vec m := by
  refine ⟨matApplyLM (m := m) (n := n) x, ?_⟩
  simpa using
    (LinearMap.continuous_of_finiteDimensional (matApplyLM (m := m) (n := n) x))

/--
Outer product `δ ⊗ x` (as a matrix in `Mat`).

This is the standard formula for the adjoint of `W ↦ W x` under Frobenius/L2 inner products.
-/
def outer {m n : Nat} (δ : Vec m) (x : Vec n) : Mat m n :=
  WithLp.toLp 2 fun i : Fin m =>
    WithLp.toLp 2 fun j : Fin n =>
      δ.ofLp i * x.ofLp j

@[simp] lemma outer_apply {m n : Nat} (δ : Vec m) (x : Vec n) (i : Fin m) (j : Fin n) :
    outer (m := m) (n := n) δ x i j = δ.ofLp i * x.ofLp j := by
  simp [outer]

/-- Coordinate formula for the Frobenius/L2 inner product on `Mat m n`. -/
lemma inner_mat_eq_sum {m n : Nat} (A B : Mat m n) :
    inner ℝ A B = ∑ i : Fin m, ∑ j : Fin n, A i j * B i j := by
  classical
  calc
    inner ℝ A B = ∑ i : Fin m, inner ℝ (A i) (B i) := by
      simp [Mat, PiLp.inner_apply]
    _ = ∑ i : Fin m, ∑ j : Fin n, A i j * B i j := by
      refine Finset.sum_congr rfl ?_
      intro i _
      simpa using (inner_eq_sum_mul (x := A i) (y := B i))

/--
Adjointness identity for `matApplyLin x`:

`⟪(W ↦ W x) dW, δ⟫ = ⟪dW, δ ⊗ x⟫`.
-/
lemma inner_matApply_eq {m n : Nat} (x : Vec n) (dW : Mat m n) (δ : Vec m) :
    inner ℝ ((matApplyLin (m := m) (n := n) x) dW) δ
      =
    inner ℝ dW (outer (m := m) (n := n) δ x) := by
  classical
  have hL :
      inner ℝ ((matApplyLin (m := m) (n := n) x) dW) δ
        = ∑ i : Fin m, ((toMatrix dW).mulVec x.ofLp) i * δ.ofLp i := by
    simp [matApplyLin, matApplyLM, inner_eq_sum_mul]
  have hR :
      inner ℝ dW (outer (m := m) (n := n) δ x)
        =
      ∑ i : Fin m, ∑ j : Fin n, dW i j * (δ.ofLp i * x.ofLp j) := by
    simp [inner_mat_eq_sum, outer]

  calc
    inner ℝ ((matApplyLin (m := m) (n := n) x) dW) δ
        = ∑ i : Fin m, ((toMatrix dW).mulVec x.ofLp) i * δ.ofLp i := hL
    _ = ∑ i : Fin m, (∑ j : Fin n, dW i j * x.ofLp j) * δ.ofLp i := by
          refine Finset.sum_congr rfl ?_
          intro i _
          simp [toMatrix, Matrix.mulVec, dotProduct]
    _ = ∑ i : Fin m, ∑ j : Fin n, dW i j * (δ.ofLp i * x.ofLp j) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          calc
            (∑ j : Fin n, dW i j * x.ofLp j) * δ.ofLp i
                = δ.ofLp i * ∑ j : Fin n, dW i j * x.ofLp j := by
                    ring
            _ = ∑ j : Fin n, δ.ofLp i * (dW i j * x.ofLp j) := by
                    simp [Finset.mul_sum]
            _ = ∑ j : Fin n, dW i j * (δ.ofLp i * x.ofLp j) := by
                    refine Finset.sum_congr rfl ?_
                    intro j _
                    ring
    _ = inner ℝ dW (outer (m := m) (n := n) δ x) := by
          simp [hR]

/-!
Main adjoint lemma:

`(W ↦ W x)† δ = δ ⊗ x`.
-/
/--
Adjoint of `W ↦ W x` under Frobenius/L2 inner products.

This is the mathematical core of the “weight gradient is outer product” rule:
`(matApplyLin x)† δ = δ ⊗ x`.
-/
lemma matApplyLin_adjoint_apply {m n : Nat} (x : Vec n) (δ : Vec m) :
    (matApplyLin (m := m) (n := n) x).adjoint δ = outer (m := m) (n := n) δ x := by
  classical
  let A := matApplyLin (m := m) (n := n) x
  have hforall :
      ∀ dW : Mat m n, inner ℝ dW (A.adjoint δ) = inner ℝ dW (outer (m := m) (n := n) δ x) := by
    intro dW
    calc
      inner ℝ dW (A.adjoint δ) = inner ℝ (A dW) δ := by
        simpa [A] using
          (ContinuousLinearMap.adjoint_inner_right (A := A) (x := dW) (y := δ))
      _ = inner ℝ dW (outer (m := m) (n := n) δ x) := by
        simpa [A] using (inner_matApply_eq (m := m) (n := n) (x := x) (dW := dW) (δ := δ))

  set u := A.adjoint δ
  set v := outer (m := m) (n := n) δ x
  have h0 : inner ℝ (u - v) (u - v) = 0 := by
    have hEq := hforall (dW := (u - v))
    have : inner ℝ (u - v) u - inner ℝ (u - v) v = 0 := by
      simpa [sub_eq_zero] using congrArg (fun t => t - inner ℝ (u - v) v) hEq
    have hinnerSub :
        inner ℝ (u - v) (u - v) = inner ℝ (u - v) u - inner ℝ (u - v) v := by
      simpa using (inner_sub_right (x := u - v) (y := u) (z := v))
    exact hinnerSub.trans this
  have huv : u - v = 0 := (inner_self_eq_zero (𝕜 := ℝ) (x := (u - v))).1 h0
  have : u = v := sub_eq_zero.mp huv
  simpa [u, v, A] using this

end
end Autograd
end Proofs
