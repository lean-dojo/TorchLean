/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Core


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

/-- Weight matrices as a real Hilbert space (Frobenius/$\ell_2$ inner product). -/
abbrev Mat (m n : Nat) := PiLp 2 (fun _ : Fin m => PiLp 2 (fun _ : Fin n => ℝ))

/-- View `Mat m n` as a Mathlib `Matrix` with the same coordinate function. -/
def toMatrix {m n : Nat} (W : Mat m n) : Matrix (Fin m) (Fin n) ℝ := fun i j => W i j

/-- `toMatrix` preserves addition. -/
theorem toMatrix_add {m n : Nat} (W1 W2 : Mat m n) :
    toMatrix (W1 + W2) = toMatrix W1 + toMatrix W2 := by
  funext i j
  rfl

/-- `toMatrix` preserves scalar multiplication. -/
theorem toMatrix_smul {m n : Nat} (a : ℝ) (W : Mat m n) :
    toMatrix (a • W) = a • toMatrix W := by
  funext i j
  rfl

/-- Linear map `W ↦ W.mulVec x` (matrix-vector product, linear in `W`). -/
def matApplyLM {m n : Nat} (x : Vec n) : Mat m n →ₗ[ℝ] Vec m :=
{ toFun := fun W => vecOfFun ((toMatrix W).mulVec x.ofLp)
  map_add' := by
    intro W1 W2
    ext i
    simp [toMatrix_add, Matrix.add_mulVec]
  map_smul' := by
    intro a W
    ext i
    simp [toMatrix_smul, Matrix.smul_mulVec, smul_eq_mul] }

/-- Continuous version of `matApplyLM`. -/
def matApplyLin {m n : Nat} (x : Vec n) : Mat m n →L[ℝ] Vec m :=
  ⟨matApplyLM (m := m) (n := n) x,
    (matApplyLM (m := m) (n := n) x).continuous_of_finiteDimensional⟩

/--
Outer product `δ ⊗ x` (as a matrix in `Mat`).

This is the standard formula for the adjoint of `W ↦ W x` under Frobenius/$\ell_2$ inner products.
-/
def outer {m n : Nat} (δ : Vec m) (x : Vec n) : Mat m n :=
  WithLp.toLp 2 fun i : Fin m =>
    WithLp.toLp 2 fun j : Fin n =>
      δ.ofLp i * x.ofLp j

/-- Entries of the outer product are the pairwise products, as expected.

This is the shape a weight gradient takes: `∂L/∂W = δ ⊗ x`. Having it as a `simp` lemma means the
adjointness proof below never has to unfold the nested `WithLp` wrappers. -/
@[simp] theorem outer_apply {m n : Nat} (δ : Vec m) (x : Vec n) (i : Fin m) (j : Fin n) :
    outer (m := m) (n := n) δ x i j = δ.ofLp i * x.ofLp j := by
  simp [outer]

/-- Coordinate formula for the Frobenius/$\ell_2$ inner product on `Mat m n`. -/
theorem inner_mat_eq_sum {m n : Nat} (A B : Mat m n) :
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
theorem inner_matApply_eq {m n : Nat} (x : Vec n) (dW : Mat m n) (δ : Vec m) :
    inner ℝ ((matApplyLin (m := m) (n := n) x) dW) δ
      =
    inner ℝ dW (outer (m := m) (n := n) δ x) := by
  classical
  rw [inner_eq_sum_mul, inner_mat_eq_sum]
  change (∑ i : Fin m, (∑ j : Fin n, dW i j * x.ofLp j) * δ.ofLp i) =
    ∑ i : Fin m, ∑ j : Fin n, dW i j * (δ.ofLp i * x.ofLp j)
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  ring

/--
Adjoint of `W ↦ W x` under Frobenius/$\ell_2$ inner products.

This is the mathematical core of the “weight gradient is outer product” rule:
`(matApplyLin x)† δ = δ ⊗ x`.
-/
theorem matApplyLin_adjoint_apply {m n : Nat} (x : Vec n) (δ : Vec m) :
    (matApplyLin (m := m) (n := n) x).adjoint δ = outer (m := m) (n := n) δ x := by
  apply ext_inner_left ℝ
  intro dW
  rw [ContinuousLinearMap.adjoint_inner_right]
  exact inner_matApply_eq x dW δ

end
end Autograd
end Proofs
