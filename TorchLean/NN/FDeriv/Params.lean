/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv
public import TorchLean.NN.Gradients.Linear
public import TorchLean.Tensor.LinearAlgebra.Real
public import TorchLean.Tensor.Euclidean

/-!
# Weight gradients as adjoints

A weight matrix `W : Tensor ℝ [m, n]` is a point of the real inner product space of tensors, and
its inner product is the Frobenius one (`TorchLean.Tensor.inner_mat_eq_sum`). For a fixed input
`x`, the map `W ↦ W x` (`TorchLean.Vec.matApplyLin x`) has the adjoint `δ ↦ δ ⊗ x`, and that outer
product is the weight gradient `TorchLean.Linear.weightsDeriv`
(`TorchLean.Vec.matApplyLin_adjoint_apply`). `TorchLean.NN.FDeriv.MLP` uses this identity to show
that the weight gradients of backprop are adjoints of `fderiv`.

## PyTorch correspondence / citations
For a linear layer `y = W x + b`, PyTorch's backward returns `∂L/∂W = δ ⊗ x`, the outer product of
the upstream gradient and the input, and `∂L/∂x = Wᵀ δ`.
https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- The matrix view `Rep.matrixEquiv` of a sum of real matrix tensors is the sum of their matrix
views. -/
theorem Tensor.matrixEquiv_add {m n : Nat} (W1 W2 : Tensor ℝ [m, n]) :
    Rep.matrixEquiv ℝ m n (W1 + W2) = Rep.matrixEquiv ℝ m n W1 + Rep.matrixEquiv ℝ m n W2 := by
  ext i j
  simp only [Rep.matrixEquiv_apply, Matrix.add_apply, Tensor.add_apply]

/-- The matrix view `Rep.matrixEquiv` of a scaled real matrix tensor is the scaled matrix view. -/
theorem Tensor.matrixEquiv_smul {m n : Nat} (a : ℝ) (W : Tensor ℝ [m, n]) :
    Rep.matrixEquiv ℝ m n (a • W) = a • Rep.matrixEquiv ℝ m n W := by
  ext i j
  simp only [Rep.matrixEquiv_apply, Rep.smul_apply, Matrix.smul_apply]

/-- The map `W ↦ W x` on weight matrices `Tensor ℝ [m, n]`, as a continuous linear map into
`Vec m`. -/
def Vec.matApplyLin {m n : Nat} (x : Vec n) : Tensor ℝ [m, n] →L[ℝ] Vec m :=
  let L : Tensor ℝ [m, n] →ₗ[ℝ] Vec m :=
    { toFun := fun W => Matrix.toEuclideanLin (Rep.matrixEquiv ℝ m n W) x
      map_add' := by
        intro W1 W2
        rw [Tensor.matrixEquiv_add, map_add, LinearMap.add_apply]
      map_smul' := by
        intro a W
        rw [Tensor.matrixEquiv_smul, map_smul, LinearMap.smul_apply, RingHom.id_apply] }
  ⟨L, L.continuous_of_finiteDimensional⟩

/-- The Frobenius inner product on `Tensor ℝ [m, n]` is `∑ i, ∑ j, A i j * B i j`, read with
`get2`. -/
theorem Tensor.inner_mat_eq_sum {m n : Nat} (A B : Tensor ℝ [m, n]) :
    inner ℝ A B = ∑ i : Fin m, ∑ j : Fin n, get2 A i j * get2 B i j := by
  rw [← Tensor.dot_eq_inner, Tensor.dot_mat_eq_sum]

/-- `⟪dW x, δ⟫ = ⟪dW, δ ⊗ x⟫`, where the outer product `δ ⊗ x` is the weight gradient
`Linear.weightsDeriv (ofFnE x) (ofFnE δ)`. -/
theorem Vec.inner_matApplyLin_eq {m n : Nat} (x : Vec n) (dW : Tensor ℝ [m, n]) (δ : Vec m) :
    inner ℝ ((matApplyLin (m := m) (n := n) x) dW) δ
      =
    inner ℝ dW (Linear.weightsDeriv (ofFnE x) (ofFnE δ)) := by
  classical
  rw [inner_eq_sum_mul, Tensor.inner_mat_eq_sum]
  simp only [Linear.weightsDeriv_apply, getScalar_ofFnE]
  simp only [get2_eq_apply]
  change (∑ i : Fin m, (∑ j : Fin n, dW (i, j, PUnit.unit) * x.ofLp j) * δ.ofLp i) =
    ∑ i : Fin m, ∑ j : Fin n, dW (i, j, PUnit.unit) * (δ.ofLp i * x.ofLp j)
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- The adjoint of `W ↦ W x` for the Frobenius inner product sends `δ` to the outer product
`δ ⊗ x`, the weight gradient `Linear.weightsDeriv (ofFnE x) (ofFnE δ)`. -/
theorem Vec.matApplyLin_adjoint_apply {m n : Nat} (x : Vec n) (δ : Vec m) :
    (matApplyLin (m := m) (n := n) x).adjoint δ
      =
    Linear.weightsDeriv (ofFnE x) (ofFnE δ) := by
  apply ext_inner_left ℝ
  intro dW
  rw [ContinuousLinearMap.adjoint_inner_right]
  exact inner_matApplyLin_eq x dW δ

end
end TorchLean
