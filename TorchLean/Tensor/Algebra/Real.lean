/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Real
public import TorchLean.Tensor.Contract

/-!
# Algebraic identities for real tensors

We prove that matrix-vector multiplication over `ℝ` is linear in the vector argument: it sends
`x + y` to the sum of the two products and `Tensor.scale x c` to the scaled product. It therefore
sends a linear combination of two vectors to the same combination of their products. A left fold
of scalar tensors under `+` equals the scalar tensor of the folded values.
-/

@[expose] public section

namespace TorchLean.Tensor

open scoped BigOperators

/-- Linearity of matrix-vector multiplication in the vector argument (addition). -/
theorem contract_matVec_add {m n : Nat} (W : Tensor ℝ [m, n]) (x y : Tensor ℝ [n]) :
    Tensor.contract W (x + y) _ _ [m] .matVec = Tensor.contract W x _ _ [m] .matVec +
        Tensor.contract W y _ _ [m] .matVec := by
  apply Tensor.ext_vector
  intro i
  simp only [getScalar_contract_matVec_eq_sum, getScalar_add, mul_add,
    Finset.sum_add_distrib]

/-- Linearity of matrix-vector multiplication in the vector argument (scaling). -/
theorem contract_matVec_scale {m n : Nat} (W : Tensor ℝ [m, n]) (x : Tensor ℝ [n]) (c : ℝ) :
    Tensor.contract W (Tensor.scale x c) _ _ [m] .matVec = Tensor.scale
        (Tensor.contract W x _ _ [m] .matVec) c := by
  apply Tensor.ext_vector
  intro i
  simp only [getScalar_contract_matVec_eq_sum, getScalar_scale,
    Finset.sum_mul, mul_assoc]

/-- Full linearity of matrix-vector multiplication in the vector argument. -/
theorem contract_matVec_combination {m n : Nat} (W : Tensor ℝ [m, n]) (x y : Tensor ℝ [n])
    (a b : ℝ) :
    Tensor.contract W (Tensor.scale x a + Tensor.scale y b) _ _ [m] .matVec =
      Tensor.scale (Tensor.contract W x _ _ [m] .matVec) a + (Tensor.scale
          (Tensor.contract W y _ _ [m] .matVec) b) := by
  rw [contract_matVec_add, contract_matVec_scale, contract_matVec_scale]

/-- Over any scalar type with an addition, adding scalar tensors in a left fold agrees with
folding their scalar values. -/
theorem foldl_add_scalar {α ι : Type} [Storage α] [Add α]
    (values : ι → Tensor α .scalar) (items : List ι) (initial : α) :
    items.foldl (fun total item => total + values item) (Tensor.scalar initial) =
      Tensor.scalar (items.foldl (fun total item => total + (values item).item) initial) := by
  induction items generalizing initial with
  | nil => rfl
  | cons item rest ih =>
      simp only [List.foldl_cons]
      rw [show Tensor.scalar initial + values item =
        Tensor.scalar (initial + (values item).item) by
          apply Tensor.ext_scalar
          rw [Tensor.item_scalar]
          exact (Tensor.add_apply (Tensor.scalar initial) (values item) PUnit.unit).trans
            (congrArg (· + (values item).item) (Tensor.item_scalar initial))]
      exact ih (initial := initial + (values item).item)

end TorchLean.Tensor
