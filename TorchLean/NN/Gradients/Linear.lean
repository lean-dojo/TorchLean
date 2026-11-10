/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Contract

/-!
# Spec-level gradient identities for the linear layer

For the linear layer `y = W x + b` with upstream gradient `δ = ∂L/∂y`, TorchLean's derivative
specification uses these tensor formulas:

- `∂L/∂W = δ ⊗ x` (outer product),
- `∂L/∂x = Wᵀ δ` (matrix-vector multiply), and
- `∂L/∂b = δ`.

The lemmas here put the specification into these forms. That a runtime backward pass is the
Fréchet derivative of the forward pass is proved under `TorchLean/NN`.

## PyTorch correspondence / citations

- `torch.nn.Linear` / `torch.nn.functional.linear` implement `y = x Wᵀ + b` with weight stored as
  shape `(out_features, in_features)` (so the math “matrix” is `W` with output rows). TorchLean’s
  `Linear` follows the same convention: `weights : Tensor α [outDim, inDim]`.
  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
- The “outer product” view of the weight gradient corresponds to the common vector formula
  `grad_W = δ ⊗ x` (PyTorch has `torch.outer` for vectors).
  https://pytorch.org/docs/stable/generated/torch.outer.html

## References
- Standard matrix calculus and backpropagation identities. No single source is needed.
-/

@[expose] public section

open TorchLean.Shape (rank_eq_length)

namespace TorchLean

open TorchLean.Tensor

/--
Entry `(i, j)` of the weight gradient `Linear.weightsDeriv x δ` is `δ i * x j`. For
`y = W x + b` and `δ = ∂L/∂y` this is the outer product `∂L/∂W = δ ⊗ x`.

For a batch, PyTorch computes the corresponding formula as a matrix product with the input batch.
-/
theorem Linear.weightsDeriv_apply
  {inDim outDim : Nat}
  (x : Tensor ℝ [inDim])
  (δ : Tensor ℝ [outDim]) (i : Fin outDim) (j : Fin inDim) :
  Tensor.get2 (weightsDeriv x δ) i j = δ.getScalar i * x.getScalar j := by
  rw [Tensor.get2_eq_apply, weightsDeriv, Tensor.zipWith_apply,
    Tensor.broadcast_apply_of_extent_eq _ _ _ (fun k => by
      match k with
      | ⟨0, _⟩ => rfl
      | ⟨k + 1, hk⟩ => exact absurd hk (by simp [Shape.rank_eq_length])),
    Tensor.broadcast_apply_of_extent_eq _ _ _ (fun k => by
      match k with
      | ⟨0, _⟩ => rfl
      | ⟨k + 1, hk⟩ => exact absurd hk (by simp [Shape.rank_eq_length])),
    Tensor.getScalar_eq_apply, Tensor.getScalar_eq_apply]
  rfl

/--
`Linear.inputDeriv W δ`, the input gradient of `y = W x + b`, is the vector-matrix contraction
of `δ` with `W`, that is `∂L/∂x = Wᵀ δ`.
-/
theorem Linear.inputDeriv_eq_contract
  {inDim outDim : Nat}
  (layer : Linear ℝ inDim outDim)
  (δ : Tensor ℝ [outDim]) :
  inputDeriv layer.weights δ =
  Tensor.contract δ layer.weights _ _ [inDim] .vecMat := rfl

/--
The complete linear backward result passes the bias cotangent through unchanged.

For `y = W x + b`, the bias gradient is `∂L/∂b = δ`.
-/
theorem Linear.backward_biasGradient
  {inDim outDim : Nat}
  (layer : Linear ℝ inDim outDim)
  (x : Tensor ℝ [inDim])
  (δ : Tensor ℝ [outDim]) :
  (backward layer x δ).parameters.bias = δ := rfl

end TorchLean
