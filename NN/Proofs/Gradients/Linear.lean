/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Linear

/-!
# Spec-level gradient identities for the linear layer

This file records the tensor formulas used by the derivative specification of a linear layer:

`y = W x + b`

namely:
- `∂L/∂W = δ ⊗ x` (outer product),
- `∂L/∂x = Wᵀ δ` (matrix-vector multiply), and
- `∂L/∂b = δ`.

These are normalization lemmas for TorchLean's derivative specification. They do not prove that a
runtime backward pass is the Fréchet derivative of the forward pass; those results live under
`NN.Proofs.Autograd`.

## PyTorch correspondence / citations

- `torch.nn.Linear` / `torch.nn.functional.linear` implement `y = x Wᵀ + b` with weight stored as
  shape `(out_features, in_features)` (so the math “matrix” is `W` with output rows). TorchLean’s
  `LinearSpec` follows the same convention: `weights : Tensor α [outDim, inDim]`.
  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
- The “outer product” view of the weight gradient corresponds to the common vector formula
  `grad_W = δ ⊗ x` (PyTorch has `torch.outer` for vectors).
  https://pytorch.org/docs/stable/generated/torch.outer.html

## References
- Standard matrix calculus / backpropagation identities; no single source is required.
-/

@[expose] public section


namespace Proofs

open Spec TorchLean
open TorchLean TorchLean.Tensor

/--
Spec identity: weight gradient for a linear layer.

For `y = W x + b`, if `δ = ∂L/∂y` then the weight gradient is

`∂L/∂W = δ ⊗ x`.

For a batch, PyTorch evaluates the corresponding formula as a matrix multiplication against the
input batch.
-/
theorem linearWeightsDerivSpec_eq_outerProductSpec
  {inDim outDim : Nat}
  (x : Tensor ℝ [inDim])
  (δ : Tensor ℝ [outDim]) :
  Spec.linearWeightsDerivSpec x δ = outerProductSpec δ x := rfl

/--
Spec identity: input gradient for a linear layer.

For `y = W x + b`, the input gradient is

`∂L/∂x = Wᵀ δ`.
-/
theorem linearInputDerivSpec_eq_vecMatMulSpec
  {inDim outDim : Nat}
  (layer : Spec.LinearSpec ℝ inDim outDim)
  (δ : Tensor ℝ [outDim]) :
  Spec.linearInputDerivSpec layer.weights δ =
  vecMatMulSpec δ layer.weights := rfl

/--
Spec identity: bias gradient for a linear layer.

For `y = W x + b`, the bias gradient is `∂L/∂b = δ`.
-/
theorem linearBiasDerivSpec_eq
  {inDim outDim : Nat}
  (x : Tensor ℝ [inDim])
  (δ : Tensor ℝ [outDim]) :
  Spec.linearBiasDerivSpec
      (Tensor.default : Tensor ℝ [outDim, inDim]) δ x = δ := rfl

end Proofs
