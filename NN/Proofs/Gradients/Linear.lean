/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Linear

/-!
# Spec-level gradient identities for the linear layer

This file states (and in several cases, proves by definitional unfolding) the “obvious” gradient
formulas for a linear layer:

`y = W x + b`

namely:
- `∂L/∂W = δ ⊗ x` (outer product),
- `∂L/∂x = Wᵀ δ` (matrix-vector multiply), and
- `∂L/∂b = δ`.

## What these theorems are (and are not)
- These are **spec-level** identities over TorchLean’s tensor encodings, not a full calculus layer
  about Frechét derivatives.
- Several proofs are `rfl` after unfolding definitions, because the corresponding specs are
  implemented directly in that form.

## PyTorch correspondence / citations

- `torch.nn.Linear` / `torch.nn.functional.linear` implement `y = x Wᵀ + b` with weight stored as
  shape `(out_features, in_features)` (so the math “matrix” is `W` with output rows). TorchLean’s
  `LinearSpec` follows the same convention: `weights : Tensor α (.dim outDim (.dim inDim .scalar))`.
  https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
- The “outer product” view of the weight gradient corresponds to the common vector formula
  `grad_W = δ ⊗ x` (PyTorch has `torch.outer` for vectors).
  https://pytorch.org/docs/stable/generated/torch.outer.html

## Why keep this file
Even when proofs are definitional, having them recorded explicitly helps:
- document the intended math semantics of the “backward specs”,
- provide simple regression checks when refactoring tensor encodings, and
- serve as stepping stones for the more advanced autograd soundness proofs in
  `NN/Proofs/Autograd/*`.

## References
- Standard matrix calculus / backpropagation identities; no single source is required.
-/

@[expose] public section


namespace Proofs

open Spec
open Tensor

/--
Spec identity: weight gradient for a linear layer.

For `y = W x + b`, if `δ = ∂L/∂y` then the weight gradient is

`∂L/∂W = δ ⊗ x`.

PyTorch mental model: this is the per-sample formula whose batched version becomes a matmul
against the input batch.
-/
theorem linear_weight_gradient_correct
  {inDim outDim : Nat}
  (x : Tensor ℝ (.dim inDim .scalar))
  (δ : Tensor ℝ (.dim outDim .scalar)) :
  Spec.linearWeightsDerivSpec x δ = outerProductSpec δ x := by
  -- Unfold definitions
  unfold Spec.linearWeightsDerivSpec outerProductSpec
  -- Split δ and x into their dim constructors
  cases δ with | dim δ_vals =>
  cases x with | dim x_vals =>
  rfl

/--
Spec identity: input gradient for a linear layer.

For `y = W x + b`, the input gradient is

`∂L/∂x = Wᵀ δ`.
-/
theorem linear_input_gradient_correct
  {inDim outDim : Nat}
  (layer : Spec.LinearSpec ℝ inDim outDim)
  (δ : Tensor ℝ (.dim outDim .scalar)) :
  Spec.linearInputDerivSpec layer.weights δ =
  vecMatMulSpec δ layer.weights := by
  -- This follows directly from the definition
  rfl

/--
Spec identity: bias gradient for a linear layer.

For `y = W x + b`, the bias gradient is `∂L/∂b = δ`.
-/
theorem linear_bias_gradient_correct
  {inDim outDim : Nat}
  (x : Tensor ℝ (.dim inDim .scalar))
  (δ : Tensor ℝ (.dim outDim .scalar)) :
  Spec.linearBiasDerivSpec (Inhabited.default) δ x = δ := by
  -- This follows directly from the definition
  rfl

/--
Shape/typing sanity check: all backward specs return tensors of the expected shapes.

This is “free” from Lean’s dependent types, but it is sometimes convenient as a lemma when writing
documentation-style proofs.
-/
theorem linear_gradients_preserve_shapes
  {inDim outDim : Nat}
  (layer : Spec.LinearSpec ℝ inDim outDim)
  (x : Tensor ℝ (.dim inDim .scalar))
  (δ : Tensor ℝ (.dim outDim .scalar)) :
  ∃ (grad_W : Tensor ℝ (.dim outDim (.dim inDim .scalar)))
    (grad_b : Tensor ℝ (.dim outDim .scalar))
    (grad_x : Tensor ℝ (.dim inDim .scalar)),
    Spec.linearWeightsDerivSpec x δ = grad_W ∧
    Spec.linearBiasDerivSpec (Inhabited.default) δ x = grad_b ∧
    Spec.linearInputDerivSpec layer.weights δ = grad_x := by
  refine ⟨Spec.linearWeightsDerivSpec x δ,
    Spec.linearBiasDerivSpec (Inhabited.default) δ x,
    Spec.linearInputDerivSpec layer.weights δ, ?_, ?_, ?_⟩ <;> rfl

/--
Mathematical correctness theorem: Linear layer gradients satisfy the chain rule.
This formalizes the core mathematical property validating backward implementation.
-/
theorem linear_gradients_mathematical_correctness
  {inDim outDim : Nat}
  (layer : Spec.LinearSpec ℝ inDim outDim)
  (x : Tensor ℝ (.dim inDim .scalar))
  (δ : Tensor ℝ (.dim outDim .scalar)) :
  -- For linear function f(x) = Wx + b with gradient δ = ∂L/∂f(x):
  -- Chain rule gives: ∂L/∂x = W^T · δ, ∂L/∂W = δ ⊗ x, ∂L/∂b = δ
  let grad_x := Spec.linearInputDerivSpec layer.weights δ
  let grad_W := Spec.linearWeightsDerivSpec x δ
  let grad_b := Spec.linearBiasDerivSpec (Inhabited.default) δ x
  -- The gradients correctly implement the mathematical chain rule
  (grad_x = vecMatMulSpec δ layer.weights) ∧
  (grad_W = outerProductSpec δ x) ∧
  (grad_b = δ) := by
  constructor
  · rfl
  constructor
  · simpa using (linear_weight_gradient_correct (x := x) (δ := δ))
  · rfl

end Proofs
