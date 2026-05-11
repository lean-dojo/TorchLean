/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Linear layer (spec layer)

This file defines a fully‑connected layer and its gradients:

- forward: `y = W x + b`
- backward: ∂L/∂W, ∂L/∂b, ∂L/∂x

Definitions are purely functional and shape‑indexed, suitable for both proofs and reuse by
autograd wrappers in `NN/Spec/Autograd`.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Add α] [Mul α] [Zero α]

/--
Linear layer specification (pure, shape-indexed).

This is the spec-level analogue of PyTorch `torch.nn.Linear` / `torch.nn.functional.linear`:
- `weights` has shape `[outDim, inDim]`,
- `bias` has shape `[outDim]`.
-/
structure LinearSpec (α : Type) (inDim outDim : Nat) where
  /-- weights. -/
  weights : Tensor α (.dim outDim (.dim inDim .scalar))
  /-- bias. -/
  bias    : Tensor α (.dim outDim .scalar)

/--
Unbatched forward pass: `y = W x + b`.

PyTorch analogue: `torch.nn.functional.linear`.
-/
def linearSpec {inDim outDim : Nat}
  (m : LinearSpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar)) :
  Tensor α (.dim outDim .scalar) :=
  addSpec (matVecMulSpec m.weights input) m.bias

/--
Batched forward pass (map the unbatched `linear_spec` over the batch axis).

Input shape:  `[batch, inDim]`
Output shape: `[batch, outDim]`

PyTorch analogue: applying `nn.Linear` to a batched tensor.
-/
def linearBatchedSpec {batch inDim outDim : Nat}
  (m : LinearSpec α inDim outDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar))) :
  Tensor α (.dim batch (.dim outDim .scalar)) :=
  match input with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => linearSpec m (batch_fn i))

/--
Gradient w.r.t. weights: `∂L/∂W = (∂L/∂y) ⊗ x` (outer product).

This is the standard linear-layer backward formula for `y = W x + b`.
-/
def linearWeightsDerivSpec {inDim outDim : Nat}
  (input : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim outDim (.dim inDim .scalar)) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      match grad_output, input with
      | Tensor.dim g_vals, Tensor.dim x_vals =>
        match g_vals i, x_vals j with
        | Tensor.scalar g, Tensor.scalar x => Tensor.scalar (g * x)
    ))

/--
Gradient w.r.t. bias: `∂L/∂b = ∂L/∂y`.

Since `y = W x + b`, the Jacobian of `y` w.r.t. `b` is the identity.
-/
def linearBiasDerivSpec {inDim outDim : Nat}
  (_dW : Tensor α (.dim outDim (.dim inDim .scalar)))
  (grad_output : Tensor α (.dim outDim .scalar))
  (_input : Tensor α (.dim inDim .scalar)) :
  Tensor α (.dim outDim .scalar) := grad_output

/--
Gradient w.r.t. input: `∂L/∂x = Wᵀ (∂L/∂y)`.

This is the standard "matmul by the transpose" rule for `y = W x + b`.
-/
def linearInputDerivSpec {inDim outDim : Nat}
  (weights : Tensor α (.dim outDim (.dim inDim .scalar)))
  (grad_output : Tensor α (.dim outDim .scalar)) :
  Tensor α (.dim inDim .scalar) :=
  vecMatMulSpec grad_output weights

/--
Batched derivatives `(∂L/∂W, ∂L/∂b, ∂L/∂x)` for a batch of size `batch + 1`.

This is a convenience wrapper that uses matrix operations to compute:
- `d_weights = (grad_outputᵀ) · input`,
- `d_bias = sum(grad_output)` over the batch axis,
- `d_input = grad_output · weights`.
-/
def batchLinearDerivSpec {batch inDim outDim : Nat}
  (weights : Tensor α (.dim outDim (.dim inDim .scalar)))
  (input : Tensor α (.dim (batch + 1) (.dim inDim .scalar)))
  (grad_output : Tensor α (.dim (batch + 1) (.dim outDim .scalar))) :
  (Tensor α (.dim outDim (.dim inDim .scalar)) ×
   Tensor α (.dim outDim .scalar) ×
   Tensor α (.dim (batch + 1) (.dim inDim .scalar))) :=

  let d_weights := matMulSpec (matrixTransposeSpec grad_output) input
  let d_bias := reduceSumAuto 0 grad_output
  let d_input := matMulSpec grad_output weights
  (d_weights, d_bias, d_input)

/--
Complete unbatched backward pass for a linear layer.

Returns `(∂L/∂W, ∂L/∂b, ∂L/∂x)` given the layer params, input `x`, and output gradient `∂L/∂y`.
-/
def linearBackwardSpec {inDim outDim : Nat}
  (layer : LinearSpec α inDim outDim)
  (input : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α (.dim outDim .scalar)) :
  (Tensor α (.dim outDim (.dim inDim .scalar)) ×
   Tensor α (.dim outDim .scalar) ×
   Tensor α (.dim inDim .scalar)) :=
  let d_weights := linearWeightsDerivSpec input grad_output
  let d_bias := linearBiasDerivSpec d_weights grad_output input
  let d_input := linearInputDerivSpec layer.weights grad_output
  (d_weights, d_bias, d_input)

/--
Accumulate two weight gradients by addition.

This is a small helper used by batching/training code.
-/
def linearGradientAccumulateSpec {inDim outDim : Nat}
  (grad1 : Tensor α (.dim outDim (.dim inDim .scalar)))
  (grad2 : Tensor α (.dim outDim (.dim inDim .scalar))) :
  Tensor α (.dim outDim (.dim inDim .scalar)) :=
  addSpec grad1 grad2

/-- Scale a weight gradient by a scalar factor (e.g. learning-rate adjustment). -/
def linearGradientScaleSpec {inDim outDim : Nat}
  (grad : Tensor α (.dim outDim (.dim inDim .scalar)))
  (scale_factor : α) :
  Tensor α (.dim outDim (.dim inDim .scalar)) :=
  scaleSpec grad scale_factor

end Spec
