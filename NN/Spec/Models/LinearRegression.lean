/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.AutogradSpec
public import NN.Spec.Core.Sequence

/-!
# Linear regression (spec model)

Defines linear regression as a dot product plus bias (one output):

`y = wᵀ x + b`

We aim to stay close to PyTorch's mental model:

- `torch.nn.Linear(in_features, out_features=1)` for the forward pass,
- `torch.nn.functional.mse_loss(..., reduction="mean")` for the MSE objective,
- an SGD-style parameter update step (as in `torch.optim.SGD`) for training.

This file is a *spec*: it states the math (forward + VJPs) with shapes tracked by the type system.
It prioritizes clarity and explicit derivatives over performance, and it does not include the
closed-form normal-equations solution.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-- Parameters for a single-output linear regression model.

PyTorch analogy: the `weights` and `bias` fields correspond to `nn.Linear(inDim, 1).weight` and
`nn.Linear(inDim, 1).bias`, but with shapes tracked in the tensor type.
-/
structure LinearRegressionSpec (α : Type) (inDim : Nat) where
  /-- weights. -/
  weights : Tensor α (.dim inDim .scalar)
  /-- bias. -/
  bias : Tensor α .scalar

/-- Forward pass for linear regression: `y = wᵀ x + b`. -/
def linearRegressionForwardSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim inDim .scalar)) :
  Tensor α .scalar :=
  let dot_product := dotSpec model.weights input
  addSpec (Tensor.scalar dot_product) model.bias

/-- Batched forward pass, applied independently to each input row. -/
def linearRegressionBatchedForwardSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar))) :
  Tensor α (.dim batch .scalar) :=
  match input with
  | Tensor.dim batch_fn =>
    Tensor.dim (fun i => linearRegressionForwardSpec model (batch_fn i))

/-- VJP contribution for `weights`: `dL/dw = x * (dL/dy)` (scalar-times-vector scaling). -/
def linearRegressionWeightsDerivSpec {inDim : Nat}
  (input : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α .scalar) :
  Tensor α (.dim inDim .scalar) :=
  scaleSpec input (Tensor.toScalar grad_output)

/-- VJP contribution for `bias`: `dL/db = dL/dy`. -/
def linearRegressionBiasDerivSpec {inDim : Nat}
  (_weights : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α .scalar)
  (_input : Tensor α (.dim inDim .scalar)) :
  Tensor α .scalar := grad_output

/-- VJP contribution for `input`: `dL/dx = w * (dL/dy)`. -/
def linearRegressionInputDerivSpec {inDim : Nat}
  (weights : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α .scalar) :
  Tensor α (.dim inDim .scalar) :=
  scaleSpec weights (Tensor.toScalar grad_output)

/-- Full backward pass returning `(dW, db, dX)` in that order. -/
def linearRegressionBackwardSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim inDim .scalar))
  (grad_output : Tensor α .scalar) :
  (Tensor α (.dim inDim .scalar) × Tensor α .scalar × Tensor α (.dim inDim .scalar)) :=
  let dW := linearRegressionWeightsDerivSpec input grad_output
  let db := linearRegressionBiasDerivSpec model.weights grad_output input
  let dX := linearRegressionInputDerivSpec model.weights grad_output
  (dW, db, dX)

/-- Batched backward pass.

This aggregates parameter gradients across the batch (a sum over `batch`), matching PyTorch's
default behavior for loss reductions like `"mean"` when you subsequently scale appropriately.
-/
def linearRegressionBatchedBackwardSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar)))
  (grad_output : Tensor α (.dim batch .scalar)) (h : batch ≠ 0) :
  (Tensor α (.dim inDim .scalar) × Tensor α .scalar × Tensor α (.dim batch (.dim inDim .scalar))) :=
  -- Gradient w.r.t. weights: sum over batch dimension
  let dW := reduceSumVec 0
    (map2SequenceVecScalarSpec (.dim inDim .scalar)
      (fun x gy => scaleSpec x (Tensor.toScalar gy))
      input grad_output) (by rfl)
  -- Gradient w.r.t. bias: sum over batch dimension
  let _ : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) :=
    Shape.validAxisInstZeroAlt h
  let db := reduceSumAuto 0 grad_output
  -- Gradient w.r.t. input: broadcast weights to each batch element
  let dX := mapSequenceVecScalarSpec
    (fun gy => scaleSpec model.weights (Tensor.toScalar gy))
    grad_output

  (dW, db, dX)

/-- Mean Squared Error loss (MSE).

PyTorch analogy: `F.mse_loss(predictions, target, reduction="mean")`.

Note: the `batch ≠ 0` hypothesis avoids dividing by zero.
-/
def mseLossSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar)))
  (target : Tensor α (.dim batch .scalar)) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := linearRegressionBatchedForwardSpec model input
  let errors := subSpec predictions target
  let squared_errors := squareSpec errors
  have _ : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) :=
              Shape.validAxisInstZeroAlt h
  let mse := reduceSumAuto 0 squared_errors
  scaleSpec mse (1 / (batch : α))

/-- Gradient of MSE w.r.t. predictions: `d/dy (mean (y - t)^2) = (2/batch) * (y - t)`.

This is only meaningful when `batch > 0` (callers typically already carry `batch ≠ 0`).
-/
def mseLossGradSpec {batch : Nat}
  (predictions : Tensor α (.dim batch .scalar))
  (target : Tensor α (.dim batch .scalar)) :
  Tensor α (.dim batch .scalar) :=
  let errors := subSpec predictions target
  scaleSpec errors (Numbers.two / (batch : α))

/-- One gradient-descent training step for linear regression. -/
def linearRegressionTrainStepSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (learning_rate : α) (h : batch ≠ 0) :
  (Tensor α .scalar × LinearRegressionSpec α inDim) :=
  -- Forward pass
  let predictions := linearRegressionBatchedForwardSpec model input
  -- Compute loss
  let loss := mseLossSpec model input target h
  -- Compute gradients
  let grad_predictions := mseLossGradSpec predictions target
  let (dW, db, _dX) := linearRegressionBatchedBackwardSpec model input grad_predictions h
  -- Update parameters
  let new_weights := subSpec model.weights (scaleSpec dW learning_rate)
  let new_bias := subSpec model.bias (scaleSpec db learning_rate)
  let updated_model := { model with weights := new_weights, bias := new_bias }
  (loss, updated_model)

/-- `OpSpec` wrapper for linear regression.

This is useful when composing the op in a spec-level AD development.
-/
def linearRegressionOpSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim) :
  OpSpec α (.dim inDim .scalar) .scalar :=
{
  forward := fun x => linearRegressionForwardSpec model x,
  backward := fun x dLdy =>
    let (_, _, dX) := linearRegressionBackwardSpec model x dLdy
    dX
}

/-- R-squared (coefficient of determination) for model evaluation.

PyTorch analogy: there is no single built-in for R² in core PyTorch; this matches the standard
definition `1 - SS_res / SS_tot`.

Note: if `SS_tot = 0` (targets are constant), this divides by zero. Many libraries treat that
as a special case; this spec keeps the plain formula.
-/
def rSquaredSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar)))
  (target : Tensor α (.dim batch .scalar)) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := linearRegressionBatchedForwardSpec model input
  let axis0 : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) :=
    Shape.validAxisInstZeroAlt h
  let target_mean := reduceMeanAuto 0 axis0 target
  let target_mean_broadcast := broadcastLike target target_mean
  let ss_res := reduceSumAuto 0 (squareSpec (subSpec predictions target))
  let ss_tot := reduceSumAuto 0 (squareSpec (subSpec target target_mean_broadcast))
  subSpec (Tensor.scalar 1) (divSpec ss_res ss_tot)

/-- Ridge regression forward pass.

Regularization changes the *objective*, not the raw prediction function, so the forward pass is
identical to ordinary linear regression.

Reference: Hoerl and Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems"
(1970). https://doi.org/10.1080/00401706.1970.10488634
-/
def ridgeRegressionForwardSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim inDim .scalar))
  (_lambda : α) :
  Tensor α .scalar :=
  linearRegressionForwardSpec model input

/-- Ridge loss: MSE plus `lambda * ||w||_2^2`. -/
def ridgeLossSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (lambda : α) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let mse := mseLossSpec model input target h
  let l2_penalty := scaleSpec (Tensor.scalar (dotSpec model.weights model.weights)) lambda
  addSpec mse l2_penalty

/-- Ridge gradient w.r.t. weights.

This is the usual batched gradient plus the derivative of `lambda * ||w||_2^2`, which contributes
`2 * lambda * w`.
-/
def ridgeWeightsDerivSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar)))
  (grad_output : Tensor α (.dim batch .scalar))
  (lambda : α) (h : batch ≠ 0) :
  Tensor α (.dim inDim .scalar) :=
  let mse_grad :=
    reduceSumVec 0
      (map2SequenceVecScalarSpec (.dim inDim .scalar)
        (fun x gy => scaleSpec x (Tensor.toScalar gy))
        input grad_output) (by rfl)
  let _ : Shape.valid_axis_inst 0 (Shape.dim batch Shape.scalar) :=
    Shape.validAxisInstZeroAlt h
  let l2_grad := scaleSpec model.weights (Numbers.two * lambda)
  addSpec mse_grad l2_grad

/-- Soft-thresholding operator (often written `S_λ`), used in proximal-gradient updates for L1.

Reference: Tibshirani, "Regression Shrinkage and Selection via the Lasso" (1996).
https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
-/
def lassoSoftThresholdSpec {inDim : Nat}
  (weights : Tensor α (.dim inDim .scalar))
  (threshold : α) :
  Tensor α (.dim inDim .scalar) :=
  mapSpec (fun w =>
    if w > threshold then w - threshold
    else if (-threshold) > w then w + threshold
    else 0) weights

/-- Lasso forward pass (same raw prediction function as ordinary linear regression). -/
def lassoRegressionForwardSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim inDim .scalar))
  (_lambda : α) :
  Tensor α .scalar :=
  linearRegressionForwardSpec model input

/-- Lasso loss: MSE plus `lambda * ||w||_1`. -/
def lassoLossSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (lambda : α) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let mse := mseLossSpec model input target h
  let l1_penalty := scaleSpec (Tensor.scalar (sumSpec (absSpec model.weights))) lambda
  addSpec mse l1_penalty

/-- Elastic net loss: a convex combination of L1 and L2 penalties.

Reference: Zou and Hastie, "Regularization and Variable Selection via the Elastic Net" (2005).
https://doi.org/10.1111/j.1467-9868.2005.00503.x
-/
def elasticNetLossSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α (.dim batch (.dim inDim .scalar)))
  (target : Tensor α (.dim batch .scalar))
  (l1_ratio : α)
  (alpha : α) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let mse := mseLossSpec model input target h
  let l1_penalty := scaleSpec (Tensor.scalar (sumSpec (absSpec model.weights))) (alpha *
    l1_ratio)
  let l2_penalty := scaleSpec (Tensor.scalar (dotSpec model.weights model.weights)) (alpha * (1 -
    l1_ratio))
  addSpec mse (addSpec l1_penalty l2_penalty)

/-!
## Polynomial features

Polynomial regression can be expressed as linear regression on a fixed feature expansion
`φ(x) = [x, x^2, ..., x^degree]` (per input coordinate). We keep this as a lightweight helper,
then reuse `linear_regression_forward_spec` on the expanded input.
-/

/-- Expand a length-`inDim` input vector into polynomial features up to `degree`.

This expansion does not include a constant feature (the model bias already plays that role).
-/
def polynomialFeaturesSpec {inDim : Nat} (degree : Nat)
  (input : Tensor α (.dim inDim .scalar)) :
  Tensor α (.dim (inDim * degree) .scalar) :=
  match input with
  | Tensor.dim values =>
    let rec expand (remaining : Nat) (acc : List α) : List α :=
      match remaining with
      | 0 => acc
      | Nat.succ d' =>
        let current_degree := degree - d'
        let new_features := (List.finRange inDim).map (fun i =>
          match values i with
          | Tensor.scalar x => x ^ (current_degree : α))
        expand d' (acc ++ new_features)
    let features := expand degree []
    Tensor.dim (fun i => Tensor.scalar (features.getD i.val 0))

/-- Forward pass for polynomial regression: expand features, then run linear regression. -/
def polynomialRegressionForwardSpec {inDim degree : Nat}
  (model : LinearRegressionSpec α (inDim * degree))
  (input : Tensor α (.dim inDim .scalar)) :
  Tensor α .scalar :=
  let expanded_input := polynomialFeaturesSpec degree input
  linearRegressionForwardSpec model expanded_input

end Spec
