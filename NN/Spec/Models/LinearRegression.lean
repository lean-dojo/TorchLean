/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Autograd.AutogradSpec
public import NN.Spec.Core.Sequence
public import Mathlib.Algebra.GroupWithZero.Nat
public import NN.Spec.Core.TensorReductionShape.Reductions

/-!
# Linear regression (spec model)

Defines linear regression as a dot product plus bias (one output):

`y = wᵀ x + b`

The corresponding PyTorch operations are:

- `torch.nn.Linear(in_features, out_features=1)` for the forward pass,
- `torch.nn.functional.mse_loss(..., reduction="mean")` for the MSE objective,
- an SGD-style parameter update step (as in `torch.optim.SGD`) for training.

This file is a *spec*: it states the math (forward + VJPs) with shapes tracked by the type system.
It prioritizes clarity and explicit derivatives over performance, and it does not include the
closed-form normal-equations solution.

## Implementation status

No API builder implements this model as a unit (`nn.linear` is the layer, not this model with its
loss and training step). No theorem relates it to runtime code.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Parameters for a single-output linear regression model.

PyTorch analogy: the `weights` and `bias` fields correspond to `nn.Linear(inDim, 1).weight` and
`nn.Linear(inDim, 1).bias`, but with shapes tracked in the tensor type.
-/
structure LinearRegressionSpec (α : Type) [TorchLean.Storage α]
    (inDim : Nat) where
  /-- Regression coefficients, one per input feature. -/
  weights : Tensor α [inDim]
  /-- Scalar intercept term. -/
  bias : Tensor α .scalar

/-- Forward pass for linear regression: `y = wᵀ x + b`. -/
def linearRegressionForwardSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [inDim]) :
  Tensor α .scalar :=
  let dotProduct := dotSpec model.weights input
  addSpec (Tensor.scalar dotProduct) model.bias

/-- Batched forward pass, applied independently to each input row. -/
def linearRegressionBatchedForwardSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [batch, inDim]) :
  Tensor α [batch] :=
  Tensor.dim (fun i => linearRegressionForwardSpec model (Tensor.unstack input i))

/-- VJP contribution for `weights`: `dL/dw = x * (dL/dy)` (scalar-times-vector scaling). -/
def linearRegressionWeightsDerivSpec {inDim : Nat}
  (input : Tensor α [inDim])
  (gradOutput : Tensor α .scalar) :
  Tensor α [inDim] :=
  scaleSpec input (Tensor.item gradOutput)

/-- VJP contribution for `bias`: `dL/db = dL/dy`. -/
def linearRegressionBiasDerivSpec {inDim : Nat}
  (_weights : Tensor α [inDim])
  (gradOutput : Tensor α .scalar)
  (_input : Tensor α [inDim]) :
  Tensor α .scalar := gradOutput

/-- VJP contribution for `input`: `dL/dx = w * (dL/dy)`. -/
def linearRegressionInputDerivSpec {inDim : Nat}
  (weights : Tensor α [inDim])
  (gradOutput : Tensor α .scalar) :
  Tensor α [inDim] :=
  scaleSpec weights (Tensor.item gradOutput)

/-- Gradients for a linear regression model.

`inputShape` is a parameter so that the unbatched and batched backward passes return the same
record: the parameter gradients have the same shape either way, and only the input gradient grows a
leading batch axis. -/
structure LinearRegressionGradients (α : Type) [TorchLean.Storage α] (inDim : Nat)
    (inputShape : Shape) where
  /-- Gradient with respect to the weight vector. -/
  weightGradient : Tensor α [inDim]
  /-- Gradient with respect to the scalar bias. -/
  biasGradient : Tensor α .scalar
  /-- Gradient with respect to the input. -/
  inputGradient : Tensor α inputShape

/-- Full backward pass for one example. -/
def linearRegressionBackwardSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [inDim])
  (gradOutput : Tensor α .scalar) :
  LinearRegressionGradients α inDim [inDim] :=
  { weightGradient := linearRegressionWeightsDerivSpec input gradOutput
    biasGradient := linearRegressionBiasDerivSpec model.weights gradOutput input
    inputGradient := linearRegressionInputDerivSpec model.weights gradOutput }

/-- Batched backward pass.

This aggregates parameter gradients across the batch (a sum over `batch`), matching PyTorch's
default behavior for loss reductions like `"mean"` when you subsequently scale appropriately.
-/
def linearRegressionBatchedBackwardSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [batch, inDim])
  (gradOutput : Tensor α [batch]) (h : batch ≠ 0) :
  LinearRegressionGradients α inDim [batch, inDim] :=
  -- Gradient w.r.t. weights: sum over batch dimension
  let dW := Tensor.reduceSum 0
    (Tensor.zipEach ([batch]) ([inDim])
      (fun x gy => scaleSpec x (Tensor.item gy)) input gradOutput)
    (Shape.hasNonemptyAxisZeroOfNe h).proof
  -- Gradient w.r.t. bias: sum over batch dimension
  let db := Tensor.reduceSum 0 gradOutput (Shape.hasNonemptyAxisZeroOfNe h).proof
  -- Gradient w.r.t. input: broadcast weights to each batch element
  let dX := Tensor.mapLeading ([batch])
    (fun gy => scaleSpec model.weights (Tensor.item gy))
    gradOutput

  { weightGradient := dW, biasGradient := db, inputGradient := dX }

/-- Mean Squared Error loss (MSE).

PyTorch analogy: `F.mse_loss(predictions, target, reduction="mean")`.

Note: the `batch ≠ 0` hypothesis avoids dividing by zero.
-/
def mseLossSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch]) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := linearRegressionBatchedForwardSpec model input
  let errors := subSpec predictions target
  let squaredErrors := squareSpec errors
  let mse := reduceSum 0 squaredErrors (Shape.hasNonemptyAxisZeroOfNe h).proof
  scaleSpec mse (1 / (batch : α))

/-- Gradient of MSE w.r.t. predictions: `d/dy (mean (y - t)^2) = (2/batch) * (y - t)`.

This is only meaningful when `batch > 0` (callers typically already carry `batch ≠ 0`).
-/
def mseLossGradSpec {batch : Nat}
  (predictions : Tensor α [batch])
  (target : Tensor α [batch]) :
  Tensor α [batch] :=
  let errors := subSpec predictions target
  scaleSpec errors (2 / (batch : α))

/-- One gradient-descent training step for linear regression. -/
def linearRegressionTrainStepSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch])
  (learningRate : α) (h : batch ≠ 0) :
  (Tensor α .scalar × LinearRegressionSpec α inDim) :=
  -- Forward pass
  let predictions := linearRegressionBatchedForwardSpec model input
  -- Compute loss
  let loss := mseLossSpec model input target h
  -- Compute gradients
  let gradPredictions := mseLossGradSpec predictions target
  let gradients := linearRegressionBatchedBackwardSpec model input gradPredictions h
  -- Update parameters
  let newWeights :=
    subSpec model.weights (scaleSpec gradients.weightGradient learningRate)
  let newBias := subSpec model.bias (scaleSpec gradients.biasGradient learningRate)
  let updatedModel := { model with weights := newWeights, bias := newBias }
  (loss, updatedModel)

/-- `OpSpec` wrapper for linear regression.

This is useful when composing the op in a spec-level AD development.
-/
def linearRegressionOpSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim) :
  OpSpec α ([inDim]) .scalar :=
{
  forward := fun x => linearRegressionForwardSpec model x,
  backward := fun x dLdy =>
    (linearRegressionBackwardSpec model x dLdy).inputGradient
}

/-- R-squared (coefficient of determination) for model evaluation.

PyTorch analogy: there is no single built-in for R² in core PyTorch; this matches the standard
definition `1 - SS_res / SS_tot`.

Note: if `SS_tot = 0` (targets are constant), this divides by zero. Many libraries treat that
as a special case; this spec keeps the plain formula.
-/
def rSquaredSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch]) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions := linearRegressionBatchedForwardSpec model input
  let leadingAxis : Shape.HasNonemptyAxis 0 (Shape.dim batch Shape.scalar) :=
    Shape.hasNonemptyAxisZeroOfNe h
  let targetMean := reduceMean 0 target leadingAxis.proof
  let targetMeanBroadcast := replicate (shape := [batch]) targetMean
  let ss_res := reduceSum 0 (squareSpec (subSpec predictions target)) leadingAxis.proof
  let ss_tot := reduceSum 0 (squareSpec (subSpec target targetMeanBroadcast)) leadingAxis.proof
  subSpec (Tensor.scalar 1) (divSpec ss_res ss_tot)

/-- Ridge regression forward pass.

Regularization changes the *objective*, not the raw prediction function, so the forward pass is
identical to ordinary linear regression.

Reference: Hoerl and Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems"
(1970). https://doi.org/10.1080/00401706.1970.10488634
-/
def ridgeRegressionForwardSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [inDim])
  (_lambda : α) :
  Tensor α .scalar :=
  linearRegressionForwardSpec model input

/-- Ridge loss: MSE plus `lambda * ||w||_2^2`. -/
def ridgeLossSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch])
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
  (input : Tensor α [batch, inDim])
  (gradOutput : Tensor α [batch])
  (lambda : α) (h : batch ≠ 0) :
  Tensor α [inDim] :=
  let mseGrad := Tensor.reduceSum 0
    (Tensor.zipEach ([batch]) ([inDim])
      (fun x gy => scaleSpec x (Tensor.item gy)) input gradOutput)
    (Shape.hasNonemptyAxisZeroOfNe h).proof
  let l2_grad := scaleSpec model.weights (2 * lambda)
  addSpec mseGrad l2_grad

/-- Soft-thresholding operator (often written `S_λ`), used in proximal-gradient updates for L1.

Reference: Tibshirani, "Regression Shrinkage and Selection via the Lasso" (1996).
https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
-/
def lassoSoftThresholdSpec {inDim : Nat}
  (weights : Tensor α [inDim])
  (threshold : α) :
  Tensor α [inDim] :=
  mapSpec (fun w =>
    if w > threshold then w - threshold
    else if (-threshold) > w then w + threshold
    else 0) weights

/-- Lasso forward pass (same raw prediction function as ordinary linear regression). -/
def lassoRegressionForwardSpec {inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [inDim])
  (_lambda : α) :
  Tensor α .scalar :=
  linearRegressionForwardSpec model input

/-- Lasso loss: MSE plus `lambda * ||w||_1`. -/
def lassoLossSpec {batch inDim : Nat}
  (model : LinearRegressionSpec α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch])
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
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch])
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
`φ(x) = [x, x^2, ..., x^degree]` (per input coordinate). We keep this as a named helper,
then reuse `linearRegressionForwardSpec` on the expanded input.
-/

/-- Expand a length-`inDim` input vector into polynomial features up to `degree`.

This expansion does not include a constant feature (the model bias already plays that role).
-/
def polynomialFeaturesSpec {inDim : Nat} (degree : Nat)
  (input : Tensor α [inDim]) :
  Tensor α [inDim * degree] :=
  Tensor.dim fun i =>
    have hProduct : 0 < inDim * degree := lt_of_le_of_lt (Nat.zero_le i.val) i.isLt
    have hInDimNe : inDim ≠ 0 := by
      intro h
      simp [h] at hProduct
    have hInDim : 0 < inDim := Nat.pos_of_ne_zero hInDimNe
    let coordinate : Fin inDim := ⟨i.val % inDim, Nat.mod_lt _ hInDim⟩
    let power := i.val / inDim + 1
    Tensor.scalar (Tensor.getScalar input coordinate ^ (power : α))

/-- Forward pass for polynomial regression: expand features, then run linear regression. -/
def polynomialRegressionForwardSpec {inDim degree : Nat}
  (model : LinearRegressionSpec α (inDim * degree))
  (input : Tensor α [inDim]) :
  Tensor α .scalar :=
  let expandedInput := polynomialFeaturesSpec degree input
  linearRegressionForwardSpec model expandedInput

end Spec
