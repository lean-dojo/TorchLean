/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.VJPOp
public import TorchLean.Tensor.Sequence
public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Layers.Loss

/-!
# Linear regression (spec model)

Linear regression with one output, a dot product plus bias:

`y = wᵀ x + b`

The corresponding PyTorch operations are:

- `torch.nn.Linear(in_features, out_features=1)` for the forward pass,
- `torch.nn.functional.mse_loss(..., reduction="mean")` for the MSE objective,
- an SGD-style parameter update step (as in `torch.optim.SGD`) for training.

The definitions state the math, the forward pass and its VJPs, with shapes tracked by the type
system and derivatives written out by hand for clarity over speed. Besides the model with its MSE
training step and R², there are a polynomial feature expansion and ridge, lasso and elastic-net
objectives, whose penalties leave the bias out. The closed-form normal-equations solution is not
included.

In PyTorch 2.8 terms, `trainStep` is one step of `torch.optim.SGD` at its defaults (`momentum=0`,
`weight_decay=0`) on that mean MSE. On one row with prediction 3 and target 1, the bias drops by
`4 * lr`.

## Implementation status

No API builder implements this model as a unit: `nn.linear` builds the layer alone, without this
loss and training step. No theorem relates the model to runtime code.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Parameters for a single-output linear regression model.

PyTorch analogy: `weights` and `bias` correspond to `nn.Linear(inDim, 1).weight` and
`nn.Linear(inDim, 1).bias`, which PyTorch stores with shapes `[1, inDim]` and `[1]`. Here they are
a vector and a scalar, with the shapes in the tensor type.
-/
structure LinearRegression (α : Type) [Storage α]
    (inDim : Nat) where
  /-- Regression coefficients, one per input feature. -/
  weights : Tensor α [inDim]
  /-- Scalar intercept term. -/
  bias : Tensor α .scalar

/-- Forward pass for linear regression: `y = wᵀ x + b`. -/
def LinearRegression.forward {inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [inDim]) :
  Tensor α .scalar :=
  let dotProduct := dot model.weights input
  Tensor.scalar dotProduct + model.bias

/-- Gradients for a linear regression model.

`inputShape` is a parameter so that the unbatched and batched backward passes return the same
record: the parameter gradients have the same shape either way, and only the input gradient grows a
leading batch axis. -/
structure LinearRegression.Grads (α : Type) [Storage α] (inDim : Nat)
    (inputShape : Shape) where
  /-- Gradient with respect to the weight vector. -/
  weightGradient : Tensor α [inDim]
  /-- Gradient with respect to the scalar bias. -/
  biasGradient : Tensor α .scalar
  /-- Gradient with respect to the input. -/
  inputGradient : Tensor α inputShape

/-- Backward pass for one example with upstream gradient `g`: the weight gradient is `g * x`, the
bias gradient `g` and the input gradient `g * w`. -/
def LinearRegression.backward {inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [inDim])
  (gradOutput : Tensor α .scalar) :
  Grads α inDim [inDim] :=
  { weightGradient := Tensor.scale input (Tensor.item gradOutput)
    biasGradient := gradOutput
    inputGradient := Tensor.scale model.weights (Tensor.item gradOutput) }

/-- Batched backward pass. The parameter gradients sum the per-example gradients over `batch`,
and the input gradient keeps one row per example.

With the upstream gradient of a `"mean"` loss, which carries the `1 / batch` factor, the sums are
the parameter gradients that PyTorch accumulates.
-/
def LinearRegression.backwardBatch {batch inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [batch, inDim])
  (gradOutput : Tensor α [batch]) :
  Grads α inDim [batch, inDim] :=
  -- dW = Σ_b g_b x_b
  let dW := Tensor.castShape (Tensor.reduce .sum (Tensor.zipEach ([batch]) ([inDim])
      (fun x gy => Tensor.scale x (Tensor.item gy)) input gradOutput) [⟨0, by simp⟩] false)
          (Shape.reduce_singleton _ _)
  -- db = Σ_b g_b
  let db := Tensor.castShape (Tensor.reduce .sum gradOutput [⟨0, by simp⟩] false)
      (Shape.reduce_singleton _ _)
  -- Row b of dX is g_b w.
  let dX := Tensor.mapLeading ([batch])
    (fun gy => Tensor.scale model.weights (Tensor.item gy))
    gradOutput

  { weightGradient := dW, biasGradient := db, inputGradient := dX }

/-- One gradient-descent step on the mean squared error, returning the loss of `model` on the
batch and the model after the update `θ ← θ - lr * ∇θ`. -/
def LinearRegression.trainStep {batch inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch])
  (lr : α) :
  (Tensor α .scalar × LinearRegression α inDim) :=
  let predictions : Tensor α [batch] :=
    Tensor.mapLeading [batch] (forward model) input
  let loss := Tensor.scalar (Tensor.meanSquaredError predictions target)
  let gradPredictions := meanSquaredErrorBackward predictions target
  let gradients := backwardBatch model input gradPredictions
  let newWeights :=
    model.weights - Tensor.scale gradients.weightGradient lr
  let newBias := model.bias - Tensor.scale gradients.biasGradient lr
  let updatedModel := { model with weights := newWeights, bias := newBias }
  (loss, updatedModel)

/-- Linear regression with fixed parameters as an `VJPOp` from `[inDim]` to a scalar, for
composing with other ops in spec-level AD.

The backward pass returns the input gradient only, so the weights and bias act as constants.
-/
def LinearRegression.vjpOp {inDim : Nat}
  (model : LinearRegression α inDim) :
  VJPOp α ([inDim]) .scalar :=
{
  forward := fun x => forward model x,
  backward := fun x dLdy =>
    (backward model x dLdy).inputGradient
}

/-- R-squared (coefficient of determination) `1 - SS_res / SS_tot` for model evaluation on a
nonempty batch (`h : batch ≠ 0`).

Core PyTorch has no single built-in for R², and the definition here is the standard one. If
`SS_tot = 0`, as for constant targets in exact arithmetic, the division by zero is kept: on IEEE
floats it gives `-∞`, or NaN when `SS_res = 0` as well. Many libraries treat that case specially.
-/
def LinearRegression.rSquared {batch inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch]) (h : batch ≠ 0) :
  Tensor α .scalar :=
  let predictions : Tensor α [batch] :=
    Tensor.mapLeading [batch] (forward model) input
  let leadingAxis : Shape.HasNonemptyAxis 0 (Shape.dim batch Shape.scalar) :=
    Shape.hasNonemptyAxis_zero_of_ne h
  let targetMean := Tensor.map (Context.divNat · batch)
    (castShape (reduce .sum target [⟨0, leadingAxis.proof.axisInBounds.proof⟩] false)
      (Shape.reduce_singleton _ _))
  let targetMeanBroadcast := Tensor.full [batch] targetMean.item
  let ss_res := castShape
      (reduce .sum (Tensor.square (predictions - target))
      [⟨0, leadingAxis.proof.axisInBounds.proof⟩] false) (Shape.reduce_singleton _ _)
  let ss_tot := castShape
      (reduce .sum (Tensor.square (target - targetMeanBroadcast))
      [⟨0, leadingAxis.proof.axisInBounds.proof⟩] false) (Shape.reduce_singleton _ _)
  Tensor.scalar 1 - (ss_res / ss_tot)

/-- Ridge loss: MSE plus `lambda * ||w||_2^2`.

Regularization changes the training objective. For fixed parameters, predictions use
`forward` and do not take a regularization coefficient. `sklearn.linear_model.Ridge`
(scikit-learn 1.9.1 docs) sums the squared errors, so its `alpha` on `n` rows is our
`lambda = alpha / n`, and `alpha = 1` on 50 rows is `lambda = 0.02`.

Reference: Hoerl and Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems"
(1970), Technometrics 12(1), 55-67. https://doi.org/10.1080/00401706.1970.10488634
-/
def LinearRegression.ridgeLoss {batch inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch])
  (lambda : α) :
  Tensor α .scalar :=
  let predictions : Tensor α [batch] :=
    Tensor.mapLeading [batch] (forward model) input
  let mse := Tensor.scalar (Tensor.meanSquaredError predictions target)
  let l2_penalty := Tensor.scale (Tensor.scalar (dot model.weights model.weights)) lambda
  mse + l2_penalty

/-- Ridge gradient with respect to the weights, for `gradOutput` the gradient of the MSE term
with respect to the predictions: `Σ_b g_b x_b`, as in `LinearRegression.backwardBatch`, plus
`2 * lambda * w` from the penalty.
-/
def LinearRegression.ridgeWeightsDeriv {batch inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [batch, inDim])
  (gradOutput : Tensor α [batch])
  (lambda : α) :
  Tensor α [inDim] :=
  let mseGrad : Tensor α [inDim] :=
    Tensor.castShape (Tensor.reduce .sum (Tensor.zipEach ([batch]) ([inDim])
      (fun x gy => Tensor.scale x (Tensor.item gy)) input gradOutput) [⟨0, by simp⟩] false)
          (Shape.reduce_singleton _ _)
  let l2_grad := Tensor.scale model.weights (2 * lambda)
  mseGrad + l2_grad

/-- Soft-thresholding operator (often written `S_λ`), used in proximal-gradient updates for L1.

For `threshold ≥ 0`, entries in `[-threshold, threshold]` become `0` and the others move toward
zero by `threshold`. A NaN entry fails both comparisons and becomes `0` too.

Reference: Tibshirani, "Regression Shrinkage and Selection via the Lasso" (1996), Journal of the
Royal Statistical Society, Series B 58(1), 267-288.
https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
-/
def LinearRegression.lassoSoftThreshold {inDim : Nat}
  (weights : Tensor α [inDim])
  (threshold : α) :
  Tensor α [inDim] :=
  Tensor.map (fun w =>
    if w > threshold then w - threshold
    else if (-threshold) > w then w + threshold
    else 0) weights

/-- Lasso loss: MSE plus `lambda * ||w||_1`.

As with `LinearRegression.ridgeLoss`, the penalty belongs to the training objective, and predictions
use `forward`. `sklearn.linear_model.Lasso` scales the squared error by `1 / (2 n)`, so our
`lambda` is twice its `alpha`.
-/
def LinearRegression.lassoLoss {batch inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch])
  (lambda : α) :
  Tensor α .scalar :=
  let predictions : Tensor α [batch] :=
    Tensor.mapLeading [batch] (forward model) input
  let mse := Tensor.scalar (Tensor.meanSquaredError predictions target)
  let l1_penalty := Tensor.scale (Tensor.scalar (Tensor.sum (Tensor.abs model.weights))) lambda
  mse + l1_penalty

/-- Elastic net loss: MSE plus `alpha * l1_ratio * ||w||_1 + alpha * (1 - l1_ratio) * ||w||_2^2`,
which is `alpha` times a convex combination of the two penalties when `0 ≤ l1_ratio ≤ 1`.

The sklearn `ElasticNet` estimator uses the same parameter names, and its objective halves both the
squared error and the L2 term.

Reference: Zou and Hastie, "Regularization and Variable Selection via the Elastic Net" (2005),
Journal of the Royal Statistical Society, Series B 67(2), 301-320.
https://doi.org/10.1111/j.1467-9868.2005.00503.x
-/
def LinearRegression.elasticNetLoss {batch inDim : Nat}
  (model : LinearRegression α inDim)
  (input : Tensor α [batch, inDim])
  (target : Tensor α [batch])
  (l1_ratio : α)
  (alpha : α) :
  Tensor α .scalar :=
  let predictions : Tensor α [batch] :=
    Tensor.mapLeading [batch] (forward model) input
  let mse := Tensor.scalar (Tensor.meanSquaredError predictions target)
  let l1_penalty := Tensor.scale (Tensor.scalar (Tensor.sum (Tensor.abs model.weights))) (alpha *
    l1_ratio)
  let l2_penalty := Tensor.scale (Tensor.scalar (dot model.weights model.weights)) (alpha * (1 -
    l1_ratio))
  mse + (l1_penalty + l2_penalty)

/-!
## Polynomial features

Polynomial regression can be expressed as linear regression on a fixed feature expansion
`φ(x) = [x, x^2, ..., x^degree]` (per input coordinate). `LinearRegression.polynomialFeatures` is
that expansion, and `LinearRegression.polynomialForward` applies `LinearRegression.forward` to the
expanded input.
-/

/-- Expand a length-`inDim` input vector into polynomial features up to `degree`.

There is no constant feature, because the model bias plays that role. Features are ordered by
degree, with all input coordinates at power one followed by all at power two, and so on. Degree
zero or an empty input gives an empty output.
-/
def LinearRegression.polynomialFeatures {inDim : Nat} (degree : Nat)
  (input : Tensor α [inDim]) :
  Tensor α [inDim * degree] :=
  -- Reuse each power vector across its coordinates, advancing it once per degree.
  -- Multiplication preserves polynomial derivatives at zero and negative inputs,
  -- including all coefficients of nested Dual scalars.
  (Sequence.mapAccum (inDim * degree) input fun i powers =>
    have hProduct : 0 < inDim * degree := lt_of_le_of_lt (Nat.zero_le i.val) i.isLt
    have hInDimNe : inDim ≠ 0 := by
      intro h
      simp [h] at hProduct
    have hInDim : 0 < inDim := Nat.pos_of_ne_zero hInDimNe
    let coordinate : Fin inDim := ⟨i.val % inDim, Nat.mod_lt _ hInDim⟩
    let powers :=
      if i.val ≠ 0 ∧ coordinate.val = 0 then powers * input else powers
    (powers, Tensor.getScalar powers coordinate)).2

/-- Forward pass for polynomial regression: expand features, then run linear regression. -/
def LinearRegression.polynomialForward {inDim degree : Nat}
  (model : LinearRegression α (inDim * degree))
  (input : Tensor α [inDim]) :
  Tensor α .scalar :=
  let expandedInput := polynomialFeatures degree input
  forward model expandedInput

end TorchLean
