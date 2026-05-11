/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation

/-!
# Logistic regression (spec model)

This file implements a small, deterministic logistic regression baseline.

Model (binary classification):

- logits: `z = X w + b`
- probabilities: `p = σ(z)` where `σ` is the logistic sigmoid

PyTorch mental model:

- parameters correspond to `nn.Linear(p, 1)` (weights + bias),
- probabilities correspond to `torch.sigmoid(logits)`,
- training is a simple gradient-descent loop (similar to `torch.optim.SGD`), written in a
  simple, explicit style rather than tuned for performance.

Notes:
- We augment the input matrix with a column of ones to represent the intercept term.
- This is reference/spec code: it prioritizes clarity and auditability over performance.

Numerical note:
PyTorch often uses `BCEWithLogitsLoss` for stability (it works directly on logits without forming
`sigmoid` explicitly). Here we keep the math explicit.
-/

@[expose] public section


variable {α : Type} [Context α]

open Spec
open Tensor
open Activation
open MathFunctions
open Numbers

/-- Parameters for logistic regression: a weight vector `w` and scalar intercept `b`.

We store `intercept : α` separately rather than folding it into `weights`, but `fitLogistic`
internally learns `(p + 1)` parameters by augmenting the input with a trailing column of ones.
-/
structure LogisticRegression (p n : ℕ) (α : Type) where
  /-- `p`-dimensional weight vector `w`. -/
  weights : Tensor α (.dim p .scalar)
  /-- Scalar intercept term `b`. -/
  intercept : α

/-- Augment an `n × p` design matrix with a final column of ones.

This lets us represent the affine model `X w + b` as a single matrix-vector product with a
`(p + 1)`-vector of parameters.
-/
def augmentWithOnes {n p : ℕ} (X : Tensor α (.dim n (.dim p .scalar))) :
  Tensor α (.dim n (.dim (p + 1) .scalar)) :=
  Tensor.dim (fun i =>
    let row := getAtSpec X ⟨i.val, i.isLt⟩
    Tensor.dim (fun j =>
      if h : j.val < p then
        -- Original features.
        getAtSpec row ⟨j.val, h⟩
      else
        -- Final "bias feature" (j = p).
        Tensor.scalar 1))

/-- Gradient of the logistic negative log-likelihood, expressed as `Xᵀ (σ(Xw) - y)`.

This is the standard expression used for (unregularized) logistic regression under labels
`y ∈ {0,1}`. We do not divide by `n` here; callers can rescale if they want the mean loss.
-/
def computeLogGradient {n p : ℕ} (X : Tensor α (.dim n (.dim (p + 1) .scalar)))
  (y : Tensor α (.dim n .scalar)) (w : Tensor α (.dim (p + 1) .scalar)) :
  Tensor α (.dim (p + 1) .scalar) :=
  let predictions := sigmoidSpec (matVecMulSpec X w)
  let error := subSpec predictions y
  vecMatMulSpec error X

/-- Fit logistic regression by plain gradient descent (structural recursion).

This is a simple deterministic baseline that is easy to reason about. It does not attempt to match
optimized solvers (LBFGS/Newton/IRLS); it is a small reference implementation that can be
instantiated over different scalar backends.
-/
def fitLogistic {n p : ℕ} (X : Tensor α (.dim n (.dim p .scalar)))
  (y : Tensor α (.dim n .scalar)) (learning_rate : α) (iterations : Nat) :
  LogisticRegression p n α :=
  -- Augment X with a column of ones for the intercept term
  let X_aug := augmentWithOnes X

  -- Initialize weights with zeros
  let initial_weights := fill (0 : α) (.dim (p + 1) .scalar)

  -- Implement gradient descent (structural recursion for predictable runtime)
  let rec gradient_descent (iter : Nat) (weights : Tensor α (.dim (p + 1) .scalar)) :
      Tensor α (.dim (p + 1) .scalar) :=
    match iter with
    | 0 => weights
    | Nat.succ k =>
        let gradient := computeLogGradient X_aug y weights
        let scaled_gradient := scaleSpec gradient learning_rate
        let new_weights := subSpec weights scaled_gradient
        gradient_descent k new_weights

  -- Run gradient descent
  let final_weights := gradient_descent iterations initial_weights

  -- Extract weights and intercept
  let weights := Tensor.dim (fun i => getAtSpec final_weights ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩)
  let intercept := getAtSpec final_weights ⟨p, Nat.lt_succ_self p⟩

  { weights := weights, intercept := toScalar intercept }

/-- Predict probabilities `σ(Xw + b)` for each row in `X`. -/
def predictProba {n p : ℕ} (model : LogisticRegression p n α)
  (X : Tensor α (.dim n (.dim p .scalar))) : Tensor α (.dim n .scalar) :=
  let linear_pred := matVecMulSpec X model.weights
  let bias_term := fill model.intercept (.dim n .scalar)
  let combined := addSpec linear_pred bias_term
  sigmoidSpec combined

/-- Convert probabilities to hard labels using a threshold (default `0.5`). -/
def logPredict {n p : ℕ} (model : LogisticRegression p n α)
  (X : Tensor α (.dim n (.dim p .scalar))) (threshold : α := (1 : α) / (Numbers.two : α)) :
  Tensor α (.dim n .scalar) :=
  let probabilities := predictProba model X
  mapSpec (fun prob => if prob > threshold then (1 : α) else (0 : α)) probabilities
