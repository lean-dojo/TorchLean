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

PyTorch analogue:

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

## Implementation status

No API builder implements this model, and no theorem is proved about it. It is a reference
definition only.
-/

@[expose] public section


variable {α : Type} [TorchLean.Storage α] [Context α]

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Activation
open MathFunctions

/-- Parameters for logistic regression: a weight vector `w` and scalar intercept `b`.

We store `intercept : α` separately rather than folding it into `weights`, but `fitLogistic`
internally learns `(p + 1)` parameters by augmenting the input with a trailing column of ones.
-/
structure LogisticRegression (p n : ℕ) (α : Type) [TorchLean.Storage α] where
  /-- `p`-dimensional weight vector `w`. -/
  weights : Tensor α [p]
  /-- Scalar intercept term `b`. -/
  intercept : α

/-- Augment an `n × p` design matrix with a final column of ones.

This lets us represent the affine model `X w + b` as a single matrix-vector product with a
`(p + 1)`-vector of parameters.
-/
def augmentWithOnes {n p : ℕ} (X : Tensor α [n, p]) :
  Tensor α [n, p + 1] :=
  Tensor.dim (fun i =>
    let row := get X ⟨i.val, i.isLt⟩
    Tensor.dim (fun j =>
      if h : j.val < p then
        -- Original features.
        get row ⟨j.val, h⟩
      else
        -- Final "bias feature" (j = p).
        Tensor.scalar 1))

/-- Gradient of the logistic negative log-likelihood, expressed as `Xᵀ (σ(Xw) - y)`.

This is the standard expression used for (unregularized) logistic regression under labels
`y ∈ {0,1}`. We do not divide by `n` here; callers can rescale if they want the mean loss.
-/
def computeLogGradient {n p : ℕ} (X : Tensor α [n, p + 1])
  (y : Tensor α [n]) (w : Tensor α [p + 1]) :
  Tensor α [p + 1] :=
  let predictions := sigmoidSpec (matVecMulSpec X w)
  let error := subSpec predictions y
  vecMatMulSpec error X

/-- Fit logistic regression by plain gradient descent (structural recursion).

This is a simple deterministic baseline that is easy to reason about. It does not attempt to match
optimized solvers (LBFGS/Newton/IRLS); it is a small reference implementation that can be
instantiated over different scalar backends.
-/
def fitLogistic {n p : ℕ} (X : Tensor α [n, p])
  (y : Tensor α [n]) (learningRate : α) (iterations : Nat) :
  LogisticRegression p n α :=
  -- Augment X with a column of ones for the intercept term
  let X_aug := augmentWithOnes X

  -- Initialize weights with zeros
  let initialWeights := Tensor.full (.dim (p + 1) .scalar) (0 : α)

  -- Implement gradient descent (structural recursion for predictable runtime)
  let rec gradient_descent (iter : Nat) (weights : Tensor α [p + 1]) :
      Tensor α [p + 1] :=
    match iter with
    | 0 => weights
    | Nat.succ k =>
        let gradient := computeLogGradient X_aug y weights
        let scaledGradient := scaleSpec gradient learningRate
        let newWeights := subSpec weights scaledGradient
        gradient_descent k newWeights

  -- Run gradient descent
  let finalWeights := gradient_descent iterations initialWeights

  -- Extract weights and intercept
  let weights := Tensor.dim (fun i => get finalWeights ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩)
  let intercept := get finalWeights ⟨p, Nat.lt_succ_self p⟩

  { weights := weights, intercept := item intercept }

/-- Predict probabilities `σ(Xw + b)` for each row in `X`. -/
def predictProba {n p : ℕ} (model : LogisticRegression p n α)
  (X : Tensor α [n, p]) : Tensor α [n] :=
  let linearPred := matVecMulSpec X model.weights
  let biasTerm := Tensor.full (.dim n .scalar) model.intercept
  let combined := addSpec linearPred biasTerm
  sigmoidSpec combined

/-- Convert probabilities to hard labels using a threshold (default `0.5`). -/
def logPredict {n p : ℕ} (model : LogisticRegression p n α)
  (X : Tensor α [n, p]) (threshold : α := (1 : α) / (2 : α)) :
  Tensor α [n] :=
  let probabilities := predictProba model X
  mapSpec (fun prob => if prob > threshold then (1 : α) else (0 : α)) probabilities
