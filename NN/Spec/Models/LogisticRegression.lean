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
structure LogisticRegression (p : ℕ) (α : Type) [TorchLean.Storage α] where
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
  let logits := matVecMulSpec X w
  -- Near zero, σ(z) - y = (1/2 - y) + tanh(z/2)/2. Keep the two parts separate:
  -- rounding their sum near ±1/2 can discard the small contribution that remains after
  -- positive and negative examples cancel in a balanced batch.
  let offsets := mapSpec (fun logit =>
    if Context.gtBool (1 : α) (MathFunctions.abs logit) then tanh (logit / 2) / 2 else 0) logits
  let residuals : Tensor α [n] := Tensor.dim fun i =>
    let logit := Tensor.getScalar logits i
    let target := Tensor.getScalar y i
    Tensor.scalar <| if Context.gtBool (1 : α) (MathFunctions.abs logit) then (1 : α) / 2 - target
      -- Away from zero, preserve the small exponential tail for a positive target.
      -- Subtracting one from a sigmoid that has rounded to one would erase it.
      -- A unit primal can still carry a target tangent. Keep its subtraction so the
      -- residual retains derivative -1 with respect to that target.
      else if target == (1 : α) then (1 - target) - Activation.Math.sigmoidSpec (-logit)
      else Activation.Math.sigmoidSpec logit - target
  -- Compensate both sums independently. Combining the parts only after the reduction
  -- preserves the centered contribution without changing the summed-loss convention.
  let accumulate (total correction term : α) : α × α :=
    let adjusted := term - correction
    let next := total + adjusted
    (next, (next - total) - adjusted)
  Tensor.dim fun j =>
    let ((residualSum, _), (offsetSum, _)) :=
      (List.finRange n).foldl (fun ((residualSum, residualCorrection),
          (offsetSum, offsetCorrection)) i =>
        let feature := Spec.get2 X i j
        (accumulate residualSum residualCorrection (Tensor.getScalar residuals i * feature),
          accumulate offsetSum offsetCorrection (Tensor.getScalar offsets i * feature)))
        (((0 : α), (0 : α)), ((0 : α), (0 : α)))
    Tensor.scalar (residualSum + offsetSum)

/-- A training label outside the binary encoding expected by logistic regression. -/
inductive LogisticRegression.FitError where
  /-- The zero-based row whose label is neither `0` nor `1`. -/
  | invalidLabel (row : Nat)
  deriving Repr, BEq

/-- Fit binary logistic regression, checking the label encoding before any gradient step.

Each target must equal `0` or `1` under the scalar context's equality operation. In particular,
signed SVM labels and soft targets are rejected here. The lower-level `computeLogGradient`
remains the explicit mathematical expression for callers studying other target conventions.

The objective is a sum over observations, so duplicating the dataset doubles the gradient.
An empty dataset has zero gradient and returns the zero initial parameters.
-/
def fitLogistic {n p : ℕ} (X : Tensor α [n, p]) (y : Tensor α [n])
    (learningRate : α) (iterations : Nat) :
    Except LogisticRegression.FitError (LogisticRegression p α) := do
  for i in List.finRange n do
    let label := Tensor.getScalar y i
    unless label == (0 : α) || label == (1 : α) do
      throw (.invalidLabel i.val)
  -- Augment X with a column of ones for the intercept term
  let augmentedInputs := augmentWithOnes X

  -- Initialize weights with zeros
  let initialWeights := Tensor.full (.dim (p + 1) .scalar) (0 : α)

  -- Implement gradient descent (structural recursion for predictable runtime)
  let rec gradientDescent (iter : Nat) (weights : Tensor α [p + 1]) :
      Tensor α [p + 1] :=
    match iter with
    | 0 => weights
    | Nat.succ k =>
        let gradient := computeLogGradient augmentedInputs y weights
        let scaledGradient := scaleSpec gradient learningRate
        let newWeights := subSpec weights scaledGradient
        gradientDescent k newWeights

  -- Run gradient descent
  let finalWeights := gradientDescent iterations initialWeights

  -- Extract weights and intercept
  let weights := Tensor.dim (fun i => get finalWeights ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩)
  let intercept := get finalWeights ⟨p, Nat.lt_succ_self p⟩

  return { weights := weights, intercept := item intercept }

/-- Predict the probability of label `1` for each row of `X`.

Only the number of features must agree with the fitted weights. The prediction batch may
have any number of rows; an empty batch returns an empty probability tensor.
-/
def LogisticRegression.predictProba {batch p : ℕ} (model : LogisticRegression p α)
  (X : Tensor α [batch, p]) : Tensor α [batch] :=
  let linearPred := matVecMulSpec X model.weights
  let biasTerm := Tensor.full (.dim batch .scalar) model.intercept
  let combined := addSpec linearPred biasTerm
  sigmoidSpec combined

/-- Predict binary labels for a batch, using `0.5` as the default probability threshold.

A probability strictly above the threshold gives label `1`; equality gives label `0`.
The inference batch may have any number of rows, independently of the training batch.
-/
def LogisticRegression.predict {batch p : ℕ} (model : LogisticRegression p α)
  (X : Tensor α [batch, p]) (threshold : α := (1 : α) / (2 : α)) :
  Tensor α [batch] :=
  let probabilities := model.predictProba X
  mapSpec (fun prob => if prob > threshold then (1 : α) else (0 : α)) probabilities

/-- Predict the probability of label `1` for one feature vector.

This uses the one-row batch operation, so its scalar arithmetic and rounding order agree with
`predictProba` on the same observation.
-/
def LogisticRegression.predictProbaOne {p : ℕ} (model : LogisticRegression p α)
  (x : Tensor α [p]) : α :=
  Tensor.getScalar (model.predictProba (Tensor.dim fun (_ : Fin 1) => x)) ⟨0, by decide⟩

/-- Predict one binary label, with the same threshold and tie rule as `predict`. -/
def LogisticRegression.predictOne {p : ℕ} (model : LogisticRegression p α)
  (x : Tensor α [p]) (threshold : α := (1 : α) / (2 : α)) : α :=
  Tensor.getScalar (model.predict (Tensor.dim fun (_ : Fin 1) => x) threshold) ⟨0, by decide⟩
