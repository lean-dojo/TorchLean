/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Contract

/-!
# Logistic regression (spec model)

A small, deterministic logistic regression baseline.

Model (binary classification):

- logits: `z = X w + b`
- probabilities: `p = σ(z)` where `σ` is the logistic sigmoid

PyTorch analogue:

- parameters correspond to `nn.Linear(p, 1)` (weights + bias),
- probabilities correspond to `torch.sigmoid(logits)`,
- training is a plain gradient-descent loop (similar to `torch.optim.SGD`), written for clarity
  and auditability over speed.

Training represents the intercept by appending a column of ones to the input matrix.

`lossGradient` is the gradient in `w` of `torch.nn.BCEWithLogitsLoss` with `reduction="sum"` on
the logits `X w`, derived by hand for the model of Bishop's Section 4.3.2. That loss's default
`"mean"` divides it by `n`.

Numerical note:
PyTorch often uses `BCEWithLogitsLoss` for stability, because it works on the logits and never
forms the sigmoid. `lossGradient` forms the residual `σ(z) - y` and controls its rounding:
for `|z| < 1` it sums `1/2 - y` and `tanh(z/2)/2` separately, and for target `1` with `|z| ≥ 1`
it computes `σ(z) - 1` as `-σ(-z)`.

References:

- Bishop (2006), "Pattern Recognition and Machine Learning", Springer, Section 4.3.2 (Logistic
  regression). https://doi.org/10.1007/978-0-387-45528-0
- `torch.nn.BCEWithLogitsLoss` (PyTorch 2.8 docs).
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html

## Implementation status

No API builder implements this model, and no theorem is proved about it. It is a reference
definition only.
-/

@[expose] public section


namespace TorchLean

open FloatLib.Numerics (MathFunctions)

variable {α : Type} [Storage α] [Context α]

open TorchLean.Tensor

/-- Parameters for logistic regression: a weight vector `w` and a scalar intercept `b`.

The intercept is a separate field, while `LogisticRegression.fit` learns all `p + 1` parameters
together on inputs with a trailing column of ones and splits them at the end.
-/
structure LogisticRegression (p : ℕ) (α : Type) [Storage α] where
  /-- `p`-dimensional weight vector `w`. -/
  weights : Tensor α [p]
  /-- Scalar intercept term `b`. -/
  intercept : α

/-- Append a column of ones to an `n × p` design matrix, so that the affine model `X w + b`
becomes one matrix-vector product with the `(p + 1)`-vector of parameters `[w, b]`.
-/
def LogisticRegression.augmentWithOnes {α : Type} [Storage α] [One α]
    {n p : ℕ} (X : Tensor α [n, p]) :
  Tensor α [n, p + 1] :=
  Tensor.dim (fun i =>
    let row := Tensor.unstack X ⟨i.val, i.isLt⟩
    Tensor.dim (fun j =>
      if h : j.val < p then
        Tensor.unstack row ⟨j.val, h⟩
      else
        -- Final "bias feature" (j = p).
        Tensor.scalar 1))

/-- Gradient `Xᵀ (σ(Xw) - y)` of the logistic negative log-likelihood, summed over the `n`
observations, for labels `y ∈ {0,1}` and without regularization.

Callers who want the gradient of the mean loss divide by `n`. Each coordinate is a pair of
Kahan-compensated sums over the rows. At `w = 0` every probability is `1/2`, so a single row with
label `1` and feature `2` gives `-1` in that coordinate.
-/
def LogisticRegression.lossGradient {n p : ℕ} (X : Tensor α [n, p + 1])
  (y : Tensor α [n]) (w : Tensor α [p + 1]) :
  Tensor α [p + 1] :=
  let logits := Tensor.contract X w _ _ [n] .matVec
  -- For |z| < 1, σ(z) - y = (1/2 - y) + tanh(z/2)/2. Keep the two parts separate:
  -- rounding their sum near ±1/2 can discard the small contribution left after
  -- positive and negative examples cancel in a balanced batch.
  let offsets := Tensor.map (fun logit =>
    if Context.gtBool (1 : α) (MathFunctions.abs logit) then MathFunctions.tanh (logit / 2) / 2
    else 0) logits
  let residuals : Tensor α [n] := Tensor.dim fun i =>
    let logit := Tensor.getScalar logits i
    let target := Tensor.getScalar y i
    Tensor.scalar <| if Context.gtBool (1 : α) (MathFunctions.abs logit) then (1 : α) / 2 - target
      -- For |z| ≥ 1 and target 1, the residual σ(z) - 1 is computed as -σ(-z), keeping the small
      -- exponential tail that subtracting one would erase once σ(z) rounds to one. A target with
      -- primal 1 can carry a tangent, and the `1 - target` term, zero in value, gives the residual
      -- its derivative -1 with respect to that target.
      else if target == (1 : α) then (1 - target) - Context.sigmoid (-logit)
      else Context.sigmoid logit - target
  -- Kahan-compensate the two sums separately and add them only after the reduction, which keeps
  -- the centered contribution and leaves the gradient a plain sum over observations.
  let accumulate (total correction term : α) : α × α :=
    let adjusted := term - correction
    let next := total + adjusted
    (next, (next - total) - adjusted)
  Tensor.dim fun j =>
    let ((residualSum, _), (offsetSum, _)) :=
      (List.finRange n).foldl (fun ((residualSum, residualCorrection),
          (offsetSum, offsetCorrection)) i =>
        let feature := Tensor.get2 X i j
        (accumulate residualSum residualCorrection (Tensor.getScalar residuals i * feature),
          accumulate offsetSum offsetCorrection (Tensor.getScalar offsets i * feature)))
        (((0 : α), (0 : α)), ((0 : α), (0 : α)))
    Tensor.scalar (residualSum + offsetSum)

/-- A training label outside the binary encoding expected by logistic regression. -/
inductive LogisticRegression.FitError where
  /-- The zero-based row whose label is neither `0` nor `1`. -/
  | invalidLabel (row : Nat)
  deriving Repr, BEq

/-- Fit binary logistic regression by `iterations` full-batch gradient-descent steps with rate
`lr` from zero parameters, after checking the label encoding.

Each target must equal `0` or `1` under the scalar context's equality, so signed SVM labels and
soft targets give `invalidLabel` with the first offending row. Callers studying other target
conventions can use `lossGradient`, which accepts any targets.

The objective is a sum over observations, so duplicating the dataset doubles the gradient.
An empty dataset has zero gradient and returns the zero initial parameters.
-/
def LogisticRegression.fit {n p : ℕ} (X : Tensor α [n, p]) (y : Tensor α [n])
    (lr : α) (iterations : Nat) :
    Except LogisticRegression.FitError (LogisticRegression p α) := do
  for i in List.finRange n do
    let label := Tensor.getScalar y i
    unless label == (0 : α) || label == (1 : α) do
      throw (.invalidLabel i.val)
  let augmentedInputs := augmentWithOnes X

  let initialWeights := Tensor.full (.dim (p + 1) .scalar) (0 : α)

  -- Gradient descent by structural recursion on the step count, for a predictable runtime.
  let rec
    /-- `iter` full-batch gradient-descent steps with rate `lr` from `weights`. -/
    gradientDescent (iter : Nat) (weights : Tensor α [p + 1]) :
        Tensor α [p + 1] :=
      match iter with
      | 0 => weights
      | Nat.succ k =>
          let gradient := lossGradient augmentedInputs y weights
          let scaledGradient := Tensor.scale gradient lr
          let newWeights := weights - scaledGradient
          gradientDescent k newWeights

  let finalWeights := gradientDescent iterations initialWeights

  let weights := Tensor.dim (fun i => Tensor.unstack finalWeights ⟨i.val, Nat.lt_succ_of_lt i.isLt⟩)
  let intercept := Tensor.unstack finalWeights ⟨p, Nat.lt_succ_self p⟩

  return { weights := weights, intercept := item intercept }

/-- Probability `σ(X w + b)` of label `1` for each row of `X`.

Only the feature count `p` must match the fitted weights. The batch may have any number of rows,
and an empty batch gives an empty tensor.
-/
def LogisticRegression.predictProba {batch p : ℕ} (model : LogisticRegression p α)
  (X : Tensor α [batch, p]) : Tensor α [batch] :=
  let linearPred := Tensor.contract X model.weights _ _ [batch] .matVec
  let biasTerm := Tensor.full (.dim batch .scalar) model.intercept
  let combined := linearPred + biasTerm
  Tensor.sigmoid combined

/-- Predict binary labels for a batch, using `0.5` as the default probability threshold.

A probability strictly above the threshold gives label `1`, and every other value gives `0`,
including a probability equal to the threshold or NaN. The batch may have any number of rows,
whatever the size of the training batch.
-/
def LogisticRegression.predict {batch p : ℕ} (model : LogisticRegression p α)
  (X : Tensor α [batch, p]) (threshold : α := (1 : α) / (2 : α)) :
  Tensor α [batch] :=
  let probabilities := model.predictProba X
  Tensor.map (fun prob => if prob > threshold then (1 : α) else (0 : α)) probabilities

/-- Probability of label `1` for one feature vector, computed as a one-row batch so that its
arithmetic and rounding order agree with `predictProba` on the same observation. -/
def LogisticRegression.predictProbaOne {p : ℕ} (model : LogisticRegression p α)
  (x : Tensor α [p]) : α :=
  Tensor.getScalar (model.predictProba (Tensor.dim fun (_ : Fin 1) => x)) ⟨0, by decide⟩

/-- Predict one binary label, with the same threshold and tie rule as `predict`. -/
def LogisticRegression.predictOne {p : ℕ} (model : LogisticRegression p α)
  (x : Tensor α [p]) (threshold : α := (1 : α) / (2 : α)) : α :=
  Tensor.getScalar (model.predict (Tensor.dim fun (_ : Fin 1) => x) threshold) ⟨0, by decide⟩

end TorchLean
