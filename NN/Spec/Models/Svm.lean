/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.Reductions

/-!
# Support Vector Machines (spec models)

This file provides a small **linear SVM** baseline with explicit gradients.

PyTorch analogue:

- scoring function: `score = X @ w + b` (like `nn.Linear(p, 1)` without an activation),
- loss: hinge loss on signed labels `y ∈ {−1, +1}`:
  `loss_i = max(0, 1 - y_i * score_i)`,
- optimization: a small deterministic gradient descent loop (not an optimized solver).

There are two "layers" in this file:

- `LinearSVM`: the clean mathematical model + objective + backward pass (VJP-style gradients);
- `fitLinearSVM`/`SVM.predict`: a small training + prediction wrapper used by runtime checks and
  examples.

Classic SVM literature often uses `C` for the weight on the hinge term. This implementation instead
uses the equivalent primal form with an explicit L2 coefficient named `lambda`.

References:
- Cortes and Vapnik, "Support-Vector Networks", 1995.
- Vapnik, "The Nature of Statistical Learning Theory", 1995/1998.

## Implementation status

No API builder implements this model, and no theorem is proved about it. It is a reference
definition only.
-/

@[expose] public section


variable {α : Type} [TorchLean.Storage α] [Context α]
variable [DecidableRel ((· > ·) : α → α → Prop)]

open Spec TorchLean
open TorchLean TorchLean.Tensor
open MathFunctions

/-! ## Linear SVM (primal) -/

/-- Linear SVM parameters: a weight vector `w` and bias `b`.

We intentionally keep "training hyperparameters" (regularization strength, learning rate, etc.)
out of the parameter record; those are *choices about an optimizer*, not part of the model itself.
-/
structure LinearSVM (p : Nat) (α : Type) [TorchLean.Storage α] where
  /-- Normal vector of the separating hyperplane. -/
  w : Tensor α [p]
  /-- Bias, or intercept, of the separating hyperplane. -/
  b : α

/-- Decision function `f(x) = w·x + b`. -/
def LinearSVM.decision {p : Nat} (m : LinearSVM p α) (x : Tensor α [p]) : α :=
  Tensor.dotSpec m.w x + m.b

/-- Batch decision values for `X : (n×p)`. -/
def LinearSVM.decisionBatch {n p : Nat} (m : LinearSVM p α) (X : Tensor α [n, p]) :
  Tensor α [n] :=
  Tensor.dim (fun i =>
    Tensor.scalar (LinearSVM.decision m (get X i)))

/-- Hinge loss per example: `ℓ_i = max(0, 1 - y_i * f(x_i))`.

We write it using `if` rather than `max` to make the "active-set" logic explicit. -/
def hingeLossPerExample (score y : α) : α :=
  let oneMinusMargin := (1 : α) - (y * score)
  if oneMinusMargin > (0 : α) then oneMinusMargin else 0

/-- Mean hinge loss over a dataset. -/
def hingeLossMean {n : Nat} (scores : Tensor α [n]) (y : Tensor α [n]) :
  α :=
  let losses : Tensor α [n] :=
    Tensor.dim (fun i =>
      let s := item (get scores i)
      let yi := item (get y i)
      Tensor.scalar (hingeLossPerExample s yi))
  meanSpec losses

/-- L2-regularized SVM objective (primal, soft-margin style).

We use the common objective
$\frac12\lambda\lVert w\rVert^2+\operatorname{mean}(\text{hinge loss})$.
-/
def LinearSVM.objective {n p : Nat} (lambda : α) (m : LinearSVM p α)
  (X : Tensor α [n, p]) (y : Tensor α [n]) : α :=
  let scores := LinearSVM.decisionBatch (n := n) m X
  let hinge := hingeLossMean scores y
  -- Form each scaled square before summing. Computing ‖w‖² first can overflow even
  -- when a small regularization coefficient makes the objective representable.
  -- For |w| > 1, halve w before applying lambda; for |w| ≤ 1, apply lambda first.
  -- This also avoids prematurely halving a subnormal regularization coefficient.
  let penalty := (List.finRange p).foldl (fun total i =>
    let weight := Tensor.getScalar m.w i
    let term := if abs weight > (1 : α) then
      (lambda * (weight / (2 : α))) * weight
    else (lambda * weight) * (weight / (2 : α))
    total + term) 0
  penalty + hinge

/-!
### Backward pass

For the objective

`L(w,b) = ½λ‖w‖² + (1/n) Σ max(0, 1 - y_i (w·x_i + b))`

the gradients are:

- `∂L/∂w = λ w + (1/n) Σ [margin_i < 1] * (-y_i x_i)`
- `∂L/∂b = (1/n) Σ [margin_i < 1] * (-y_i)`

We also return `∂L/∂X` because it is sometimes useful for sensitivity analysis.

PyTorch analogy: this is what autograd would compute for
`0.5*λ*||w||^2 + mean(relu(1 - y*(X@w+b)))`, except we write it out explicitly.
-/

/--
Backward/VJP for the linear SVM objective.

Returns `(dw, db, dX)` where:
- `dw : ∂L/∂w`
- `db : ∂L/∂b`
- `dX : ∂L/∂X` (sometimes useful for sensitivity analysis)
-/
def LinearSVM.backward
  {n p : Nat}
  (lambda : α)
  (m : LinearSVM p α)
  (X : Tensor α [n, p])
  (y : Tensor α [n]) :
  (Tensor α [p] × α × Tensor α [n, p]) :=

  let nα : α := (n : α)
  let invN : α := 1 / (Max.max nα Context.defaultEpsilon)

  -- Regularization contribution: λ w
  let regDw : Tensor α [p] := scaleSpec m.w lambda

  -- Accumulate hinge contributions for `w` and `b` by folding over the dataset.
  -- The hinge is active exactly when `1 - y_i * score_i > 0`, i.e. when the margin is < 1.
  let (dwHinge, dbHinge) :=
    (List.finRange n).foldl
      (fun (acc : Tensor α [p] × α) idx =>
        let dw := acc.1
        let db := acc.2
        let xi := get X idx
        let yi := item (get y idx)
        let score := LinearSVM.decision m xi
        let oneMinusMargin := (1 : α) - (yi * score)
        if decide (oneMinusMargin > (0 : α)) then
          (addSpec dw (scaleSpec xi (-yi)), db + (-yi))
        else
          (dw, db))
      (Tensor.full (.dim p .scalar) 0, (0 : α))

  -- Per-example input gradients.
  let dXHinge : Tensor α [n, p] :=
    Tensor.dim (fun idx =>
      let xi := get X idx
      let yi := item (get y idx)
      let score := LinearSVM.decision m xi
      let oneMinusMargin := (1 : α) - (yi * score)
      let active : Bool := decide (oneMinusMargin > (0 : α))
      if active then
        scaleSpec m.w (-yi)
      else
        Tensor.full (.dim p .scalar) 0)

  -- Scale hinge part by 1/n, and combine with regularization.
  let dw := addSpec regDw (scaleSpec dwHinge invN)
  let db := dbHinge * invN
  let dX := scaleSpec dXHinge invN
  (dw, db, dX)

/-!
## A Small Training Wrapper (Gradient Descent)

The `LinearSVM` definitions above are enough for "spec math".
For examples/tests, it is convenient to package a trained parameter pair together with a simple
predictor, so we provide:

- `SVM`: a small record holding `(weights, bias)` and a heuristic support-vector index tensor,
- `fitLinearSVM`: deterministic gradient descent using `LinearSVM.backward`,
- `SVM.predict`: sign prediction as `±1`.
-/

/--
Small trained SVM bundle for examples/tests.

This is not a full SMO-style solver; it is a deterministic gradient-descent baseline that is
useful as a reference model in the TorchLean spec layer.
-/
structure SVM (p n : ℕ) (α : Type) [TorchLean.Storage α] where
  /-- Normal vector `w` of the separating hyperplane. -/
  weights : Tensor α [p]
  /-- Bias/intercept term `b`. -/
  bias : α
  /--
  One entry per training row: its index when the margin is near `1`, or the sentinel `n`
  otherwise. Filter out `n` before using entries to index the training data.
  -/
  supportVectorIndices : Tensor Nat [n]

/--
Heuristic support-vector index extractor.

We mark an example as a "support vector" if its margin is close to `1`. The output has one entry
per training row, containing that row's index or the sentinel `n` for a non-support row.
It is not a compact list of valid indices. This is only meant for introspection and examples
(it is not used by the optimizer).
-/
def findSupportVectorIndices {n p : Nat}
  (X : Tensor α [n, p])
  (y : Tensor α [n])
  (finalWeights : Tensor α [p])
  (finalBias : α) :
  Tensor Nat [n] :=

  Tensor.dim (fun i =>
    let x_i := get X i
    let y_i := item (get y i)
    let margin := y_i * (Tensor.dotSpec finalWeights x_i + finalBias)
    if abs (margin - (1 : α)) < ((1 / 10) : α) then
      Tensor.scalar i.val  -- support vector index
    else
      Tensor.scalar n      -- sentinel value, meaning "not a support vector"
  )

/-- A training label outside the signed encoding used by the hinge objective. -/
inductive SVM.FitError where
  /-- The zero-based row whose label is neither `-1` nor `1`. -/
  | invalidLabel (row : Nat)
  deriving Repr, BEq

/-- Fit a linear SVM after checking that every label is `-1` or `1`.

The check uses the scalar context's equality operation and precedes every optimization step,
including a request for zero steps. Binary `0`/`1` labels therefore cannot silently change the
hinge objective. `LinearSVM.objective` and `LinearSVM.backward` remain explicit formulas for
mathematical work with a supplied parameter record.
-/
def fitLinearSVM {n p : ℕ} (X : Tensor α [n, p]) (y : Tensor α [n])
    (learningRate : α) (lambda : α) (iterations : Nat) :
    Except SVM.FitError (SVM p n α) := do
  for i in List.finRange n do
    let label := Tensor.getScalar y i
    unless label == (-(1 : α)) || label == (1 : α) do
      throw (.invalidLabel i.val)
  -- Initialize weights with zeros
  let initialWeights : Tensor α [p] := Tensor.full [p] (0 : α)
  let initialBias := (0 : α)

  -- Implement gradient descent (structural recursion for predictable runtime)
  let rec gradientDescent (iter : Nat) (weights : Tensor α [p]) (bias : α) :
      (Tensor α [p] × α) :=
    match iter with
    | 0 => (weights, bias)
    | Nat.succ k =>
        let m : LinearSVM p α := { w := weights, b := bias }
        let (gradW, gradB, _dX) := LinearSVM.backward (n := n) (p := p) (lambda := lambda) m X y
        let newWeights := subSpec weights (scaleSpec gradW learningRate)
        let newBias := bias - learningRate * gradB
        gradientDescent k newWeights newBias

  -- Run gradient descent
  let (finalWeights, finalBias) := gradientDescent iterations initialWeights initialBias

  let supportVectorIndices := findSupportVectorIndices X y finalWeights finalBias

  return {
    weights := finalWeights
    bias := finalBias
    supportVectorIndices := supportVectorIndices }

/-- Predict signed labels `±1` using the learned hyperplane.

The training size `n` indexes the stored support-vector information. Prediction only needs the
weights and bias, so the inference batch has an independent size. A zero decision value gives
label `-1`; an empty batch returns an empty label tensor.
-/
def SVM.predict {n batch p : ℕ} (model : SVM p n α)
  (X : Tensor α [batch, p]) : Tensor α [batch] :=
  Tensor.dim (fun i =>
    let decisionValue := Tensor.dotSpec model.weights (get X i) + model.bias
    if decisionValue > (0 : α) then
      Tensor.scalar (1 : α)
    else
      -- Tie-breaking at `0` is arbitrary; we choose `-1` for determinism.
      Tensor.scalar (-(1 : α)))

/-- Predict a signed label for one feature vector using the batch predictor's arithmetic. -/
def SVM.predictOne {n p : ℕ} (model : SVM p n α) (x : Tensor α [p]) : α :=
  Tensor.getScalar (model.predict (Tensor.dim fun (_ : Fin 1) => x)) ⟨0, by decide⟩

namespace Kernel
/-- Linear kernel: `k(x, y) = x·y`. -/
def linear {p : ℕ} (x y : Tensor α [p]) : α :=
  Tensor.dotSpec x y

/-- Polynomial kernel: `k(x, y) = (x·y + c)^degree` (naive power for generic `α`). -/
def polynomial {p : ℕ} (degree : Nat) (c : α) (x y : Tensor α [p]) : α :=
  let dot := Tensor.dotSpec x y
  -- Generic `α` does not provide a `Float.pow`-style operation, so use recursive multiplication.
  let rec powRec (base : α) (exp : Nat) : α :=
    match exp with
    | 0 => (1 : α)
    | 1 => base
    | n + 1 => base * powRec base n
  powRec (dot + c) degree

/-- RBF kernel: `k(x, y) = exp(-gamma * ||x - y||^2)`.

When the squared distance is finite, evaluate the usual distance-then-exponential expression.
Finite feature differences can overflow that distance even when multiplication by a small
`gamma` gives a representable exponent. In that case, multiply each difference by `gamma`
before its second factor, then sum the weighted squares.
-/
def rbf {p : ℕ} (gamma : α) (x y : Tensor α [p]) {h : p ≠ 0} : α :=
  let diff := subSpec x y
  let squared := mulSpec diff diff
  let leadingAxis := (Shape.hasNonemptyAxisZeroOfNe h).proof
  let squaredDist := reduceSum 0 squared leadingAxis
  let distScalar := item squaredDist
  if distScalar - distScalar == 0 then
    -- Preserve the existing arithmetic and rounding whenever the distance is finite.
    exp (-gamma * distScalar)
  else
    let finiteDifferences := (List.finRange p).all fun i =>
      let value := Tensor.getScalar diff i
      value - value == 0
    if finiteDifferences && (gamma - gamma == 0) then
      -- Applying gamma before the square avoids an overflowing unweighted distance.
      -- This also keeps gamma zero well-defined for finite differences: each weighted
      -- term is zero, rather than multiplying zero by an already infinite distance.
      let weightedSquares := mapSpec (fun value => (gamma * value) * value) diff
      exp (-item (reduceSum 0 weightedSquares leadingAxis))
    else
      -- Nonfinite differences or gamma retain the original expression and its NaN or
      -- infinity propagation; they are not an overflow of otherwise finite inputs.
      exp (-gamma * distScalar)

end Kernel
