/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Support Vector Machines (spec models)

This file provides a small **linear SVM** baseline with explicit gradients.

PyTorch mental model:

- scoring function: `score = X @ w + b` (like `nn.Linear(p, 1)` without an activation),
- loss: hinge loss on signed labels `y ∈ {−1, +1}`:
  `loss_i = max(0, 1 - y_i * score_i)`,
- optimization: a small deterministic gradient descent loop (not an optimized solver).

There are two "layers" in this file:

- `LinearSVM`: the clean mathematical model + objective + backward pass (VJP-style gradients);
- `fitLinearSVM`/`predict`: a small training + prediction wrapper used by smoke tests and demos.

Note on naming: classic SVM literature often uses a parameter `C` that weights the hinge term.
In this file, `fitLinearSVM` takes a parameter named `C`, but we use it as the L2
regularization strength (the `lambda` in `LinearSVM.backward`) to keep the baseline small.

References:
- Cortes and Vapnik, "Support-Vector Networks", 1995.
- Vapnik, "The Nature of Statistical Learning Theory", 1995/1998.
-/

@[expose] public section


variable {α : Type} [Context α]
variable [DecidableRel ((· > ·) : α → α → Prop)]

open Spec
open Tensor
open MathFunctions
open Numbers

/-! ## Linear SVM (primal) -/

/-- Linear SVM parameters: a weight vector `w` and bias `b`.

We intentionally keep "training hyperparameters" (regularization strength, learning rate, etc.)
out of the parameter record; those are *choices about an optimizer*, not part of the model itself.
-/
structure LinearSVM (p : Nat) (α : Type) where
  /-- w. -/
  w : Tensor α (.dim p .scalar)
  /-- b. -/
  b : α

/-- Decision function `f(x) = w·x + b`. -/
def LinearSVM.decision {p : Nat} (m : LinearSVM p α) (x : Tensor α (.dim p .scalar)) : α :=
  Tensor.dotSpec m.w x + m.b

/-- Batch decision values for `X : (n×p)`. -/
def LinearSVM.decisionBatch {n p : Nat} (m : LinearSVM p α) (X : Tensor α (.dim n (.dim p
  .scalar))) :
  Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i =>
    Tensor.scalar (LinearSVM.decision m (getAtSpec X i)))

/-- Hinge loss per example: `ℓ_i = max(0, 1 - y_i * f(x_i))`.

We write it using `if` rather than `max` to make the "active-set" logic explicit. -/
def hingeLossPerExample (score y : α) : α :=
  let one_minus_margin := (1 : α) - (y * score)
  if one_minus_margin > (0 : α) then one_minus_margin else 0

/-- Mean hinge loss over a dataset. -/
def hingeLossMean {n : Nat} (scores : Tensor α (.dim n .scalar)) (y : Tensor α (.dim n .scalar)) :
  α :=
  let losses : Tensor α (.dim n .scalar) :=
    Tensor.dim (fun i =>
      let s := toScalar (getAtSpec scores i)
      let yi := toScalar (getAtSpec y i)
      Tensor.scalar (hingeLossPerExample s yi))
  meanSpec losses

/-- L2-regularized SVM objective (primal, soft-margin style).

We use the common "½λ‖w‖² + mean hinge" form.
-/
def LinearSVM.objective {n p : Nat} (lambda : α) (m : LinearSVM p α)
  (X : Tensor α (.dim n (.dim p .scalar))) (y : Tensor α (.dim n .scalar)) : α :=
  let scores := LinearSVM.decisionBatch (n := n) m X
  let hinge := hingeLossMean scores y
  let wnorm2 : α := Tensor.dotSpec m.w m.w
  (Numbers.pointfive * lambda * wnorm2) + hinge

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
  (X : Tensor α (.dim n (.dim p .scalar)))
  (y : Tensor α (.dim n .scalar)) :
  (Tensor α (.dim p .scalar) × α × Tensor α (.dim n (.dim p .scalar))) :=

  let nα : α := (n : α)
  let invN : α := 1 / (Max.max nα Numbers.epsilon)

  -- Regularization contribution: λ w
  let reg_dw : Tensor α (.dim p .scalar) := scaleSpec m.w lambda

  -- Accumulate hinge contributions for `w` and `b` by folding over the dataset.
  -- The hinge is active exactly when `1 - y_i * score_i > 0`, i.e. when the margin is < 1.
  let (dw_hinge, db_hinge) :=
    (List.finRange n).foldl
      (fun (acc : Tensor α (.dim p .scalar) × α) idx =>
        let dw := acc.1
        let db := acc.2
        let xi := getAtSpec X idx
        let yi := toScalar (getAtSpec y idx)
        let score := LinearSVM.decision m xi
        let one_minus_margin := (1 : α) - (yi * score)
        if decide (one_minus_margin > (0 : α)) then
          (addSpec dw (scaleSpec xi (-yi)), db + (-yi))
        else
          (dw, db))
      (fill 0 (.dim p .scalar), (0 : α))

  -- Per-example input gradients.
  let dX_hinge : Tensor α (.dim n (.dim p .scalar)) :=
    Tensor.dim (fun idx =>
      let xi := getAtSpec X idx
      let yi := toScalar (getAtSpec y idx)
      let score := LinearSVM.decision m xi
      let one_minus_margin := (1 : α) - (yi * score)
      let active : Bool := decide (one_minus_margin > (0 : α))
      if active then
        scaleSpec m.w (-yi)
      else
        fill 0 (.dim p .scalar))

  -- Scale hinge part by 1/n, and combine with regularization.
  let dw := addSpec reg_dw (scaleSpec dw_hinge invN)
  let db := db_hinge * invN
  let dX := scaleSpec dX_hinge invN
  (dw, db, dX)

/-!
## A Small Training Wrapper (Gradient Descent)

The `LinearSVM` definitions above are enough for "spec math".
For demos/tests, it is convenient to package a trained parameter pair together with a simple
predictor, so we provide:

- `SVM`: a small record holding `(weights, bias)` and a heuristic support-vector index tensor,
- `fitLinearSVM`: deterministic gradient descent using `LinearSVM.backward`,
- `predict`: sign prediction as `±1`.
-/

/--
Small trained SVM bundle for demos/tests.

This is not a full SMO-style solver; it is a deterministic gradient-descent baseline that is
useful as a reference model in the TorchLean spec layer.
-/
structure SVM (p n : ℕ) (α : Type) where
  /-- Normal vector `w` of the separating hyperplane. -/
  weights : Tensor α (.dim p .scalar)
  /-- Bias/intercept term `b`. -/
  bias : α
  /-- Heuristic support-vector indices (approximate: margin near `1`). -/
  supportVectorIndices : Tensor Nat (.dim n .scalar)

/--
Heuristic support-vector index extractor.

We mark an example as a "support vector" if its margin is close to `1`. This is only meant for
introspection and demos (it is not used by the optimizer).
-/
def findSupportVectorIndices {n p : Nat}
  (X : Tensor α (.dim n (.dim p .scalar)))
  (y : Tensor α (.dim n .scalar))
  (final_weights : Tensor α (.dim p .scalar))
  (final_bias : α) :
  Tensor Nat (.dim n .scalar) :=

  Tensor.dim (fun i =>
    let x_i := getAtSpec X i
    let y_i := toScalar (getAtSpec y i)
    let margin := y_i * (Tensor.dotSpec final_weights x_i + final_bias)
    if abs (margin - (1 : α)) < (pointone : α) then
      Tensor.scalar i.val  -- support vector index
    else
      Tensor.scalar n      -- sentinel value, meaning "not a support vector"
  )

-- Fit method using gradient descent for linear SVM
/--
Fit a linear SVM by deterministic gradient descent on the primal objective.

Parameters:
- `learning_rate`: gradient step size
- `C`: regularization strength (treated as `lambda`)
- `iterations`: number of GD steps
-/
def fitLinearSVM {n p : ℕ} (X : Tensor α (.dim n (.dim p .scalar))) (y : Tensor α (.dim n .scalar))
                 (learning_rate : α) (C : α) (iterations : Nat) : SVM p n α :=
  -- Initialize weights with zeros
  let initial_weights : Tensor α (.dim p .scalar) := fill (0 : α) (.dim p .scalar)
  let initial_bias := (0 : α)

  -- Implement gradient descent (structural recursion for predictable runtime)
  let rec gradient_descent (iter : Nat) (weights : Tensor α (.dim p .scalar)) (bias : α) :
      (Tensor α (.dim p .scalar) × α) :=
    match iter with
    | 0 => (weights, bias)
    | Nat.succ k =>
        let m : LinearSVM p α := { w := weights, b := bias }
        let (grad_w, grad_b, _dX) := LinearSVM.backward (n := n) (p := p) (lambda := C) m X y
        let new_weights := subSpec weights (scaleSpec grad_w learning_rate)
        let new_bias := bias - learning_rate * grad_b
        gradient_descent k new_weights new_bias

  -- Run gradient descent
  let (final_weights, final_bias) := gradient_descent iterations initial_weights initial_bias

  let supportVectorIndices := findSupportVectorIndices X y final_weights final_bias

  { weights := final_weights, bias := final_bias, supportVectorIndices := supportVectorIndices }

-- Predict method for linear SVM
/-- Predict signed labels `±1` for a batch `X` using the learned hyperplane. -/
def predict {n p : ℕ} (model : SVM p n α) (X : Tensor α (.dim n (.dim p .scalar))) : Tensor α (.dim
  n .scalar) :=
  Tensor.dim (fun i =>
    let decision_value := Tensor.dotSpec model.weights (getAtSpec X i) + model.bias
    if decision_value > (0 : α) then
      Tensor.scalar (1 : α)
    else
      -- Tie-breaking at `0` is arbitrary; we choose `-1` for determinism.
      Tensor.scalar (-(1 : α)))

-- Kernel functions
namespace Kernel
  -- Linear kernel: k(x, y) = x·y
/-- Linear kernel: `k(x, y) = x·y`. -/
def linear {p : ℕ} (x y : Tensor α (.dim p .scalar)) : α :=
  Tensor.dotSpec x y

-- Polynomial kernel: k(x, y) = (x·y + c)^d
/-- Polynomial kernel: `k(x, y) = (x·y + c)^degree` (naive power for generic `α`). -/
def polynomial {p : ℕ} (degree : Nat) (c : α) (x y : Tensor α (.dim p .scalar)) : α :=
  let dot := Tensor.dotSpec x y
  -- Generic `α` does not provide a `Float.pow`-style operation, so use recursive multiplication.
  let rec pow_rec (base : α) (exp : Nat) : α :=
    match exp with
    | 0 => (1 : α)
    | 1 => base
    | n + 1 => base * pow_rec base n
  pow_rec (dot + c) degree

-- RBF kernel: k(x, y) = exp(-γ||x-y||^2)
/-- RBF kernel: `k(x, y) = exp(-gamma * ||x - y||^2)`. -/
def rbf {p : ℕ} (gamma : α) (x y : Tensor α (.dim p .scalar)) {h : p ≠ 0} : α :=
  let diff := subSpec x y
  let squared := mulSpec diff diff
  -- Provide explicit proof that axis 0 is valid for Shape.dim p Shape.scalar
  have _ : Shape.valid_axis_inst 0 (Shape.dim p Shape.scalar) :=
    Shape.validAxisInstZeroAlt h
  let squaredDist := reduceSumAuto 0 squared
  let dist_scalar := toScalar squaredDist
  exp ((-gamma * dist_scalar))

end Kernel
