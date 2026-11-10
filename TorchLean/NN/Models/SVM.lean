/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce
public import TorchLean.Tensor.Layers.Loss

/-!
# Support Vector Machines (spec models)

A small linear SVM baseline with hand-written gradients. In PyTorch terms the score is
`score = X @ w + b` (an `nn.Linear(p, 1)` without activation), the loss is the hinge loss
`loss_i = max(0, 1 - y_i * score_i)` on signed labels `y ∈ {−1, +1}`, and training is a small
deterministic gradient-descent loop.

The file has two layers. `SVM.Linear` holds the mathematical model, its objective and the backward
pass (VJP-style gradients). `SVM.fit` and `SVM.predict` wrap it into a small training and
prediction interface. The kernels in `SVM.Kernel` stand alone, and `SVM.fit` trains the linear
primal without calling them. They are scikit-learn's `polynomial_kernel` with `gamma = 1` and
`coef0 = c`, and its `rbf_kernel`.

Classic SVM literature often puts a weight `C` on the hinge term. The objective here is the
equivalent primal form with an L2 coefficient named `lambda`. Dividing scikit-learn's SVC primal
`½‖w‖² + C Σ ζ_i` by `C n` gives ours with `lambda = 1 / (C n)`, so `C = 1` on 100 rows is
`lambda = 0.01`. Neither form penalizes the bias.

References:
- Cortes and Vapnik (1995), "Support-Vector Networks", Machine Learning 20(3), 273-297.
  https://doi.org/10.1007/BF00994018
- Vapnik (1995), "The Nature of Statistical Learning Theory", Springer.
  https://doi.org/10.1007/978-1-4757-2440-0
- scikit-learn 1.9.1 user guide, Section 1.4.7.1 (SVC primal and dual), and the pairwise
  `polynomial_kernel` and `rbf_kernel` docs. https://scikit-learn.org/1.9/modules/svm.html
- PyTorch 2.8 docs, Autograd mechanics, "Gradients for non-differentiable functions".
  https://docs.pytorch.org/docs/2.8/notes/autograd.html

## Implementation status

No API builder implements this model, and no theorem is proved about it. It is a reference
definition only.
-/

@[expose] public section


namespace TorchLean

variable {α : Type} [Storage α] [Context α]
variable [DecidableRel ((· > ·) : α → α → Prop)]

open TorchLean.Tensor
open FloatLib.Numerics.MathFunctions (abs exp)

/-! ## Linear SVM (primal) -/

/-- Linear SVM parameters: a weight vector `w` and a bias `b`.

Training hyperparameters such as the regularization strength and the learning rate belong to the
optimizer, so the parameter record leaves them out.
-/
structure SVM.Linear (p : Nat) (α : Type) [Storage α] where
  /-- Normal vector of the separating hyperplane. -/
  w : Tensor α [p]
  /-- Bias, or intercept, of the separating hyperplane. -/
  b : α

/-- Decision function `f(x) = w·x + b`. -/
def SVM.Linear.decision {p : Nat} (m : Linear p α) (x : Tensor α [p]) : α :=
  Tensor.dot m.w x + m.b

/-- Batch decision values for `X : (n×p)`. -/
def SVM.Linear.decisionBatch {n p : Nat} (m : Linear p α) (X : Tensor α [n, p]) :
  Tensor α [n] :=
  Tensor.dim (fun i =>
    Tensor.scalar (Linear.decision m (Tensor.unstack X i)))

/-- L2-regularized SVM objective (primal, soft-margin style).

It is the common objective $\frac12\lambda\lVert w\rVert^2+\operatorname{mean}(\text{hinge loss})$,
where the hinge loss is `Tensor.hingeLoss` of the scores against the signed labels.
-/
def SVM.Linear.objective {n p : Nat} (lambda : α) (m : Linear p α)
  (X : Tensor α [n, p]) (y : Tensor α [n]) : α :=
  let scores := Linear.decisionBatch (n := n) m X
  let hinge := Tensor.hingeLoss scores y
  -- Form each scaled square before summing: computing ‖w‖² first can overflow even when a
  -- small regularization coefficient makes the objective representable. For |w| > 1, w is
  -- halved before lambda is applied, and for |w| ≤ 1, lambda is applied first. Neither
  -- branch halves lambda, which would lose precision when lambda is subnormal.
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

the chain rule through the scores `s_i = w·x_i + b` gives, with `g = ∂L/∂s` the subgradient
`Tensor.hingeLossBackward` (`-y_i / n` where the margin is below one, zero elsewhere, a NaN at a NaN
margin):

- `∂L/∂w = λ w + Σ g_i x_i`
- `∂L/∂b = Σ g_i`

The result also includes `∂L/∂X`, whose row `i` is `g_i w`, because it is sometimes useful for
sensitivity analysis.

PyTorch analogy: autograd computes the same gradients for
`0.5*λ*||w||^2 + mean(relu(1 - y*(X@w+b)))`, and here they are written out by hand. At margin `1`
the hinge has a kink, and our `g_i = 0` there is the subgradient of minimum norm, the rule
PyTorch's autograd notes give for convex functions.
-/

/-- Backward pass of `Linear.objective`, returning `(∂L/∂w, ∂L/∂b, ∂L/∂X)`. -/
def SVM.Linear.backward
  {n p : Nat}
  (lambda : α)
  (m : Linear p α)
  (X : Tensor α [n, p])
  (y : Tensor α [n]) :
  (Tensor α [p] × α × Tensor α [n, p]) :=
  -- The subgradient of the mean hinge loss with respect to each score.
  let dScore := Tensor.hingeLossBackward (Linear.decisionBatch m X) y
  -- `X^T dScore`, folded over the dataset, and the regularization contribution `λ w`.
  let dwHinge := (List.finRange n).foldl
    (fun acc idx => acc + Tensor.scale (Tensor.unstack X idx) (Tensor.getScalar dScore idx))
    (Tensor.full (.dim p .scalar) 0)
  let dw := Tensor.scale m.w lambda + dwHinge
  let db := Tensor.sum dScore
  -- Per-example input gradients: row `i` is `dScore_i w`.
  let dX : Tensor α [n, p] :=
    Tensor.dim fun idx => Tensor.scale m.w (Tensor.getScalar dScore idx)
  (dw, db, dX)

/-!
## A Small Training Wrapper (Gradient Descent)

`SVM.fit` runs deterministic gradient descent with `SVM.Linear.backward` and packages the
result as an `SVM` record: the weights, the bias and a heuristic support-vector index tensor.
`SVM.predict` turns decision values into labels `±1`.
-/

/--
A linear SVM fitted by `SVM.fit`: the separating hyperplane and heuristic support-vector
indices. The fit is a deterministic gradient-descent baseline for the spec layer, with no
SMO-style solver.
-/
structure SVM (p n : ℕ) (α : Type) [Storage α] where
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
Heuristic support-vector indices: row `i` counts as a support vector when its margin
`y_i (w·x_i + b)` is within `1/10` of `1`.

The output has one entry per training row, holding that row's index or the sentinel `n` for a
non-support row, so its entries need filtering before they index the training data. The indices
are for introspection, and the optimizer never reads them. scikit-learn's SVM guide calls the
samples within the margin boundaries the support vectors, so a row at margin `0.5` counts there
and not here, and a row at `1.05` counts here and not there.
-/
def SVM.findSupportVectorIndices {n p : Nat}
  (X : Tensor α [n, p])
  (y : Tensor α [n])
  (finalWeights : Tensor α [p])
  (finalBias : α) :
  Tensor Nat [n] :=

  Tensor.dim (fun i =>
    let x_i := Tensor.unstack X i
    let y_i := item (Tensor.unstack y i)
    let margin := y_i * (Tensor.dot finalWeights x_i + finalBias)
    if abs (margin - (1 : α)) < ((1 / 10) : α) then
      Tensor.scalar i.val
    else
      Tensor.scalar n      -- sentinel value, meaning "not a support vector"
  )

/-- A training label outside the signed encoding used by the hinge objective. -/
inductive SVM.FitError where
  /-- The zero-based row whose label is neither `-1` nor `1`. -/
  | invalidLabel (row : Nat)
  deriving Repr, BEq

/-- Fit a linear SVM by `iterations` gradient-descent steps from zero weights and bias, after
checking that every label is `-1` or `1`.

The check uses the scalar context's equality and runs before any optimization step, even when
`iterations = 0`, so binary `0`/`1` labels cannot silently change the hinge objective.
`Linear.objective` and `Linear.backward` are plain formulas over a supplied parameter
record and take the labels as given.
-/
def SVM.fit {n p : ℕ} (X : Tensor α [n, p]) (y : Tensor α [n])
    (lr : α) (lambda : α) (iterations : Nat) :
    Except SVM.FitError (SVM p n α) := do
  for i in List.finRange n do
    let label := Tensor.getScalar y i
    unless label == (-(1 : α)) || label == (1 : α) do
      throw (.invalidLabel i.val)
  let initialWeights : Tensor α [p] := Tensor.full [p] (0 : α)
  let initialBias := (0 : α)

  -- Gradient descent by structural recursion on the step count, for a predictable runtime.
  let rec
    /-- `iter` gradient-descent steps with rate `lr` from `weights` and `bias`, with the gradients
    of `Linear.backward`. -/
    gradientDescent (iter : Nat) (weights : Tensor α [p]) (bias : α) :
        (Tensor α [p] × α) :=
      match iter with
      | 0 => (weights, bias)
      | Nat.succ k =>
          let m : Linear p α := { w := weights, b := bias }
          let (gradW, gradB, _dX) := Linear.backward (n := n) (p := p) (lambda := lambda) m X y
          let newWeights := weights - Tensor.scale gradW lr
          let newBias := bias - lr * gradB
          gradientDescent k newWeights newBias

  let (finalWeights, finalBias) := gradientDescent iterations initialWeights initialBias

  let supportVectorIndices := findSupportVectorIndices X y finalWeights finalBias

  return {
    weights := finalWeights
    bias := finalBias
    supportVectorIndices := supportVectorIndices }

/-- Predict signed labels `±1` using the learned hyperplane.

The training size `n` indexes the stored support-vector information. Prediction only needs the
weights and bias, so the inference batch has an independent size. Zero, negative and NaN decision
values give label `-1`, and an empty batch gives an empty label tensor.
-/
def SVM.predict {n batch p : ℕ} (model : SVM p n α)
  (X : Tensor α [batch, p]) : Tensor α [batch] :=
  Tensor.dim (fun i =>
    let decisionValue := Tensor.dot model.weights (Tensor.unstack X i) + model.bias
    if decisionValue > (0 : α) then
      Tensor.scalar (1 : α)
    else
      -- Ties at `0` go to `-1`, an arbitrary but fixed choice.
      Tensor.scalar (-(1 : α)))

/-- Predict a signed label for one feature vector using the batch predictor's arithmetic. -/
def SVM.predictOne {n p : ℕ} (model : SVM p n α) (x : Tensor α [p]) : α :=
  Tensor.getScalar (model.predict (Tensor.dim fun (_ : Fin 1) => x)) ⟨0, by decide⟩

namespace SVM.Kernel
/-- Polynomial kernel: `k(x, y) = (x·y + c)^degree` (naive power for generic `α`). -/
def polynomial {p : ℕ} (degree : Nat) (c : α) (x y : Tensor α [p]) : α :=
  let dot := Tensor.dot x y
  -- Fix the multiplication order for a natural degree. The scalar `Pow α α` operation
  -- does not specify this schedule, including the multiplication-free degree-one case.
  let rec
    /-- `base ^ exp` as the right-nested product `base * (base * ⋯ * base)`, `1` for `exp = 0`. -/
    powRec (base : α) (exp : Nat) : α :=
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
def rbf {p : ℕ} (gamma : α) (x y : Tensor α [p]) : α :=
  let diff := x - y
  let squared := diff * diff
  let squaredDist := castShape (reduce .sum squared [⟨0, by simp⟩] false)
      (Shape.reduce_singleton _ _)
  let distScalar := item squaredDist
  if distScalar - distScalar == 0 then
    -- A finite distance takes the usual formula, with its usual rounding.
    exp (-gamma * distScalar)
  else
    let finiteDifferences := (List.finRange p).all fun i =>
      let value := Tensor.getScalar diff i
      value - value == 0
    if finiteDifferences && (gamma - gamma == 0) then
      -- Applying gamma before the square avoids an overflowing unweighted distance. With gamma
      -- zero and finite differences each weighted term is zero, while the unweighted form
      -- would compute `0 * ∞ = NaN`.
      let weightedSquares := Tensor.map (fun value => (gamma * value) * value) diff
      exp (-item
          (castShape (reduce .sum weightedSquares [⟨0, by simp⟩] false)
          (Shape.reduce_singleton _ _)))
    else
      -- The weighted form only helps when finite differences overflow on squaring. A
      -- nonfinite difference or gamma keeps the usual expression and its NaN or infinity
      -- propagation.
      exp (-gamma * distScalar)

end SVM.Kernel

end TorchLean
