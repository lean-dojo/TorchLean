/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation

/-!
# Loss functions (spec layer)

This file defines a small collection of common losses (and their gradients) in a way that is:

- shape-generic: a loss takes `Tensor α s` and reduces it to a scalar `α`,
- explicit about reduction: most losses here are "mean over all elements",
- easy to line up with PyTorch terminology when you read training code.

In PyTorch you'll often see two layers:

- a low-level, elementwise loss (for example, Huber loss),
- plus a reduction (`mean` or `sum`).

TorchLean's spec layer mirrors that idea: most definitions are written as an elementwise formula
followed by a global mean over the shape.
-/

@[expose] public section


open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor
open MathFunctions

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Enumeration of supported loss families used by configuration records. -/
inductive LossType
| mse                    -- Mean Squared Error
| mae                    -- Mean Absolute Error
| huber                  -- Huber Loss
| crossEntropy           -- Cross-Entropy Loss
| hinge                  -- Hinge Loss
| poisson                -- Poisson Loss
| cosineSimilarity       -- Cosine Similarity Loss
| logCosh                -- Log-Cosh Loss

/-- Loss configuration record that names the selected loss family. -/
structure Loss where
  /-- Selected loss family for this configuration. -/
  lossType : LossType
  -- Note: regularization would be added here if needed

/-- Configuration selecting mean-squared-error loss. -/
def Loss.mse : Loss :=
  { lossType := LossType.mse }

/-- Configuration selecting mean-absolute-error loss. -/
def Loss.mae : Loss :=
  { lossType := LossType.mae }

/-- Configuration selecting Huber loss. -/
def Loss.huber : Loss :=
  { lossType := LossType.huber }

/-- Cross-entropy loss configuration. -/
def Loss.crossEntropy : Loss :=
  { lossType := LossType.crossEntropy }

/-- Configuration selecting hinge loss. -/
def Loss.hinge : Loss :=
  { lossType := LossType.hinge }

/-- Poisson loss configuration. -/
def Loss.poisson : Loss :=
  { lossType := LossType.poisson }

/-- Cosine similarity loss configuration. -/
def Loss.cosineSimilarity : Loss :=
  { lossType := LossType.cosineSimilarity }

/-- Log-cosh loss configuration. -/
def Loss.logCosh : Loss :=
  { lossType := LossType.logCosh }

-- Pure loss function specifications

/-- Sum all tensor elements into a single scalar. -/
def toScalarSpec {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {s : Shape} : Tensor α s → α :=
  sumSpec

/-- Mean of a scalar that conceptually came from a tensor with shape `s`.

The denominator is `TorchLean.Tensor.meanDenominator`, the same totalized element count the tensor
reductions use, so the loss layer and `mean` agree on what an empty tensor averages to. -/
def meanOver {α : Type} [Div α] [NatCast α] {s : Shape} (x : α) : α :=
  x / (meanDenominator s : α)

/-- Number of slices orthogonal to `axis` in a tensor shape.

Classification losses sum along the selected class dimension and average over every other
dimension. Thus a class vector has one slice, a matrix of shape `(batch, classes)` with class
dimension `1` has `batch` slices, and a tensor may instead place its class dimension anywhere in
the shape. -/
def axisSliceCount (s : Shape) (axis : Nat) [Shape.AxisInBounds axis s] : Nat :=
  Spec.Shape.size (shapeAfterSum s axis)

/-- Totalized denominator for a mean over slices orthogonal to `axis`. -/
def axisMeanDenom (s : Shape) (axis : Nat) [Shape.AxisInBounds axis s] : Nat :=
  if axisSliceCount s axis = 0 then 1 else axisSliceCount s axis

/-- Divide a classification loss by the number of slices orthogonal to `axis`. -/
def meanOverAxisSlices {α : Type} [Div α] [NatCast α] {s : Shape}
    (axis : Nat) [Shape.AxisInBounds axis s] (x : α) : α :=
  x / (axisMeanDenom s axis : α)

/-- Mean squared error: average of $(\mathtt{predicted}-\mathtt{target})^2$. -/
def mseSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Sub α] [Mul α] [Div α] [Zero α] [NatCast α]
    {s : Shape} (predicted target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let squared := mulSpec diff diff
  meanOver (s := s) (toScalarSpec squared)

/-- Derivative of `mseSpec` with respect to `predicted`. -/
def mseDerivSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Sub α] [Mul α] [Div α] [One α] [NatCast α]
    {s : Shape} (predicted target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  -- Corresponds to PyTorch's `MSELoss(reduction="mean")`.
  -- d/dpred ( (1/N) * Σᵢ (predᵢ - tgtᵢ)^2 ) = (2/N) * (pred - tgt)
  let n : α := (meanDenominator s : α)
  scaleSpec diff (((1 : α) + 1) / n)

/-- Mean absolute error: average of `|predicted - target|`. -/
def maeSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let abs_diff := absSpec diff
  meanOver (s := s) (toScalarSpec abs_diff)

/-- Derivative of `maeSpec` w.r.t. `predicted` (subgradient via sign). -/
def maeDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  -- Corresponds to PyTorch's `L1Loss(reduction="mean")`.
  -- This is a subgradient at 0.
  let grad :=
    mapSpec (fun x => if x > (0 : α) then (1 : α) else if x < (0 : α) then -(1 : α) else (0 : α))
      diff
  scaleSpec grad (1 / (meanDenominator s : α))

/--
Huber loss with transition parameter `delta`.

Elementwise, for residual $d=\mathtt{pred}-\mathtt{target}$:

- if $\lvert d\rvert<\delta$: $\tfrac12d^2$
- otherwise: $\delta(\lvert d\rvert-\tfrac12\delta)$

Then we take a mean over all elements.

This is PyTorch's `HuberLoss` convention. It differs from `SmoothL1Loss` by a factor of `delta`.
The Huber interpretation requires `delta > 0`.
-/
def huberSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) (delta : α := (1 : α)) : α
  :=
  let diff := subSpec predicted target
  let abs_diff := absSpec diff
  let per_elem := mapSpec (fun x =>
    if x < delta then
      (x * x) / 2
    else
      delta * (x - delta / 2)) abs_diff
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative of `huberSpec` w.r.t. `predicted`. -/
def huberDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) (delta : α := (1 :
  α)) : Tensor α s :=
  let diff := subSpec predicted target
  -- Subgradient at `|d| = delta` is fine for spec purposes; we pick the natural piecewise form.
  let grad :=
    mapSpec (fun d =>
      let ad := if d > (0 : α) then d else -d
      if ad < delta then d
      else if d > (0 : α) then delta
      else if d < (0 : α) then -delta
      else (0 : α)
    ) diff
  scaleSpec grad (1 / (meanDenominator s : α))

/--
Cross-entropy between distributions (probabilities).

This is closest to PyTorch when you already have probabilities `q` (e.g. after a softmax) and a
probability target `p` (e.g. one-hot or label-smoothed), and you want:

$$
\operatorname{CE}(p,q)
=-\operatorname{mean}_r\sum_c p_{rc}\log q_{rc},
$$

where `c` ranges along the selected class dimension and `r` ranges over all remaining dimensions.
A lone class vector is one distribution and is not divided by its number of classes.

Each predicted probability is clamped to `[epsilon, 1 - epsilon]` before the logarithm. A one-hot
prediction that exactly matches a one-hot target therefore has loss `-log(1 - epsilon)`, which is
small but not `0`.

PyTorch's `F.cross_entropy` typically takes logits and does `log_softmax + NLLLoss`; that is a
different API surface than this "probabilities in, scalar out" spec.
-/
def crossEntropySpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (predicted : Tensor α s) (target : Tensor α s) (epsilon : α := Context.defaultEpsilon) : α :=
  -- Sum over each class distribution, then average over all dimensions other than `axis`.
  let clamp01 := fun x : α =>
    let x := if x > epsilon then x else epsilon
    if x < (1 : α) - epsilon then x else (1 : α) - epsilon
  let q := mapSpec clamp01 predicted
  let logq := logSpec q
  let total := sumSpec (mulSpec target logq)
  meanOverAxisSlices (s := s) axis (-total)

/-- Derivative of `crossEntropySpec` w.r.t. `predicted`. -/
def crossEntropyDerivSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (predicted : Tensor α s) (target : Tensor α s) (epsilon : α := Context.defaultEpsilon) :
    Tensor α s :=
  -- The forward clamp is locally constant outside `(epsilon, 1 - epsilon)`, so its branch
  -- derivative is zero there. At the two clipping kinks this definition selects the zero
  -- subgradient. Inside the interval the derivative is the usual `-target / predicted`.
  let grad := map2Spec (fun q p =>
    if q > epsilon then
      if q < (1 : α) - epsilon then -p / q else 0
    else
      0) predicted target
  scaleSpec grad (1 / (axisMeanDenom s axis : α))

/--
Cross-entropy on logits (stable log-softmax form).

This matches the common PyTorch decomposition:

$$
\operatorname{cross\_entropy}(\mathtt{logits},\mathtt{target})
=-\operatorname{mean}_r\sum_c
  \mathtt{target}_{rc}\operatorname{logsoftmax}(\mathtt{logits})_{rc},
$$

where the dimension selected by `axis` contains classes and `r` ranges over the remaining
dimensions. This is PyTorch's `reduction="mean"` convention for one-hot or soft distribution
targets.

Unlike `crossEntropySpec`, this takes *logits* and uses `Activation.logSoftmaxSpec` for
numerical stability.

Probability targets usually sum to one along `axis`, as in one-hot or label-smoothed targets.
The same formula also accepts arbitrary target weights. Their sum scales the contribution of
that slice; the reduction still divides by the number of slices, not by the total target weight. -/
def crossEntropyLogitsSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits : Tensor α s) (target : Tensor α s) : α :=
  let logp := Activation.logSoftmaxSpec (α := α) (s := s) axis logits
  let total := sumSpec (mulSpec target logp)
  meanOverAxisSlices (s := s) axis (-total)

/-- Derivative of `crossEntropyLogitsSpec` with respect to the logits.

For each class slice, differentiating the log-normalizer contributes
`softmax(logits) * sum(target)`. The direct logit term contributes `-target`. Keeping that
slice sum makes the derivative agree with the loss for weighted, unnormalized, and zero targets.
For a probability target, its sum is one and the familiar `softmax(logits) - target` follows.

The reduction drops the class axis and then restores that exact axis. Ordinary trailing-axis
broadcasting would mix slices when classes occupy an outer or middle dimension. Empty class
axes use the same zero-sum convention as the loss. -/
def crossEntropyLogitsDerivSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits : Tensor α s) (target : Tensor α s) :
    Tensor α s :=
  let probs := Activation.softmaxSpec (α := α) (s := s) axis logits
  let targetMass := reduceDim sumSpec axis target
  let grad := subSpec (mulSpec probs (broadcastAfterSum s axis targetMass)) target
  scaleSpec grad (1 / (axisMeanDenom s axis : α))

/--
Hinge loss (binary margin loss), elementwise then mean-reduced:

$\operatorname{hinge}(x,y)=\operatorname{mean}_i\max(0,1-y_i x_i)$.

This matches the usual SVM-style hinge loss. (PyTorch exposes similar behavior via margin-style
losses such as `HingeEmbeddingLoss` / `MultiMarginLoss`, but the exact signature differs.)
-/
def hingeSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let margin := mulSpec predicted target
  let per_elem := mapSpec (fun m =>
    let v := (1 : α) - m
    if v > (0 : α) then v else (0 : α)
  ) margin
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative/subgradient of `hingeSpec` w.r.t. `predicted`. -/
def hingeDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let margin := mulSpec predicted target
  -- Subgradient: if `1 - y*x > 0` then `d/dx = -y`, else 0. Then mean-reduce.
  let active := mapSpec (fun m => if (1 : α) - m > (0 : α) then (1 : α) else (0 : α)) margin
  let grad := mulSpec active (negSpec target)
  scaleSpec grad (1 / (meanDenominator s : α))

/--
Poisson negative log-likelihood (log-input form), elementwise then mean-reduced:

If `predicted` represents `log(rate)` and `target` is a nonnegative count,
then (up to an additive constant that does not affect gradients):

$\operatorname{loss}_i=\exp(\mathtt{pred}_i)-\mathtt{target}_i\mathtt{pred}_i$.

This corresponds to PyTorch's `PoissonNLLLoss(log_input=true, full=false)` at the math level.
-/
def poissonSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let exp_pred := mapSpec MathFunctions.exp predicted
  let target_times_pred := mulSpec target predicted
  let per_elem := subSpec exp_pred target_times_pred
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative of `poissonSpec` w.r.t. `predicted`. -/
def poissonDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  -- d/dpred [exp(pred) - target*pred] = exp(pred) - target, then mean-reduce.
  let exp_pred := mapSpec MathFunctions.exp predicted
  let grad := subSpec exp_pred target
  scaleSpec grad (1 / (meanDenominator s : α))

/-- Cosine similarity loss: `1 - cos(predicted, target)` (reduced-to-scalar). -/
def cosineSimilaritySpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let dot_product := mulSpec predicted target
  let pred_squared := mulSpec predicted predicted
  let target_squared := mulSpec target target
  let dot_sum := toScalarSpec dot_product
  let pred_norm := MathFunctions.sqrt (toScalarSpec pred_squared)
  let target_norm := MathFunctions.sqrt (toScalarSpec target_squared)
  let pred_norm_safe := if pred_norm > epsilon then pred_norm else epsilon
  let target_norm_safe := if target_norm > epsilon then target_norm else epsilon
  let cosine_sim := dot_sum / (pred_norm_safe * target_norm_safe)
  (1 : α) - cosine_sim

/--
Derivative of `cosineSimilaritySpec` w.r.t. `predicted`.

If $\cos=(p\mathbin{\cdot}t)/(\lVert p\rVert\lVert t\rVert)$ and
$\operatorname{loss}=1-\cos$, then (for nonzero norms):

$$
\frac{\partial\operatorname{loss}}{\partial p}
=\frac{p\mathbin{\cdot}t}{\lVert p\rVert^3\lVert t\rVert}p
 -\frac{1}{\lVert p\rVert\lVert t\rVert}t.
$$

We use `epsilon` to avoid division by zero (similar to common "eps" handling in PyTorch code).
-/
def cosineSimilarityDerivSpec {s : Shape}
  (predicted : Tensor α s) (target : Tensor α s) (epsilon : α := Context.defaultEpsilon) :
    Tensor α s :=
  let dot_sum := toScalarSpec (mulSpec predicted target)
  let pred_sq_sum := toScalarSpec (mulSpec predicted predicted)
  let target_sq_sum := toScalarSpec (mulSpec target target)
  let pred_norm := MathFunctions.sqrt pred_sq_sum
  let target_norm := MathFunctions.sqrt target_sq_sum
  let pred_norm_safe := if pred_norm > epsilon then pred_norm else epsilon
  let target_norm_safe := if target_norm > epsilon then target_norm else epsilon
  let denom := pred_norm_safe * target_norm_safe
  -- When `pred_norm ≤ epsilon`, the denominator selected by the forward pass is locally constant
  -- with respect to `predicted`; the radial derivative term is therefore zero on that branch.
  let c1 :=
    if pred_norm > epsilon then
      dot_sum / (pred_norm_safe * pred_norm_safe * pred_norm_safe * target_norm_safe)
    else
      0
  let term1 := scaleSpec predicted c1
  let term2 := scaleSpec target (1 / denom)
  subSpec term1 term2

/-- Log-cosh loss (reduced-to-scalar): `log(cosh(predicted - target))`. -/
def logCoshSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let per_elem := mapSpec (fun d => MathFunctions.log (MathFunctions.cosh d)) diff
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative of `logCoshSpec` w.r.t. `predicted`. -/
def logCoshDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  let grad := mapSpec MathFunctions.tanh diff
  scaleSpec grad (1 / (meanDenominator s : α))

/--
Binary cross-entropy on scalars (probabilities), with clipping to avoid `log(0)`.

This matches the core formula behind PyTorch's `BCELoss` when `predicted` is already a probability
(not a logit):

$$
\operatorname{BCE}(p,y)
=-\left(y\log p+(1-y)\log(1-p)\right).
$$

Assumption: `target` is in `[0, 1]`. We do not clip the target; we only clip `predicted`.
-/
def binaryCrossEntropySpec (predicted : α) (target : α)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let p := if predicted > epsilon then predicted else epsilon
  let p := if p < (1 : α) - epsilon then p else (1 : α) - epsilon
  let log_p := MathFunctions.log p
  let log_one_minus_p := MathFunctions.log ((1 : α) - p)
  let t := target * log_p + ((1 : α) - target) * log_one_minus_p
  (0 : α) - t

/-- Selected derivative of `binaryCrossEntropySpec` w.r.t. `predicted`.

The clipped forward function is not differentiable at `epsilon` or `1 - epsilon`; this definition
chooses zero at those two kinks and on the clipped exterior branches. -/
def binaryCrossEntropyDerivSpec (predicted : α) (target : α)
    (epsilon : α := Context.defaultEpsilon) :
  α :=
  -- As in `crossEntropyDerivSpec`, use the derivative of the selected clamp branch.
  if predicted > epsilon then
    if predicted < (1 : α) - epsilon then
      (predicted - target) / (predicted * ((1 : α) - predicted))
    else
      0
  else
    0

/-- Tensor BCE (probabilities), elementwise then mean-reduced. -/
def binaryCrossEntropyTensorSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let per_elem := map2Spec (fun p y => binaryCrossEntropySpec (predicted := p) (target := y)
    (epsilon := epsilon))
      predicted target
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative of `binaryCrossEntropyTensorSpec` w.r.t. `predicted`. -/
def binaryCrossEntropyTensorDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α
  s)
    (epsilon : α := Context.defaultEpsilon) : Tensor α s :=
  let grad := map2Spec (fun p y => binaryCrossEntropyDerivSpec (predicted := p) (target := y)
    (epsilon := epsilon))
      predicted target
  scaleSpec grad (1 / (meanDenominator s : α))

end Spec
