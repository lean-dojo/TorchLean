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

The teacher-logit gradient uses logit differences, centering, and a residual correction as in
Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and Cross-Entropy"
(https://arxiv.org/abs/2610.31642). Its near-field expression retains small gradients that would
be lost by subtracting two separately rounded probability vectors.
-/

@[expose] public section


open TorchLean

namespace Spec
open TorchLean TorchLean.Tensor
open MathFunctions

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Weight a difference, distributing the multiplication when subtraction overflows.
Ordinary finite differences keep their evaluation order. For opposite-sign finite operands,
weighting each operand first can keep the result representable without halving small tangents. -/
def weightedDifference {α : Type} [Zero α] [Sub α] [Mul α] [BEq α]
    (weight left right : α) : α :=
  let difference := left - right
  if difference - difference == 0 then weight * difference
  else weight * left - weight * right

/-- Totalized denominator for a mean over slices orthogonal to `axis`.

Classification losses sum along the selected class dimension and average over every other
dimension. Thus a class vector has one slice, a matrix of shape `(batch, classes)` with class
dimension `1` has `batch` slices, and a tensor may instead place its class dimension anywhere in
the shape. The shared `meanDenominator` supplies the empty-slice convention. -/
def axisMeanDenom (s : Shape) (axis : Nat) [Shape.AxisInBounds axis s] : Nat :=
  meanDenominator (shapeAfterSum s axis)

/-- Divide a classification loss by the number of slices orthogonal to `axis`. -/
def meanOverAxisSlices {α : Type} [Div α] [NatCast α] {s : Shape}
    (axis : Nat) [Shape.AxisInBounds axis s] (x : α) : α :=
  x / (axisMeanDenom s axis : α)

/-- Mean squared error: average of $(\mathtt{predicted}-\mathtt{target})^2$. -/
def mseSpec {α : Type} [TorchLean.Storage α]
    [Add α] [Sub α] [Mul α] [Div α] [Zero α] [NatCast α]
    {s : Shape} (predicted target : Tensor α s) : α :=
  Tensor.meanSquaredError predicted target

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
  meanSpec abs_diff

/-- Derivative of `maeSpec` w.r.t. `predicted` (subgradient via sign). -/
def maeDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  -- Corresponds to PyTorch's `L1Loss(reduction="mean")`.
  -- This is a subgradient at 0.
  let grad :=
    Tensor.map (fun x => if x > (0 : α) then (1 : α) else if x < (0 : α) then -(1 : α) else (0 : α))
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

The quadratic branch halves before squaring. If a finite prediction and target have an
overflowing difference, the linear branch instead weights both operands before subtraction.
Nonfinite inputs and genuinely unrepresentable losses or reductions can still overflow.
-/
def huberSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) (delta : α := (1 : α)) : α
  :=
  let per_elem := map2Spec (fun p t =>
    let difference := p - t
    let x := MathFunctions.abs difference
    if x < delta then
      -- Halve before squaring so a representable half-square need not overflow first.
      (x / 2) * x
    else
      let direct := delta * (x - delta / 2)
      if difference - difference == 0 || !(delta > 0) then direct
      else
        let weighted := MathFunctions.abs (weightedDifference delta p t)
        if weighted - weighted == 0 then weighted - (delta / 2) * delta
        else direct) predicted target
  meanSpec per_elem

/-- Derivative of `huberSpec` w.r.t. `predicted`. -/
def huberDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) (delta : α := (1 :
  α)) : Tensor α s :=
  let diff := subSpec predicted target
  -- Subgradient at `|d| = delta` is fine for spec purposes; we pick the natural piecewise form.
  let grad :=
    Tensor.map (fun d =>
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
  let q := Tensor.map clamp01 predicted
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

namespace CrossEntropy

/-- Weighted log-probabilities for one class vector. Keep the original logits available to
reconstruct a weighted term when the unweighted log-probability overflows. -/
def vectorTerms {n : Nat} (logits target : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => logits
  | n + 1 =>
      let logp := Activation.logSoftmaxVecSpec logits
      let pivot : Fin (n + 1) :=
        Fin.cast (by simp [Shape.size]) (Tensor.argmax (by simp [Shape.size]) logits)
      let largest := logits.getScalar pivot
      let normalizer := -logp.getScalar pivot
      Tensor.ofFn fun i =>
        let weight := target.getScalar i
        let value := logp.getScalar i
        if value - value == 0 then weight * value
        else -(weight * normalizer + weightedDifference weight largest (logits.getScalar i))

/-- Apply the weighted-log-probability calculation independently to each final-axis vector. -/
def innermostTerms : {s : Shape} → Tensor α s → Tensor α s → Tensor α s
  | .scalar, _, _ => Tensor.scalar 0
  | .dim _ .scalar, logits, target => vectorTerms logits target
  | .dim _ inner, logits, target => Tensor.dim fun i =>
      innermostTerms (s := inner) (Tensor.unstack logits i) (Tensor.unstack target i)

/-- Weighted log-probabilities along `axis`, restored to the original tensor layout before
the loss reduction. This preserves the existing accumulation order for non-trailing axes. -/
def terms {s : Shape} (axis : Nat) (logits target : Tensor α s) : Tensor α s :=
  let swaps := Shape.moveAxisToInnermostSwaps s.rank axis
  let values := innermostTerms (Tensor.permuteByAdjacentSwaps logits swaps)
    (Tensor.permuteByAdjacentSwaps target swaps)
  let restored := Tensor.permuteByAdjacentSwaps values swaps.reverse
  Shape.applyAdjacentSwaps_reverse s swaps ▸ restored

end CrossEntropy

/--
Cross-entropy on logits, retaining the original logits when a weighted term needs rescaling.

This matches the common PyTorch decomposition:

$$
\operatorname{cross\_entropy}(\mathtt{logits},\mathtt{target})
=-\operatorname{mean}_r\sum_c
  \mathtt{target}_{rc}\operatorname{logsoftmax}(\mathtt{logits})_{rc},
$$

where the dimension selected by `axis` contains classes and `r` ranges over the remaining
dimensions. This is PyTorch's `reduction="mean"` convention for one-hot or soft distribution
targets.

Unlike `crossEntropySpec`, this takes *logits* and uses `Activation.logSoftmaxSpec`. Ordinary
finite log-probabilities are multiplied as usual. When a log-probability is nonfinite, its
weighted term is reconstructed from the shifted normalizer and a rescaled logit difference.
Thus zero or small weights need not multiply an overflowing unweighted difference. No target
mask discards target derivatives. Nonfinite inputs or overflowing reductions can still yield NaN.

Probability targets usually sum to one along `axis`, as in one-hot or label-smoothed targets.
The same formula also accepts arbitrary target weights. Their sum scales the contribution of
that slice; the mean still divides by the number of slices, not by the total target weight.
Set `mean := false` to sum the contributions without averaging. -/
def crossEntropyLogitsSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits : Tensor α s) (target : Tensor α s) (mean : Bool := true) : α :=
  let total := sumSpec (CrossEntropy.terms axis logits target)
  if mean then meanOverAxisSlices (s := s) axis (-total) else -total

/-- Derivative of `crossEntropyLogitsSpec` with respect to the logits.

For each class slice, differentiating the log-normalizer contributes
`softmax(logits) * sum(target)`. The direct logit term contributes `-target`. Keeping that
slice sum makes the derivative agree with the loss for weighted, unnormalized, and zero targets.
For a probability target, its sum is one and the familiar `softmax(logits) - target` follows.
Evaluation uses the shared log-softmax VJP rather than forming this difference directly;
the pivot's complementary probability is retained when its exponential rounds to one.

The reduction drops the class axis and then restores that exact axis. Ordinary trailing-axis
broadcasting would mix slices when classes occupy an outer or middle dimension. Empty class
axes use the same zero-sum convention as the loss. -/
def crossEntropyLogitsDerivSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits : Tensor α s) (target : Tensor α s) (mean : Bool := true) :
    Tensor α s :=
  let logp := Activation.logSoftmaxSpec (α := α) (s := s) axis logits
  let grad := Activation.logSoftmaxBackwardSpec axis logp (Tensor.map (fun x => -x) target)
  if mean then scaleSpec grad (1 / (axisMeanDenom s axis : α)) else grad

/-- Target-weight derivative of logits cross-entropy, multiplied by an upstream scalar seed.
Reconstructing the weighted log-probability before reduction also handles zero or small seeds
without first materializing a possibly overflowing unweighted target derivative. -/
def CrossEntropy.targetGradient {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits : Tensor α s) (seed : α := 1) (mean : Bool := true) : Tensor α s :=
  let terms := CrossEntropy.terms axis logits (Tensor.full s seed)
  if mean then Tensor.map (fun value => -value / (axisMeanDenom s axis : α)) terms
  else negSpec terms

/-- Cross-entropy against a fixed teacher's logits, averaged over class slices.

Unlike probability targets, teacher logits retain the information needed to compute small
student–teacher gradient differences. Zero teacher weights contribute zero even when a student
log-probability overflows to negative infinity. The teacher is held fixed during differentiation.
-/
def distillSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (student teacher : Tensor α s) : α :=
  let p := Activation.softmaxSpec axis teacher
  let logq := Activation.logSoftmaxSpec axis student
  let terms := map2Spec (fun weight logp => if weight == 0 then 0 else weight * logp) p logq
  meanOverAxisSlices (s := s) axis (-sumSpec terms)

/-- Student gradient of `distillSpec`, evaluated from logit differences.

Center the difference under the teacher distribution, then compute its log-normalizer with
`expm1` and `log1p`. Near equal logits this avoids subtracting rounded probabilities. Coordinates
far from the teacher use the ordinary probability difference instead, avoiding `0 * ∞` in a
small teacher weight times a large exponential. The final correction removes reduction residuals.
The mean uses the same class-slice denominator as `crossEntropyLogitsSpec`.
-/
def distillDerivSpec {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (student teacher : Tensor α s) : Tensor α s :=
  let p := Activation.softmaxSpec axis teacher
  let q := Activation.softmaxSpec axis student
  let delta := subSpec student teacher
  let center := broadcastAfterSum s axis (reduceDim sumSpec axis (mulSpec p delta))
  let d := subSpec delta center
  let correction := Tensor.map Activation.Math.log1pSpec
    (reduceDim sumSpec axis (mulSpec p (Tensor.map Activation.Math.expm1Spec d)))
  let logRatio := subSpec d (broadcastAfterSum s axis correction)
  let near := map2Spec (fun weight ratio =>
    if MathFunctions.abs ratio < 1 then weight * Activation.Math.expm1Spec ratio else 0) p logRatio
  let far := map2Spec (fun ratio difference =>
    if MathFunctions.abs ratio < 1 then 0 else difference) logRatio (subSpec q p)
  let difference := addSpec near far
  let residual := broadcastAfterSum s axis (reduceDim sumSpec axis difference)
  scaleSpec (subSpec difference (mulSpec q residual)) (1 / (axisMeanDenom s axis : α))

/--
Hinge loss (binary margin loss), elementwise then mean-reduced:

$\operatorname{hinge}(x,y)=\operatorname{mean}_i\max(0,1-y_i x_i)$.

This matches the usual SVM-style hinge loss. (PyTorch exposes similar behavior via margin-style
losses such as `HingeEmbeddingLoss` / `MultiMarginLoss`, but the exact signature differs.)
-/
def hingeSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let margin := mulSpec predicted target
  let per_elem := Tensor.map (fun m =>
    let v := (1 : α) - m
    if v > (0 : α) then v else (0 : α)
  ) margin
  meanSpec per_elem

/-- Derivative/subgradient of `hingeSpec` w.r.t. `predicted`. -/
def hingeDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let margin := mulSpec predicted target
  -- Subgradient: if `1 - y*x > 0` then `d/dx = -y`, else 0. Then mean-reduce.
  let active := Tensor.map (fun m => if (1 : α) - m > (0 : α) then (1 : α) else (0 : α)) margin
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
  let exp_pred := Tensor.map MathFunctions.exp predicted
  let target_times_pred := mulSpec target predicted
  let per_elem := subSpec exp_pred target_times_pred
  meanSpec per_elem

/-- Derivative of `poissonSpec` w.r.t. `predicted`. -/
def poissonDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  -- d/dpred [exp(pred) - target*pred] = exp(pred) - target, then mean-reduce.
  let exp_pred := Tensor.map MathFunctions.exp predicted
  let grad := subSpec exp_pred target
  scaleSpec grad (1 / (meanDenominator s : α))

namespace cosineSimilaritySpec

/-- Scale and scaled, epsilon-clipped norm. For positive epsilon over the reals, both are
positive and their product is the clipped L2 norm. Numerical callers divide by them separately. -/
def normFactors {s : Shape} (x : Tensor α s) (epsilon : α) : α × α :=
  let scale := Context.stopGradient (max (Tensor.maxAbs x) (MathFunctions.abs epsilon))
  let squared := Tensor.map (fun value => let scaled := value / scale; scaled * scaled) x
  let norm := MathFunctions.sqrt (sumSpec squared)
  let cutoff := epsilon / scale
  (scale, if norm > cutoff then norm else cutoff)

end cosineSimilaritySpec

/-- Cosine similarity loss: `1 - cos(predicted, target)` (reduced-to-scalar).

Normalize each tensor before the dot product, using a shared scale before squaring. This avoids
forming raw dot products or products of large norms. The norm floor remains `epsilon`, not an
additive regularizer. Pass a positive, finite epsilon representable in the scalar type.
Nonfinite inputs and an overflowing scaled reduction can still produce nonfinite results. -/
def cosineSimilaritySpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let (predScale, predNorm) := cosineSimilaritySpec.normFactors predicted epsilon
  let (targetScale, targetNorm) := cosineSimilaritySpec.normFactors target epsilon
  let normalizedPred := Tensor.map (fun value => (value / predScale) / predNorm) predicted
  let normalizedTarget := Tensor.map (fun value => (value / targetScale) / targetNorm) target
  (1 : α) - sumSpec (mulSpec normalizedPred normalizedTarget)

/--
Derivative of `cosineSimilaritySpec` w.r.t. `predicted`.

If $\cos=(p\mathbin{\cdot}t)/(\lVert p\rVert\lVert t\rVert)$ and
$\operatorname{loss}=1-\cos$, then (for nonzero norms):

$$
\frac{\partial\operatorname{loss}}{\partial p}
=\frac{p\mathbin{\cdot}t}{\lVert p\rVert^3\lVert t\rVert}p
 -\frac{1}{\lVert p\rVert\lVert t\rVert}t.
$$

The shared scaled norms avoid forming the fourth-degree denominator in this expression.
When the predicted norm is at or below `epsilon`, its clipped denominator is held constant
and the radial term is zero. The same positive, finite epsilon requirement applies as above.
-/
def cosineSimilarityDerivSpec {s : Shape}
  (predicted : Tensor α s) (target : Tensor α s) (epsilon : α := Context.defaultEpsilon) :
    Tensor α s :=
  let (predScale, predNorm) := cosineSimilaritySpec.normFactors predicted epsilon
  let (targetScale, targetNorm) := cosineSimilaritySpec.normFactors target epsilon
  let normalizedPred := Tensor.map (fun value => (value / predScale) / predNorm) predicted
  let normalizedTarget := Tensor.map (fun value => (value / targetScale) / targetNorm) target
  let similarity := sumSpec (mulSpec normalizedPred normalizedTarget)
  map2Spec (fun x t =>
    let radial := if predNorm > epsilon / predScale then similarity * x else 0
    ((radial - t) / predNorm) / predScale) normalizedPred normalizedTarget

/-- Log-cosh loss (reduced-to-scalar): `log(cosh(predicted - target))`. -/
def logCoshSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let per_elem := Tensor.map (fun d =>
    let a := MathFunctions.abs d
    if a == d || a == -d then
      if a < 1 then
        -- cosh(a) = 1 + 2 sinh²(a/2); log1p retains the small loss near zero.
        let halfSinh := MathFunctions.sinh (a / 2)
        Activation.Math.log1pSpec (2 * halfSinh * halfSinh)
      else
        -- Factoring out exp(a) avoids overflowing cosh(a) in the tails.
        a + Activation.Math.log1pSpec (MathFunctions.exp (-(2 * a))) - MathFunctions.log 2
    else
      MathFunctions.log (MathFunctions.cosh d)) diff
  meanSpec per_elem

/-- Derivative of `logCoshSpec` w.r.t. `predicted`. -/
def logCoshDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  let grad := Tensor.map MathFunctions.tanh diff
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
The safeguard must remain positive in the chosen scalar type. At the upper clamp, retain it
directly as the complementary probability: subtracting `1 - epsilon` from one could otherwise
produce zero when the upper clamp rounds to one. Inside the interval, `log1p (-p)` retains the
negative-class loss when `1 - p` would round to one.
-/
def binaryCrossEntropySpec (predicted : α) (target : α)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let p := if predicted > epsilon then predicted else epsilon
  let (p, logComplement) :=
    if p < (1 : α) - epsilon then (p, Context.log1p (-p))
    else ((1 : α) - epsilon, MathFunctions.log epsilon)
  let log_p := MathFunctions.log p
  let t := target * log_p + ((1 : α) - target) * logComplement
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
  meanSpec per_elem

/-- Derivative of `binaryCrossEntropyTensorSpec` w.r.t. `predicted`. -/
def binaryCrossEntropyTensorDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α
  s)
    (epsilon : α := Context.defaultEpsilon) : Tensor α s :=
  let grad := map2Spec (fun p y => binaryCrossEntropyDerivSpec (predicted := p) (target := y)
    (epsilon := epsilon))
      predicted target
  scaleSpec grad (1 / (meanDenominator s : α))

end Spec
