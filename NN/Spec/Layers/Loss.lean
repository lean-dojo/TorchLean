/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Core.TensorReductionShape

/-!
# Loss functions (spec layer)

This file defines a small collection of common losses (and their gradients) in a way that is:

- shape-generic: a loss takes `Tensor α s` and reduces it to a scalar `α`,
- explicit about reduction: most losses here are "mean over all elements",
- easy to line up with PyTorch terminology when you read training code.

In PyTorch you'll often see two layers:

- a low-level, elementwise loss (e.g. `smooth_l1_loss` / "Huber"),
- plus a reduction (`mean` or `sum`).

TorchLean's spec layer mirrors that idea: most definitions are written as an elementwise formula
followed by a global mean over the shape.
-/

@[expose] public section


namespace Spec
open Tensor
open MathFunctions
open Numbers

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [LE α]

/-- Enumeration of supported loss families (for lightweight configuration). -/
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
structure Loss (α : Type) (n p : ℕ) where
  /-- loss Type. -/
  lossType : LossType
  -- Note: regularization would be added here if needed

/-- MSE loss configuration. -/
def Loss.mse {α : Type} {n p : ℕ} : Loss α n p :=
  { lossType := LossType.mse }

/-- MAE loss configuration. -/
def Loss.mae {α : Type} {n p : ℕ} : Loss α n p :=
  { lossType := LossType.mae }

/-- Huber loss configuration. -/
def Loss.huber {α : Type} {n p : ℕ} : Loss α n p :=
  { lossType := LossType.huber }

/-- Cross-entropy loss configuration. -/
def Loss.crossEntropy {α : Type} {n p : ℕ} : Loss α n p :=
  { lossType := LossType.crossEntropy }

/-- Hinge loss configuration. -/
def Loss.hinge {α : Type} {n p : ℕ} : Loss α n p :=
  { lossType := LossType.hinge }

/-- Poisson loss configuration. -/
def Loss.poisson {α : Type} {n p : ℕ} : Loss α n p :=
  { lossType := LossType.poisson }

/-- Cosine similarity loss configuration. -/
def Loss.cosineSimilarity {α : Type} {n p : ℕ} : Loss α n p :=
  { lossType := LossType.cosineSimilarity }

/-- Log-cosh loss configuration. -/
def Loss.logCosh {α : Type} {n p : ℕ} : Loss α n p :=
  { lossType := LossType.logCosh }

-- Pure loss function specifications

/-- Sum all tensor elements into a single scalar. -/
def toScalarSpec {s : Shape} : Tensor α s → α :=
  sumSpec

/-- Mean of a scalar that conceptually came from a tensor with shape `s`. -/
def meanOver {s : Shape} (x : α) : α :=
  x / (Shape.size s : α)

/-- Mean squared error: average of `(predicted - target)^2`. -/
def mseSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let squared := mulSpec diff diff
  meanOver (s := s) (toScalarSpec squared)

/-- Derivative of `mse_spec` w.r.t. `predicted`. -/
def mseDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  -- PyTorch mental model: `MSELoss(reduction="mean")`.
  -- d/dpred ( (1/N) * Σᵢ (predᵢ - tgtᵢ)^2 ) = (2/N) * (pred - tgt)
  let n : α := (Shape.size s : α)
  scaleSpec diff (Numbers.two / n)

/-- Mean absolute error: average of `|predicted - target|`. -/
def maeSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let abs_diff := absSpec diff
  meanOver (s := s) (toScalarSpec abs_diff)

/-- Derivative of `mae_spec` w.r.t. `predicted` (subgradient via sign). -/
def maeDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  -- PyTorch mental model: `L1Loss(reduction="mean")`.
  -- This is a subgradient at 0.
  let grad :=
    mapSpec (fun x => if x > (0 : α) then (1 : α) else if x < (0 : α) then -(1 : α) else (0 : α))
      diff
  scaleSpec grad (1 / (Shape.size s : α))

/--
Huber / SmoothL1 loss (PyTorch's `smooth_l1_loss`) with parameter `delta`.

Elementwise, for residual `d = pred - target`:

- if `|d| < delta`: `0.5 * d^2 / delta`
- else:           `|d| - 0.5 * delta`

Then we take a mean over all elements.
-/
def huberSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) (delta : α := (1 : α)) : α
  :=
  let diff := subSpec predicted target
  let abs_diff := absSpec diff
  let per_elem := mapSpec (fun x =>
    if x < delta then
      (x * x) / (Numbers.two * delta)
    else
      x - delta / Numbers.two) abs_diff
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative of `huber_spec` w.r.t. `predicted`. -/
def huberDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) (delta : α := (1 :
  α)) : Tensor α s :=
  let diff := subSpec predicted target
  -- Subgradient at `|d| = delta` is fine for spec purposes; we pick the natural piecewise form.
  let grad :=
    mapSpec (fun d =>
      let ad := if d > (0 : α) then d else -d
      if ad < delta then d / delta else if d > (0 : α) then (1 : α) else if d < (0 : α) then -(1 :
        α) else (0 : α)
    ) diff
  scaleSpec grad (1 / (Shape.size s : α))

/--
Cross-entropy between distributions (probabilities).

This is closest to PyTorch when you already have probabilities `q` (e.g. after a softmax) and a
probability target `p` (e.g. one-hot or label-smoothed), and you want:

`CE(p, q) = -mean_i p_i * log(q_i)`.

PyTorch's `F.cross_entropy` typically takes logits and does `log_softmax + NLLLoss`; that is a
different API surface than this "probabilities in, scalar out" spec.
-/
def crossEntropySpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) (epsilon : α :=
  Numbers.epsilon) : α :=
  -- Standard cross-entropy between distributions:
  --   CE(p, q) = - (1/N) * Σᵢ pᵢ log(qᵢ),
  -- where `target` is `p` and `predicted` is `q` (typically softmax probabilities).
  let clamp01 := fun x : α =>
    let x := if x > epsilon then x else epsilon
    if x < (1 : α) - epsilon then x else (1 : α) - epsilon
  let q := mapSpec clamp01 predicted
  let logq := logSpec q
  let total := sumSpec (mulSpec target logq)
  meanOver (s := s) (-total)

/-- Derivative of `cross_entropy_spec` w.r.t. `predicted`. -/
def crossEntropyDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) (epsilon : α
  := Numbers.epsilon) :
    Tensor α s :=
  -- d/dqᵢ [ -(1/N) * Σ pᵢ log(qᵢ) ] = -(1/N) * pᵢ / qᵢ
  let clamp01 := fun x : α =>
    let x := if x > epsilon then x else epsilon
    if x < (1 : α) - epsilon then x else (1 : α) - epsilon
  let q := mapSpec clamp01 predicted
  let grad := divSpec (negSpec target) q
  scaleSpec grad (1 / (Shape.size s : α))

/--
Cross-entropy on logits (stable log-softmax form).

This matches the common PyTorch decomposition:

`cross_entropy(logits, target) = -mean_i target_i * log_softmax(logits)_i`.

Unlike `crossEntropySpec`, this takes *logits* and uses `Activation.logSoftmaxSpec` for
numerical stability.

Note: this spec assumes each last-axis `target` slice is a probability distribution (sums to 1),
as in one-hot or label-smoothed targets. -/
def crossEntropyLogitsSpec {s : Shape} (logits : Tensor α s) (target : Tensor α s) : α :=
  let logp := Activation.logSoftmaxSpec (α := α) (s := s) logits
  let total := sumSpec (mulSpec target logp)
  meanOver (s := s) (-total)

/-- Derivative of `cross_entropy_logits_spec` w.r.t. `logits`. -/
def crossEntropyLogitsDerivSpec {s : Shape} (logits : Tensor α s) (target : Tensor α s) :
    Tensor α s :=
  -- When `target` is a distribution over the last axis, the gradient is the familiar:
  --   d/dlogits = softmax(logits) - target
  -- followed by the global mean reduction.
  let probs := Activation.softmaxSpec (α := α) (s := s) logits
  let grad := subSpec probs target
  scaleSpec grad (1 / (Shape.size s : α))

/--
Hinge loss (binary margin loss), elementwise then mean-reduced:

`hinge(x, y) = mean_i max(0, 1 - y_i * x_i)`.

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

/-- Derivative/subgradient of `hinge_spec` w.r.t. `predicted`. -/
def hingeDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let margin := mulSpec predicted target
  -- Subgradient: if `1 - y*x > 0` then `d/dx = -y`, else 0. Then mean-reduce.
  let active := mapSpec (fun m => if (1 : α) - m > (0 : α) then (1 : α) else (0 : α)) margin
  let grad := mulSpec active (negSpec target)
  scaleSpec grad (1 / (Shape.size s : α))

/--
Poisson negative log-likelihood (log-input form), elementwise then mean-reduced:

If `predicted` represents `log(rate)` and `target` is a nonnegative count,
then (up to an additive constant that does not affect gradients):

`loss_i = exp(pred_i) - target_i * pred_i`.

This corresponds to PyTorch's `PoissonNLLLoss(log_input=true, full=false)` at the math level.
-/
def poissonSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let exp_pred := mapSpec MathFunctions.exp predicted
  let target_times_pred := mulSpec target predicted
  let per_elem := subSpec exp_pred target_times_pred
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative of `poisson_spec` w.r.t. `predicted`. -/
def poissonDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  -- d/dpred [exp(pred) - target*pred] = exp(pred) - target, then mean-reduce.
  let exp_pred := mapSpec MathFunctions.exp predicted
  let grad := subSpec exp_pred target
  scaleSpec grad (1 / (Shape.size s : α))

/-- Cosine similarity loss: `1 - cos(predicted, target)` (reduced-to-scalar). -/
def cosineSimilaritySpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Numbers.epsilon) : α :=
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
Derivative of `cosine_similarity_spec` w.r.t. `predicted`.

If `cos = (p·t)/(|p||t|)` and `loss = 1 - cos`, then (for nonzero norms):

`∂loss/∂p = (p·t) / (|p|^2 |t|) * p - 1/(|p||t|) * t`.

We use `epsilon` to avoid division by zero (similar to common "eps" handling in PyTorch code).
-/
def cosineSimilarityDerivSpec {s : Shape}
  (predicted : Tensor α s) (target : Tensor α s) (epsilon : α := Numbers.epsilon) : Tensor α s :=
  let dot_sum := toScalarSpec (mulSpec predicted target)
  let pred_sq_sum := toScalarSpec (mulSpec predicted predicted)
  let target_sq_sum := toScalarSpec (mulSpec target target)
  let pred_norm := MathFunctions.sqrt pred_sq_sum
  let target_norm := MathFunctions.sqrt target_sq_sum
  let pred_norm_safe := if pred_norm > epsilon then pred_norm else epsilon
  let target_norm_safe := if target_norm > epsilon then target_norm else epsilon
  let denom := pred_norm_safe * target_norm_safe
  let c1 := dot_sum / (pred_norm_safe * pred_norm_safe * target_norm_safe)
  let term1 := scaleSpec predicted c1
  let term2 := scaleSpec target (1 / denom)
  subSpec term1 term2

/-- Log-cosh loss (reduced-to-scalar): `log(cosh(predicted - target))`. -/
def logCoshSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let diff := subSpec predicted target
  let per_elem := mapSpec (fun d => MathFunctions.log (MathFunctions.cosh d)) diff
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative of `log_cosh_spec` w.r.t. `predicted`. -/
def logCoshDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let diff := subSpec predicted target
  let grad := mapSpec MathFunctions.tanh diff
  scaleSpec grad (1 / (Shape.size s : α))

/--
Binary cross-entropy on scalars (probabilities), with clipping to avoid `log(0)`.

This matches the core formula behind PyTorch's `BCELoss` when `predicted` is already a probability
(not a logit):

`BCE(p, y) = - ( y*log(p) + (1-y)*log(1-p) )`.

Assumption: `target` is in `[0, 1]`. We do not clip the target; we only clip `predicted`.
-/
def binaryCrossEntropySpec (predicted : α) (target : α) (epsilon : α := Numbers.epsilon) : α :=
  let p := if predicted > epsilon then predicted else epsilon
  let p := if p < (1 : α) - epsilon then p else (1 : α) - epsilon
  let log_p := MathFunctions.log p
  let log_one_minus_p := MathFunctions.log ((1 : α) - p)
  let t := target * log_p + ((1 : α) - target) * log_one_minus_p
  (0 : α) - t

/-- Derivative of `binary_cross_entropy_spec` w.r.t. `predicted`. -/
def binaryCrossEntropyDerivSpec (predicted : α) (target : α) (epsilon : α := Numbers.epsilon) :
  α :=
  let p := if predicted > epsilon then predicted else epsilon
  let p := if p < (1 : α) - epsilon then p else (1 : α) - epsilon
  -- d/dp BCE(p, y) = (p - y) / (p*(1-p))
  (p - target) / (p * ((1 : α) - p))

/-- Tensor BCE (probabilities), elementwise then mean-reduced. -/
def binaryCrossEntropyTensorSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Numbers.epsilon) : α :=
  let per_elem := map2Spec (fun p y => binaryCrossEntropySpec (predicted := p) (target := y)
    (epsilon := epsilon))
      predicted target
  meanOver (s := s) (toScalarSpec per_elem)

/-- Derivative of `binary_cross_entropy_tensor_spec` w.r.t. `predicted`. -/
def binaryCrossEntropyTensorDerivSpec {s : Shape} (predicted : Tensor α s) (target : Tensor α
  s)
    (epsilon : α := Numbers.epsilon) : Tensor α s :=
  let grad := map2Spec (fun p y => binaryCrossEntropyDerivSpec (predicted := p) (target := y)
    (epsilon := epsilon))
      predicted target
  scaleSpec grad (1 / (Shape.size s : α))

end Spec
