/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.Layers.Softmax

/-!
# Loss functions (spec layer)

Common training losses, each taking tensors of any shape `s` to a scalar, and their gradients.
Most follow PyTorch's split into an elementwise loss and a reduction: the elementwise terms are
added in row-major order and divided once by the entry count `meanDenominator s`, which is one
for a shape with no entries. `huberLoss` can return the plain sum, and the losses with a class axis
divide by the number of slices orthogonal to it (`axisMeanDenom`).

We wrote the losses by hand. Five follow the PyTorch 2.8 docs of their modules at the default
`reduction='mean'`: `torch.nn.MSELoss`, `torch.nn.L1Loss`, `torch.nn.HuberLoss`,
`torch.nn.PoissonNLLLoss` and `torch.nn.BCELoss`. The Huber term is the one of Huber, "Robust
Estimation of a Location Parameter", Ann. Math. Statist. 35(1):73-101, 1964
(https://doi.org/10.1214/aoms/1177703732). Binary cross-entropy is where we part ways with torch:
`BinaryCrossEntropy.term` clips the probability into `[epsilon, 1 - epsilon]`, with the default
`Context.defaultEpsilon` (`1e-6` for `Float`), while `BCELoss` clamps each logarithm at `-100`.

The student gradient against teacher logits (`distillLossBackward`) uses logit differences,
centering, and a residual correction as in Barron, Gnanapragasam and Tschannen, "A Fix for Softmax
and Cross-Entropy" (2026, arXiv:2610.31642), eq. (9) of Sec. 3, falling back to `q - p` where the
log-ratio has magnitude at least one, as they do. Its near-field expression retains small
gradients that would be lost by subtracting two separately rounded probability vectors.
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean
open TorchLean.Tensor Shape

variable {α : Type} [Storage α] [Context α]

/-- Weight a difference, distributing the multiplication when subtraction overflows.
Ordinary finite differences keep their evaluation order. For opposite-sign finite operands,
weighting each operand first can keep the result representable without halving small tangents. -/
def Tensor.weightedDifference {α : Type} [Zero α] [Sub α] [Mul α] [BEq α]
    (weight left right : α) : α :=
  let difference := left - right
  if difference - difference == 0 then weight * difference
  else weight * left - weight * right

/-- Totalized denominator for a mean over the slices orthogonal to `dim`: the number of entries of
`s` with axis `dim` removed, or one when there are none (`meanDenominator`).

Classification losses sum along the class axis `dim` and average over every other axis. A class
vector has one slice, a `(batch, classes)` matrix with class axis `1` has `batch` slices, and the
class axis can sit anywhere in the shape. -/
def Tensor.axisMeanDenom (s : Shape) (dim : Fin s.rank) : Nat :=
  meanDenominator (s.eraseIdx dim)

/-- A mean over slices divides by a positive count. -/
theorem Tensor.axisMeanDenom_pos (s : Shape) (dim : Fin s.rank) : 0 < axisMeanDenom s dim :=
  Tensor.meanDenominator_pos _

/-- Divide a classification loss once by the number of slices orthogonal to `dim`. -/
def Tensor.meanOverAxisSlices {α : Type} [Context α] {s : Shape} (dim : Fin s.rank) (x : α) : α :=
  Context.divNat x (axisMeanDenom s dim)

/-- Derivative of `Tensor.meanSquaredError` with respect to `predicted`, PyTorch's
`MSELoss(reduction="mean")`: `2 d / n` at each difference `d`, for `n = meanDenominator s`. A
finite `d + d` is divided once by `n`, and otherwise `d / n` is doubled. In a binary format both
give `2 d / n` rounded once, because a finite `d + d` is exact and so is doubling `d / n` for
`n ≥ 2`. The one exception is an overflowing `d + d` with a subnormal `d / n`, which needs
`n > 2 ^ (emax - emin)`: `2 ^ 253` in binary32, `2 ^ 2045` in binary64. -/
def Tensor.meanSquaredErrorBackward {α : Type} [Storage α] [Context α]
    {s : Shape} (predicted target : Tensor α s) : Tensor α s :=
  Tensor.map (fun d =>
    let twice := d + d
    if Context.isFinite twice then Context.divNat twice (meanDenominator s)
    else
      let quotient := Context.divNat d (meanDenominator s)
      quotient + quotient) (predicted - target)

/-- Mean absolute error: average of `|predicted - target|`. -/
def Tensor.l1Loss {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let diff := predicted - target
  let abs_diff := Tensor.abs diff
  Context.divNat (Tensor.sum abs_diff) (meanDenominator s)

/-- The derivative of `l1Loss` with respect to `predicted`: the sign of each difference, `+0` at a
zero or a NaN, divided by the number of entries. A NaN target thus gives `+0` where the loss is
NaN. -/
def Tensor.l1LossBackward {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : Tensor α s :=
  let diff := predicted - target
  -- Corresponds to PyTorch's `L1Loss(reduction="mean")`.
  -- The sign below is `+0` at both zeros (a subgradient of `|d|`) and at a NaN difference, where
  -- `Context.sign`, and so `Op.abs`'s pullback, gives a NaN.
  let grad :=
    Tensor.map (fun x => if x > (0 : α) then (1 : α) else if x < (0 : α) then -(1 : α) else (0 : α))
      diff
  Tensor.map (Context.divNat · (meanDenominator s)) grad

/-- The Huber loss of one prediction against its target with threshold `delta` (PyTorch's
`HuberLoss` before its reduction): at the magnitude `x` of the difference, `(x / 2) * x` below
`delta` and `delta * (x - delta / 2)` otherwise. Where a finite prediction and target have an
overflowing difference and `delta > 0`, the linear branch weights both operands before subtracting
(`weightedDifference`). -/
def Huber.term (predicted target : α) (delta : α := 1) : α :=
  let difference := predicted - target
  let x := MathFunctions.abs difference
  if x < delta then
    -- Halve before squaring so a representable half-square need not overflow first.
    (x / 2) * x
  else
    let direct := delta * (x - delta / 2)
    if difference - difference == 0 || !(delta > 0) then direct
    else
      let weighted := MathFunctions.abs (weightedDifference delta predicted target)
      if weighted - weighted == 0 then weighted - (delta / 2) * delta
      else direct

/--
Huber loss with transition parameter `delta`, in PyTorch's `HuberLoss` convention, which differs
from `SmoothL1Loss` by a factor of `delta`. Each entry is `Huber.term`, for the residual
$d=\mathtt{pred}-\mathtt{target}$:

- if $\lvert d\rvert<\delta$: $\tfrac12d^2$
- otherwise: $\delta(\lvert d\rvert-\tfrac12\delta)$

The entries are averaged, or summed when `mean` is false. The Huber interpretation requires
`delta > 0`. Nonfinite inputs, and losses or sums too large to represent, give nonfinite results.
-/
def Tensor.huberLoss {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (delta : α := (1 : α))
    (mean : Bool := true) : α :=
  let per_elem := zipWith (fun p t => Huber.term p t delta) predicted target
  if mean then Context.divNat (Tensor.sum per_elem) (meanDenominator s) else Tensor.sum per_elem

/-- The derivative of `huberLoss predicted target delta mean` with respect to `predicted` at each
difference `d`: `d` where `|d| < delta`, else `delta` or `-delta` by the sign of `d`, and `+0` at a
NaN `d` (where the loss is NaN) and at `d = ±0` when `delta` is zero, negative or NaN. It is
divided by the number of entries when `mean`. -/
def Tensor.huberLossBackward {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (delta : α := (1 : α)) (mean : Bool := true) : Tensor α s :=
  let diff := predicted - target
  -- Both pieces have the slope `d` at `|d| = delta`, so the loss is differentiable there.
  let grad :=
    Tensor.map (fun d =>
      let ad := if d > (0 : α) then d else -d
      if ad < delta then d
      else if d > (0 : α) then delta
      else if d < (0 : α) then -delta
      -- Reached at a NaN `d`, and at `d = ±0` when `delta` is zero, negative or NaN. A NaN `d` or
      -- `delta` makes the loss NaN, and at the other zeros `+0` lies between the one-sided slopes
      -- `delta` and `-delta`.
      else (0 : α)
    ) diff
  if mean then Tensor.map (Context.divNat · (meanDenominator s)) grad else grad

namespace CrossEntropy

/-- Target-weighted log-probabilities of one class vector. Where an unweighted log-probability is
not finite, the weighted term is rebuilt from the logits: the negated sum of the weight times the
shifted normalizer and the weighted gap between the largest logit and this one. -/
def vectorTerms {n : Nat} (logits target : Tensor α [n]) : Tensor α [n] :=
  match n with
  | 0 => logits
  | n + 1 =>
      let logp := LogSoftmax.vec logits
      let pivot : Fin (n + 1) := Softmax.pivot logits
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

/-- Weighted log-probabilities along `axis`, in the layout of the input. An axis out of range
applies `innermostTerms` to the tensors as they are. -/
def terms {s : Shape} (axis : Nat) (logits target : Tensor α s) : Tensor α s :=
  if h : axis < s.rank then Tensor.alongAxis₂ ⟨axis, h⟩ innermostTerms logits target
  else innermostTerms logits target

/-- Entry `c` of `terms dim logits target` is the entry of `vectorTerms` on the fibers of `logits`
and `target` through `c`, at the position of `c` along `dim`. -/
theorem terms_apply {s : Shape} (dim : Fin s.rank) (logits target : Tensor α s) (c : s.Coord) :
    terms dim logits target c =
      (vectorTerms (logits.fiber dim c) (target.fiber dim c)).getScalar
        (Coord.axisEquiv dim (shape := s) c).1 := by
  rw [terms, dite_eq_left dim.isLt]
  exact Tensor.alongAxis₂_apply innermostTerms (fun _ _ => rfl) dim logits target c

end CrossEntropy

/-- Cross-entropy against a fixed teacher's logits, averaged over class slices.

The teacher enters as logits because they carry the information needed for small student-teacher
gradient differences, which probability targets lose. A zero teacher weight contributes zero even
where a student log-probability overflows to negative infinity. The teacher is held fixed during
differentiation.
-/
def Tensor.distillLoss {s : Shape} (dim : Fin s.rank) (student teacher : Tensor α s) : α :=
  let p := Tensor.softmax teacher dim
  let logq := Tensor.logSoftmax student dim
  let terms := zipWith (fun weight logp => if weight == 0 then 0 else weight * logp) p logq
  meanOverAxisSlices dim (-Tensor.sum terms)

/-- Student gradient of `distillLoss`, `(q - p) / n` for the student and teacher softmaxes `q` and
`p` and `n = Tensor.axisMeanDenom s dim`, evaluated from logit differences.

The difference `student - teacher` is centered under the teacher distribution, and `expm1` and
`log1p` give its log-normalizer, leaving the log-ratio `log (q / p)`. Where the log-ratio lies in
`(-1, 1)`, the difference is computed as `p * expm1 (log (q / p))`, so near-equal logits never
subtract two rounded probabilities. Farther out it is the ordinary `q - p`, which avoids `0 * ∞`
from a small teacher weight times a large exponential. A final correction removes the residual
sum along `dim` in proportion to `q`.
-/
def Tensor.distillLossBackward {s : Shape} (dim : Fin s.rank) (student teacher : Tensor α s) :
    Tensor α s :=
  let p := Tensor.softmax teacher dim
  let q := Tensor.softmax student dim
  let delta := student - teacher
  let shift := castShape (reduce .sum (p * delta) [dim] false) (Shape.reduce_singleton _ _)
  let d := delta - Tensor.broadcast shift s _ (s.eraseIdx_broadcastDims dim)
  let correction := Tensor.map Context.log1p
    (castShape (reduce .sum (p * Tensor.map Context.expm1 d) [dim] false)
      (Shape.reduce_singleton _ _))
  let logRatio := d - Tensor.broadcast correction s _ (s.eraseIdx_broadcastDims dim)
  let near := zipWith (fun weight ratio =>
    if MathFunctions.abs ratio < 1 then weight * Context.expm1 ratio else 0) p logRatio
  let far := zipWith (fun ratio difference =>
    if MathFunctions.abs ratio < 1 then 0 else difference) logRatio (q - p)
  let difference := near + far
  let residual := castShape (reduce .sum difference [dim] false) (Shape.reduce_singleton _ _)
  Tensor.map (Context.divNat · (axisMeanDenom s dim))
    (difference - (q * Tensor.broadcast residual s _ (s.eraseIdx_broadcastDims dim)))

/--
Hinge loss (binary margin loss), elementwise then mean-reduced:

$\operatorname{hinge}(x,y)=\operatorname{mean}_i\max(0,1-y_i x_i)$.

Each term is IEEE 754-2019 `maximum (1 - x y, 0)` (`Context.maximum`), a NaN where the margin
`x y` is a NaN. At the kink, where the margin is one, the term is the constant zero, so dual
numbers get the tangent zero there, the subgradient that `hingeLossBackward` takes.

It is the usual SVM hinge loss. PyTorch's margin losses such as `HingeEmbeddingLoss` and
`MultiMarginLoss` behave similarly with different signatures.
-/
def Tensor.hingeLoss {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let margin := predicted * target
  let per_elem := Tensor.map (fun m =>
    let v := (1 : α) - m
    if v == 0 then 0 else Context.maximum v 0
  ) margin
  Context.divNat (Tensor.sum per_elem) (meanDenominator s)

/-- Subgradient of `hingeLoss` with respect to `predicted`: `-y` times the indicator of
`1 - x y > 0`, divided once by `meanDenominator s`. The indicator is zero at the kink, where the
margin `x y` is one, and is a NaN at a NaN margin, so the gradient is a NaN there. -/
def Tensor.hingeLossBackward {s : Shape} (predicted : Tensor α s) (target : Tensor α s) :
    Tensor α s :=
  let margin := predicted * target
  let active := Tensor.map (fun m =>
    let v := (1 : α) - m
    if v > 0 then 1 else if v == v then 0 else v) margin
  let grad := active * Tensor.neg target
  Tensor.map (Context.divNat · (meanDenominator s)) grad

/--
Poisson negative log-likelihood in log-input form, averaged over the entries. When `predicted`
holds `log(rate)` and `target` a nonnegative count, the loss of each entry, up to an additive
constant that does not affect gradients, is

$\operatorname{loss}_i=\exp(\mathtt{pred}_i)-\mathtt{target}_i\mathtt{pred}_i$.

Mathematically it is PyTorch's `PoissonNLLLoss(log_input=True, full=False)`.
-/
def Tensor.poissonNLLLoss {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let exp_pred := Tensor.map MathFunctions.exp predicted
  let target_times_pred := target * predicted
  let per_elem := exp_pred - target_times_pred
  Context.divNat (Tensor.sum per_elem) (meanDenominator s)

/-- Derivative of `poissonNLLLoss` w.r.t. `predicted`. -/
def Tensor.poissonNLLLossBackward {s : Shape} (predicted : Tensor α s) (target : Tensor α
    s) : Tensor α s :=
  -- d/dpred [exp(pred) - target*pred] = exp(pred) - target, then mean-reduce.
  let exp_pred := Tensor.map MathFunctions.exp predicted
  let grad := exp_pred - target
  Tensor.map (Context.divNat · (meanDenominator s)) grad

namespace CosineLoss

/-- The scale `max (maxAbs x) |epsilon|`, held constant under differentiation
(`Context.stopGradient`), and the L2 norm of `x / scale` floored at `epsilon / scale`. For positive
epsilon over the reals both are positive and their product is the clipped L2 norm. Numerical
callers divide by them separately. -/
def normFactors {s : Shape} (x : Tensor α s) (epsilon : α) : α × α :=
  let scale := Context.stopGradient (max (Tensor.maxAbs x) (MathFunctions.abs epsilon))
  let squared := Tensor.map (fun value => let scaled := value / scale; scaled * scaled) x
  let norm := MathFunctions.sqrt (Tensor.sum squared)
  let cutoff := epsilon / scale
  (scale, if norm > cutoff then norm else cutoff)

/-- The cosine loss `1 - cos(predicted, target)` of two tensors read as one fiber each, the
value of `TorchLean.Tensor.cosineLoss` at a fiber.

Each tensor is divided by its own scale before squaring and then by its scaled norm
(`normFactors`), so no raw dot product or product of large norms is formed. Epsilon floors each
norm and is never added to it, and a nonpositive epsilon sets no floor. Pass a finite epsilon
representable in the scalar type.

When both norms are positive and above their floors, the normalized vectors `p̂` and `t̂` are unit
vectors over the reals, for every epsilon, and the loss is computed as `‖p̂ - t̂‖² / 2`, which
equals `1 - p̂ · t̂` there. Subtracting the rounded dot product from one would cancel for nearly
aligned vectors, whose loss is about `θ² / 2` at an angle `θ` while the rounding error of the dot
product is about the unit roundoff. At a zero norm or a norm at its floor the loss is
`1 - p̂ · t̂`, and over the reals the two branches agree
(`TorchLean.CosineLoss.fiberLoss_eq_one_sub_dot`). Nonfinite inputs and an overflowing scaled
reduction can produce nonfinite results. -/
def fiberLoss {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let (predScale, predNorm) := normFactors predicted epsilon
  let (targetScale, targetNorm) := normFactors target epsilon
  let normalizedPred := Tensor.map (fun value => (value / predScale) / predNorm) predicted
  let normalizedTarget := Tensor.map (fun value => (value / targetScale) / targetNorm) target
  if predNorm > 0 ∧ predNorm > epsilon / predScale ∧ targetNorm > 0 ∧
      targetNorm > epsilon / targetScale then
    let difference := normalizedPred - normalizedTarget
    Tensor.sum (difference * difference) / 2
  else
    (1 : α) - Tensor.sum (normalizedPred * normalizedTarget)

/--
Derivative of `fiberLoss` with respect to `predicted`.

If $\cos=(p\mathbin{\cdot}t)/(\lVert p\rVert\lVert t\rVert)$ and
$\operatorname{loss}=1-\cos$, then (for nonzero norms):

$$
\frac{\partial\operatorname{loss}}{\partial p}
=\frac{p\mathbin{\cdot}t}{\lVert p\rVert^3\lVert t\rVert}p
 -\frac{1}{\lVert p\rVert\lVert t\rVert}t.
$$

Computed from the normalized vectors as `((p̂ · t̂) p̂ - t̂) / ‖p‖`, the expression never forms its
fourth-degree denominator. When the predicted norm is at or below `epsilon`, its clipped
denominator is held constant and the radial term is zero. Pass a positive, finite epsilon
representable in the scalar type.
-/
def fiberLossDeriv {s : Shape}
  (predicted : Tensor α s) (target : Tensor α s) (epsilon : α := Context.defaultEpsilon) :
    Tensor α s :=
  let (predScale, predNorm) := normFactors predicted epsilon
  let (targetScale, targetNorm) := normFactors target epsilon
  let normalizedPred := Tensor.map (fun value => (value / predScale) / predNorm) predicted
  let normalizedTarget := Tensor.map (fun value => (value / targetScale) / targetNorm) target
  let similarity := Tensor.sum (normalizedPred * normalizedTarget)
  zipWith (fun x t =>
    let radial := if predNorm > epsilon / predScale then similarity * x else 0
    ((radial - t) / predNorm) / predScale) normalizedPred normalizedTarget

end CosineLoss

/-- Log-cosh loss: the mean of `log(cosh(predicted - target))` over the entries. -/
def Tensor.logCoshLoss {s : Shape} (predicted : Tensor α s) (target : Tensor α s) : α :=
  let diff := predicted - target
  let per_elem := Tensor.map (fun d =>
    let a := MathFunctions.abs d
    if a == d || a == -d then
      if a < 1 then
        -- cosh(a) = 1 + 2 sinh²(a/2), and log1p keeps the small loss near zero.
        let halfSinh := MathFunctions.sinh (a / 2)
        Context.log1p (2 * halfSinh * halfSinh)
      else
        -- Factoring out exp(a) avoids overflowing cosh(a) in the tails.
        a + Context.log1p (MathFunctions.exp (-(2 * a))) - MathFunctions.log 2
    else
      MathFunctions.log (MathFunctions.cosh d)) diff
  Context.divNat (Tensor.sum per_elem) (meanDenominator s)

/-- Derivative of `logCoshLoss` w.r.t. `predicted`. -/
def Tensor.logCoshLossBackward {s : Shape} (predicted : Tensor α s) (target : Tensor α s) :
    Tensor α s :=
  let diff := predicted - target
  let grad := Tensor.map MathFunctions.tanh diff
  Tensor.map (Context.divNat · (meanDenominator s)) grad

/--
Binary cross-entropy of one predicted probability against its target, with `predicted` clipped
into `[epsilon, 1 - epsilon]` so that no logarithm sees zero. It is the formula behind PyTorch's
`BCELoss`, which takes `predicted` as a probability:

$$
\operatorname{BCE}(p,y)
=-\left(y\log p+(1-y)\log(1-p)\right).
$$

Assumption: `target` is in `[0, 1]`. We do not clip the target. We only clip `predicted`.
The safeguard must remain positive in the chosen scalar type. At the upper clamp the probability
is `1 - epsilon` and its complement `epsilon`. Both logarithms are taken without rounding
`1 - epsilon`, as `log1p (-epsilon)` and `log epsilon`, since `1 - epsilon` can round to one.
Inside the interval, `log1p (-p)` retains the negative-class loss when `1 - p` would round to one.
A NaN `predicted` fails `predicted > epsilon` and is read as `epsilon`, so the term is finite for a
finite target.
-/
def BinaryCrossEntropy.term (predicted : α) (target : α)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let p := if predicted > epsilon then predicted else epsilon
  let (logP, logComplement) :=
    if p < (1 : α) - epsilon then (MathFunctions.log p, Context.log1p (-p))
    else (Context.log1p (-epsilon), MathFunctions.log epsilon)
  let t := target * logP + ((1 : α) - target) * logComplement
  (0 : α) - t

/-- Selected derivative of `BinaryCrossEntropy.term` w.r.t. `predicted`.

The clipped forward function is not differentiable at `epsilon` or `1 - epsilon`. This definition
chooses zero at those two kinks and on the clipped exterior branches, and for a NaN `predicted`,
which fails both comparisons. -/
def BinaryCrossEntropy.termBackward (predicted : α) (target : α)
    (epsilon : α := Context.defaultEpsilon) :
  α :=
  -- As in `Tensor.crossEntropyBackward`, use the derivative of the selected clamp branch.
  if predicted > epsilon then
    if predicted < (1 : α) - epsilon then
      (predicted - target) / (predicted * ((1 : α) - predicted))
    else
      0
  else
    0

/-- Tensor BCE (probabilities), elementwise then mean-reduced. -/
def Tensor.binaryCrossEntropy {s : Shape} (predicted : Tensor α s) (target : Tensor α s)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let per_elem := zipWith (fun p y => BinaryCrossEntropy.term (predicted := p) (target := y)
    (epsilon := epsilon))
      predicted target
  Context.divNat (Tensor.sum per_elem) (meanDenominator s)

/-- Derivative of `binaryCrossEntropy` w.r.t. `predicted`. -/
def Tensor.binaryCrossEntropyBackward {s : Shape} (predicted : Tensor α s) (target : Tensor α
  s)
    (epsilon : α := Context.defaultEpsilon) : Tensor α s :=
  let grad := zipWith (fun p y => BinaryCrossEntropy.termBackward (predicted := p) (target := y)
    (epsilon := epsilon))
      predicted target
  Tensor.map (Context.divNat · (meanDenominator s)) grad

end TorchLean
