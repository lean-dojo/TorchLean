/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Optimizers

Optimizers for TorchLean runtime training.

This file implements the *core math* of common gradient-based optimizers as pure functions on
typed tensors `Tensor α s`.

Why “pure functions”?

In PyTorch, optimizers mutate parameters in-place and keep state in Python objects.
In TorchLean, we want the update rule itself to be explicit and easy to reuse:
- eager examples can call the update directly,
- the runtime training engine can store state in maps keyed by parameter ids,
- and proofs can refer to the same update equations.

The intent is to mimic the standard textbook formulas closely. We do not try to reproduce every
implementation detail of `torch.optim.*` (e.g. foreach kernels, fused updates, or every optional
flag); those live at a different layer than the math we specify here.

How this file fits with the runtime and API:
- this file owns the scalar-polymorphic, per-tensor update equations;
- `NN.Runtime.Autograd.Model.Optim` lifts those equations to runtime parameter lists; and
- `NN.API.Runtime` exposes ergonomic `optim.sgd`, `optim.adam`, and related configuration helpers.

With this separation, the formula appears once while runtime adapters and API
configuration can evolve independently around it.

Why each optimizer has its own `State` structure:
- Lean structures do not inherit from one another the way Python classes do.
- Optimizer state is not uniform: SGD stores only `lr`, momentum SGD stores a
  buffer, Adam/AdamW store two moment buffers and a step counter, Adadelta stores gradient/update
  EMAs, Muon carries an orthogonalization backend, and GaLore-style projected updates carry a
  projection backend.
- Keeping these as separate typed states makes impossible states unrepresentable. For example, an
  SGD state cannot accidentally contain a stale Adam `v` buffer, and AdamW cannot forget its
  decoupled `weightDecay` coefficient.

The generic abstraction lives one layer up:
- `Runtime.Autograd.Model.Optim.Optimizer` packages `init`/`step` for shape-indexed parameter
  lists, like a typed analogue of a PyTorch optimizer object.
- `Runtime.Autograd.Train.OptimizerState` handles dynamic parameter groups and checkpoint-style
  maps for the training-loop API.

The result is a collection of canonical state records rather than an inheritance hierarchy.

References (original algorithms / common variants):
- AdaGrad (Duchi–Hazan–Singer, 2011): https://jmlr.org/papers/v12/duchi11a.html
- RMSProp (Hinton lecture notes; widely used variant):
  https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- Adam (Kingma–Ba, 2015): https://arxiv.org/abs/1412.6980
- AdamW / decoupled weight decay (Loshchilov–Hutter, 2019): https://arxiv.org/abs/1711.05101
- Adadelta (Zeiler, 2012): https://arxiv.org/abs/1212.5701
- SGD + momentum in deep learning practice (Sutskever et al., 2013): https://arxiv.org/abs/1301.4083
- GaLore / low-rank gradient projection (Zhao et al., 2024): https://arxiv.org/abs/2403.03507
- Muon-style momentum with orthogonalized matrix updates (Jordan et al., 2024):
  https://kellerjordan.github.io/posts/muon/

PyTorch references (for API/parameter naming):
- `torch.optim` overview: https://pytorch.org/docs/stable/optim.html
- `torch.optim.SGD`: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
- `torch.optim.Adagrad`: https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
- `torch.optim.RMSprop`: https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
- `torch.optim.Adam`: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
- `torch.optim.AdamW`: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- `torch.optim.Adadelta`: https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
-/

@[expose] public section


namespace Optim
open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-- Optimizer state and parameters produced by one tensor update. -/
structure Step (α : Type) [TorchLean.Storage α]
    (shape : Shape) (OptimizerState : Type) where
  /-- Optimizer state to use for the next update. -/
  optimizerState : OptimizerState
  /-- Updated parameter tensor. -/
  parameters : Tensor α shape

/--
Integer exponentiation for scalar optimizer coefficients.

We use an explicit `Nat → α` recursion instead of `x ^ (n : Nat)` because `Context α`
provides `Pow α α` (for runtime scalar exponentiation), but not `Pow α Nat`.
-/
def scalarPowNat {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => scalarPowNat x n * x

/--
Adam bias-correction powers with their coefficients and step count in the type.

The certificates preserve the left-associated multiplication order of `scalarPowNat`, including
for rounded scalars without associative multiplication. Changing a coefficient or counter requires
a cache for the new indices; scalar Boolean equality is never used to validate a cache.
-/
structure AdamPowers {α : Type} [One α] [Mul α] (beta1 beta2 : α) (stepCount : Nat) where
  /-- First moment bias-correction power. -/
  first : α
  /-- Second moment bias-correction power. -/
  second : α
  /-- The first power follows the canonical scalar recurrence. -/
  first_eq : first = scalarPowNat beta1 stepCount
  /-- The second power follows the canonical scalar recurrence. -/
  second_eq : second = scalarPowNat beta2 stepCount

attribute [simp] AdamPowers.first_eq AdamPowers.second_eq

/-- Reconstruct bias-correction powers, for initialization or a restored counter. -/
def AdamPowers.compute {α : Type} [One α] [Mul α] (beta1 beta2 : α)
    (stepCount : Nat) : AdamPowers beta1 beta2 stepCount :=
  ⟨scalarPowNat beta1 stepCount, scalarPowNat beta2 stepCount, rfl, rfl⟩

/-- Advance both powers with exactly one multiplication per coefficient. -/
def AdamPowers.advance {α : Type} [One α] [Mul α] {beta1 beta2 : α}
    {stepCount : Nat} (powers : AdamPowers beta1 beta2 stepCount) :
    AdamPowers beta1 beta2 (stepCount + 1) :=
  ⟨powers.first * beta1, powers.second * beta2,
    by rw [powers.first_eq]; rfl, by rw [powers.second_eq]; rfl⟩

/-! ## Shared equations -/

/-- Next momentum buffer $\mu b+g$. -/
def updateMomentumBuffer {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (momentumBuffer : Tensor α s) (momentum : α)
    (gradients : Tensor α s) : Tensor α s :=
  addSpec (scaleSpec momentumBuffer momentum) gradients

/--
Elementwise adaptive learning-rate tensor
$\mathtt{learningRate}/(\sqrt{\mathtt{denominator}}+\varepsilon)$.

This is shared by AdaGrad/RMSProp/Adam-style optimizers.
-/
def adaptiveLearningRate {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (learningRate epsilon : α) (denominator : Tensor α s) : Tensor α s :=
  divSpec (Tensor.full s learningRate) (addSpec (sqrtSpec denominator) (Tensor.full s epsilon))

/-- Single-pass form of `adaptiveLearningRate`. The reference definition builds two constant
tensors and three intermediates per call; this one maps the scalar formula once. -/
def adaptiveLearningRateFused {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (learningRate epsilon : α) (denominator : Tensor α s) : Tensor α s :=
  mapSpec (fun v => learningRate / (MathFunctions.sqrt (Max.max v 0) + epsilon)) denominator

/-- Compiled code runs the single-pass form. Proofs keep unfolding the reference definition. -/
@[csimp] theorem adaptiveLearningRate_eq_fused :
    @adaptiveLearningRate = @adaptiveLearningRateFused := by
  funext α storage context decide s learningRate epsilon denominator
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [adaptiveLearningRate, adaptiveLearningRateFused, divSpec, addSpec, sqrtSpec, mapSpec,
    map2Spec, Tensor.map, Tensor.full]

/-! ## SGD -/

/--
SGD state (per parameter tensor).

We only store the learning rate here.
-/
structure SGD.State (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Learning rate. -/
  learningRate : α

/--
Initialize SGD state.

The parameter tensor is unused; we keep it in the signature so optimizers share the same
“init from parameters” calling convention.
-/
def SGD.init {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (learningRate : α) (_ : Tensor α s) : SGD.State α s :=
  { learningRate := learningRate }

/--
One SGD step: `p ← p - lr * g`.

PyTorch analogy: the core of `torch.optim.SGD` without momentum/weight-decay extras.
-/
def SGD.update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : SGD.State α s) (parameters gradients : Tensor α s) :
    Step α s (SGD.State α s) :=
  { optimizerState := state
    parameters := subSpec parameters (scaleSpec gradients state.learningRate) }

/-! ## Momentum SGD -/

/--
Momentum SGD state (per parameter tensor).

We store a momentum buffer and a momentum coefficient $\mu$.
Update rule:

- $b\gets\mu b+g$,
- $p\gets p-\mathtt{learningRate}\,b$.

This matches PyTorch's SGD momentum behavior when `dampening = 0` and `nesterov = false`.
-/
structure MomentumSGD.State (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Learning rate. -/
  learningRate : α
  /-- Momentum coefficient $\mu$. -/
  momentum : α
  /-- Momentum buffer. -/
  momentumBuffer : Tensor α s

/-- Initialize momentum SGD with a zero buffer. -/
def MomentumSGD.init {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (learningRate momentum : α) (_ : Tensor α s) : MomentumSGD.State α s :=
  { learningRate := learningRate, momentum := momentum, momentumBuffer := Tensor.full s 0 }

/-- One momentum-SGD step. -/
def MomentumSGD.update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : MomentumSGD.State α s) (parameters gradients : Tensor α s) :
    Step α s (MomentumSGD.State α s) :=
  let nextMomentumBuffer :=
    updateMomentumBuffer state.momentumBuffer state.momentum gradients
  { optimizerState := { state with momentumBuffer := nextMomentumBuffer }
    parameters := subSpec parameters (scaleSpec nextMomentumBuffer state.learningRate) }

/-! ## AdaGrad -/

/--
AdaGrad state (per parameter tensor).

We store an accumulator $G$ of squared gradients (same shape as the parameters). The effective
step size is scaled by $1/(\sqrt G+\varepsilon)$.
-/
structure AdaGrad.State (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Base learning rate. -/
  learningRate : α
  /-- Numerical stability constant $\varepsilon$. -/
  epsilon : α
  /-- Accumulated squared gradients. -/
  squaredGradientSum : Tensor α s

/-- Initialize AdaGrad with zero accumulator. -/
def AdaGrad.init {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (learningRate epsilon : α) (_ : Tensor α s) : AdaGrad.State α s :=
  { learningRate := learningRate, epsilon := epsilon, squaredGradientSum := Tensor.full s 0 }

/-- One AdaGrad step. -/
def AdaGrad.update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : AdaGrad.State α s) (parameters gradients : Tensor α s) :
    Step α s (AdaGrad.State α s) :=
  let squaredGradients := squareSpec gradients
  let nextSquaredGradientSum := addSpec state.squaredGradientSum squaredGradients
  let effectiveLearningRate :=
    adaptiveLearningRate state.learningRate state.epsilon nextSquaredGradientSum
  { optimizerState := { state with squaredGradientSum := nextSquaredGradientSum }
    parameters := subSpec parameters (mulSpec effectiveLearningRate gradients) }

/-! ## RMSProp -/

/--
RMSProp state (per parameter tensor).

We store an exponential moving average of squared gradients.
-/
structure RMSProp.State (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Learning rate. -/
  learningRate : α
  /-- Decay coefficient for the EMA of $g^2$ (often called `alpha`). -/
  decay : α
  /-- Numerical stability constant $\varepsilon$. -/
  epsilon : α
  /-- EMA of squared gradients. -/
  squaredGradientAverage : Tensor α s

/-- Initialize RMSProp with zero accumulator. -/
def RMSProp.init {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (learningRate decay epsilon : α) (_ : Tensor α s) : RMSProp.State α s :=
  { learningRate := learningRate
    decay := decay
    epsilon := epsilon
    squaredGradientAverage := Tensor.full s 0 }

/-- One RMSProp step. -/
def RMSProp.update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : RMSProp.State α s) (parameters gradients : Tensor α s) :
    Step α s (RMSProp.State α s) :=
  let squaredGradients := squareSpec gradients
  let nextSquaredGradientAverage := addSpec
    (scaleSpec state.squaredGradientAverage state.decay)
    (scaleSpec squaredGradients (1 - state.decay))
  let effectiveLearningRate :=
    adaptiveLearningRate state.learningRate state.epsilon nextSquaredGradientAverage
  { optimizerState := { state with squaredGradientAverage := nextSquaredGradientAverage }
    parameters := subSpec parameters (mulSpec effectiveLearningRate gradients) }

/-! ## Adam -/

/--
Adam state (per parameter tensor).

We store first and second moment averages, a step counter, and certified bias-correction powers.
The default powers reconstruct the original recurrence when creating a state at a nonzero step.
When updating either beta or the counter in an existing record, also set `powers` to
`AdamPowers.compute` at the new values. Updates to other fields preserve the cache.
-/
structure Adam.State (α : Type) [TorchLean.Storage α] [One α] [Mul α] (s : Shape) where
  /-- Learning rate. -/
  learningRate : α
  /-- First moment decay $\beta_1$. -/
  beta1 : α
  /-- Second moment decay $\beta_2$. -/
  beta2 : α
  /-- Numerical stability constant $\varepsilon$. -/
  epsilon : α
  /-- First moment EMA. -/
  firstMoment : Tensor α s
  /-- Second moment EMA. -/
  secondMoment : Tensor α s
  /-- Step counter (used for bias correction). -/
  stepCount : Nat
  /-- Cached powers for these coefficients and this counter. -/
  powers : AdamPowers beta1 beta2 stepCount := AdamPowers.compute beta1 beta2 stepCount

/-- Initialize Adam with zero moments and a zero step count. -/
def Adam.init {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (learningRate beta1 beta2 epsilon : α) (_ : Tensor α s) : Adam.State α s :=
  {
    learningRate := learningRate,
    beta1 := beta1,
    beta2 := beta2,
    epsilon := epsilon,
    firstMoment := Tensor.full s 0,
    secondMoment := Tensor.full s 0,
    stepCount := 0
  }

/--
One Adam step.

Equations (elementwise):

- $m\gets\beta_1m+(1-\beta_1)g$,
- $v\gets\beta_2v+(1-\beta_2)g^2$,
- $\widehat m\gets m/(1-\beta_1^t)$,
- $\widehat v\gets v/(1-\beta_2^t)$,
- $p\gets p-\mathtt{lr}\,\widehat m/(\sqrt{\widehat v}+\varepsilon)$.

The $\varepsilon$ placement matches Kingma and Ba: it is added after $\sqrt{\widehat v}$.
-/
def Adam.update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : Adam.State α s) (parameters gradients : Tensor α s) :
    Step α s (Adam.State α s) :=
  let nextStepCount := state.stepCount + 1
  let nextPowers := state.powers.advance
  let nextFirstMoment :=
    addSpec (scaleSpec state.firstMoment state.beta1)
      (scaleSpec gradients (1 - state.beta1))
  let nextSecondMoment :=
    addSpec (scaleSpec state.secondMoment state.beta2)
      (scaleSpec (squareSpec gradients) (1 - state.beta2))
  let correctedFirstMoment :=
    scaleSpec nextFirstMoment (1 / (1 - nextPowers.first))
  let correctedSecondMoment :=
    scaleSpec nextSecondMoment (1 / (1 - nextPowers.second))
  let effectiveLearningRate :=
    adaptiveLearningRate state.learningRate state.epsilon correctedSecondMoment
  { optimizerState :=
      { state with
        firstMoment := nextFirstMoment
        secondMoment := nextSecondMoment
        stepCount := nextStepCount
        powers := nextPowers }
    parameters :=
      subSpec parameters (mulSpec effectiveLearningRate correctedFirstMoment) }

/-- Adam increments its step counter by one on every update. -/
@[simp] theorem Adam.update_stepCount {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : Adam.State α s) (parameters gradients : Tensor α s) :
    (Adam.update state parameters gradients).optimizerState.stepCount =
      state.stepCount + 1 := by
  simp [Adam.update]

/-! ## AdamW -/

/--
AdamW state (per parameter tensor).

AdamW is “Adam + decoupled weight decay”. Weight decay is applied as a
separate parameter decay term rather than being folded into the gradient that feeds the moments.
Its certified powers follow `Adam.State`: replacing either beta or the counter also requires
`powers := AdamPowers.compute` at the new values.
-/
structure AdamW.State (α : Type) [TorchLean.Storage α] [One α] [Mul α] (s : Shape) where
  /-- Learning rate. -/
  learningRate : α
  /-- First moment decay $\beta_1$. -/
  beta1 : α
  /-- Second moment decay $\beta_2$. -/
  beta2 : α
  /-- Numerical stability constant $\varepsilon$. -/
  epsilon : α
  /-- Weight decay coefficient `wd`. -/
  weightDecay : α
  /-- First moment EMA. -/
  firstMoment : Tensor α s
  /-- Second moment EMA. -/
  secondMoment : Tensor α s
  /-- Step counter (used for bias correction). -/
  stepCount : Nat
  /-- Cached powers for these coefficients and this counter. -/
  powers : AdamPowers beta1 beta2 stepCount := AdamPowers.compute beta1 beta2 stepCount

/-- Initialize AdamW state for a parameter tensor (moments start at `0`). -/
def AdamW.init {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (learningRate weightDecay beta1 beta2 epsilon : α) (_ : Tensor α s) : AdamW.State α s :=
  {
    learningRate := learningRate,
    weightDecay := weightDecay,
    beta1 := beta1,
    beta2 := beta2,
    epsilon := epsilon,
    firstMoment := Tensor.full s 0,
    secondMoment := Tensor.full s 0,
    stepCount := 0
  }

/--
One AdamW step.

We implement the decoupled form from the AdamW paper:
- update Adam moments using the *raw* gradient `g`,
- apply weight decay directly to the parameters (`p ← p - lr * wd * p`),
- then apply the Adam update.

This is the same single-step ordering used by `torch.optim.AdamW`.
-/
def AdamW.update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : AdamW.State α s) (parameters gradients : Tensor α s) :
    Step α s (AdamW.State α s) :=
  let nextStepCount := state.stepCount + 1
  let nextPowers := state.powers.advance
  let nextFirstMoment :=
    addSpec (scaleSpec state.firstMoment state.beta1)
      (scaleSpec gradients (1 - state.beta1))
  let nextSecondMoment :=
    addSpec (scaleSpec state.secondMoment state.beta2)
      (scaleSpec (squareSpec gradients) (1 - state.beta2))
  let correctedFirstMoment :=
    scaleSpec nextFirstMoment (1 / (1 - nextPowers.first))
  let correctedSecondMoment :=
    scaleSpec nextSecondMoment (1 / (1 - nextPowers.second))
  let effectiveLearningRate :=
    adaptiveLearningRate state.learningRate state.epsilon correctedSecondMoment
  let decayedParameters :=
    subSpec parameters
      (scaleSpec parameters (state.learningRate * state.weightDecay))
  { optimizerState :=
      { state with
        firstMoment := nextFirstMoment
        secondMoment := nextSecondMoment
        stepCount := nextStepCount
        powers := nextPowers }
    parameters :=
      subSpec decayedParameters (mulSpec effectiveLearningRate correctedFirstMoment) }

/-- AdamW increments its step counter by one on every update. -/
@[simp] theorem AdamW.update_stepCount {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : AdamW.State α s) (parameters gradients : Tensor α s) :
    (AdamW.update state parameters gradients).optimizerState.stepCount =
      state.stepCount + 1 := by
  simp [AdamW.update]

/-! ## Adadelta -/

/--
Adadelta state (per parameter tensor).

We store two EMAs:
- `squaredGradientAverage`,
- `squaredUpdateAverage`.
-/
structure Adadelta.State (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Learning rate (often set to `1` in some presentations; we keep it explicit). -/
  learningRate : α
  /-- Decay coefficient $\rho$. -/
  rho : α
  /-- Numerical stability constant $\varepsilon$. -/
  epsilon : α
  /-- EMA of squared gradients. -/
  squaredGradientAverage : Tensor α s
  /-- EMA of squared updates. -/
  squaredUpdateAverage : Tensor α s

/-- Initialize Adadelta state for a parameter tensor (EMAs start at `0`). -/
def Adadelta.init {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (learningRate rho epsilon : α) (_ : Tensor α s) : Adadelta.State α s :=
  { learningRate := learningRate
    rho := rho
    epsilon := epsilon
    squaredGradientAverage := Tensor.full s 0
    squaredUpdateAverage := Tensor.full s 0 }

/--
One Adadelta step.

Elementwise equations:

- $v\gets\rho v+(1-\rho)g^2$,
- $\Delta p\gets
  \dfrac{\sqrt{u+\varepsilon}}{\sqrt{v+\varepsilon}}\odot g$,
- $p\gets p-\mathtt{lr}\,\Delta p$,
- $u\gets\rho u+(1-\rho)(\Delta p)^2$.

The $\varepsilon$ placement is inside the RMS terms, matching Zeiler's Adadelta update.
-/
def Adadelta.update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : Adadelta.State α s) (parameters gradients : Tensor α s) :
    Step α s (Adadelta.State α s) :=
  let squaredGradients := squareSpec gradients
  let nextSquaredGradientAverage :=
    addSpec (scaleSpec state.squaredGradientAverage state.rho)
      (scaleSpec squaredGradients (1 - state.rho))

  let rms := fun v : α => MathFunctions.sqrt (Max.max (v + state.epsilon) 0)
  let gradientRms := mapSpec rms nextSquaredGradientAverage
  let updateRms := mapSpec rms state.squaredUpdateAverage

  let ratio := divSpec updateRms gradientRms
  let parameterUpdate := mulSpec ratio gradients
  let nextParameters :=
    subSpec parameters (scaleSpec parameterUpdate state.learningRate)

  let nextSquaredUpdateAverage :=
    addSpec (scaleSpec state.squaredUpdateAverage state.rho)
      (scaleSpec (squareSpec parameterUpdate) (1 - state.rho))
  { optimizerState :=
      { state with
        squaredGradientAverage := nextSquaredGradientAverage
        squaredUpdateAverage := nextSquaredUpdateAverage }
    parameters := nextParameters }

/-! ## Projected / low-rank gradient transforms -/

namespace GaLore

/--
A shape-safe gradient projector.

GaLore-style training periodically builds a low-rank subspace for a large matrix parameter,
projects the gradient into that subspace, runs a base optimizer there, and lifts the update back to
the original parameter shape. This record is the algebraic interface; the
expensive policy that computes or refreshes the projector belongs to the runtime layer.
-/
structure Projector (α : Type) [TorchLean.Storage α] (full low : Shape) where
  /-- Project a full gradient into the low-rank optimizer space. -/
  project : Tensor α full → Tensor α low
  /-- Lift a low-rank update back to the full parameter shape. -/
  lift : Tensor α low → Tensor α full

/-- Identity projector, used when projected SGD is requested without a projection backend. -/
def identityProjector {α : Type} [TorchLean.Storage α] {s : Shape} : Projector α s s :=
  { project := id, lift := id }

/--
GaLore-style projected SGD state for one tensor.

This is not a full GaLore implementation by itself: it specifies the update once a projector is
available. A practical trainer still needs a refresh schedule and a way to build projectors for
large matrix parameters.
-/
structure SGDState (α : Type) [TorchLean.Storage α] (full low : Shape) where
  /-- Learning rate used after the gradient has been projected and lifted. -/
  learningRate : α
  /-- Current gradient projector. -/
  projector : Projector α full low

/-- One projected-SGD update: `p ← p - learningRate * lift(project(g))`. -/
def update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {full low : Shape} (state : SGDState α full low)
    (parameters gradients : Tensor α full) :
    Step α full (SGDState α full low) :=
  { optimizerState := state
    parameters :=
      subSpec parameters
        (scaleSpec
          (state.projector.lift (state.projector.project gradients))
          state.learningRate) }

end GaLore

/-! ## Muon-style orthogonalized momentum -/

namespace Muon

/--
Orthogonalization backend for a matrix-shaped update.

Muon uses a momentum buffer and then replaces the raw momentum direction by an approximately
orthogonalized update, commonly via Newton-Schulz iterations. TorchLean keeps this as an explicit
backend so the pure update rule is testable before CUDA kernels are introduced.
-/
structure Orthogonalizer (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Convert a momentum buffer into the direction used for the parameter update. -/
  apply : Tensor α s → Tensor α s

/-- The identity orthogonalizer, used when Muon is requested without a matrix backend. -/
def identityOrthogonalizer {α : Type} [TorchLean.Storage α] {s : Shape} :
    Orthogonalizer α s :=
  { apply := id }

/-- Per-parameter state for Muon-style momentum with an explicit orthogonalization backend. -/
structure State (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- Learning rate. -/
  learningRate : α
  /-- Momentum coefficient. -/
  momentum : α
  /-- Momentum buffer. -/
  momentumBuffer : Tensor α s
  /-- Backend that turns the momentum buffer into the update direction. -/
  orthogonalizer : Orthogonalizer α s

/-- Initialize Muon-style state with a zero momentum buffer. -/
def init {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (learningRate momentum : α) (orthogonalizer : Orthogonalizer α s)
    (_ : Tensor α s) : State α s :=
  { learningRate := learningRate
    momentum := momentum
    momentumBuffer := Tensor.full s 0
    orthogonalizer := orthogonalizer }

/--
One Muon-style update:
- update the momentum buffer,
- orthogonalize the buffer,
- subtract the scaled orthogonalized direction.

For actual Muon, use a matrix-shaped `s` and a Newton-Schulz orthogonalizer. The generic shape here
keeps the definition reusable for tests and for future batched matrix layouts.
-/
def update {α : Type} [TorchLean.Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : State α s) (parameters gradients : Tensor α s) :
    Step α s (State α s) :=
  let nextMomentumBuffer :=
    updateMomentumBuffer state.momentumBuffer state.momentum gradients
  let direction := state.orthogonalizer.apply nextMomentumBuffer
  { optimizerState := { state with momentumBuffer := nextMomentumBuffer }
    parameters := subSpec parameters (scaleSpec direction state.learningRate) }

end Muon

end Optim
