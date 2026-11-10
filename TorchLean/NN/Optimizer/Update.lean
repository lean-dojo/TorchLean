/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise

/-!
# Optimizers

This module holds the update equations of SGD, momentum SGD, AdaGrad, RMSProp, Adam, AdamW,
Adadelta, GaLore-style projected SGD and Muon-style momentum, as pure functions on typed tensors
`Tensor α s`. Each optimizer has a state record and an update, which takes the state, a parameter
tensor and its gradient and returns the next state and parameters as a `Step`.

PyTorch's optimizers mutate parameters in place and keep their state in Python objects. We write
each update once, as a function, and its callers share it:
- examples, tests and `nn.sgdStep` call an update directly,
- the runtime optimizers of `TorchLean.NN.Optimizer` keep one state per
  parameter in a shape-indexed `Pack`, and their `Optimizer`
  packages `init` and `step` as a typed analogue of a PyTorch optimizer object,
- and proofs such as `TorchLean.Verify.RuntimeApprox.NF.Optimizers` compare the same equations
  over ℝ and in the rounded-real model `NF`.

`TorchLean.NN.Optimizer.Constructor` exposes the configurations `optim.sgd`, `optim.adam` and the
rest. The updates follow the textbook formulas. PyTorch's foreach and fused kernels have no
counterpart here, and neither do flags such as `nesterov`, `amsgrad` and `centered`. On eager CUDA,
SGD without momentum, Adam and AdamW can run as device-resident updates (`NativeOptimizer`) in place
of these functions.

Each `State` holds the fields its update reads, so the records differ: SGD keeps only `lr`,
momentum SGD a buffer, AdaGrad a sum of squared gradients and RMSProp their moving average. Adam
keeps two moment averages, a step counter and cached bias-correction powers (`Powers`), and
Adadelta keeps averages of squared gradients and of squared updates. As separate types, the
records rule out mixed states: an SGD state has no `secondMoment` field, and an `AdamW.State`
can't be built without its `weightDecay`. `AdamW.State` extends `Adam.State` with that decay
coefficient, `Muon.State` extends `MomentumSGD.State` with the orthogonalizer, and
`GaLore.SGDState` is a learning rate and a projector.

The SGD, momentum SGD, projected SGD and Muon updates need only `Add`, `Sub` and `Mul` on the
scalar, with no order and no square root. The Muon orthogonalizer and the GaLore projector bring
their own maps, and every `init` needs at most `Zero`, `One` and `Mul`. The adaptive updates
(AdaGrad, RMSProp, Adam, AdamW, Adadelta) take `Context α` and a decidable `>` for their square
roots and divisions.

We wrote each update by hand from its entry in the list below and then read it against its
PyTorch 2.8 class (GaLore and Muon have none). Besides the flags above, no record has `maximize`,
`dampening` or `lr_decay`, and only `AdamW.State` has a weight decay, so each update is the torch
step with those arguments at their defaults, with AdamW's `weightDecay` as torch's `weight_decay`.
Over ℝ the two sides agree. In floating point some operations are grouped differently from torch
2.8 on the CPU, and the last bits can differ with them, for example:
- we divide by $\sigma+\varepsilon$ and then multiply by `lr`, while the kernel of torch's
  `addcdiv_` multiplies by the step size before it divides,
- we square a gradient and then weight it, and the kernel of `addcmul_` forms `value * g * g`
  from the left,
- torch's single-tensor Adam updates the first moment with `lerp_`, takes
  `step_size = lr / bias_correction1` and `sqrt(v) / sqrt(bias_correction2) + eps`, and computes
  `beta1**step` afresh at each step, where we correct both moments first, from the running
  products of `Adam.Powers`.

No TorchLeanTest test runs `torch.optim` on these functions.
`TorchLean.Test.CUDA.Trainer.checkAdamSpec` checks the CUDA Adam and AdamW steps against
`Adam.update` and `AdamW.update` at a few scalar inputs with zero weight decay, in `Float32` and
`Float`: word for word, except that any NaN matches any NaN.

References for the algorithms and their common variants:
- AdaGrad: Duchi, Hazan and Singer (2011), "Adaptive Subgradient Methods for Online Learning and
  Stochastic Optimization", JMLR 12(61):2121-2159, eq. (5) and Figure 1:
  https://jmlr.org/papers/v12/duchi11a.html
- RMSProp: Hinton, with Srivastava and Swersky, Neural Networks for Machine Learning, lecture 6e,
  "rmsprop: Divide the gradient by a running average of its recent magnitude" (slides):
  https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
- Adam: Kingma and Ba (2015), "Adam: A Method for Stochastic Optimization", ICLR 2015,
  Algorithm 1: https://arxiv.org/abs/1412.6980
- AdamW: Loshchilov and Hutter (2019), "Decoupled Weight Decay Regularization", ICLR 2019,
  Algorithm 2 and Proposition 1: https://arxiv.org/abs/1711.05101
- Adadelta: Zeiler (2012), "ADADELTA: An Adaptive Learning Rate Method", arXiv:1212.5701,
  eq. (14) and Algorithm 1: https://arxiv.org/abs/1212.5701
- SGD with momentum: Sutskever, Martens, Dahl and Hinton (2013), "On the importance of
  initialization and momentum in deep learning", ICML 2013, PMLR 28(3):1139-1147, Section 2,
  eqs. (1) and (2): https://proceedings.mlr.press/v28/sutskever13.html
- GaLore: Zhao, Zhang, Chen, Wang, Anandkumar and Tian (2024), "GaLore: Memory-Efficient LLM
  Training by Gradient Low-Rank Projection", ICML 2024, Algorithms 1 and 2:
  https://arxiv.org/abs/2403.03507
- Muon: Jordan, Jin, Boza, You, Cesista, Newhouse and Bernstein (2024), "Muon: An optimizer for
  hidden layers in neural networks", blog post, 8 December 2024, Algorithm 2 and
  `newtonschulz5`: https://kellerjordan.github.io/posts/muon/

PyTorch 2.8 documentation, for API and parameter names:
- `torch.optim` overview: https://docs.pytorch.org/docs/2.8/optim.html
- `torch.optim.SGD`: https://docs.pytorch.org/docs/2.8/generated/torch.optim.SGD.html
- `torch.optim.Adagrad`: https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adagrad.html
- `torch.optim.RMSprop`: https://docs.pytorch.org/docs/2.8/generated/torch.optim.RMSprop.html
- `torch.optim.Adam`: https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html
- `torch.optim.AdamW`: https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html
- `torch.optim.Adadelta`: https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adadelta.html
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean
open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

/-- Optimizer state and parameters produced by one tensor update. -/
structure OptimizerStep (α : Type) [Storage α]
    (shape : Shape) (OptimizerState : Type) where
  /-- Optimizer state to use for the next update. -/
  optimizerState : OptimizerState
  /-- Updated parameter tensor. -/
  parameters : Tensor α shape

/--
Adam bias-correction powers with their coefficients and step count in the type.

Each power is `npowRec stepCount beta`, the left-associated product (`Context α` has `Pow α α`
but no `Pow α Nat`). The certificates keep this multiplication order, including for rounded
scalars without associative multiplication. A cache is valid only for the coefficients and step
count in its type, and no cache is validated by scalar Boolean equality.
-/
structure Adam.Powers {α : Type} [One α] [Mul α] (beta1 beta2 : α) (stepCount : Nat) where
  /-- First moment bias-correction power. -/
  first : α
  /-- Second moment bias-correction power. -/
  second : α
  /-- The first power follows the canonical scalar recurrence. -/
  first_eq : first = npowRec stepCount beta1
  /-- The second power follows the canonical scalar recurrence. -/
  second_eq : second = npowRec stepCount beta2

attribute [simp] Adam.Powers.first_eq Adam.Powers.second_eq

/-- Reconstruct bias-correction powers, for initialization or a restored counter. -/
def Adam.Powers.compute {α : Type} [One α] [Mul α] (beta1 beta2 : α)
    (stepCount : Nat) : Powers beta1 beta2 stepCount :=
  ⟨npowRec stepCount beta1, npowRec stepCount beta2, rfl, rfl⟩

/-- Advance both powers by one step, multiplying each on the right by its coefficient. -/
def Adam.Powers.advance {α : Type} [One α] [Mul α] {beta1 beta2 : α}
    {stepCount : Nat} (powers : Powers beta1 beta2 stepCount) :
    Powers beta1 beta2 (stepCount + 1) :=
  ⟨powers.first * beta1, powers.second * beta2,
    by rw [powers.first_eq]; rfl, by rw [powers.second_eq]; rfl⟩

/-! ## Shared equations -/

/-- Each entry's square scaled by `weight`: `(x * x) * weight` when `x * x` is finite, and
`(x * weight) * x` otherwise, so an overflowing square can give a finite weighted value. -/
def Optimizer.weightedSquare {α : Type} [Storage α] [Zero α] [Sub α] [Mul α] [BEq α]
    {s : Shape} (values : Tensor α s) (weight : α) : Tensor α s :=
  Tensor.map (fun x =>
    let squared := x * x
    if squared - squared == 0 then squared * weight else (x * weight) * x) values

/-- Divide `moment` by the bias-correction `denominator`, scaling by `1 / denominator` when that
reciprocal is finite and dividing each entry by `denominator` when it is not. -/
def Optimizer.correctMoment {α : Type} [Storage α] [Context α] {s : Shape}
    (moment : Tensor α s) (denominator : α) : Tensor α s :=
  let inverse := 1 / denominator
  if inverse - inverse == 0 then Tensor.scale moment inverse
  else Tensor.map (fun x => x / denominator) moment

/-- The root $\sqrt{\max(\widehat v,0)}$ of each entry $\widehat v$ of the corrected second
moment. Where $\widehat v$ is not finite and `denominator` is positive, the root is
$\sqrt{\max(v,0)}/\sqrt{\mathtt{denominator}}$ from the uncorrected entry $v$, so an entry whose
correction overflows can have a finite root. -/
def Optimizer.correctedRMS {α : Type} [Storage α] [Context α] {s : Shape}
    (moment : Tensor α s) (denominator : α) : Tensor α s :=
  let corrected := correctMoment moment denominator
  zipWith (fun x v =>
    if v - v == 0 || !(denominator > 0) then MathFunctions.sqrt (Max.max v 0)
    else MathFunctions.sqrt (Max.max x 0) / MathFunctions.sqrt denominator) moment corrected

/-- The update $(m/(\sigma+\varepsilon))\,\mathtt{lr}$ for each entry $m$ of `moment` and the
matching entry $\sigma$ of `standardDeviation`, normalized before the learning rate is applied.

When the scalar's `Numeric.QuotientArithmetic` is supported and the quotient or the product
reports range loss, the entry is replayed from the finite coefficient trees of $m$, `lr` and
$\sigma+\varepsilon$ with the scalar's checked rounding. Replay recovers an update lost by either
association of the product and quotient, its dual coefficients included. Nonfinite inputs, a zero
divisor and unsupported carriers keep the ordinary result, and replay cannot undo rounding already
present in the inputs.
-/
def Optimizer.normalizedUpdate {α : Type} [Storage α] [Context α] {s : Shape}
    (lr epsilon : α) (moment standardDeviation : Tensor α s) : Tensor α s :=
  let ops : Numeric.QuotientArithmetic α := inferInstance
  zipWith (fun x root =>
    let denominator := root + epsilon
    if !ops.supported then (x / denominator) * lr
    else
      let (quotient, divisionFlags) := ops.divWithFlags x denominator
      let (ordinary, productFlags) := ops.mulWithFlags quotient lr
      if divisionFlags ||| productFlags == 0 then ordinary
      else
        match ops.encode x, ops.encode lr, ops.encode denominator with
        | some a, some b, some c =>
            if c.primal == 0 then ordinary
            else
              let coefficients := Numeric.QuotientCoefficients.div
                (Numeric.QuotientCoefficients.mul a b) c
              match ops.decode coefficients with
              | some repaired =>
                  if coefficients.primal == 0 then
                    ops.copyPrimal ((x * lr) / denominator) repaired
                  else repaired
              | none => ordinary
        | _, _, _ => ordinary) moment standardDeviation

/-- The next momentum buffer $b\mu+g$, each buffer entry multiplied on the right by `momentum` as
in `Tensor.scale`. -/
def Optimizer.updateMomentumBuffer {α : Type} [Storage α] [Add α] [Mul α] {s : Shape}
    (momentumBuffer : Tensor α s) (momentum : α)
    (gradients : Tensor α s) : Tensor α s :=
  Tensor.scale momentumBuffer momentum + gradients

/-! ## SGD -/

/--
SGD state (per parameter tensor).

The state holds only the learning rate.
-/
structure SGD.State (α : Type) [Storage α] (s : Shape) where
  /-- Learning rate. -/
  lr : α

/--
Initialize SGD state.

The parameter tensor is unused. It is in the signature so that all optimizers share the same
“init from parameters” calling convention.
-/
def SGD.init {α : Type} [Storage α] {s : Shape}
  (lr : α) (_ : Tensor α s) : SGD.State α s :=
  { lr := lr }

/--
One SGD step: `p ← p - g * lr`, with the learning rate on the right of each gradient element.

It is the step of `torch.optim.SGD` with zero `momentum` and `weight_decay`.
-/
def SGD.update {α : Type} [Storage α] [Sub α] [Mul α] {s : Shape}
    (state : SGD.State α s) (parameters gradients : Tensor α s) :
    OptimizerStep α s (SGD.State α s) :=
  { optimizerState := state
    parameters := parameters - Tensor.scale gradients state.lr }

/-! ## Momentum SGD -/

/--
Momentum SGD state (per parameter tensor).

It stores a momentum buffer $b$ and a momentum coefficient $\mu$, and a step updates the buffer,
then the parameters:

- $b\gets b\mu+g$,
- $p\gets p-b\,\mathtt{lr}$.

PyTorch's SGD momentum computes the same update when `dampening = 0` and `nesterov = false`.
Sutskever et al. (2013), eqs. (1) and (2), scale each gradient by the learning rate as it enters
the velocity. With a constant `lr` both forms give the same iterates, and when `lr` changes
between steps ours rescales the whole buffer, the form the `torch.optim.SGD` docs describe.
-/
structure MomentumSGD.State (α : Type) [Storage α] (s : Shape) where
  /-- Learning rate. -/
  lr : α
  /-- Momentum coefficient $\mu$. -/
  momentum : α
  /-- Momentum buffer. -/
  momentumBuffer : Tensor α s

/-- Initialize momentum SGD with a zero buffer.

Torch copies the first gradient into its buffer. We start from zero, so the first buffer is
$0\mu+g$, the same value for a finite $\mu$ except that a `-0` gradient entry becomes `+0` when
$\mu$ is `+0` or positive. -/
def MomentumSGD.init {α : Type} [Storage α] [Zero α] {s : Shape}
  (lr momentum : α) (_ : Tensor α s) : MomentumSGD.State α s :=
  { lr := lr, momentum := momentum, momentumBuffer := Tensor.full s 0 }

/-- One momentum-SGD step. -/
def MomentumSGD.update {α : Type} [Storage α] [Add α] [Sub α] [Mul α]
    {s : Shape}
    (state : MomentumSGD.State α s) (parameters gradients : Tensor α s) :
    OptimizerStep α s (MomentumSGD.State α s) :=
  let nextMomentumBuffer :=
    Optimizer.updateMomentumBuffer state.momentumBuffer state.momentum gradients
  { optimizerState := { state with momentumBuffer := nextMomentumBuffer }
    parameters := parameters - Tensor.scale nextMomentumBuffer state.lr }

/-! ## AdaGrad -/

/--
AdaGrad state (per parameter tensor).

It stores an accumulator $G$ of squared gradients, of the same shape as the parameters. The
effective step size is scaled by $1/(\sqrt G+\varepsilon)$.

This is the diagonal case of eq. (5) in Duchi, Hazan and Singer,
$H_t=\delta I+\mathrm{diag}(G_t)^{1/2}$, with their $\delta$ as `epsilon`, outside the root as in
`torch.optim.Adagrad`. Our zero start is torch's default `initial_accumulator_value=0`.
-/
structure AdaGrad.State (α : Type) [Storage α] (s : Shape) where
  /-- Base learning rate. -/
  lr : α
  /-- Numerical stability constant $\varepsilon$. -/
  epsilon : α
  /-- Accumulated squared gradients. -/
  squaredGradientSum : Tensor α s

/-- Initialize AdaGrad with a zero accumulator. -/
def AdaGrad.init {α : Type} [Storage α] [Zero α] {s : Shape}
  (lr epsilon : α) (_ : Tensor α s) : AdaGrad.State α s :=
  { lr := lr, epsilon := epsilon, squaredGradientSum := Tensor.full s 0 }

/-- One AdaGrad step. -/
def AdaGrad.update {α : Type} [Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : AdaGrad.State α s) (parameters gradients : Tensor α s) :
    OptimizerStep α s (AdaGrad.State α s) :=
  let squaredGradients := Tensor.square gradients
  let nextSquaredGradientSum := state.squaredGradientSum + squaredGradients
  { optimizerState := { state with squaredGradientSum := nextSquaredGradientSum }
    parameters := parameters -
      (Optimizer.normalizedUpdate state.lr state.epsilon gradients
        (Tensor.sqrt nextSquaredGradientSum .clampNegative)) }

/-! ## RMSProp -/

/--
RMSProp state (per parameter tensor).

It stores an exponential moving average of squared gradients.

We took the update from Hinton's lecture 6e slide, which credits Tijmen Tieleman (unpublished),
weights the mean square 0.9 and 0.1 and adds no $\varepsilon$. Ours adds `epsilon` after the
square root, where `torch.optim.RMSprop` adds its `eps`.
-/
structure RMSProp.State (α : Type) [Storage α] (s : Shape) where
  /-- Learning rate. -/
  lr : α
  /-- Decay coefficient for the EMA of $g^2$ (often called `alpha`). -/
  decay : α
  /-- Numerical stability constant $\varepsilon$. -/
  epsilon : α
  /-- EMA of squared gradients. -/
  squaredGradientAverage : Tensor α s

/-- Initialize RMSProp with a zero average of squared gradients. -/
def RMSProp.init {α : Type} [Storage α] [Zero α] {s : Shape}
  (lr decay epsilon : α) (_ : Tensor α s) : RMSProp.State α s :=
  { lr := lr
    decay := decay
    epsilon := epsilon
    squaredGradientAverage := Tensor.full s 0 }

/-- One RMSProp step. -/
def RMSProp.update {α : Type} [Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : RMSProp.State α s) (parameters gradients : Tensor α s) :
    OptimizerStep α s (RMSProp.State α s) :=
  let nextSquaredGradientAverage := Tensor.scale state.squaredGradientAverage state.decay +
    Optimizer.weightedSquare gradients (1 - state.decay)
  { optimizerState := { state with squaredGradientAverage := nextSquaredGradientAverage }
    parameters := parameters -
      (Optimizer.normalizedUpdate state.lr state.epsilon gradients
        (Tensor.sqrt nextSquaredGradientAverage .clampNegative)) }

/-! ## Adam -/

/--
Adam state (per parameter tensor).

It stores first and second moment averages, a step counter and certified bias-correction powers.
The default `powers` is `Powers.compute`, so a state created at a nonzero step gets the powers
of the recurrence. A record update that changes either beta or the counter must also set `powers`
to `Powers.compute` at the updated values, and updates to other fields keep the cache.
-/
structure Adam.State (α : Type) [Storage α] [One α] [Mul α] (s : Shape) where
  /-- Learning rate. -/
  lr : α
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
  powers : Powers beta1 beta2 stepCount := Powers.compute beta1 beta2 stepCount

/-- Initialize Adam with zero moments and a zero step count. -/
def Adam.init {α : Type} [Storage α] [Zero α] [One α] [Mul α] {s : Shape}
  (lr beta1 beta2 epsilon : α) (_ : Tensor α s) : Adam.State α s :=
  {
    lr := lr,
    beta1 := beta1,
    beta2 := beta2,
    epsilon := epsilon,
    firstMoment := Tensor.full s 0,
    secondMoment := Tensor.full s 0,
    stepCount := 0
  }

/--
One Adam step.

Elementwise, with $t$ the updated step count:

- $m\gets\beta_1m+(1-\beta_1)g$,
- $v\gets\beta_2v+(1-\beta_2)g^2$,
- $\widehat m\gets m/(1-\beta_1^t)$,
- $\widehat v\gets v/(1-\beta_2^t)$,
- $p\gets p-\mathtt{lr}\,\widehat m/(\sqrt{\widehat v}+\varepsilon)$.

We copied these five lines by hand from Algorithm 1 of the Adam paper, with `lr` for its $\alpha$.
The $\varepsilon$ placement matches Kingma and Ba: it is added after $\sqrt{\widehat v}$.
Weighted squares, moment corrections and the corrected square root avoid unnecessary overflow.
The learning rate is applied after normalization, and supported scalars replay the finite
coefficient trees when that product or the quotient loses range (`Optimizer.normalizedUpdate`).
Stored moments and final updates can overflow the chosen scalar format.
-/
def Adam.update {α : Type} [Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : Adam.State α s) (parameters gradients : Tensor α s) :
    OptimizerStep α s (Adam.State α s) :=
  let nextStepCount := state.stepCount + 1
  let nextPowers := state.powers.advance
  let nextFirstMoment :=
    Tensor.scale state.firstMoment state.beta1 +
      Tensor.scale gradients (1 - state.beta1)
  let nextSecondMoment :=
    Tensor.scale state.secondMoment state.beta2 +
      Optimizer.weightedSquare gradients (1 - state.beta2)
  let correctedFirstMoment :=
    Optimizer.correctMoment nextFirstMoment (1 - nextPowers.first)
  let correctedRoot :=
    Optimizer.correctedRMS nextSecondMoment (1 - nextPowers.second)
  { optimizerState :=
      { state with
        firstMoment := nextFirstMoment
        secondMoment := nextSecondMoment
        stepCount := nextStepCount
        powers := nextPowers }
    parameters :=
      parameters -
        Optimizer.normalizedUpdate state.lr state.epsilon correctedFirstMoment correctedRoot }

/-- Adam increments its step counter by one on every update. -/
@[simp] theorem Adam.update_stepCount {α : Type} [Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : Adam.State α s) (parameters gradients : Tensor α s) :
    (Adam.update state parameters gradients).optimizerState.stepCount =
      state.stepCount + 1 := rfl

/-! ## AdamW -/

/--
AdamW state (per parameter tensor).

AdamW is “Adam + decoupled weight decay”: the decay acts on the parameters as a separate term, and
the gradient that feeds the moments carries no decay. The state is an `Adam.State` with the decay
coefficient, and its certified powers follow `Adam.State`, so replacing either beta or the counter
also requires `powers := Adam.Powers.compute` at the updated values.
-/
structure AdamW.State (α : Type) [Storage α] [One α] [Mul α] (s : Shape)
    extends Adam.State α s where
  /-- Weight decay coefficient `wd`. -/
  weightDecay : α

/-- Initialize AdamW state for a parameter tensor (moments start at `0`). -/
def AdamW.init {α : Type} [Storage α] [Zero α] [One α] [Mul α] {s : Shape}
    (lr weightDecay beta1 beta2 epsilon : α) (parameters : Tensor α s) : AdamW.State α s :=
  { toState := Adam.init lr beta1 beta2 epsilon parameters, weightDecay := weightDecay }

/--
One AdamW step.

It follows the decoupled form of the AdamW paper:
- apply weight decay to the parameters (`p ← p - lr * wd * p`),
- apply the canonical Adam update to the decayed parameters, using the *raw* gradient `g`,
- keep the AdamW decay coefficient with the updated moments and powers.

`torch.optim.AdamW` orders a single step the same way.

Line 12 of the paper's Algorithm 2 subtracts $\eta_t\lambda\theta_{t-1}$ with `lr` in the role of
$\eta_t\alpha$, so our `wd` is their $\lambda/\alpha$, and torch's default `weight_decay=0.01` at
`lr=1e-3` is their $\lambda=10^{-5}$ when $\eta_t=1$. At `wd = 0` torch skips its
`1 - lr * weight_decay` factor, while our `p - p * (lr * wd)` turns `±∞` into NaN and `-0` into
`+0`, the results that `TorchLean.Test.CUDA.Trainer.checkAdamSpec` checks for the CUDA AdamW.
-/
def AdamW.update {α : Type} [Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : AdamW.State α s) (parameters gradients : Tensor α s) :
    OptimizerStep α s (AdamW.State α s) :=
  let decayedParameters :=
    parameters -
      Tensor.scale parameters (state.lr * state.weightDecay)
  let result := Adam.update state.toState decayedParameters gradients
  { optimizerState := { state with toState := result.optimizerState }
    parameters := result.parameters }

/-- AdamW increments its step counter by one on every update. -/
@[simp] theorem AdamW.update_stepCount {α : Type} [Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : AdamW.State α s) (parameters gradients : Tensor α s) :
    (AdamW.update state parameters gradients).optimizerState.stepCount =
      state.stepCount + 1 := rfl

/-! ## Adadelta -/

/--
Adadelta state (per parameter tensor).

It stores two EMAs:
- `squaredGradientAverage`,
- `squaredUpdateAverage`.
-/
structure Adadelta.State (α : Type) [Storage α] (s : Shape) where
  /-- Learning rate. Zeiler's update has none, which corresponds to `1`. -/
  lr : α
  /-- Decay coefficient $\rho$. -/
  rho : α
  /-- Numerical stability constant $\varepsilon$. -/
  epsilon : α
  /-- EMA of squared gradients. -/
  squaredGradientAverage : Tensor α s
  /-- EMA of squared updates. -/
  squaredUpdateAverage : Tensor α s

/-- Initialize Adadelta state for a parameter tensor (EMAs start at `0`). -/
def Adadelta.init {α : Type} [Storage α] [Zero α] {s : Shape}
  (lr rho epsilon : α) (_ : Tensor α s) : Adadelta.State α s :=
  { lr := lr
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
With `lr = 1` these lines are eq. (14) and Algorithm 1 of Zeiler (2012). For any other `lr` we
follow `torch.optim.Adadelta`, which also feeds the unscaled $\Delta p$ into the update average.
-/
def Adadelta.update {α : Type} [Storage α] [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : Adadelta.State α s) (parameters gradients : Tensor α s) :
    OptimizerStep α s (Adadelta.State α s) :=
  let nextSquaredGradientAverage :=
    Tensor.scale state.squaredGradientAverage state.rho +
      Optimizer.weightedSquare gradients (1 - state.rho)

  let rms := fun v : α => MathFunctions.sqrt (Max.max (v + state.epsilon) 0)
  let gradientRms := Tensor.map rms nextSquaredGradientAverage
  let updateRms := Tensor.map rms state.squaredUpdateAverage

  let ratio := updateRms / gradientRms
  let parameterUpdate := ratio * gradients
  let nextParameters :=
    parameters - Tensor.scale parameterUpdate state.lr

  let nextSquaredUpdateAverage :=
    Tensor.scale state.squaredUpdateAverage state.rho +
      Optimizer.weightedSquare parameterUpdate (1 - state.rho)
  { optimizerState :=
      { state with
        squaredGradientAverage := nextSquaredGradientAverage
        squaredUpdateAverage := nextSquaredUpdateAverage }
    parameters := nextParameters }

/-! ## GaLore-style projected SGD -/

namespace GaLore

/--
A shape-safe gradient projector.

GaLore-style training periodically builds a low-rank subspace for a large matrix parameter,
projects the gradient into that subspace, runs a base optimizer there, and lifts the update back to
the original parameter shape. The record holds only the two maps, and computing or refreshing a
projector, the costly part, is left to the caller.
-/
structure Projector (α : Type) [Storage α] (full low : Shape) where
  /-- Project a full gradient into the low-rank optimizer space. -/
  project : Tensor α full → Tensor α low
  /-- Lift a low-rank update back to the full parameter shape. -/
  lift : Tensor α low → Tensor α full

/-- Identity projector, used when projected SGD is requested without a projection backend. -/
def identityProjector {α : Type} [Storage α] {s : Shape} : Projector α s s :=
  { project := id, lift := id }

/--
GaLore-style projected SGD state for one tensor.

It holds the learning rate and the projector, which `GaLore.update` leaves unchanged. Full GaLore
also needs a refresh schedule and a way to build projectors for large matrix parameters, which lie
outside this state.
-/
structure SGDState (α : Type) [Storage α] (full low : Shape) where
  /-- Learning rate used after the gradient has been projected and lifted. -/
  lr : α
  /-- Current gradient projector. -/
  projector : Projector α full low

/-- One projected-SGD update: `p ← p - lift(project(g)) * lr`.

We wrote it by hand as the SGD case of Algorithm 1 in Zhao et al. (2024), scaling by `lr` after
the lift, the same step whenever `lift` is linear. Their Algorithm 2 runs Adam in the compact space
and multiplies the lifted step by a scale $\alpha$, which a `lift` can carry. -/
def update {α : Type} [Storage α] [Sub α] [Mul α]
    {full low : Shape} (state : SGDState α full low)
    (parameters gradients : Tensor α full) :
    OptimizerStep α full (SGDState α full low) :=
  { optimizerState := state
    parameters :=
      parameters -
        (Tensor.scale
          (state.projector.lift (state.projector.project gradients))
          state.lr) }

end GaLore

/-! ## Muon-style orthogonalized momentum -/

namespace Muon

/--
Orthogonalization backend for a matrix-shaped update.

Muon uses a momentum buffer and then replaces the raw momentum direction by an approximately
orthogonalized update, commonly via Newton-Schulz iterations. The backend is a field of the state,
so one update rule serves different orthogonalization algorithms.
-/
structure Orthogonalizer (α : Type) [Storage α] (s : Shape) where
  /-- Convert a momentum buffer into the direction used for the parameter update. -/
  apply : Tensor α s → Tensor α s

/-- The identity orthogonalizer, used when Muon is requested without a matrix backend. -/
def identityOrthogonalizer {α : Type} [Storage α] {s : Shape} :
    Orthogonalizer α s :=
  { apply := id }

/-- Per-parameter state for Muon-style momentum: a momentum-SGD state with an orthogonalization
backend. -/
structure State (α : Type) [Storage α] (s : Shape)
    extends MomentumSGD.State α s where
  /-- Backend that turns the momentum buffer into the update direction. -/
  orthogonalizer : Orthogonalizer α s

/-- Initialize Muon-style state with a zero momentum buffer. -/
def init {α : Type} [Storage α] [Zero α] {s : Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (parameters : Tensor α s) : State α s :=
  { toState := MomentumSGD.init lr momentum parameters, orthogonalizer := orthogonalizer }

/--
One Muon-style update:
- update the momentum buffer,
- orthogonalize the buffer,
- subtract the orthogonalized direction scaled by the learning rate.

For actual Muon, use a matrix-shaped `s` and a Newton-Schulz orthogonalizer. The shape is generic
so that tests and other matrix layouts can reuse the definition.

With that backend these are lines 4 to 6 of Algorithm 2 in Jordan et al. (2024), `lr` for $\eta$
and `momentum` for $\mu$. Their post finds Nesterov-style momentum a bit better in every case they
tested, and we keep the plain buffer of Algorithm 2.
-/
def update {α : Type} [Storage α] [Add α] [Sub α] [Mul α] {s : Shape}
    (state : State α s) (parameters gradients : Tensor α s) :
    OptimizerStep α s (State α s) :=
  let nextMomentumBuffer :=
    Optimizer.updateMomentumBuffer state.momentumBuffer state.momentum gradients
  let direction := state.orthogonalizer.apply nextMomentumBuffer
  { optimizerState := { state with momentumBuffer := nextMomentumBuffer }
    parameters := parameters - Tensor.scale direction state.lr }

end Muon

end TorchLean
