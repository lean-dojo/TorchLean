/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.TensorOps

/-!
# Optimizers

Optimizers for TorchLean runtime training.

This file implements the *core math* of common gradient-based optimizers as pure functions on
typed tensors `Tensor α s`.

Why “pure functions”?

In PyTorch, optimizers mutate parameters in-place and keep state in Python objects.
In TorchLean, we want the update rule itself to be explicit and easy to reuse:
- eager demos can call the update directly,
- the runtime training engine can store state in maps keyed by parameter ids,
- and proofs can refer to the same update equations.

The intent is to mimic the standard textbook formulas closely. We do not try to reproduce every
implementation detail of `torch.optim.*` (e.g. foreach kernels, fused updates, or every optional
flag); those live at a different layer than the math we specify here.

Where this file sits in the stack:
- this file owns the scalar-polymorphic, per-tensor update equations;
- `NN.Runtime.Autograd.TorchLean.Optim` lifts those equations to runtime parameter lists; and
- `NN.API.Runtime` exposes ergonomic `optim.sgd`, `optim.adam`, and related configuration helpers.

That separation is deliberate: the formula appears once, while runtime adapters and public API
configuration can evolve independently around it.

Why each optimizer has its own `State` structure:
- Lean structures do not inherit from one another the way Python classes do.
- More importantly, optimizer state is not uniform: SGD stores only `lr`, momentum SGD stores a
  buffer, Adam/AdamW store two moment buffers and a step counter, Adadelta stores gradient/update
  EMAs, and Muon/GaLore carry backend functions.
- Keeping these as separate typed states makes impossible states unrepresentable. For example, an
  SGD state cannot accidentally contain a stale Adam `v` buffer, and AdamW cannot forget its
  decoupled `weight_decay` coefficient.

The generic abstraction lives one layer up:
- `Runtime.Autograd.TorchLean.Optim.Optimizer` packages `init`/`step` for shape-indexed parameter
  lists, like a typed analogue of a PyTorch optimizer object.
- `Runtime.Autograd.Train.OptimizerState` handles dynamic parameter groups and checkpoint-style
  maps for the training-loop API.

So this file intentionally favors small canonical state records over an inheritance hierarchy.

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
open Spec
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/--
Integer exponentiation for scalar optimizer coefficients.

We use an explicit `Nat → α` recursion instead of `x ^ (n : Nat)` because `Context α`
provides `Pow α α` (for runtime scalar exponentiation), but not `Pow α Nat`.
-/
def scalarPowNat {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => scalarPowNat x n * x

/-- Scalar exponentiation starts at `1`. -/
@[simp] theorem scalarPowNat_zero {α : Type} [One α] [Mul α] (x : α) :
    scalarPowNat x 0 = 1 := by
  rfl

/-- Successor case for scalar exponentiation. -/
@[simp] theorem scalarPowNat_succ {α : Type} [One α] [Mul α] (x : α) (n : Nat) :
    scalarPowNat x (n + 1) = scalarPowNat x n * x := by
  rfl

/-! ## Shared utilities -/
namespace OptimizerUtils

/-- Momentum-style buffer update `μ * buf + g`. -/
def updateMomentumBuf {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (buf : Tensor α s) (momentum : α) (grads : Tensor α s) : Tensor α s :=
  addSpec (scaleSpec buf momentum) grads

/--
Elementwise “adaptive learning rate” tensor `lr / (sqrt(denom) + ε)`.

This is shared by AdaGrad/RMSProp/Adam-style optimizers.
-/
def mkAdaptiveLR {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr epsilon : α) (denom : Tensor α s) : Tensor α s :=
  divSpec (fill lr s) (addSpec (sqrtSpec denom) (fill epsilon s))

end OptimizerUtils

/-! ## SGD -/

/--
SGD state (per parameter tensor).

We only store the learning rate here.
-/
structure SGD.State (α : Type) (s : Shape) where
  /-- Learning rate. -/
  lr : α

/--
Initialize SGD state.

The parameter tensor is unused; we keep it in the signature so optimizers share the same
“init from parameters” calling convention.
-/
def SGD.init {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (lr : α) (_ : Tensor α s) : SGD.State α s :=
  { lr := lr }

/-- SGD initialization records exactly the requested learning rate. -/
@[simp] theorem SGD.init_lr {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr : α) (params : Tensor α s) :
    (SGD.init lr params).lr = lr := by
  rfl

/--
One SGD step: `p ← p - lr * g`.

PyTorch analogy: the core of `torch.optim.SGD` without momentum/weight-decay extras.
-/
def SGD.update {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (state : SGD.State α s) (params : Tensor α s) (grads : Tensor α s) : Tensor α s :=
  subSpec params (scaleSpec grads state.lr)

/-! ## Momentum SGD -/

/--
Momentum SGD state (per parameter tensor).

We store a momentum buffer `buf` and a momentum coefficient `μ`.
Update rule:
- `buf ← μ buf + g`
- `p ← p - lr * buf`

This matches PyTorch's SGD momentum behavior when `dampening = 0` and `nesterov = false`.
-/
structure MomentumSGD.State (α : Type) (s : Shape) where
  /-- Learning rate. -/
  lr : α
  /-- Momentum coefficient `μ`. -/
  momentum : α
  /-- Momentum buffer `buf`. -/
  buf : Tensor α s

/-- Initialize momentum SGD with a zero buffer. -/
def MomentumSGD.init {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (lr : α) (momentum : α) (_ : Tensor α s) : MomentumSGD.State α s :=
  { lr := lr, momentum := momentum, buf := fill 0 s }

/-- Momentum-SGD starts with a zero momentum buffer. -/
@[simp] theorem MomentumSGD.init_buf {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr momentum : α) (params : Tensor α s) :
    (MomentumSGD.init lr momentum params).buf = fill 0 s := by
  rfl

/-- One momentum-SGD step (returns updated state and parameters). -/
def MomentumSGD.update {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (state : MomentumSGD.State α s) (params : Tensor α s) (grads : Tensor α s) : (MomentumSGD.State α
    s × Tensor α s) :=
  let newBuf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads
  let newParams := subSpec params (scaleSpec newBuf state.lr)
  ({ state with buf := newBuf }, newParams)

/-! ## AdaGrad -/

/--
AdaGrad state (per parameter tensor).

We store an accumulator `G` of squared gradients (same shape as the parameters). The effective
step size is scaled by `1 / (sqrt(G) + ε)`.
-/
structure AdaGrad.State (α : Type) (s : Shape) where
  /-- Base learning rate. -/
  lr : α
  /-- Numerical stability constant `ε`. -/
  epsilon : α
  /-- Accumulated squared gradients. -/
  accumulator : Tensor α s

/-- Initialize AdaGrad with zero accumulator. -/
def AdaGrad.init {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (lr : α) (epsilon : α) (_ : Tensor α s) : AdaGrad.State α s :=
  { lr := lr, epsilon := epsilon, accumulator := fill 0 s }

/-- AdaGrad starts with a zero squared-gradient accumulator. -/
@[simp] theorem AdaGrad.init_accumulator {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr epsilon : α) (params : Tensor α s) :
    (AdaGrad.init lr epsilon params).accumulator = fill 0 s := by
  rfl

/-- One AdaGrad step (returns updated state and parameters). -/
def AdaGrad.update {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (state : AdaGrad.State α s) (params : Tensor α s) (grads : Tensor α s) : (AdaGrad.State α s ×
    Tensor α s) :=
  let squaredGrads := squareSpec grads
  let newAccumulator := addSpec state.accumulator squaredGrads
  let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon newAccumulator
  let newParams := subSpec params (mulSpec adaptiveLR grads)
  ({ state with accumulator := newAccumulator }, newParams)

/-! ## RMSProp -/

/--
RMSProp state (per parameter tensor).

We store an EMA of squared gradients (`accumulator`), often called `square_avg` in PyTorch code.
-/
structure RMSProp.State (α : Type) (s : Shape) where
  /-- Learning rate. -/
  lr : α
  /-- Decay coefficient for the EMA of `g²` (often called `alpha`). -/
  decay : α
  /-- Numerical stability constant `ε`. -/
  epsilon : α
  /-- EMA of squared gradients. -/
  accumulator : Tensor α s

/-- Initialize RMSProp with zero accumulator. -/
def RMSProp.init {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (lr : α) (decay : α) (epsilon : α) (_ : Tensor α s) : RMSProp.State α s :=
  { lr := lr, decay := decay, epsilon := epsilon, accumulator := fill 0 s }

/-- RMSProp starts with a zero running average of squared gradients. -/
@[simp] theorem RMSProp.init_accumulator {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr decay epsilon : α) (params : Tensor α s) :
    (RMSProp.init lr decay epsilon params).accumulator = fill 0 s := by
  rfl

/-- One RMSProp step (returns updated state and parameters). -/
def RMSProp.update {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (state : RMSProp.State α s) (params : Tensor α s) (grads : Tensor α s) : (RMSProp.State α s ×
    Tensor α s) :=
  let squaredGrads := squareSpec grads
  let newAccumulator := addSpec
                        (scaleSpec state.accumulator state.decay)
                        (scaleSpec squaredGrads (1 - state.decay))
  let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon newAccumulator
  let newParams := subSpec params (mulSpec adaptiveLR grads)
  ({ state with accumulator := newAccumulator }, newParams)

/-! ## Adam -/

/--
Adam state (per parameter tensor).

We store first/second moment EMAs (`m`, `v`) and a step counter `t` used for bias correction.
-/
structure Adam.State (α : Type) (s : Shape) where
  /-- Learning rate. -/
  lr : α
  /-- First moment decay `β₁`. -/
  beta1 : α
  /-- Second moment decay `β₂`. -/
  beta2 : α
  /-- Numerical stability constant `ε`. -/
  epsilon : α
  /-- First moment EMA. -/
  m : Tensor α s
  /-- Second moment EMA. -/
  v : Tensor α s
  /-- Step counter (used for bias correction). -/
  t : Nat

/-- Initialize Adam with `m = 0`, `v = 0`, and `t = 0`. -/
def Adam.init {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (lr : α) (beta1 : α) (beta2 : α) (epsilon : α) (_ : Tensor α s) : Adam.State α s :=
  {
    lr := lr,
    beta1 := beta1,
    beta2 := beta2,
    epsilon := epsilon,
    m := fill 0 s,
    v := fill 0 s,
    t := 0
  }

/-- Adam starts at step `0`. -/
@[simp] theorem Adam.init_t {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr beta1 beta2 epsilon : α) (params : Tensor α s) :
    (Adam.init lr beta1 beta2 epsilon params).t = 0 := by
  rfl

/-- Adam starts with a zero first-moment buffer. -/
@[simp] theorem Adam.init_m {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr beta1 beta2 epsilon : α) (params : Tensor α s) :
    (Adam.init lr beta1 beta2 epsilon params).m = fill 0 s := by
  rfl

/-- Adam starts with a zero second-moment buffer. -/
@[simp] theorem Adam.init_v {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr beta1 beta2 epsilon : α) (params : Tensor α s) :
    (Adam.init lr beta1 beta2 epsilon params).v = fill 0 s := by
  rfl

/--
One Adam step (returns updated state and parameters).

Equations (elementwise):
- `m ← β₁ m + (1-β₁) g`
- `v ← β₂ v + (1-β₂) g²`
- `m̂ ← m / (1-β₁ᵗ)`
- `v̂ ← v / (1-β₂ᵗ)`
- `p ← p - lr * m̂ / (sqrt(v̂) + ε)`

The `ε` placement matches Kingma and Ba: it is added after `sqrt(v̂)`.
-/
def Adam.update {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (state : Adam.State α s) (params : Tensor α s) (grads : Tensor α s) : (Adam.State α s × Tensor α
    s) :=
  -- Increment timestep
  let t' := state.t + 1

  -- Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grads
  let m' := addSpec (scaleSpec state.m state.beta1) (scaleSpec grads (1 - state.beta1))

  -- Update biased second moment estimate: v = beta2 * v + (1 - beta2) * grads^2
  let v' := addSpec (scaleSpec state.v state.beta2) (scaleSpec (squareSpec grads) (1 -
    state.beta2))

  -- Compute bias-corrected first moment estimate: m_hat = m / (1 - beta1^t)
  let m_hat := scaleSpec m' (1 / (1 - scalarPowNat state.beta1 t'))

  -- Compute bias-corrected second moment estimate: v_hat = v / (1 - beta2^t)
  let v_hat := scaleSpec v' (1 / (1 - scalarPowNat state.beta2 t'))

  -- Compute adaptive learning rates: adjusted_lr = lr / (sqrt(v_hat) + epsilon)
  let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon v_hat

  -- Update parameters: params = params - adjusted_lr * m_hat
  let newParams := subSpec params (mulSpec adaptiveLR m_hat)

  ({ state with m := m', v := v', t := t' }, newParams)

/-- Adam increments its step counter by one on every update. -/
@[simp] theorem Adam.update_t {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : Adam.State α s) (params grads : Tensor α s) :
    (Adam.update state params grads).1.t = state.t + 1 := by
  simp [Adam.update]

/-! ## AdamW -/

/--
AdamW state (per parameter tensor).

AdamW is “Adam + decoupled weight decay”. The key point is that weight decay is applied as a
separate parameter decay term rather than being folded into the gradient that feeds the moments.
-/
structure AdamW.State (α : Type) (s : Shape) where
  /-- Learning rate. -/
  lr : α
  /-- First moment decay `β₁`. -/
  beta1 : α
  /-- Second moment decay `β₂`. -/
  beta2 : α
  /-- Numerical stability constant `ε`. -/
  epsilon : α
  /-- Weight decay coefficient `wd`. -/
  weight_decay : α
  /-- First moment EMA. -/
  m : Tensor α s
  /-- Second moment EMA. -/
  v : Tensor α s
  /-- Step counter (used for bias correction). -/
  t : Nat

/-- Initialize AdamW state for a parameter tensor (moments start at `0`). -/
def AdamW.init {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (lr : α) (weight_decay : α) (beta1 : α) (beta2 : α) (epsilon : α) (_ : Tensor α s) : AdamW.State α
    s :=
  {
    lr := lr,
    weight_decay := weight_decay,
    beta1 := beta1,
    beta2 := beta2,
    epsilon := epsilon,
    m := fill 0 s,
    v := fill 0 s,
    t := 0
  }

/-- AdamW initialization records the requested decoupled weight-decay coefficient. -/
@[simp] theorem AdamW.init_weight_decay {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr weightDecay beta1 beta2 epsilon : α) (params : Tensor α s) :
    (AdamW.init lr weightDecay beta1 beta2 epsilon params).weight_decay = weightDecay := by
  rfl

/-- AdamW starts at step `0`. -/
@[simp] theorem AdamW.init_t {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr weightDecay beta1 beta2 epsilon : α) (params : Tensor α s) :
    (AdamW.init lr weightDecay beta1 beta2 epsilon params).t = 0 := by
  rfl

/--
One AdamW step (returns updated state and parameters).

We implement the decoupled form from the AdamW paper:
- update Adam moments using the *raw* gradient `g`,
- apply weight decay directly to the parameters (`p ← p - lr * wd * p`),
- then apply the Adam update.

This is the same single-step ordering used by `torch.optim.AdamW`.
-/
def AdamW.update {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (state : AdamW.State α s) (params : Tensor α s) (grads : Tensor α s) : (AdamW.State α s × Tensor α
    s) :=
  let t' := state.t + 1

  -- Moments from the *raw* gradient.
  let m' := addSpec (scaleSpec state.m state.beta1) (scaleSpec grads (1 - state.beta1))
  let v' := addSpec (scaleSpec state.v state.beta2) (scaleSpec (squareSpec grads) (1 -
    state.beta2))

  -- Compute bias-corrected estimates
  let m_hat := scaleSpec m' (1 / (1 - scalarPowNat state.beta1 t'))
  let v_hat := scaleSpec v' (1 / (1 - scalarPowNat state.beta2 t'))

  -- Compute adaptive learning rate
  let adaptiveLR := OptimizerUtils.mkAdaptiveLR state.lr state.epsilon v_hat

  -- Decoupled weight decay on parameters, then the Adam step.
  let decayedParams := subSpec params (scaleSpec params (state.lr * state.weight_decay))
  let newParams := subSpec decayedParams (mulSpec adaptiveLR m_hat)

  ({ state with m := m', v := v', t := t' }, newParams)

/-- AdamW increments its step counter by one on every update. -/
@[simp] theorem AdamW.update_t {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : AdamW.State α s) (params grads : Tensor α s) :
    (AdamW.update state params grads).1.t = state.t + 1 := by
  simp [AdamW.update]

/-! ## Adadelta -/

/--
Adadelta state (per parameter tensor).

We store two EMAs:
- `v`: EMA of squared gradients,
- `u`: EMA of squared updates.
-/
structure Adadelta.State (α : Type) (s : Shape) where
  /-- Learning rate (often set to `1` in some presentations; we keep it explicit). -/
  lr : α
  /-- Decay coefficient `ρ`. -/
  rho : α
  /-- Numerical stability constant `ε`. -/
  epsilon : α
  /-- EMA of squared gradients. -/
  v : Tensor α s
  /-- EMA of squared updates. -/
  u : Tensor α s

/-- Initialize Adadelta state for a parameter tensor (EMAs start at `0`). -/
def Adadelta.init {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (lr : α) (rho : α) (epsilon : α) (_ : Tensor α s) : Adadelta.State α s :=
  { lr := lr, rho := rho, epsilon := epsilon, v := fill 0 s, u := fill 0 s }

/-- Adadelta starts with a zero squared-gradient EMA. -/
@[simp] theorem Adadelta.init_v {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr rho epsilon : α) (params : Tensor α s) :
    (Adadelta.init lr rho epsilon params).v = fill 0 s := by
  rfl

/-- Adadelta starts with a zero squared-update EMA. -/
@[simp] theorem Adadelta.init_u {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr rho epsilon : α) (params : Tensor α s) :
    (Adadelta.init lr rho epsilon params).u = fill 0 s := by
  rfl

/--
One Adadelta step (returns updated state and parameters).

Elementwise equations:
- `v ← ρ v + (1-ρ) g²`
- `Δp ← - lr * (sqrt(u + ε) / sqrt(v + ε)) ⊙ g`
- `p ← p + Δp`
- `u ← ρ u + (1-ρ) (Δp)²`

The `ε` placement is inside the RMS terms, matching Zeiler's Adadelta update.
-/
def Adadelta.update {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
  (state : Adadelta.State α s) (params : Tensor α s) (grads : Tensor α s) : (Adadelta.State α s ×
    Tensor α s) :=
  let squaredGrads := squareSpec grads
  let newV := addSpec (scaleSpec state.v state.rho) (scaleSpec squaredGrads (1 - state.rho))

  let epsT : Tensor α s := fill state.epsilon s
  let rmsV := sqrtSpec (addSpec newV epsT)
  let rmsU := sqrtSpec (addSpec state.u epsT)

  let ratio := divSpec rmsU rmsV
  let delta := scaleSpec (mulSpec ratio grads) (-state.lr)
  let newParams := addSpec params delta

  let newU := addSpec (scaleSpec state.u state.rho) (scaleSpec (squareSpec delta) (1 -
    state.rho))
  ({ state with v := newV, u := newU }, newParams)

/-! ## Projected / low-rank optimizers -/

namespace GaLore

/--
A shape-safe gradient projector.

GaLore-style training periodically builds a low-rank subspace for a large matrix parameter,
projects the gradient into that subspace, runs a base optimizer there, and lifts the update back to
the original parameter shape. This record is deliberately only the algebraic interface: the
expensive policy that computes or refreshes the projector belongs to the runtime layer.
-/
structure Projector (α : Type) (full low : Shape) where
  /-- Project a full gradient into the low-rank optimizer space. -/
  project : Tensor α full → Tensor α low
  /-- Lift a low-rank update back to the full parameter shape. -/
  lift : Tensor α low → Tensor α full

/-- Identity projector, useful for tests and for the theorem that projected SGD reduces to SGD. -/
def identityProjector {α : Type} {s : Shape} : Projector α s s :=
  { project := id, lift := id }

/--
GaLore-style projected SGD state for one tensor.

This is not a full GaLore implementation by itself: it specifies the update once a projector is
available. A practical trainer still needs a refresh schedule and a way to build projectors for
large matrix parameters.
-/
structure SGDState (α : Type) (full low : Shape) where
  /-- Learning rate used after the gradient has been projected and lifted. -/
  lr : α
  /-- Current gradient projector. -/
  projector : Projector α full low

/-- One projected-SGD update: `p ← p - lr * lift(project(g))`. -/
def projectedSGDUpdate {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {full low : Shape} (state : SGDState α full low)
    (params : Tensor α full) (grads : Tensor α full) : Tensor α full :=
  subSpec params (scaleSpec (state.projector.lift (state.projector.project grads)) state.lr)

/--
With the identity projector, projected SGD is exactly ordinary SGD.

This is the main sanity check for the GaLore extension point: adding a projection backend cannot
silently change the base optimizer when the backend is the identity.
-/
theorem projectedSGDUpdate_identity_eq_sgd {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr : α) (params grads : Tensor α s) :
    projectedSGDUpdate
        { lr := lr, projector := identityProjector (α := α) (s := s) }
        params grads =
      SGD.update { lr := lr } params grads := by
  rfl

end GaLore

/-! ## Muon-style orthogonalized momentum -/

namespace Muon

/--
Orthogonalization backend for a matrix-shaped update.

Muon uses a momentum buffer and then replaces the raw momentum direction by an approximately
orthogonalized update, commonly via Newton-Schulz iterations. TorchLean keeps this as an explicit
backend so the pure update rule is testable before CUDA kernels are introduced.
-/
structure Orthogonalizer (α : Type) (s : Shape) where
  /-- Convert a momentum buffer into the direction used for the parameter update. -/
  apply : Tensor α s → Tensor α s

/-- The identity orthogonalizer; with this backend Muon reduces to momentum SGD. -/
def identityOrthogonalizer {α : Type} {s : Shape} : Orthogonalizer α s :=
  { apply := id }

/-- Per-parameter state for Muon-style momentum with an explicit orthogonalization backend. -/
structure State (α : Type) (s : Shape) where
  /-- Learning rate. -/
  lr : α
  /-- Momentum coefficient. -/
  momentum : α
  /-- Momentum buffer. -/
  buf : Tensor α s
  /-- Backend that turns the momentum buffer into the update direction. -/
  orthogonalizer : Orthogonalizer α s

/-- Initialize Muon-style state with a zero momentum buffer. -/
def init {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s) (_ : Tensor α s) : State α s :=
  { lr := lr, momentum := momentum, buf := fill 0 s, orthogonalizer := orthogonalizer }

/--
One Muon-style update:
- update the momentum buffer,
- orthogonalize the buffer,
- subtract the scaled orthogonalized direction.

For actual Muon, use a matrix-shaped `s` and a Newton-Schulz orthogonalizer. The generic shape here
keeps the definition reusable for tests and for future batched matrix layouts.
-/
def update {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (state : State α s) (params : Tensor α s) (grads : Tensor α s) : (State α s × Tensor α s) :=
  let newBuf := OptimizerUtils.updateMomentumBuf state.buf state.momentum grads
  let direction := state.orthogonalizer.apply newBuf
  let newParams := subSpec params (scaleSpec direction state.lr)
  ({ state with buf := newBuf }, newParams)

/--
With the identity orthogonalizer, Muon's parameter update is exactly momentum SGD's parameter
update.

The state records are different because Muon carries an orthogonalizer backend, but the parameter
direction is the same when that backend is `id`.
-/
theorem update_identity_param_eq_momentumSGD {α : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (lr momentum : α) (buf params grads : Tensor α s) :
    (update
        { lr := lr
          momentum := momentum
          buf := buf
          orthogonalizer := identityOrthogonalizer (α := α) (s := s) }
        params grads).2 =
      (MomentumSGD.update { lr := lr, momentum := momentum, buf := buf } params grads).2 := by
  rfl

end Muon

end Optim
