/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Dynamics
public import TorchLean.NN.Models.Diffusion
public import TorchLean.Tensor.Context.Real

/-!
# Probability-flow ODE (spec layer)

A small continuous-time VP schedule (linear $\beta(t)$) and an epsilon-shifted probability-flow
ODE drift field, using an $\varepsilon_\theta(x,t)$ model.

The ODE belongs in the spec layer because it is a deterministic dynamical system derived from the
same diffusion model, and it is the natural interface for inference-time verification tooling
(corridor certificates, IBP/CROWN bounds on the RHS, etc.).

The implementation is scalar-polymorphic (`Context α`), so it can be executed with `Float`,
`Float32` or the configured binary32 type `ExecFloat.Binary 8 23`, run in CPU software at a chosen
precision with `FloatLib.Floats.ExecFloat.Binary`, and reasoned about with `ℝ` or the
noncomputable rounded-real model `FloatLib.Floats.Formats.Flocq.NF`.

We took the VP SDE (eq. (11)) and its probability-flow ODE (eq. (13)) from Song et al. (2021) and
wrote `pfODERhs` by hand. The score is replaced by `-ε̂ / σ(t)`, the score of their Gaussian
perturbation kernel (eq. (29)) at `x = √ᾱ(t) x₀ + σ(t) ε`, and `alphaBar` is that kernel's mean
factor squared. Their linear `β(t)` runs from `β̄_min = 0.1` to `β̄_max = 20` (Appendix C), our
`beta0` and `beta1`. They sample with a black-box Dormand-Prince solver (Section 4.3), and
Appendix C ends VP sampling at `t = 10⁻³`. `pfODESampleEuler` takes fixed Euler steps down to
`t = 0`, so with `steps = 1000` it calls the model at `t = 1, 0.999, ..., 0.001`.
`pfODEEulerSystem_step` holds by definition, and nothing here is checked against torch.

References:
- Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben
  Poole (2021), "Score-Based Generative Modeling through Stochastic Differential Equations",
  ICLR 2021, arXiv:2011.13456, eqs. (11), (13) and (29), Section 4.3 and Appendix C.
  Its marginal-equivalence result concerns the exact score and the unshifted continuous ODE. The
  module uses a learned-score approximation, an epsilon shift and discrete Euler steps, and no
  theorem here relates those approximations to the SDE marginals.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Diffusion

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Continuous-time linear VP schedule on $t\in[0,1]$:
$\beta(t)=\beta_0+t(\beta_1-\beta_0)$. -/
structure VPLinearSchedule (α : Type) where
  /-- $\beta(0)$. -/
  beta0 : α
  /-- $\beta(1)$. -/
  beta1 : α

namespace VPLinearSchedule

/-- Linear interpolation $\beta(t)$ on $t\in[0,1]$. -/
def beta (sch : VPLinearSchedule α) (t : α) : α :=
  sch.beta0 + t * (sch.beta1 - sch.beta0)

/--
Closed-form $\bar\alpha(t)$ for the VP SDE with linear $\beta(t)$:

$$
\bar\alpha(t)
=\exp\!\left(-\int_0^t\beta(s)\,ds\right)
=\exp\!\left[-\left(\beta_0t+\tfrac12(\beta_1-\beta_0)t^2\right)\right].
$$
-/
def alphaBar (sch : VPLinearSchedule α) (t : α) : α :=
  let half : α := (1 / 2)
  -- Writing `t*t` for `t^2` avoids relying on numeral coercions into arbitrary backends.
  let intBeta : α := sch.beta0 * t + half * (sch.beta1 - sch.beta0) * (t * t)
  MathFunctions.exp (-intBeta)

/-- $\sigma(t)=\sqrt{1-\bar\alpha(t)}$ (clamped to stay total). -/
def sigma (sch : VPLinearSchedule α) (t : α) : α :=
  sqrtNonneg (1 - sch.alphaBar t)

end VPLinearSchedule

variable {s : Shape}

/--
Probability-flow ODE drift for a VP schedule, expressed via an $\varepsilon_\theta(x,t)$ model.

For VP SDE:

$$
dx=-\tfrac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dW.
$$

The probability-flow ODE is:

$$
dx=\left[-\tfrac12\beta(t)x-\tfrac12\beta(t)\operatorname{score}(x,t)\right]dt.
$$

Using the $\varepsilon$-parameterization, an approximate score is
$\operatorname{score}\approx-\hat\varepsilon/\sigma(t)$, so:

$$
dx=\left[
  -\tfrac12\beta(t)x
  +\tfrac12\frac{\beta(t)}{\sigma(t)}\hat\varepsilon(x,t)
\right]dt.
$$

The implementation divides by $\sigma(t)+\mathtt{TorchLean.Context.defaultEpsilon}$ in place of
$\sigma(t)$. The shift is part of the denotation, over the reals too, and for arbitrary schedules
and contexts the shifted denominator can be zero.
-/
def pfODERhs (sch : VPLinearSchedule α) (model : EpsModel α s) (x : Tensor α s) (t : α) :
    Tensor α s :=
  let β : α := sch.beta t
  let σ : α := sch.sigma t
  let epsHat : Tensor α s := model.eps x t
  let drift_x : Tensor α s := Tensor.scale x ((-1 / 2) * β)
  let drift_eps : Tensor α s :=
    Tensor.scale epsHat ((1 / 2) * safeDiv β σ)
  drift_x + drift_eps

/--
One explicit Euler step for an ODE $x'=f(x,t)$:

$x_{\mathrm{next}}=x+dt\,f(x,t)$.

To integrate the probability-flow ODE *backwards* from $t=1$ to $t=0$, use a negative $dt$.
-/
def eulerStep (f : Tensor α s → α → Tensor α s) (x : Tensor α s) (t dt : α) : Tensor α s :=
  x + Tensor.scale (f x t) dt

/--
Deterministic probability-flow sampler using Euler integration on a uniform grid.

Inputs:
- `steps`: number of Euler steps (typically large, e.g. 1000),
- `x1`: initial state at $t=1$ (typically standard normal noise).

For `steps > 0`, the counter runs from `steps` down to `1`. Each time is the exact ratio
`counter / steps` cast once into the scalar, and `dt = -(1 / steps)` likewise (binary backends
round each ratio once). Over the reals this is the descending grid
$1,(\mathtt{steps}-1)/\mathtt{steps},\ldots,1/\mathtt{steps}$. Zero steps return `x1` unchanged.
-/
def pfODESampleEuler (sch : VPLinearSchedule α) (model : EpsModel α s)
    (steps : Nat) (x1 : Tensor α s) : Tensor α s :=
  match steps with
  | 0 => x1
  | Nat.succ steps' =>
      let n : Nat := Nat.succ steps'
      let dt : α := -((1 / n : Rat) : α)
      Fin.foldr n (fun i x =>
        let t : α := (((i.val + 1 : Nat) / n : Rat) : α)
        eulerStep (pfODERhs sch model) x t dt) x1

/--
Real-valued probability-flow Euler step as a `Dynamics.System`.

At a fixed time and step size, Euler integration is an autonomous discrete update on the current
sample, so the trajectory, fixed-point and contraction lemmas of `TorchLean.NN.Dynamics`
apply.
-/
noncomputable def pfODEEulerSystem (sch : VPLinearSchedule ℝ)
    (model : EpsModel ℝ s) (t dt : ℝ) : Dynamics.System s where
  step := fun x =>
    eulerStep (α := ℝ) (s := s)
      (pfODERhs (α := ℝ) (s := s) sch model) x t dt

/-- The probability-flow ODE system steps by one explicit Euler step of its right-hand side. -/
@[simp] theorem pfODEEulerSystem_step (sch : VPLinearSchedule ℝ)
    (model : EpsModel ℝ s) (t dt : ℝ) (x : Tensor ℝ s) :
    (pfODEEulerSystem (s := s) sch model t dt).step x =
      eulerStep (α := ℝ) (s := s)
        (pfODERhs (α := ℝ) (s := s) sch model) x t dt := by
  rfl

end TorchLean.Diffusion
