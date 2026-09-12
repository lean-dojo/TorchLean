/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Dynamics.System
public import NN.Spec.Generative.Diffusion.Core
public import NN.Spec.Core.Context.Real

/-!
# Probability-flow ODE (spec layer)

This file defines a small continuous-time VP schedule (linear $\beta(t)$) and the corresponding
probability-flow ODE drift field, using an $\varepsilon_\theta(x,t)$ model.

Why include this in the spec layer:
- the ODE is a deterministic dynamical system derived from the same diffusion model, and
- it is the natural interface for inference-time verification tooling (corridor certificates,
  IBP/CROWN bounds on the RHS, etc.).

We keep the implementation scalar-polymorphic (`Context α`) so it can be:
- executed with `Float` / `IEEE32Exec` / `NeuralFloat`, and
- reasoned about with `ℝ`.

References (informal pointers):
- Song et al. (2021), "Score-Based Generative Modeling through Stochastic Differential Equations".
  The VP SDE and its probability-flow ODE share the same marginals.
-/

@[expose] public section

namespace Generative.Diffusion

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Continuous-time linear VP schedule on $t\in[0,1]$:
$\beta(t)=\beta_0+t(\beta_1-\beta_0)$. -/
structure VPLinearSchedule (α : Type) [TorchLean.Storage α] [Context α] where
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
  -- Use `t*t` instead of `t^2` to avoid relying on numeral coercions into arbitrary backends.
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
-/
def pfOdeRhs (sch : VPLinearSchedule α) (model : EpsModel α s) (x : Tensor α s) (t : α) :
    Tensor α s :=
  let β : α := sch.beta t
  let σ : α := sch.sigma t
  let epsHat : Tensor α s := model.eps x t
  let drift_x : Tensor α s := Tensor.scaleSpec x ((-1 / 2) * β)
  let drift_eps : Tensor α s :=
    Tensor.scaleSpec epsHat ((1 / 2) * safeDiv β σ)
  drift_x + drift_eps

/--
One explicit Euler step for an ODE $x'=f(x,t)$:

$x_{\mathrm{next}}=x+dt\,f(x,t)$.

To integrate the probability-flow ODE *backwards* from $t=1$ to $t=0$, use a negative $dt$.
-/
def eulerStep (f : Tensor α s → α → Tensor α s) (x : Tensor α s) (t dt : α) : Tensor α s :=
  x + Tensor.scaleSpec (f x t) dt

/--
Deterministic probability-flow sampler using Euler integration on a uniform grid.

Inputs:
- `steps`: number of Euler steps (typically large, e.g. 1000),
- `x1`: initial state at $t=1$ (typically standard normal noise).

We integrate backwards in time on the grid:
$t_i=1-i/\mathtt{steps}$, with $dt=-1/\mathtt{steps}$.
-/
def pfOdeSampleEuler (sch : VPLinearSchedule α) (model : EpsModel α s)
    (steps : Nat) (x1 : Tensor α s) : Tensor α s :=
  match steps with
  | 0 => x1
  | Nat.succ steps' =>
      let n : Nat := Nat.succ steps'
      let dt : α := -((1 : α) / (n : α))
      let rec loop : Nat → Tensor α s → Tensor α s
        | 0, x => x
        | Nat.succ i, x =>
            -- The recursive counter runs `n, n-1, ..., 1`, so these are the descending
            -- times `1, (n-1)/n, ..., 1/n`.
            let t : α := ((Nat.succ i : Nat) : α) / (n : α)
            loop i (eulerStep (α := α) (s := s) (pfOdeRhs (α := α) (s := s) sch model) x t dt)
      loop n x1

/--
Real-valued probability-flow Euler step as a `DynamicalSystem`.

This is the formal hook used by trajectory/fixed-point/contraction lemmas in
`Spec.Dynamics.System`: at a fixed time and step size, Euler integration is an autonomous
discrete update on the current sample.
-/
noncomputable def pfOdeEulerSystem (sch : VPLinearSchedule SpecScalar)
    (model : EpsModel SpecScalar s) (t dt : SpecScalar) : Spec.Dynamics.DynamicalSystem s where
  step := fun x =>
    eulerStep (α := SpecScalar) (s := s)
      (pfOdeRhs (α := SpecScalar) (s := s) sch model) x t dt

/-- The probability-flow ODE system steps by one explicit Euler step of its right-hand side. -/
@[simp] theorem pfOdeEulerSystem_step (sch : VPLinearSchedule SpecScalar)
    (model : EpsModel SpecScalar s) (t dt : SpecScalar) (x : SpecTensor s) :
    (pfOdeEulerSystem (s := s) sch model t dt).step x =
      eulerStep (α := SpecScalar) (s := s)
        (pfOdeRhs (α := SpecScalar) (s := s) sch model) x t dt := by
  rfl

end Generative.Diffusion
