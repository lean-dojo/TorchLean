/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Dynamics.System
public import NN.Spec.Generative.Diffusion.Core

/-!
# Probability-flow ODE (spec layer)

This file defines a small continuous-time VP schedule (linear `β(t)`) and the corresponding
probability-flow ODE drift field, using an `ε_θ(x,t)` model.

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

open Spec
open Tensor

variable {α : Type} [Context α]

/-- Continuous-time linear VP schedule on `t ∈ [0,1]`: `β(t) = β0 + t(β1-β0)`. -/
structure VPLinearSchedule (α : Type) [Context α] where
  /-- `β(0)`. -/
  beta0 : α
  /-- `β(1)`. -/
  beta1 : α

namespace VPLinearSchedule

/-- Linear interpolation `β(t)` on `t ∈ [0,1]`. -/
def beta (sch : VPLinearSchedule α) (t : α) : α :=
  sch.beta0 + t * (sch.beta1 - sch.beta0)

/--
Closed-form `ᾱ(t)` for the VP SDE with linear `β(t)`:

`ᾱ(t) = exp(-∫₀ᵗ β(s) ds) = exp(-(β0 t + 0.5 (β1-β0) t^2))`.
-/
def alphaBar (sch : VPLinearSchedule α) (t : α) : α :=
  let half : α := Numbers.pointfive
  -- Use `t*t` instead of `t^2` to avoid relying on numeral coercions into arbitrary backends.
  let intBeta : α := sch.beta0 * t + half * (sch.beta1 - sch.beta0) * (t * t)
  MathFunctions.exp (-intBeta)

/-- `σ(t) = sqrt(1-ᾱ(t))` (clamped to stay total). -/
def sigma (sch : VPLinearSchedule α) (t : α) : α :=
  sqrtNonneg (1 - sch.alphaBar t)

end VPLinearSchedule

variable {s : Shape}

/--
Probability-flow ODE drift for a VP schedule, expressed via an `ε_θ(x,t)` model.

For VP SDE:

`dx = -0.5 β(t) x dt + sqrt(β(t)) dW`

The probability-flow ODE is:

`dx = (-0.5 β(t) x - β(t) score(x,t)) dt`.

Using the ε-parameterization, an approximate score is `score ≈ -(1/σ(t)) ε̂`, so:

`dx = (-0.5 β(t) x + (β(t)/σ(t)) ε̂(x,t)) dt`.
-/
def pfOdeRhs (sch : VPLinearSchedule α) (model : EpsModel α s) (x : Tensor α s) (t : α) :
    Tensor α s :=
  let β : α := sch.beta t
  let σ : α := sch.sigma t
  let epsHat : Tensor α s := model.eps x t
  let drift_x : Tensor α s := Tensor.scaleSpec x (Numbers.neg_point_five * β)
  let drift_eps : Tensor α s := Tensor.scaleSpec epsHat (safeDiv β σ)
  drift_x + drift_eps

/--
One explicit Euler step for an ODE `x' = f(x,t)`:

`xNext = x + dt * f(x,t)`.

To integrate the probability-flow ODE *backwards* from `t=1` to `t=0`, use a negative `dt`.
-/
def eulerStep (f : Tensor α s → α → Tensor α s) (x : Tensor α s) (t dt : α) : Tensor α s :=
  x + Tensor.scaleSpec (f x t) dt

/--
Deterministic probability-flow sampler using Euler integration on a uniform grid.

Inputs:
- `steps`: number of Euler steps (typically large, e.g. 1000),
- `x1`: initial state at `t = 1` (typically standard normal noise).

We integrate backwards in time on the grid:
`t_i = 1 - i/steps`, with `dt = -1/steps`.
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
            let t : α := (1 : α) - ((i : α) / (n : α))
            loop i (eulerStep (α := α) (s := s) (pfOdeRhs (α := α) (s := s) sch model) x t dt)
      loop n x1

/--
Real-valued probability-flow Euler step as a `DynamicalSystem`.

This is the formal hook used by trajectory/fixed-point/contraction lemmas in
`NN.Spec.Dynamics.System`: at a fixed time and step size, Euler integration is an autonomous
discrete update on the current sample.
-/
noncomputable def pfOdeEulerSystem (sch : VPLinearSchedule SpecScalar) (model : EpsModel SpecScalar s)
    (t dt : SpecScalar) : NN.Spec.Dynamics.DynamicalSystem s where
  step := fun x =>
    eulerStep (α := SpecScalar) (s := s)
      (pfOdeRhs (α := SpecScalar) (s := s) sch model) x t dt

@[simp] theorem pfOdeEulerSystem_step (sch : VPLinearSchedule SpecScalar)
    (model : EpsModel SpecScalar s) (t dt : SpecScalar) (x : SpecTensor s) :
    (pfOdeEulerSystem (s := s) sch model t dt).step x =
      eulerStep (α := SpecScalar) (s := s)
        (pfOdeRhs (α := SpecScalar) (s := s) sch model) x t dt := by
  rfl

end Generative.Diffusion
