/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.TensorOps

/-!
# Diffusion core (spec layer)

This module defines the common vocabulary used by TorchLean's diffusion / flow specs:

- `EpsModel`: an `ε_θ(x,t)` denoiser interface (noise prediction), and
- a couple of **total** scalar helpers (`sqrtNonneg`, `safeDiv`) that keep specs robust across
  scalar backends.

Design notes:

- `EpsModel` is deliberately backbone-independent: it is a pure function of the current state `x` and a
  scalar time parameter `t` (typically normalized into `[0,1]`).
- Schedules and samplers remain separate modules so we can reuse the same denoiser with:
  - discrete DDPM / DDIM-style samplers, and
  - continuous-time probability-flow ODE samplers.

References (informal pointers):

- Ho, Jain, Abbeel (2020), "Denoising Diffusion Probabilistic Models" (DDPM).
- Song et al. (2021), "Score-Based Generative Modeling through Stochastic Differential Equations"
  (continuous-time VP SDE and probability-flow ODE).
-/

@[expose] public section

namespace Generative.Diffusion

open Spec
open Tensor

variable {α : Type} [Context α]

/-- A safe square root used by diffusion schedules and samplers: `sqrt(max x 0)`.

Why this helper exists:
- Some backends (intervals, IEEE models, etc.) prefer total semantics.
- In diffusion schedules, the quantities under a square root are mathematically nonnegative
  (e.g. `ᾱ(t)` and `1-ᾱ(t)`), but numeric backends can still produce small negative values.
-/
def sqrtNonneg (x : α) : α :=
  MathFunctions.sqrt (Max.max x 0)

/-- Safe scalar division with epsilon protection: `x / (y + ε)`.

This is primarily used to avoid `1/0` in edge cases like `t = 0` or degenerate schedules.
-/
def safeDiv (x y : α) : α :=
  x / (y + Numbers.epsilon)

/-- Noise-prediction model interface: `ε_θ(x,t)`.

The intended interpretation is "predict the noise used to construct the noisy sample `x` at time
`t`".

Notes:
- `t` is a scalar, not a discrete index. Discrete samplers can provide `t := (k/T)` or any other
  conventional embedding; continuous samplers can pass true continuous time.
- This interface does not bake in class-conditioning or text-conditioning; those can be handled by
  closing over extra context in the `eps` function, or by defining a richer model record in user
  code.
-/
structure EpsModel (α : Type) (s : Shape) [Context α] where
  /-- Predict `ε` from a noisy sample `x` at scalar time `t`. -/
  eps : Tensor α s → α → Tensor α s

end Generative.Diffusion
