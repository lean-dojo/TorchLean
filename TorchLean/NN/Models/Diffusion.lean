/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Datatype

/-!
# Diffusion core (spec layer)

The vocabulary shared by TorchLean's diffusion and flow specs:

- `EpsModel`, an $\varepsilon_\theta(x,t)$ denoiser interface (noise prediction), and
- the scalar helpers `sqrtNonneg` (square root of the argument clamped at zero) and `safeDiv`
  (division by an epsilon-shifted denominator). Both are total, and on floating-point backends
  they can return nonfinite values.

`EpsModel` is independent of any backbone: it is a pure function of the current state `x` and a
scalar time parameter `t` (typically normalized into `[0,1]`). Schedules and samplers live in
separate modules, so one denoiser serves both discrete DDPM / DDIM-style samplers and
continuous-time probability-flow ODE samplers.

`EpsModel` is the `ε_θ(x_t, t)` of Ho, Jain and Abbeel, their Section 3.2. The two helpers come
from no paper. They're our guards, and on `Float`, `Float32` and `ℝ` the shift
`Context.defaultEpsilon` is `1e-6`, so even `safeDiv 1 1` is `1 / 1.000001 ≈ 0.999999`.

References:

- Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020), "Denoising Diffusion Probabilistic Models",
  NeurIPS 2020, arXiv:2006.11239, Section 3.2.
- Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben
  Poole (2021), "Score-Based Generative Modeling through Stochastic Differential Equations",
  ICLR 2021, arXiv:2011.13456 (continuous-time VP SDE and probability-flow ODE).
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Diffusion

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- The clamped square root $\sqrt{\max(x,0)}$ used by diffusion schedules and samplers.

The quantities under a schedule's square roots, such as $\bar\alpha(t)$ and $1-\bar\alpha(t)$, are
nonnegative in exact arithmetic, but numeric backends can produce small negative values, where an
IEEE square root would return NaN.
-/
def sqrtNonneg {α : Type} [Zero α] [Max α] [MathFunctions α] (x : α) : α :=
  MathFunctions.sqrt (Max.max x 0)

/-- Epsilon-shifted scalar division: $x/(y+\varepsilon)$.

The shift regularizes small denominators in schedules and samplers. `Context` imposes no
nonzero-denominator law, and the shifted denominator is zero at $y=-\varepsilon$.
-/
def safeDiv (x y : α) : α :=
  x / (y + Context.defaultEpsilon)

/-- Noise-prediction model interface: $\varepsilon_\theta(x,t)$ predicts the noise used to
construct the noisy sample `x` at time `t`.

The time `t` is a scalar. Discrete samplers can pass `t := (k/T)` or another conventional
embedding of the step index, and continuous samplers pass the continuous time. Class or text
conditioning is left to the caller, who can close over extra context in the `eps` function or
define a richer model record.
-/
structure EpsModel (α : Type) (s : Shape) [Storage α] where
  /-- Predict $\varepsilon$ from a noisy sample $x$ at scalar time $t$. -/
  eps : Tensor α s → α → Tensor α s

end TorchLean.Diffusion
