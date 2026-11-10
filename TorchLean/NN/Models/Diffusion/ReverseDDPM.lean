/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Diffusion
public import TorchLean.NN.Models.Diffusion.Schedule
public import TorchLean.Tensor.Arithmetic

/-!
# Reverse DDPM sampler (spec layer)

A standard **ε-prediction** reverse sampler for discrete VP/DDPM schedules:

- `ddpmStep` takes one reverse step $x_t\to x_{t-1}$ with a given noise input $z$, and
- `ddpmSample` runs all $T$ reverse steps, given a noise stream $z_0,\ldots,z_{T-1}$.

Everything is scalar-polymorphic (`Context α`). The definitions execute with `Float`, `Float32` or
the configured binary32 type `ExecFloat.Binary 8 23`, run in CPU software at a chosen precision
with `FloatLib.Floats.ExecFloat.Binary`, and serve proofs over `ℝ` or the noncomputable
`FloatLib.Floats.Formats.Flocq.NF`.

`ddpmStep` follows Algorithm 2 of Ho, Jain and Abbeel (2020), written by hand. The mean is their
eq. (11) with `1 - α_t = β_t`, and the noise scale is `σ_t = √β̃_t` with `β̃_t` from eq. (7).
Section 3.2 reports similar results for `σ_t² = β_t` and `σ_t² = β̃_t`, and we took the second.
Algorithm 2 sets `z = 0` on the last step. We don't branch: at `k = 0` the factor `1 - ᾱ_0` is
`0`, so `σ_t = 0` and a finite `z` drops out. Their network sees the step `t` and ours sees
`t / T`. With their `β_1 = 1e-4` the shifted division turns the first noise coefficient
`β_1 / √(1 - ᾱ_1) = 0.01` into `1e-4 / 0.010001 ≈ 0.009999`. `ddpmSample_zero_steps` proves the
`T = 0` case, and no other theorem or torch test covers the sampler.

References:

- Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020), "Denoising Diffusion Probabilistic Models",
  NeurIPS 2020, arXiv:2006.11239, Algorithm 2, eqs. (7) and (11), Section 3.2.
-/

@[expose] public section

namespace TorchLean.Diffusion

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
variable {T : Nat} {s : Shape}

/--
One reverse DDPM step $x_t\to x_{t-1}$ with the noise $z$ given as an argument (meant as a draw
from $\mathcal{N}(0,I)$).

The step index `k : Fin T` names the transition $t=k+1\to k$, and the model sees the time $t/T$
(`VPSchedule.timeOfIndex`). Every coefficient division adds `Context.defaultEpsilon` to its
denominator. The shift is part of this sampler's formula, and it does not make denominators
nonzero or outputs finite.
-/
def ddpmStep (sched : VPSchedule α T) (model : EpsModel α s)
    (k : Fin T) (x_t : Tensor α s) (z : Tensor α s) : Tensor α s :=
  let t : Fin (T + 1) := k.succ
  let αbar_t : α := sched.alphaBar t
  let αbar_prev : α := sched.alphaBar (Fin.castSucc k)
  let β_t : α := sched.beta k
  let α_t : α := 1 - β_t
  let sqrt_α_t : α := sqrtNonneg α_t
  let inv_sqrt_α_t : α := safeDiv 1 sqrt_α_t

  let tScalar : α := VPSchedule.timeOfIndex (α := α) (T := T) t
  let epsHat : Tensor α s := model.eps x_t tScalar

  let sqrt_one_minus_αbar_t : α := sqrtNonneg (1 - αbar_t)
  let coeff : α := safeDiv β_t sqrt_one_minus_αbar_t
  let mean : Tensor α s :=
    Tensor.scale (x_t - Tensor.scale epsHat coeff) inv_sqrt_α_t

  -- Shifted reverse variance: β̃_t = ((1-ᾱ_{t-1})/(1-ᾱ_t+ε)) * β_t.
  let β_tilde : α := safeDiv (1 - αbar_prev) (1 - αbar_t) * β_t
  let σ_t : α := sqrtNonneg β_tilde
  mean + Tensor.scale z σ_t

/--
Run the full reverse DDPM sampler for `T` steps.

Inputs:
- $x_T$: starting state (typically standard normal noise),
- `noise`: per-step noise stream `z_k` for `k = 0..T-1`.

Output:
- the terminal sample $x_0$.

Order note:
- `noise (T-1)` is used first (for the step $T\to T-1$),
- `noise 0` is used last (for the step $1\to0$).
-/
def ddpmSample (sched : VPSchedule α T) (model : EpsModel α s)
    (x_T : Tensor α s) (noise : Fin T → Tensor α s) : Tensor α s :=
  Fin.foldr T
    (fun k x => ddpmStep (α := α) (T := T) (s := s) sched model k x (noise k)) x_T

end TorchLean.Diffusion
