/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion.Core
public import NN.Spec.Generative.Diffusion.Schedule
public import NN.Spec.Core.TensorOps

/-!
# Reverse DDPM sampler (spec layer)

This file defines a standard **ε-prediction** reverse sampler step for discrete VP/DDPM
schedules.

We expose:

- `ddpmStep`: one reverse step $x_t\to x_{t-1}$ with explicit noise input $z$,
- `ddpmSample`: run all $T$ reverse steps, given a noise stream $z_0,\ldots,z_{T-1}$.

We keep everything scalar-polymorphic (`Context α`). The intended use is:

- execute with `Float`/`IEEE32Exec`/`NeuralFloat` for concrete runs, and
- reuse the same definitions with `ℝ` in proofs.

References (informal pointers):

- Ho, Jain, Abbeel (2020), DDPM, Algorithm 2 (reverse process).
-/

@[expose] public section

namespace Generative.Diffusion

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]
variable {T : Nat} {s : Shape}

/--
Reconstruct $x_0$ from $x_t$ and an already computed noise prediction.

$$
x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\varepsilon}
          {\sqrt{\bar\alpha_t}}.
$$

`safeDiv` adds the context's epsilon to the denominator, as in `x0Pred`. Passing the prediction
explicitly lets DDIM use the same tensor for this reconstruction and the direction toward the
previous sample.
-/
def x0PredFromEps (sched : VPSchedule α T) (x_t epsHat : Tensor α s)
    (t : Fin (T + 1)) : Tensor α s :=
  let αbar : α := sched.alphaBar t
  let c1 : α := sqrtNonneg (1 - αbar)
  let c0 : α := sqrtNonneg αbar
  let num := x_t - Tensor.scaleSpec epsHat c1
  Tensor.scaleSpec num (safeDiv 1 c0)

/--
Predict $x_0$ by evaluating the denoiser at time `t / T` and applying `x0PredFromEps`.

Call `x0PredFromEps` directly when another part of the sampler already needs the same denoiser
output. Both entry points use the same coefficient arithmetic and denominator protection.
-/
def x0Pred (sched : VPSchedule α T) (model : EpsModel α s) (x_t : Tensor α s)
    (t : Fin (T + 1)) : Tensor α s :=
  let tScalar : α := VPSchedule.timeOfIndex (α := α) (T := T) t
  let epsHat : Tensor α s := model.eps x_t tScalar
  x0PredFromEps sched x_t epsHat t

/--
One reverse DDPM step $x_t\to x_{t-1}$ with explicit noise $z$ (intended as
$\mathcal{N}(0,I)$).

We index reverse steps by `k : Fin T` corresponding to the transition $t=k+1\to k$.

Implementation details:
- time embedding passed to the model is $t/T$ (see `VPSchedule.timeOfIndex`).
- we use epsilon-protected scalar division in the coefficient formulas to stay total.
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
    Tensor.scaleSpec (x_t - Tensor.scaleSpec epsHat coeff) inv_sqrt_α_t

  -- Variance for the reverse step: β̃_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t.
  let β_tilde : α := safeDiv (1 - αbar_prev) (1 - αbar_t) * β_t
  let σ_t : α := sqrtNonneg β_tilde
  mean + Tensor.scaleSpec z σ_t

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
  (List.finRange T).foldr
    (fun k x => ddpmStep (α := α) (T := T) (s := s) sched model k x (noise k)) x_T

end Generative.Diffusion
