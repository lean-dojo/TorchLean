/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion.ReverseDDPM
public import NN.Spec.Dynamics.System

import Mathlib.Data.List.FinRange

/-!
# Reverse DDIM sampler (spec layer)

DDIM (Denoising Diffusion Implicit Models) can be viewed as a **deterministic** sampler that
reuses the same denoiser `ε_θ(x,t)` but removes per-step noise.

This file provides the `η = 0` variant (fully deterministic), which is often used as a simple
"flow-like" sampler derived from the same diffusion model.

Reference (informal pointer):
- Song, Meng, Ermon (2021), "Denoising Diffusion Implicit Models" (DDIM).
-/

@[expose] public section

namespace Generative.Diffusion

open Spec
open Tensor

variable {α : Type} [Context α]
variable {T : Nat} {s : Shape}

/--
One deterministic DDIM step `x_t -> x_{t-1}` (η = 0).

We reuse the same `x0Pred` reconstruction as DDPM and then recompose `x_{t-1}` using the
forward-process coefficients at time `t-1`:

`x_{t-1} = sqrt(ᾱ_{t-1}) * x0_pred + sqrt(1-ᾱ_{t-1}) * ε̂`.
-/
def ddimStep (sched : VPSchedule α T) (model : EpsModel α s)
    (k : Fin T) (x_t : Tensor α s) : Tensor α s :=
  let t : Fin (T + 1) := k.succ
  let tPrev : Fin (T + 1) := Fin.castSucc k
  let x0Hat : Tensor α s := x0Pred (α := α) (T := T) (s := s) sched model x_t t

  let tScalar : α := VPSchedule.timeOfIndex (α := α) (T := T) t
  let epsHat : Tensor α s := model.eps x_t tScalar

  let αbar_prev : α := sched.alphaBar tPrev
  let c0 : α := sqrtNonneg αbar_prev
  let c1 : α := sqrtNonneg (1 - αbar_prev)
  Tensor.scaleSpec x0Hat c0 + Tensor.scaleSpec epsHat c1

/-- Run the full deterministic DDIM sampler for `T` steps (η = 0). -/
def ddimSample (sched : VPSchedule α T) (model : EpsModel α s) (x_T : Tensor α s) : Tensor α s :=
  (List.finRange T).foldr (fun k x => ddimStep (α := α) (T := T) (s := s) sched model k x) x_T

/--
Real-valued DDIM transition as a `DynamicalSystem`.

`DynamicalSystem` is fixed to `SpecScalar = ℝ`, so this adapter gives DDIM samplers the same
trajectory/fixed-point API used by SSMs and other discrete systems.
-/
noncomputable def ddimStepSystem (sched : VPSchedule SpecScalar T) (model : EpsModel SpecScalar s)
    (k : Fin T) : NN.Spec.Dynamics.DynamicalSystem s where
  step := fun x => ddimStep (α := SpecScalar) (T := T) (s := s) sched model k x

@[simp] theorem ddimStepSystem_step (sched : VPSchedule SpecScalar T)
    (model : EpsModel SpecScalar s) (k : Fin T) (x : SpecTensor s) :
    (ddimStepSystem (T := T) (s := s) sched model k).step x =
      ddimStep (α := SpecScalar) (T := T) (s := s) sched model k x := by
  rfl

end Generative.Diffusion
