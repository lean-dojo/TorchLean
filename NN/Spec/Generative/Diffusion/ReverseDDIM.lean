/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Dynamics.System
public import NN.Spec.Generative.Diffusion.ReverseDDPM
public import NN.Spec.Core.Context.Real

/-!
# Reverse DDIM sampler (spec layer)

DDIM (Denoising Diffusion Implicit Models) can be viewed as a **deterministic** sampler that
reuses the same denoiser $\varepsilon_\theta(x,t)$ but removes per-step noise.

This file provides the $\eta=0$ variant (fully deterministic), which is often used as a simple
"flow-like" sampler derived from the same diffusion model.

Reference (informal pointer):
- Song, Meng, Ermon (2021), "Denoising Diffusion Implicit Models" (DDIM).
-/

@[expose] public section

namespace Generative.Diffusion

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]
variable {T : Nat} {s : Shape}

/--
One deterministic DDIM step $x_t\to x_{t-1}$ ($\eta=0$).

Evaluate the denoiser once, reconstruct $x_0$ with `x0PredFromEps`, and reuse that prediction in
the direction term. The coefficients are those of the forward process at time $t-1$:

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\widehat{x}_0
  +\sqrt{1-\bar\alpha_{t-1}}\,\hat\varepsilon.
$$
-/
def ddimStep (sched : VPSchedule α T) (model : EpsModel α s)
    (k : Fin T) (x_t : Tensor α s) : Tensor α s :=
  let t : Fin (T + 1) := k.succ
  let tPrev : Fin (T + 1) := Fin.castSucc k
  let tScalar : α := VPSchedule.timeOfIndex (α := α) (T := T) t
  let epsHat : Tensor α s := model.eps x_t tScalar
  let x0Hat : Tensor α s := x0PredFromEps sched x_t epsHat t

  let αbar_prev : α := sched.alphaBar tPrev
  let c0 : α := sqrtNonneg αbar_prev
  let c1 : α := sqrtNonneg (1 - αbar_prev)
  Tensor.scaleSpec x0Hat c0 + Tensor.scaleSpec epsHat c1

/--
Sharing the denoiser output preserves the reconstruction-and-direction formula.

This equality uses the same tensor operations in the same order, so it holds for every scalar
context, including floating-point contexts. It does not require field identities or a claim about
the compiler's treatment of repeated pure function calls.
-/
theorem ddimStep_eq_x0Pred (sched : VPSchedule α T) (model : EpsModel α s)
    (k : Fin T) (x_t : Tensor α s) :
    ddimStep sched model k x_t =
      Tensor.scaleSpec (x0Pred sched model x_t k.succ)
          (sqrtNonneg (sched.alphaBar (Fin.castSucc k))) +
        Tensor.scaleSpec (model.eps x_t (VPSchedule.timeOfIndex (T := T) k.succ))
          (sqrtNonneg (1 - sched.alphaBar (Fin.castSucc k))) := by
  rfl

/-- Run the full deterministic DDIM sampler for $T$ steps ($\eta=0$). -/
def ddimSample (sched : VPSchedule α T) (model : EpsModel α s) (x_T : Tensor α s) : Tensor α s :=
  (List.finRange T).foldr (fun k x => ddimStep (α := α) (T := T) (s := s) sched model k x) x_T

/--
Real-valued DDIM transition as a `DynamicalSystem`.

`DynamicalSystem` is fixed to `SpecScalar = ℝ`, so this adapter gives DDIM samplers the same
trajectory/fixed-point API used by SSMs and other discrete systems.
-/
noncomputable def ddimStepSystem (sched : VPSchedule SpecScalar T) (model : EpsModel SpecScalar s)
    (k : Fin T) : Spec.Dynamics.DynamicalSystem s where
  step := fun x => ddimStep (α := SpecScalar) (T := T) (s := s) sched model k x

/-- The DDIM system steps by one `ddimStep`, so system-level lemmas transfer to the sampler. -/
@[simp] theorem ddimStepSystem_step (sched : VPSchedule SpecScalar T)
    (model : EpsModel SpecScalar s) (k : Fin T) (x : SpecTensor s) :
    (ddimStepSystem (T := T) (s := s) sched model k).step x =
      ddimStep (α := SpecScalar) (T := T) (s := s) sched model k x := by
  rfl

end Generative.Diffusion
