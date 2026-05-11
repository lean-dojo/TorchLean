/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion.Schedule

/-!
# Forward noising process (spec layer)

This module defines the standard VP/DDPM forward noising transformation:

`x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε`

where `ε` is intended to be standard normal noise (in runtime usage), but at the spec level is
treated as an explicit input tensor.
-/

@[expose] public section

namespace Generative.Diffusion

open Spec
open Tensor

variable {α : Type} [Context α]
variable {T : Nat} {s : Shape}

/--
Forward-process sampling `q(x_t | x_0)` for a discrete VP schedule.

Inputs:
- `x0`: clean data sample,
- `t`: discrete time index in `0..T`,
- `eps`: explicit noise tensor (intended as `N(0,I)`).

Output:
- the noisy sample `x_t`.

This function is **pure** and total; any probabilistic interpretation is handled in the
`MLTheory` layer (mathlib) or at runtime by sampling `eps`.
-/
def qSample (sched : VPSchedule α T) (x0 : Tensor α s) (t : Fin (T + 1)) (eps : Tensor α s) :
    Tensor α s :=
  let αbar : α := sched.alphaBar t
  let c0 : α := sqrtNonneg αbar
  let c1 : α := sqrtNonneg (1 - αbar)
  Tensor.scaleSpec x0 c0 + Tensor.scaleSpec eps c1

end Generative.Diffusion
