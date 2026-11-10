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
# Forward noising process (spec layer)

The standard VP/DDPM forward noising transformation:

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon
$$

In runtime use $\varepsilon$ is meant to be standard normal noise, and the spec takes it as an
input tensor.

From eq. (4) of Ho, Jain and Abbeel (2020), written by hand in the reparameterized form of their
Section 3.2. No theorem relates `qSample` to the Gaussian law `forwardNoising`, and no torch test
runs it.

References:
- Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020), "Denoising Diffusion Probabilistic Models",
  NeurIPS 2020, arXiv:2006.11239, eq. (4) and Section 3.2.
-/

@[expose] public section

namespace TorchLean.Diffusion

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
variable {T : Nat} {s : Shape}

/--
Forward-process sampling at one cumulative coefficient $\bar\alpha$:
$\sqrt{\max(\bar\alpha,0)}\,x_0+\sqrt{\max(1-\bar\alpha,0)}\,\varepsilon$.

Runnable code passes a coefficient read from its stored table, with no schedule to recompute it
from.
-/
def qSampleAt (alphaBar : α) (x0 eps : Tensor α s) : Tensor α s :=
  Tensor.scale x0 (sqrtNonneg alphaBar) + Tensor.scale eps (sqrtNonneg (1 - alphaBar))

/--
Forward-process sampling $q(x_t\mid x_0)$ for a discrete VP schedule.

Inputs:
- `x0`: clean data sample,
- `t`: discrete time index in `0..T`,
- `eps`: noise tensor (meant as a draw from `N(0,I)`).

The output is the noisy sample $x_t$, which is `qSampleAt` at the schedule's $\bar\alpha_t$. The
function is **pure** and total, and the runtime samples `eps`. Its probabilistic reading is the
Gaussian law `forwardNoising` of `TorchLean.NN.Models.Diffusion.ForwardProcess.Gaussian`, which
no theorem relates to `qSample`.
-/
def qSample (sched : VPSchedule α T) (x0 : Tensor α s) (t : Fin (T + 1)) (eps : Tensor α s) :
    Tensor α s :=
  qSampleAt (sched.alphaBar t) x0 eps

end TorchLean.Diffusion
