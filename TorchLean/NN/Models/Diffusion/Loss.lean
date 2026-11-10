/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Layers.Loss
public import TorchLean.NN.Models.Diffusion.ForwardProcess

/-!
# The ε-prediction diffusion loss

`epsPredLoss` is the DDPM ε-prediction objective: the mean squared error between the model's
noise prediction for `qSample sched x0 t eps` at time `t` and the noise `eps`. It lives apart from
`TorchLean.Tensor.Layers.Loss` because that library is model-agnostic.

We wrote it from eq. (14) of Ho, Jain and Abbeel (2020), the objective of their Algorithm 1, for
one draw of `t` and `ε`. Algorithm 1 draws `t` uniformly from `1..T`, and here the caller picks
it. They take a squared norm, and we average over the `n` entries of `s`, so our value is theirs
divided by `n`. That average is `torch.nn.MSELoss` (PyTorch 2.8 docs) with its default
`reduction='mean'`. No theorem or torch test covers `epsPredLoss`.

References:
- Jonathan Ho, Ajay Jain, and Pieter Abbeel (2020), "Denoising Diffusion Probabilistic Models",
  NeurIPS 2020, arXiv:2006.11239, eq. (14) and Algorithm 1.
- `torch.nn.MSELoss` (PyTorch 2.8 docs), `reduction='mean'`.
-/

@[expose] public section

namespace TorchLean.Diffusion

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
variable {T : Nat} {s : Shape}

/--
$\varepsilon$-prediction (DDPM-style) loss:
$\operatorname{MSE}(\hat\varepsilon_\theta(x_t,t),\varepsilon)$.

Inputs:
- `x0`: clean sample,
- `t`: discrete time index,
- `eps`: the noise used to build `x_t` with `qSample` (meant to be standard normal),
- `model`: ε-prediction model.

The model is evaluated at the scalar time `VPSchedule.timeOfIndex t`. The loss is the classic DDPM
training objective, up to constant weighting choices.
-/
def epsPredLoss (sched : VPSchedule α T) (model : EpsModel α s)
    (x0 : Tensor α s) (t : Fin (T + 1)) (eps : Tensor α s) : α :=
  let x_t := qSample (α := α) (T := T) (s := s) sched x0 t eps
  let tScalar : α := VPSchedule.timeOfIndex (α := α) (T := T) t
  let epsHat := model.eps x_t tScalar
  Tensor.meanSquaredError (α := α) (shape := s) epsHat eps

end TorchLean.Diffusion
