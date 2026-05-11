/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion.ForwardProcess
public import NN.Spec.Layers.Loss

/-!
# Diffusion training losses (spec layer)

This module defines the standard diffusion losses as *named wrappers* around existing spec
losses (primarily MSE).

Why keep this separate from `NN.Spec.Layers.Loss`:
- the core loss library is model-agnostic,
- diffusion introduces conventions like "ε-prediction loss", which are best surfaced with
  domain-specific names.
-/

@[expose] public section

namespace Generative.Diffusion

open Spec
open Tensor

variable {α : Type} [Context α]
variable {T : Nat} {s : Shape}

/--
ε-prediction (DDPM-style) loss: `MSE( ε̂_θ(x_t, t), ε )`.

Inputs:
- `x0`: clean sample,
- `t`: discrete time index,
- `eps`: explicit noise used to build `x_t` (intended as standard normal),
- `model`: ε-prediction model.

This is the classic DDPM training objective (up to constant weighting choices).
-/
def epsPredLoss (sched : VPSchedule α T) (model : EpsModel α s)
    (x0 : Tensor α s) (t : Fin (T + 1)) (eps : Tensor α s) : α :=
  let x_t := qSample (α := α) (T := T) (s := s) sched x0 t eps
  let tScalar : α := VPSchedule.timeOfIndex (α := α) (T := T) t
  let epsHat := model.eps x_t tScalar
  Spec.mseSpec (α := α) (s := s) epsHat eps

end Generative.Diffusion
