/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorGrad

/-!
# GradientUtils

Gradient utilities for TorchLean runtime training.

These utilities are defined in terms of the canonical TensorGrad operations where possible.
The spec layer already contains the scalar-polymorphic definitions of clipping and simple
reductions, keeping runtime optimizer behavior aligned with the spec definitions.

This runtime file provides:
- short names that read like optimizer code,
- a place to attach PyTorch analogies and citations.

So this file is intentionally a thin runtime vocabulary layer, not a second implementation of
gradient clipping. If the math changes, it should change in the spec layer first.

PyTorch analogies:
- global-norm clipping: `torch.nn.utils.clip_grad_norm_`
- value clipping: `torch.clamp`
- percentile/quantile-based clipping (conceptual): `torch.quantile(abs(g), q)` then clamp

References:
- PyTorch `clip_grad_norm_`:
  https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- PyTorch `clamp`: https://pytorch.org/docs/stable/generated/torch.clamp.html
- PyTorch `quantile`: https://pytorch.org/docs/stable/generated/torch.quantile.html
- Pascanu–Mikolov–Bengio (2013), gradient clipping for RNN training stability:
  https://arxiv.org/abs/1211.5063
-/

@[expose] public section


namespace Optim

open Spec
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-! ## Norms -/

/-- Squared L2 norm: `‖g‖₂² = ∑ᵢ gᵢ²`. -/
def l2NormSq {s : Shape} (g : Tensor α s) : α :=
  sumSpec (squareSpec g)

/-- L2 norm: `‖g‖₂ = sqrt(∑ᵢ gᵢ²)`. -/
def l2Norm {s : Shape} (g : Tensor α s) : α :=
  MathFunctions.sqrt (l2NormSq (α := α) g)

/-! ## Clipping -/

/--
Global-norm clipping: if `‖g‖₂ > maxNorm`, rescale `g` so that `‖g‖₂ = maxNorm`.

Mathematically:
`g ← g * (maxNorm / ‖g‖₂)` when `‖g‖₂` exceeds the threshold.
-/
def clipByNorm {s : Shape} (g : Tensor α s) (maxNorm : α) : Tensor α s :=
  Spec.clipGradientsSpec g maxNorm

/-- Elementwise value clipping: `gᵢ ← clamp(gᵢ, minVal, maxVal)`. -/
def clipByValue {s : Shape} (g : Tensor α s) (minVal maxVal : α) : Tensor α s :=
  Spec.clipByValueSpec g minVal maxVal

/--
Percentile-driven clipping: compute a bound from `abs(g)` and clamp to `[-b, b]`.

This is only executable when `<` on `α` is decidable (e.g. `Float`, `IEEE32Exec`).
-/
def clipByPercentile {s : Shape} (g : Tensor α s) (pct : Nat) [DecidableLT α] : Tensor α s :=
  Spec.clipByPercentileSpec g pct

end Optim
