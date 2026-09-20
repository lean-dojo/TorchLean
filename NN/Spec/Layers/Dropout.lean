/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Dropout (deterministic spec)

Dropout is traditionally randomized: each element is kept with probability `keep = 1 - p`.
In this repository we often want a *deterministic* spec that still documents the intended meaning,
so downstream models can choose explicit inference-time or mask-driven dropout semantics.

We therefore expose two deterministic variants:

- `dropoutInferenceSpec p x = x`, matching evaluation mode for inverted dropout.

- `dropoutMaskedSpec p mask x`
  A fully deterministic "training-style" dropout that takes the mask explicitly. Kept entries are
  scaled by `1 / max(1 - p, ε)`; when `1 - p ≤ 0` (so `p ≥ 1`) every entry is dropped and the
  output is zero, matching `torch.nn.Dropout(1.0)`.

How this differs from PyTorch:

- `torch.nn.Dropout(p)` uses **inverted dropout** during training: `y = mask * x / (1 - p)`,
  and becomes identity during evaluation (`y = x`).
- The spec layer here avoids randomness. If you want something close to PyTorch *training*
  semantics,
  use `dropoutMaskedSpec` and pass the mask explicitly. For evaluation semantics, use
  `dropoutInferenceSpec`.

Gradients:

- We treat `p` and `mask` as non-differentiable inputs. The backward specs only return the gradient
  with respect to `x`.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Evaluation-mode dropout. The probability is retained in the signature because it belongs to
the layer configuration, but evaluation itself is the identity. -/
def dropoutInferenceSpec {s : Shape} (p : α) (x : Tensor α s) : Tensor α s :=
  let _ := p
  x

/-- Backward/VJP for evaluation-mode dropout: the cotangent is unchanged. -/
def dropoutInferenceBackwardSpec {s : Shape} (p : α) (gradOutput : Tensor α s) : Tensor α s :=
  let _ := p
  gradOutput

/-- Scale applied to kept entries by masked dropout: `1 / max(1 - p, ε)` when `1 - p > 0`, and
`0` otherwise.

The zero branch makes `p = 1` (and any `p ≥ 1`) drop every entry, as `torch.nn.Dropout(1.0)`
does; without it the clamp to `ε` would return `x / ε` for kept entries. The clamp only matters
on the open interval `1 - ε < p < 1`, where PyTorch would divide by the tiny `1 - p` instead. -/
def dropoutKeepScale (p : α) : α :=
  let keep : α := (1 : α) - p
  if keep > 0 then (1 : α) / Max.max keep Context.defaultEpsilon else 0

/-- Deterministic training-style dropout with an explicit mask.

If `mask[i] = true`, keep element `x[i]` scaled by `dropoutKeepScale p`, otherwise drop it to `0`.
For `p ≥ 1` the scale is `0`, so the output is the zero tensor regardless of the mask. -/
def dropoutMaskedSpec {s : Shape} (p : α) (mask : Tensor Bool s) (x : Tensor α s) : Tensor α s :=
  let scale : α := dropoutKeepScale p
  map2Spec (fun xi mi => if mi then xi * scale else 0) x mask

/-- Backward/VJP for `dropoutMaskedSpec` with respect to `x`.

This mirrors the forward: gradients are masked and (in the kept positions) multiplied by
`dropoutKeepScale p`, which is `0` when `p ≥ 1`. -/
def dropoutMaskedBackwardSpec {s : Shape} (p : α) (mask : Tensor Bool s) (gradOutput : Tensor α
  s) : Tensor α s :=
  let scale : α := dropoutKeepScale p
  map2Spec (fun gi mi => if mi then gi * scale else 0) gradOutput mask

end Spec
