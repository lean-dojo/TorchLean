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

We therefore expose two simple, deterministic variants:

- `dropout_inference_spec p x = (1 - p) * x`
  This is a deterministic "shrink activations" surrogate. It corresponds to the *expected value*
  of **non-inverted** dropout training (`y = mask * x` with `E[mask] = keep`).

- `dropout_masked_spec p mask x`
  A fully deterministic "training-style" dropout that takes the mask explicitly. We use safe
  scaling by `max(keep, ε)` so it is always defined even if `p ≈ 1`.

How this differs from PyTorch:

- `torch.nn.Dropout(p)` uses **inverted dropout** during training: `y = mask * x / (1 - p)`,
  and becomes identity during evaluation (`y = x`).
- The spec layer here avoids randomness. If you want something close to PyTorch *training*
  semantics,
  use `dropout_masked_spec` and pass the mask explicitly. If you want a cheap deterministic effect,
  use `dropout_inference_spec`.

Gradients:

- We treat `p` and `mask` as non-differentiable inputs. The backward specs only return the gradient
  with respect to `x`.
-/

@[expose] public section


namespace Spec

open Tensor
open Numbers

variable {α : Type} [Context α]

/-- Deterministic "dropout-like" scaling: `y = keep * x` with `keep = 1 - p`.

This is *not* PyTorch's `eval()` behavior for `nn.Dropout` (which is the identity under inverted
dropout). We keep this around because it is a simple deterministic knob that many specs use. -/
def dropoutInferenceSpec {s : Shape} (p : α) (x : Tensor α s) : Tensor α s :=
  scaleSpec x ((1 : α) - p)

/-- Backward/VJP for `dropout_inference_spec` with respect to `x`: `dL/dx = keep * dL/dy`. -/
def dropoutInferenceBackwardSpec {s : Shape} (p : α) (grad_output : Tensor α s) : Tensor α s :=
  scaleSpec grad_output ((1 : α) - p)

/-- Deterministic training-style dropout with an explicit mask.

If `mask[i] = true`, keep element `x[i]`, otherwise drop it to `0`.
We use inverted-dropout scaling `x / keepSafe` with `keepSafe = max(1 - p, ε)`. -/
def dropoutMaskedSpec {s : Shape} (p : α) (mask : Tensor Bool s) (x : Tensor α s) : Tensor α s :=
  let keep : α := (1 : α) - p
  let keepSafe : α := Max.max keep Numbers.epsilon
  map2Spec (fun xi mi => if mi then xi / keepSafe else 0) x mask

/-- Backward/VJP for `dropout_masked_spec` with respect to `x`.

This mirrors the forward: gradients are masked and (in the kept positions) scaled by `1/keepSafe`.
  -/
def dropoutMaskedBackwardSpec {s : Shape} (p : α) (mask : Tensor Bool s) (grad_output : Tensor α
  s) : Tensor α s :=
  let keep : α := (1 : α) - p
  let keepSafe : α := Max.max keep Numbers.epsilon
  map2Spec (fun gi mi => if mi then gi / keepSafe else 0) grad_output mask

end Spec
