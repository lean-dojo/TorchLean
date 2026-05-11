/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Attention
public import NN.Spec.Module.SpecModule

/-!
# Attention module wrappers

This file wraps a few attention blocks as `NNModuleSpec`s so we can:

- compose them with `SpecChain` (shape-safe pipelines), and
- attach simple export/pretty-print metadata for demos.

The wrapper below builds a self-attention context with `Q=K=V=x` and no mask, which matches the
common "encoder block" usage. More specialized variants (cross-attention, causal masks, etc.) are
defined at the layer-spec level in `NN/Spec/Layers/Attention.lean`.

In PyTorch terms, the core computation is scaled dot-product self-attention:
`softmax(QK^T / sqrt(d)) V`, and newer PyTorch exposes it as
`torch.nn.functional.scaled_dot_product_attention`.

This wrapper stays intentionally narrow: it is self-attention only (`Q=K=V=x`) with no causal mask.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec
open Shape

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Self-attention block (`Q=K=V=x`, no mask) as an `NNModuleSpec`. -/
def ScaledDotProductAttentionModuleSpec
  (n dModel : Nat) (h1 : n ≠ 0) :
  NNModuleSpec α (.dim n (.dim dModel .scalar)) (.dim n (.dim dModel .scalar)) :=
{ forward := fun x =>
    let ctx : AttentionContext α n n dModel h1 h1 :=
      { Q := x
        K := x
        V := x
        bc_sum_to_target := buildBcProof n
        mask := none }
    scaledDotProductAttention (α := α) ctx,
  kind := "ScaledDotProductSelfAttention",
  export_func := {
    toPyTorch := s!"ScaledDotProductSelfAttention(d_model={dModel})",
    dimensions := (n, dModel)
  } }

end Spec
