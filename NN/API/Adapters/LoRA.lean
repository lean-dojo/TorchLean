/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor
public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Core.TensorOps

/-!
# LoRA Adapters

LoRA is a parameter-efficient adapter for linear layers: instead of updating the full weight matrix
`W`, training learns two small matrices `A` and `B` and uses

`W_eff = W + scale * (A * B)`.

The convention in this file matches the rest of TorchLean's linear specs, where row-batch inputs
multiply a weight matrix on the right. If a base linear layer uses `W : inDim × outDim`, then:

- `A : inDim × rank` maps the input dimension into the adapter rank,
- `B : rank × outDim` maps the rank dimension back to the output dimension,
- `loraDelta A B scale : inDim × outDim` is the trainable low-rank delta.

Reference: Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models”,
https://arxiv.org/abs/2106.09685.
-/

@[expose] public section


namespace NN
namespace API
namespace Adapters

open _root_.Spec
open _root_.Spec.Tensor

namespace LoRA

/--
LoRA adapter parameters for a linear weight matrix of shape `inDim × outDim`.

The usual LoRA scaling is `alpha / rank`; TorchLean keeps the final scalar as an explicit `scale`
argument so callers can choose that convention, a schedule, or a test value.
-/
structure Params (α : Type) (inDim rank outDim : Nat) where
  /-- Down projection from the model input dimension into the adapter rank. -/
  A : Tensor α (.dim inDim (.dim rank .scalar))
  /-- Up projection from the adapter rank into the model output dimension. -/
  B : Tensor α (.dim rank (.dim outDim .scalar))

/-- The low-rank matrix `scale * (A * B)` added to a base linear weight. -/
def delta {α : Type} [Add α] [Mul α] [Zero α]
    {inDim rank outDim : Nat} (p : Params α inDim rank outDim) (scale : α) :
    Tensor α (.dim inDim (.dim outDim .scalar)) :=
  scaleSpec (matMulSpec p.A p.B) scale

/-- Apply a LoRA adapter to a base linear weight matrix. -/
def effectiveWeight {α : Type} [Add α] [Mul α] [Sub α] [Zero α]
    {inDim rank outDim : Nat}
    (base : Tensor α (.dim inDim (.dim outDim .scalar)))
    (p : Params α inDim rank outDim) (scale : α) :
    Tensor α (.dim inDim (.dim outDim .scalar)) :=
  addSpec base (delta p scale)

/-- Run a batched linear projection using the base weight plus the LoRA delta. -/
def linear {α : Type} [Add α] [Mul α] [Sub α] [Zero α]
    {batch inDim rank outDim : Nat}
    (x : Tensor α (.dim batch (.dim inDim .scalar)))
    (base : Tensor α (.dim inDim (.dim outDim .scalar)))
    (p : Params α inDim rank outDim) (scale : α) :
    Tensor α (.dim batch (.dim outDim .scalar)) :=
  matMulSpec x (effectiveWeight base p scale)

end LoRA

end Adapters
end API
end NN
