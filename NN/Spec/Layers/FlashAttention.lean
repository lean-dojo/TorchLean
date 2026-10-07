/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Attention

/-!
# FlashAttention Semantic Contract

FlashAttention is an IO-aware implementation strategy for scaled dot-product attention: it tiles
the attention computation and maintains online softmax summaries so the full `n × n` attention
matrix does not need to be materialized. TorchLean separates the semantic contract from execution:

- this file gives the proof layer semantic contract for a fused FlashAttention operator;
- the current CUDA runtime composes attention in Lean over LibTorch numerical primitives
  (`NN/Runtime/Autograd/Engine/LibTorch/Ops/Attention.lean`), retaining the capsule identity
  `libtorch.direct_attention`; it materializes full score matrices and saves probabilities;
- the numerical FFI boundary is documented separately because Lean does not verify LibTorch or
  CUDA code. This runtime does not select a fused FlashAttention implementation.

The fused operation has the same denotation as standard masked scaled dot-product
attention over the spec scalar. Different tile sizes are runtime scheduling choices, not semantic
choices.

## Refinement target

The forward and backward contracts reuse standard attention directly. Their equality theorems
let a downstream refinement certificate use either name without duplicating the attention
calculation. Tile sizes remain implementation metadata; this file defines no online softmax
recurrence, tile loops, or memory-traffic model.

The current LibTorch composition is tested operationally. Its foreign numerical calls remain
outside Lean's kernel; a fused implementation needs its own refinement argument.

References:
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, "FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness", arXiv:2205.14135.
- Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning",
  arXiv:2307.08691.
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao,
  "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision",
  arXiv:2407.08608.
-/

@[expose] public section

open TorchLean

namespace Spec

variable {α : Type} [TorchLean.Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Runtime tiling metadata for a FlashAttention-style fused implementation.

The spec-level denotation below intentionally ignores these fields: they describe how an
implementation schedules work, not what mathematical function the operator computes.
-/
structure FlashAttentionConfig where
  /-- Query block size used by a tiled implementation. `0` means "backend default". -/
  blockQ : Nat
  /-- Key/value block size used by a tiled implementation. `0` means "backend default". -/
  blockK : Nat
deriving DecidableEq, Repr

namespace FlashAttentionConfig

/-- Backend-default tiling. -/
def default : FlashAttentionConfig :=
  { blockQ := 0, blockK := 0 }

end FlashAttentionConfig

/-- Semantic FlashAttention forward operator.

Tile sizes do not change the result. A fused implementation must refine the same masked
scaled-dot-product attention calculation as an unfused implementation.
-/
def flashAttention
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α [nQ, dModel] :=
  let _ := config
  scaledDotProductAttention (α := α) ctx

/-- Semantic FlashAttention backward/VJP operator.

This is the local derivative contract a fused backward kernel should refine. The actual CUDA kernel
may recompute attention probabilities from row statistics rather than storing the full attention
matrix, but the returned adjoints must match this spec-level VJP up to the chosen floating-point
error envelope.
-/
def flashAttentionBackward
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2)
    (dOut : Tensor α [nQ, dModel]) :
    (Tensor α [nQ, dModel] ×
     Tensor α [nK, dModel] ×
     Tensor α [nK, dModel]) :=
  let _ := config
  scaledDotProductAttentionBackward (α := α) ctx dOut

/-- Forward semantic correctness of the fused FlashAttention spec. -/
@[simp] theorem flashAttention_eq_scaledDotProductAttention
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    flashAttention (α := α) config ctx = scaledDotProductAttention (α := α) ctx := rfl

/-- Backward/VJP semantic correctness of the fused FlashAttention spec. -/
@[simp] theorem flashAttentionBackward_eq_scaledDotProductAttentionBackward
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2)
    (dOut : Tensor α [nQ, dModel]) :
    flashAttentionBackward (α := α) config ctx dOut =
      scaledDotProductAttentionBackward (α := α) ctx dOut := rfl

end Spec
