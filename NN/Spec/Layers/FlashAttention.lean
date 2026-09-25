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
matrix does not need to be materialized. TorchLean models that idea in three layers:

- this file gives the proof layer semantic contract for a fused FlashAttention operator;
- the runtime path is the LibTorch CUDA attention bridge (`NN/Backend/Attention.lean`,
  `libtorch.direct_attention`);
- that FFI boundary is documented separately because Lean does not verify LibTorch or CUDA code.

The fused operation has the same denotation as standard masked scaled dot-product
attention over the spec scalar. Different tile sizes are runtime scheduling choices, not semantic
choices.

## What is proved here?

The theorems in this file are compact but important:

- `onlineSoftmaxTiledAttention_eq_scaledDotProductAttention` proves the named FlashAttention
  algorithmic contract has the same denotation as standard attention.
- `flashAttention_eq_scaledDotProductAttention` proves the fused forward operator is semantically
  equal to TorchLean's existing standard attention spec.
- `flashAttentionBackward_eq_scaledDotProductAttentionBackward` proves the fused VJP contract is
  semantically equal to the existing standard attention backward spec.

These are definitional-equality theorems because the proof layer contract spells out the same
mathematical stages as standard attention. The LibTorch attention bridge is tested against this
contract operationally and remains a runtime trust boundary.

## Why this is not a CUDA proof

The definitions below are the mathematical contract for FlashAttention. They do not claim to verify
LibTorch's attention kernels. Instead, they make the important theorem explicit:

`onlineSoftmaxTiledAttention config ctx = scaledDotProductAttention ctx`.

That is the theorem a compiler rewrite or fused backend relies on. A production IO-tiled CUDA kernel
can be swapped in under the same contract once it is tested/refined.

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

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Shape

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

/-!
## Algorithmic Contract

The original FlashAttention algorithm streams over blocks of keys/values and maintains a row-wise
online softmax summary. The exact CUDA schedule is an implementation detail, but the mathematical
result is the same as the closed-form stabilized softmax over the full masked score row.

TorchLean names the stages below so proofs and compiler passes can point at a real algorithmic
contract rather than only at an opaque fused primitive:

1. build scaled scores `QKᵀ / sqrt(d)`;
2. apply the boolean mask with true hard-mask semantics (blocked numerator is zero);
3. compute the same row-wise normalized weights that a correct online summary must produce;
4. multiply by values.

This is schedule-polymorphic: `config.blockQ` and `config.blockK` describe how a runtime may tile
the work, but they do not alter the denotation.
-/

/-- Unmasked attention scores `QKᵀ`. -/
def attentionScores
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α [nQ, nK] :=
  matMulSpec ctx.Q (swapAdjacentAxes ctx.K 0)

/-- Scaled attention scores before row normalization.

Masking is applied at the weight level by `onlineSoftmaxWeights`, using the same true hard-mask
semantics as `scaledDotProductAttention`.
-/
def scaledAttentionScores
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α [nQ, nK] :=
  let scale := attentionScaleDenom (α := α) dModel
  scaleSpec (attentionScores (α := α) ctx) (1 / scale)

/-- The row-wise softmax weights produced by the online softmax summary.

`Activation.softmaxSpec` uses the stabilized form `exp(x - rowMax) / Σ exp(x - rowMax)`.
This definition is the **denotation** that a FlashAttention implementation must refine. It is not a
formal model of Dao-style tile loops or SRAM/HBM traffic.
-/
def onlineSoftmaxWeights
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α [nQ, nK] :=
  -- Tile metadata is relevant to the runtime schedule, not to the exact normalized weights.
  let _blockQ := config.blockQ
  let _blockK := config.blockK
  let scores := scaledAttentionScores (α := α) ctx
  match ctx.mask with
  | none => Activation.softmaxSpec (α := α) 1 scores
  | some m => hardMaskedSoftmaxSpec scores m

/-- Proof-facing FlashAttention forward algorithm.

This is the mathematical result of the online/tiled schedule: row-wise softmax weights multiplied by
`V`. Runtime kernels may avoid storing the full weights, but they must refine this value.
-/
def onlineSoftmaxTiledAttention
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α [nQ, dModel] :=
  matMulSpec (onlineSoftmaxWeights (α := α) config ctx) ctx.V

/--
The proof layer FlashAttention denotation equals standard SDPA.

This theorem is useful for graph-rewrite semantics, but should not be read as a verification of a
particular CUDA implementation.
-/
@[simp] theorem onlineSoftmaxTiledAttention_eq_scaledDotProductAttention
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    onlineSoftmaxTiledAttention (α := α) config ctx =
      scaledDotProductAttention (α := α) ctx := by
  cases config
  rfl

/-- Semantic FlashAttention forward operator.

At the spec layer this is the named online/tiled algorithmic contract above. Runtime
implementations may use tiling, online softmax summaries, or a fused CUDA kernel, but they must
refine this denotation to be considered correct.
-/
def flashAttention
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α [nQ, dModel] :=
  -- The config is kept in the signature so graph rewrites and runtimes can record the intended
  -- schedule. At the denotational level, schedules must not change the mathematical result.
  onlineSoftmaxTiledAttention (α := α) config ctx

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
  -- As with the forward operator, the tile metadata belongs to the implementation schedule.
  -- The proof layer VJP is the same local derivative contract as standard SDPA.
  let _blockQ := config.blockQ
  let _blockK := config.blockK
  scaledDotProductAttentionBackward (α := α) ctx dOut

/-- Forward semantic correctness of the fused FlashAttention spec. -/
@[simp] theorem flashAttention_eq_scaledDotProductAttention
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    flashAttention (α := α) config ctx = scaledDotProductAttention (α := α) ctx := by
  simp [flashAttention]

/-- Backward/VJP semantic correctness of the fused FlashAttention spec. -/
@[simp] theorem flashAttentionBackward_eq_scaledDotProductAttentionBackward
    (config : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2)
    (dOut : Tensor α [nQ, dModel]) :
    flashAttentionBackward (α := α) config ctx dOut =
      scaledDotProductAttentionBackward (α := α) ctx dOut := by
  cases config
  rfl

end Spec
