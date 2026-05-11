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

- this file gives the proof-facing semantic contract for a fused FlashAttention operator;
- `NN/Runtime/Autograd/Engine/Cuda/Kernels.lean` exposes native CUDA/stub FFI kernels for the
  runtime path;
- the CUDA FFI boundary is documented separately because Lean does not verify CUDA machine code.

The key point is that the fused op has the same denotation as standard masked scaled dot-product
attention over the spec scalar. Different tile sizes are runtime scheduling choices, not semantic
choices.

## What is proved here?

The theorems in this file are deliberately small but important:

- `onlineSoftmaxTiledAttention_eq_scaledDotProductAttention` proves the named FlashAttention
  algorithmic contract has the same denotation as standard attention.
- `flashAttention_eq_scaledDotProductAttention` proves the fused forward operator is semantically
  equal to TorchLean's existing standard attention spec.
- `flashAttentionBackward_eq_scaledDotProductAttentionBackward` proves the fused VJP contract is
  semantically equal to the existing standard attention backward spec.
- `cudaLoopFlashAttention_eq_onlineSoftmaxTiledAttention` gives a Lean functional model of the
  native kernel loops and proves that this loop model denotes the same online/tiled contract.

These are definitional-equality theorems because the proof-facing contract spells out the same
mathematical stages as standard attention. The native CUDA implementation is tested against this
contract operationally and remains a runtime trust boundary, like the other CUDA kernels.

## Why this is not a CUDA proof

The definitions below are the mathematical contract for FlashAttention. They do not claim to verify
the native CUDA source. Instead, they make the important theorem explicit:

`onlineSoftmaxTiledAttention cfg ctx = scaledDotProductAttention ctx`.

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

open Spec
open Tensor
open Shape

namespace Spec

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

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
3. compute the same row-wise normalized weights that an online summary converges to;
4. multiply by values.

This is intentionally schedule-polymorphic: `cfg.blockQ` and `cfg.blockK` describe how a runtime may
tile the work, but they do not alter the denotation.
-/

/-- Unmasked attention scores `QKᵀ`. -/
def attentionScores
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α (.dim nQ (.dim nK .scalar)) :=
  matMulSpec ctx.Q (matrixTransposeSpec ctx.K)

/-- Scaled attention scores before row normalization.

Masking is applied at the weight level by `onlineSoftmaxWeights`, using the same true hard-mask
semantics as `scaledDotProductAttention`.
-/
def scaledAttentionScores
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α (.dim nQ (.dim nK .scalar)) :=
  let scale := MathFunctions.sqrt (dModel : α)
  scaleSpec (attentionScores (α := α) ctx) (1 / scale)

/-- The row-wise softmax weights produced by the online softmax summary.

`Activation.softmaxSpec` already uses the stabilized form `exp(x - rowMax) / Σ exp(x - rowMax)`.
This definition is the **denotation** that a FlashAttention implementation must refine. It is not a
formal model of Dao-style tile loops or SRAM/HBM traffic.
-/
def onlineSoftmaxWeights
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α (.dim nQ (.dim nK .scalar)) :=
  -- Tile metadata is relevant to the runtime schedule, not to the exact normalized weights.
  let _blockQ := cfg.blockQ
  let _blockK := cfg.blockK
  let scores := scaledAttentionScores (α := α) ctx
  match ctx.mask with
  | none => Activation.softmaxSpec (α := α) scores
  | some m => hardMaskedSoftmaxSpec scores m

/-- Proof-facing FlashAttention forward algorithm.

This is the mathematical result of the online/tiled schedule: row-wise softmax weights multiplied by
`V`. Runtime kernels may avoid storing the full weights, but they must refine this value.
-/
def onlineSoftmaxTiledAttention
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α (.dim nQ (.dim dModel .scalar)) :=
  matMulSpec (onlineSoftmaxWeights (α := α) cfg ctx) ctx.V

/-!
## CUDA Denotation

The native runtime kernel is intended to compute the same row program in a fused way:

1. for each `(batch, head, query)` row, scan keys to build masked/scaled scores;
2. compute the stabilized softmax normalization for that row;
3. accumulate `Σ_j softmax(score_j) * V_j` directly into the output.

`cudaLoopFlashAttention` is intentionally a **denotational target**, written with tensor
combinators rather than CUDA thread/block syntax. The equalities below are definitional sanity
checks: they say the named fused operator denotes standard SDPA in the spec. They do not verify the
CUDA source code, the online-softmax recurrence, or the memory-IO schedule. Those remain explicit
runtime/FFI contracts tested against this target.
-/

/-- Denotational target for the fused CUDA FlashAttention forward kernel. -/
def cudaLoopFlashAttention
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α (.dim nQ (.dim dModel .scalar)) :=
  let rowWeights := onlineSoftmaxWeights (α := α) cfg ctx
  matMulSpec rowWeights ctx.V

@[simp] theorem cudaLoopFlashAttention_eq_onlineSoftmaxTiledAttention
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    cudaLoopFlashAttention (α := α) cfg ctx =
      onlineSoftmaxTiledAttention (α := α) cfg ctx := by
  rfl

/--
The CUDA denotational target has the same spec meaning as standard SDPA.

This is intentionally a definitional theorem, not a proof about CUDA machine code or an online
softmax tile recurrence.
-/
@[simp] theorem cudaLoopFlashAttention_eq_scaledDotProductAttention
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    cudaLoopFlashAttention (α := α) cfg ctx =
      scaledDotProductAttention (α := α) ctx := by
  cases cfg
  rfl

/--
The proof-facing FlashAttention denotation equals standard SDPA.

This theorem is useful for graph-rewrite semantics, but should not be read as a verification of a
particular CUDA implementation.
-/
@[simp] theorem onlineSoftmaxTiledAttention_eq_scaledDotProductAttention
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    onlineSoftmaxTiledAttention (α := α) cfg ctx =
      scaledDotProductAttention (α := α) ctx := by
  cases cfg
  rfl

/-- Semantic FlashAttention forward operator.

At the spec layer this is the named online/tiled algorithmic contract above. Runtime
implementations may use tiling, online softmax summaries, or a fused CUDA kernel, but they must
refine this denotation to be considered correct.
-/
def flashAttention
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    Tensor α (.dim nQ (.dim dModel .scalar)) :=
  -- The config is kept in the signature so graph rewrites and runtimes can record the intended
  -- schedule. At the denotational level, schedules must not change the mathematical result.
  onlineSoftmaxTiledAttention (α := α) cfg ctx

/-- Semantic FlashAttention backward/VJP operator.

This is the local derivative contract a fused backward kernel should refine. The actual CUDA kernel
may recompute attention probabilities from row statistics rather than storing the full attention
matrix, but the returned adjoints must match this spec-level VJP up to the chosen floating-point
error envelope.
-/
def flashAttentionBackward
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2)
    (dOut : Tensor α (.dim nQ (.dim dModel .scalar))) :
    (Tensor α (.dim nQ (.dim dModel .scalar)) ×
     Tensor α (.dim nK (.dim dModel .scalar)) ×
     Tensor α (.dim nK (.dim dModel .scalar))) :=
  -- As with the forward operator, the tile metadata belongs to the implementation schedule.
  -- The proof-facing VJP is the same local derivative contract as standard SDPA.
  let _blockQ := cfg.blockQ
  let _blockK := cfg.blockK
  scaledDotProductAttentionBackward (α := α) ctx dOut

/-- Forward semantic correctness of the fused FlashAttention spec. -/
@[simp] theorem flashAttention_eq_scaledDotProductAttention
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2) :
    flashAttention (α := α) cfg ctx = scaledDotProductAttention (α := α) ctx := by
  simp [flashAttention]

/-- Backward/VJP semantic correctness of the fused FlashAttention spec. -/
@[simp] theorem flashAttentionBackward_eq_scaledDotProductAttentionBackward
    (cfg : FlashAttentionConfig)
    {nQ nK dModel : Nat} {h1 : nQ ≠ 0} {h2 : nK ≠ 0}
    (ctx : AttentionContext α nQ nK dModel h1 h2)
    (dOut : Tensor α (.dim nQ (.dim dModel .scalar))) :
    flashAttentionBackward (α := α) cfg ctx dOut =
      scaledDotProductAttentionBackward (α := α) ctx dOut := by
  cases cfg
  rfl

end Spec
