/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# BugZoo: KV-cache contracts

LLM inference-engine bug reports include cache-shift bugs, RoPE/position mismatches, shape mistakes, and
resource/configuration errors. TorchLean does not verify a full serving engine, paged
attention allocator, or multi-GPU scheduler. The useful first step is still precise: represent the
cache update as a typed tensor operation and prove the append invariant we rely on.

Reference:
- Liu et al., "A First Look at Bugs in LLM Inference Engines", 2025.

This file proves the cache append boundary. A stronger future theorem should connect cached decode
to full-sequence attention:

`decodeWithCache prefix newToken = fullAttention (prefix ++ [newToken])`

under the same mask, RoPE/position encoding, and numeric semantics.
-/

@[expose] public section

namespace NN.Examples.BugZoo.KVCache

/-- A key/value cache with an explicit sequence length and head dimension. -/
structure Cache (α : Type) (seqLen headDim : Nat) where
  /-- Cached key vectors, indexed by time. -/
  keys : Spec.Tensor α (.dim seqLen (.dim headDim .scalar))
  /-- Cached value vectors, indexed by time. -/
  values : Spec.Tensor α (.dim seqLen (.dim headDim .scalar))

/-- View one token vector as a length-one sequence. -/
def singletonToken {α : Type} {headDim : Nat}
    (x : Spec.Tensor α (.dim headDim .scalar)) :
    Spec.Tensor α (.dim 1 (.dim headDim .scalar)) :=
  Spec.Tensor.dim fun _ => x

/-- Append one token vector to a sequence cache along the time axis. -/
def appendToken {α : Type} {seqLen headDim : Nat}
    (past : Spec.Tensor α (.dim seqLen (.dim headDim .scalar)))
    (newToken : Spec.Tensor α (.dim headDim .scalar)) :
    Spec.Tensor α (.dim (seqLen + 1) (.dim headDim .scalar)) :=
  Spec.Tensor.concatDim0Spec past (singletonToken newToken)

/-- Append both key and value vectors to the KV cache. -/
def appendKV {α : Type} {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Spec.Tensor α (.dim headDim .scalar)) :
    Cache α (seqLen + 1) headDim where
  keys := appendToken cache.keys newKey
  values := appendToken cache.values newValue

/-- The newly appended key is exactly the final key in the updated cache. -/
theorem appendKV_last_key {α : Type} {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Spec.Tensor α (.dim headDim .scalar)) :
    Spec.getAtSpec (appendKV cache newKey newValue).keys
        ⟨seqLen, Nat.lt_succ_self seqLen⟩ = newKey := by
  cases cache with
  | mk keys values =>
    cases keys with
    | dim keyRows =>
      simp [appendKV, appendToken, singletonToken, Spec.Tensor.concatDim0Spec,
        Spec.getAtSpec]

/-- The newly appended value is exactly the final value in the updated cache. -/
theorem appendKV_last_value {α : Type} {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Spec.Tensor α (.dim headDim .scalar)) :
    Spec.getAtSpec (appendKV cache newKey newValue).values
        ⟨seqLen, Nat.lt_succ_self seqLen⟩ = newValue := by
  cases cache with
  | mk keys values =>
    cases values with
    | dim valueRows =>
      simp [appendKV, appendToken, singletonToken, Spec.Tensor.concatDim0Spec,
        Spec.getAtSpec]

end NN.Examples.BugZoo.KVCache
