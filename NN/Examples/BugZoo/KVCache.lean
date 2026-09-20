/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# BugZoo: KV-cache contracts

LLM inference-engine bug reports include cache-shift bugs, RoPE/position mismatches, shape mistakes,
and resource/configuration errors. TorchLean does not verify a full serving engine, paged
attention allocator, or multi-GPU scheduler. The useful first step is still precise: represent the
cache update as a typed tensor operation and prove the append invariant we rely on.

Reference:
- Liu et al., "A First Look at Bugs in LLM Inference Engines", 2025.

This file proves the cache append boundary. A stronger future theorem should connect cached decode
to full-sequence attention:

$$
\operatorname{decodeWithCache}(\mathit{prefix},\mathit{newToken})
=\operatorname{fullAttention}(\mathit{prefix}\mathbin{+\!\!+}[\mathit{newToken}]).
$$

under the same mask, RoPE/position encoding, and numeric semantics.
-/

@[expose] public section

namespace NN.Examples.BugZoo.KVCache

open TorchLean

/-- A key/value cache with an explicit sequence length and head dimension. -/
structure Cache (α : Type) [Storage α] (seqLen headDim : Nat) where
  /-- Cached key vectors, indexed by time. -/
  keys : Tensor α [seqLen, headDim]
  /-- Cached value vectors, indexed by time. -/
  values : Tensor α [seqLen, headDim]

/-- View one token vector as a length-one sequence. -/
def singletonToken {α : Type} [Storage α] {headDim : Nat}
    (x : Tensor α [headDim]) :
    Tensor α [1, headDim] :=
  Tensor.repeatLeading 1 x

/-- Reading the only position of a singleton sequence gives the token back. -/
@[simp] theorem singletonToken_get {α : Type} [Storage α] {headDim : Nat}
    (x : Tensor α [headDim]) (i : Fin 1) :
    (singletonToken x)[i] = x := by
  change Spec.get (singletonToken x) i = x
  simp [singletonToken]

/-- Append one token vector to a sequence cache along the time axis. -/
def appendToken {α : Type} [Storage α] {seqLen headDim : Nat}
    (past : Tensor α [seqLen, headDim])
    (newToken : Tensor α [headDim]) :
    Tensor α [seqLen + 1, headDim] :=
  Tensor.concat past (singletonToken newToken)

/--
The appended token lands at index `seqLen`, that is, at the end.

This is the fact an off-by-one in a KV cache would break: writing the new token over the last cached
position instead of after it. The type already forces the length to grow by one, and this lemma pins
down where the new entry goes.
-/
@[simp] theorem appendToken_last {α : Type} [Storage α] {seqLen headDim : Nat}
    (past : Tensor α [seqLen, headDim])
    (newToken : Tensor α [headDim]) :
    (appendToken past newToken)[seqLen] = newToken := by
  change
    Spec.get (appendToken past newToken) ⟨seqLen, Nat.lt_succ_self seqLen⟩ =
      newToken
  simpa [appendToken, singletonToken] using
    (Tensor.get_concat_right past (singletonToken newToken) (0 : Fin 1))

/-- Append both key and value vectors to the KV cache. -/
def appendKV {α : Type} [Storage α] {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Tensor α [headDim]) :
    Cache α (seqLen + 1) headDim where
  keys := appendToken cache.keys newKey
  values := appendToken cache.values newValue

/-- The newly appended key is exactly the final key in the updated cache. -/
theorem appendKV_last_key {α : Type} [Storage α] {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Tensor α [headDim]) :
    (appendKV cache newKey newValue).keys[seqLen] = newKey := by
  exact appendToken_last cache.keys newKey

/-- The newly appended value is exactly the final value in the updated cache. -/
theorem appendKV_last_value {α : Type} [Storage α] {seqLen headDim : Nat}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Tensor α [headDim]) :
    (appendKV cache newKey newValue).values[seqLen] = newValue := by
  exact appendToken_last cache.values newValue

end NN.Examples.BugZoo.KVCache
