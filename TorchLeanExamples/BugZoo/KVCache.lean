/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Tactic.Init

/-!
# BugZoo: KV-cache contracts

LLM inference-engine bug reports include cache-shift bugs, RoPE/position mismatches, shape mistakes,
and resource or configuration errors. TorchLean does not verify a serving engine, a paged-attention
allocator or a multi-GPU scheduler. What it can pin down is the first step: represent the cache
update as a typed tensor operation and prove the append invariant that cached decoding relies on.

Reference:
- Liu et al., "A First Look at Bugs in LLM Inference Engines", 2025.

This file proves the cache append boundary: the appended token lands at index `seqLen`
(`appendToken_last`) and every cached position keeps its entry (`appendToken_past`). It does not
prove that cached decode agrees with full-sequence attention:

$$
\operatorname{decodeWithCache}(\mathit{prefix},\mathit{newToken})
=\operatorname{fullAttention}(\mathit{prefix}\mathbin{+\!\!+}[\mathit{newToken}]).
$$
-/

@[expose] public section

namespace TorchLean.Examples.BugZoo

/-- A key/value cache whose sequence length and token shape are part of its type. -/
structure KVCache (α : Type) [Storage α] (seqLen : Nat)
    (tokenShape : Shape) where
  /-- Cached key tensors, indexed by time. -/
  keys : Tensor α (seqLen :: tokenShape)
  /-- Cached value tensors, indexed by time. -/
  values : Tensor α (seqLen :: tokenShape)

namespace KVCache

/-- Append one token tensor to a sequence cache along the time axis. -/
def appendToken {α : Type} [Storage α] {seqLen : Nat} {tokenShape : Shape}
    (past : Tensor α (seqLen :: tokenShape))
    (newToken : Tensor α tokenShape) :
    Tensor α ((seqLen + 1) :: tokenShape) :=
  Tensor.concat past (Tensor.repeatLeading 1 newToken)

/--
The appended token lands at index `seqLen`, the end of the extended sequence.

An off-by-one in a KV cache breaks this fact by writing the token over the last cached position.
The type forces the length to grow by one, and this lemma fixes where the entry goes.
-/
@[simp] theorem appendToken_last {α : Type} [Storage α] {seqLen : Nat} {tokenShape : Shape}
    (past : Tensor α (seqLen :: tokenShape))
    (newToken : Tensor α tokenShape) :
    (appendToken past newToken)[seqLen] = newToken := by
  change
    Tensor.unstack (appendToken past newToken) ⟨seqLen, Nat.lt_succ_self seqLen⟩ =
      newToken
  simpa [appendToken] using
    (Tensor.unstack_concat_right past (Tensor.repeatLeading 1 newToken) (0 : Fin 1))

/--
Every cached position keeps its entry: below `seqLen` the appended cache reads as `past`.

A cache-shift bug, which moves or overwrites cached positions on append, breaks this fact.
-/
@[simp] theorem appendToken_past {α : Type} [Storage α] {seqLen : Nat} {tokenShape : Shape}
    (past : Tensor α (seqLen :: tokenShape))
    (newToken : Tensor α tokenShape) (i : Nat) (h : i < seqLen) :
    (appendToken past newToken)[i] = past[i] := by
  change Tensor.unstack (appendToken past newToken) ⟨i, by omega⟩ = Tensor.unstack past ⟨i, h⟩
  exact Tensor.unstack_concat_left past (Tensor.repeatLeading 1 newToken) ⟨i, h⟩

/-- Append both key and value tensors to the KV cache. -/
def append {α : Type} [Storage α] {seqLen : Nat} {tokenShape : Shape}
    (cache : KVCache α seqLen tokenShape)
    (newKey newValue : Tensor α tokenShape) :
    KVCache α (seqLen + 1) tokenShape where
  keys := appendToken cache.keys newKey
  values := appendToken cache.values newValue

/-- After `append`, the key at index `seqLen` is `newKey`. -/
theorem append_last_key {α : Type} [Storage α] {seqLen : Nat} {tokenShape : Shape}
    (cache : KVCache α seqLen tokenShape)
    (newKey newValue : Tensor α tokenShape) :
    (append cache newKey newValue).keys[seqLen] = newKey := by
  exact appendToken_last cache.keys newKey

/-- After `append`, the value at index `seqLen` is `newValue`. -/
theorem append_last_value {α : Type} [Storage α] {seqLen : Nat} {tokenShape : Shape}
    (cache : KVCache α seqLen tokenShape)
    (newKey newValue : Tensor α tokenShape) :
    (append cache newKey newValue).values[seqLen] = newValue := by
  exact appendToken_last cache.values newValue

end KVCache

end TorchLean.Examples.BugZoo
