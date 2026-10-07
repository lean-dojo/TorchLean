/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# BugZoo: RoPE position accounting

LLM inference-engine bug studies report cache, RoPE, tokenizer/config, batching, and resource
boundary bugs across real serving stacks:

https://arxiv.org/abs/2506.09713

TorchLean does not model every production RoPE kernel. It can make the
position schedule explicit. A KV-cache or decode importer should not infer positions from ambient
mutable state; it should hand over a schedule that Lean can inspect.
-/

@[expose] public section

namespace NN.Examples.BugZoo.RoPEPosition

open TorchLean

/--
Append one position to the tensor schedule supplied to rotary or absolute position embeddings.

The default is the old sequence length, matching the zero-based, unshifted convention.
An offset or sliding-window schedule can supply `position` explicitly. No rotation kernel is
evaluated or verified by this function.
-/
def append {seqLen : Nat} (schedule : Tensor Nat [seqLen]) (position : Nat := seqLen) :
    Tensor Nat [seqLen + 1] :=
  Tensor.concat schedule (Tensor.full [1] position)

/-- The last slot contains the supplied position, or the old length when using the default. -/
theorem append_last {seqLen : Nat} (schedule : Tensor Nat [seqLen])
    (position : Nat := seqLen) :
    (append schedule position)[seqLen] = position := by
  change (append schedule position) (⟨seqLen, Nat.lt_succ_self seqLen⟩, ()) = position
  have h := congrArg (fun value : Tensor Nat [] => value ())
    (Tensor.get_concat_right schedule (Tensor.full [1] position) (0 : Fin 1))
  simpa [append, Tensor.full, Spec.get, Tensor.unstack,
    Tensor.Internal.Rep.unstack] using h

end NN.Examples.BugZoo.RoPEPosition
