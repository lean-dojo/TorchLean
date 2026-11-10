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
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Operations
public import TorchLean.Tensor.Tactic.Init

/-!
# BugZoo: RoPE position accounting

LLM inference-engine bug studies report cache, RoPE, tokenizer/config, batching, and resource
boundary bugs across real serving stacks:

https://arxiv.org/abs/2506.09713

TorchLean does not model every production RoPE kernel. What it can do is make the position
schedule a tensor that Lean can inspect. A KV-cache or decode importer should hand over that
schedule and never infer positions from ambient mutable state.
-/

@[expose] public section

namespace TorchLean.Examples.BugZoo.RoPEPosition

/--
Append one position to the tensor schedule supplied to rotary or absolute position embeddings.

By default the appended slot gets `seqLen`, the next zero-based position when nothing is shifted.
An offset or sliding-window schedule passes its own `position`. The function only builds the
schedule, and no rotation kernel runs here.
-/
def append {seqLen : Nat} (schedule : Tensor Nat [seqLen]) (position : Nat := seqLen) :
    Tensor Nat [seqLen + 1] :=
  Tensor.concat schedule (Tensor.full [1] position)

/-- The last slot holds `position`, which is `seqLen` by default. -/
theorem append_last {seqLen : Nat} (schedule : Tensor Nat [seqLen])
    (position : Nat := seqLen) :
    (append schedule position)[seqLen] = position := by
  change (append schedule position) (⟨seqLen, Nat.lt_succ_self seqLen⟩, ()) = position
  have h := congrArg (fun value : Tensor Nat [] => value ())
    (Tensor.unstack_concat_right schedule (Tensor.full [1] position) (0 : Fin 1))
  simpa [append,
    Tensor.unstack] using h

end TorchLean.Examples.BugZoo.RoPEPosition
