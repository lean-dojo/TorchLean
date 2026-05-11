/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

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

/-- A position schedule gives the rotary/absolute position used for every token slot. -/
structure PositionSchedule (seqLen : Nat) where
  pos : Fin seqLen → Nat

/--
Append one decode step using the canonical next position `seqLen`.

Existing positions are preserved, and the new final token is assigned the next sequence index.
-/
def appendNextPosition {seqLen : Nat} (sched : PositionSchedule seqLen) :
    PositionSchedule (seqLen + 1) where
  pos i :=
    if h : i.val < seqLen then
      sched.pos ⟨i.val, h⟩
    else
      seqLen

/-- The newly appended token gets exactly the next position, not an off-by-one cache position. -/
theorem appendNextPosition_last {seqLen : Nat} (sched : PositionSchedule seqLen) :
    (appendNextPosition sched).pos ⟨seqLen, Nat.lt_succ_self seqLen⟩ = seqLen := by
  simp [appendNextPosition]

end NN.Examples.BugZoo.RoPEPosition
