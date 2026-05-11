/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Replay
public import Mathlib.Data.Array.Extract

/-!
# Replay-Buffer Proofs

These theorems certify the structural guarantees that make the runtime replay buffer safe to use:

- empty buffers really have size zero;
- zero-capacity buffers remain empty after a push;
- pushing below capacity increments size;
- pushing at capacity preserves the configured capacity by evicting the oldest element.

This is the proof layer for the "bounded FIFO" invariant used by DQN-style replay.
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Replay

open Spec
open Runtime.RL.Replay

variable {α : Type} {obsShape : Shape} {nActions : Nat}

/-- Empty replay buffers contain no stored transitions. -/
@[simp] theorem empty_items_size (capacity : Nat) :
    ((Buffer.empty (α := α) (obsShape := obsShape) (nActions := nActions) capacity).items.size) = 0 := by
  simp [Buffer.empty]

/-- Empty replay buffers report size zero through the public `size` helper. -/
@[simp] theorem empty_size (capacity : Nat) :
    (Buffer.empty (α := α) (obsShape := obsShape) (nActions := nActions) capacity).size = 0 := by
  simp [Buffer.size]

/-- A zero-capacity replay buffer remains empty after any push. -/
@[simp] theorem push_capacity_zero_items_size
    (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions)
    (hcap : b.capacity = 0) :
    (b.push t).items.size = 0 := by
  simp [Buffer.push, hcap]

/-- If there is room for one more item, `push` increases the size by exactly one. -/
theorem push_size_of_room
    (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions)
    (hroom : b.items.size + 1 ≤ b.capacity) :
    (b.push t).items.size = b.items.size + 1 := by
  have hcap : b.capacity ≠ 0 := by
    exact Nat.ne_of_gt (lt_of_lt_of_le (Nat.succ_pos b.items.size) hroom)
  have hlt : b.items.size < b.capacity := Nat.lt_of_succ_le hroom
  simp [Buffer.push, hcap, hlt, Array.size_push]

/--
If the buffer is exactly full and has positive capacity, `push` preserves size by evicting one old
item and appending the new one.
-/
theorem push_size_of_full
    (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions)
    (hcap : 0 < b.capacity) (hfull : b.items.size = b.capacity) :
    (b.push t).items.size = b.capacity := by
  have hcapNe : b.capacity ≠ 0 := Nat.ne_of_gt hcap
  simp [Buffer.push, hcapNe, hfull, Array.size_extract, Array.size_push]

end Replay
end RL
end Proofs
