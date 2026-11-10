/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Replay

/-!
# Replay-Buffer Proofs

The runtime replay buffer behaves as a bounded FIFO queue, the invariant that DQN-style replay
relies on:

- an empty buffer has size zero,
- a zero-capacity buffer stays empty after a push,
- a push below capacity adds one to the size,
- a push at capacity keeps the size at the capacity by evicting the oldest transition,
- the remaining transitions keep their order, and the pushed transition is last.
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Replay

open RL (Transition)

variable {α : Type} {obsShape : Shape} {nActions : Nat}

/-- Empty replay buffers contain no stored transitions. -/
@[simp] theorem Buffer.empty_items_size (capacity : Nat) :
    ((Buffer.empty (α := α) (obsShape := obsShape) (nActions := nActions) capacity).items.size)
      = 0 := by
  simp [Buffer.empty]

/-- Empty replay buffers report size zero through the public `size` helper. -/
@[simp] theorem Buffer.empty_size (capacity : Nat) :
    (Buffer.empty (α := α) (obsShape := obsShape) (nActions := nActions) capacity).size = 0 := by
  simp [Buffer.size]

/-- A zero-capacity replay buffer remains empty after any push. -/
@[simp] theorem Buffer.push_size_of_capacity_eq_zero
    (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions)
    (hcap : b.capacity = 0) :
    (b.push t).items.size = 0 := by
  simp [Buffer.push, hcap]

/-- If there is room for one more item, `push` increases the size by exactly one. -/
theorem Buffer.push_size_of_room
    (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions)
    (hroom : b.items.size + 1 ≤ b.capacity) :
    (b.push t).items.size = b.items.size + 1 := by
  have hcap : b.capacity ≠ 0 := by omega
  simp [Buffer.push, hcap, hroom, Array.size_push]

/--
If the buffer is full and its capacity is positive, `push` keeps the size at the capacity: it
evicts the oldest item and appends the pushed one.
-/
theorem Buffer.push_size_of_full
    (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions)
    (hcap : 0 < b.capacity) (hfull : b.items.size = b.capacity) :
    (b.push t).items.size = b.capacity := by
  have hcapNe : b.capacity ≠ 0 := hcap.ne'
  simp [Buffer.push, hcapNe, hfull, Array.size_extract, Array.size_push]

/-- Below capacity, `push` keeps every stored transition in order and appends `t`. -/
theorem Buffer.push_contents_of_room
    (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions)
    (hroom : b.items.size + 1 ≤ b.capacity) :
    (b.push t).items.toList = b.items.toList ++ [t] := by
  have hcap : b.capacity ≠ 0 := by omega
  simp [Buffer.push, hcap, hroom]

/-- On a full buffer of positive capacity, `push` evicts only the oldest transition and appends `t`
after the others, which keep their order. -/
theorem Buffer.push_contents_of_full
    (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions)
    (hcap : 0 < b.capacity) (hfull : b.items.size = b.capacity) :
    (b.push t).items.toList = b.items.toList.drop 1 ++ [t] := by
  have hcapNe : b.capacity ≠ 0 := hcap.ne'
  have hlength : 1 ≤ b.items.toList.length := by
    have : b.items.toList.length = b.capacity := by simpa using hfull
    omega
  simp [Buffer.push, hcapNe, hfull, hcap, Array.toList_extract,
    List.drop_append_of_le_length hlength]

end Replay
end RL
end TorchLean
