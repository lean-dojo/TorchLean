/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Runtime.Autograd.TorchLean.Random

/-!
# Experience Replay Buffers

This module provides the small typed replay-buffer layer used by value-learning algorithms such as
DQN, Double DQN, DDPG, TD3, and SAC.

The design is intentionally modest:

- transitions are already typed (`Runtime.RL.Core.Transition`), so samples cannot mix observation
  shapes or action spaces;
- the buffer is a bounded FIFO array, which is enough for examples, tests, and simple off-policy
  training loops;
- sampling is deterministic from `(seed, counter)` so runs remain replayable inside Lean.

Prioritized replay belongs in a separate module with a priority tree / segment tree and a second
set of importance weights.

References:
- Lin, "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching"
  (1992), early experience replay.
- Mnih et al., "Human-level control through deep reinforcement learning" (2015), replay in DQN:
  https://doi.org/10.1038/nature14236
- Schaul et al., "Prioritized Experience Replay" (2015), future extension point:
  https://arxiv.org/abs/1511.05952
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Replay

open Spec
open Tensor

variable {α : Type} [Context α]
variable {obsShape : Shape} {nActions : Nat}

/-- Typed replay transition for tensor-valued observations and finite actions. -/
abbrev Transition (α : Type) (obsShape : Shape) (nActions : Nat) :=
  Core.Transition α obsShape nActions

/--
Bounded FIFO replay buffer.

`capacity = 0` is allowed and represents a disabled buffer; pushes then leave the buffer empty.
-/
structure Buffer (α : Type) (obsShape : Shape) (nActions : Nat) where
  /-- Maximum number of transitions retained. -/
  capacity : Nat
  /-- Stored transitions, oldest first. -/
  items : Array (Transition α obsShape nActions)
  deriving Inhabited

namespace Buffer

/-- Create an empty replay buffer with the requested capacity. -/
def empty (capacity : Nat) : Buffer α obsShape nActions :=
  { capacity := capacity, items := #[] }

/-- Current number of stored transitions. -/
def size (b : Buffer α obsShape nActions) : Nat :=
  b.items.size

/-- `true` iff the buffer contains no transitions. -/
def isEmpty (b : Buffer α obsShape nActions) : Bool :=
  b.items.isEmpty

/-- `true` iff the buffer has reached its configured capacity. -/
def isFull (b : Buffer α obsShape nActions) : Bool :=
  b.capacity != 0 && b.items.size ≥ b.capacity

/--
Push one transition, dropping the oldest item if the buffer is already full.

The invariant `items.size ≤ capacity` is maintained by construction when the buffer starts valid.
-/
def push (b : Buffer α obsShape nActions) (t : Transition α obsShape nActions) :
    Buffer α obsShape nActions :=
  if b.capacity = 0 then
    { b with items := #[] }
  else
    let withNew := b.items.push t
    if withNew.size ≤ b.capacity then
      { b with items := withNew }
    else
      { b with items := withNew.extract 1 withNew.size }

/-- Push a batch of transitions in order. -/
def pushMany (b : Buffer α obsShape nActions) (ts : Array (Transition α obsShape nActions)) :
    Buffer α obsShape nActions :=
  ts.foldl (fun acc t => acc.push t) b

/--
Read an item by wrapping the index modulo the current buffer size.

Returns `none` for an empty buffer.
-/
def getModulo? (b : Buffer α obsShape nActions) (idx : Nat) :
    Option (Transition α obsShape nActions) :=
  if b.items.isEmpty then
    none
  else
    let j := idx % b.items.size
    b.items[j]?

/--
Deterministic contiguous sample with wraparound.

This is useful for tests and for simple off-policy examples where reproducibility matters more than
statistical randomness. Empty buffers return an empty batch.
-/
def sampleContiguous (b : Buffer α obsShape nActions) (start batchSize : Nat) :
    Array (Transition α obsShape nActions) :=
  Id.run do
    let mut out := #[]
    for k in [0:batchSize] do
      match b.getModulo? (start + k) with
      | some t => out := out.push t
      | none => pure ()
    return out

/--
Deterministic pseudo-random sample from `(seed, counter)`.

The sampler intentionally returns the next counter rather than hiding mutation. It draws indices via
TorchLean's keyed uniform helper, then wraps them modulo the current buffer size. Empty buffers return
an empty batch and leave the counter unchanged.
-/
def sampleRandom (b : Buffer α obsShape nActions) (seed counter batchSize : Nat) :
    Nat × Array (Transition α obsShape nActions) :=
  if b.items.size = 0 then
    (counter, #[])
  else
    Id.run do
      let mut out := #[]
      let mut c := counter
      for _ in [0:batchSize] do
        let key := _root_.Runtime.Autograd.TorchLean.Random.keyOf seed c
        let u : Float :=
          Tensor.toScalar
            (_root_.Runtime.Autograd.TorchLean.Random.uniform (α := Float) key (s := Shape.scalar))
        let idx := ((u * Float.ofNat b.items.size).floor.toUInt64.toNat) % b.items.size
        match b.items[idx]? with
        | some t => out := out.push t
        | none => pure ()
        c := c + 1
      return (c, out)

end Buffer

end Replay
end RL
end Runtime
