/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Transition
public import TorchLean.Tensor.Random
public import TorchLean.NN.RL.Environment

/-!
# Experience Replay Buffers

A small typed replay buffer for off-policy value learning (DQN, Double DQN, DDPG, TD3 and SAC).
Transitions are `TorchLean.RL.Transition` values, so one buffer cannot mix observation shapes
or action spaces. The buffer is a bounded FIFO array, which is enough for examples, tests and
simple training loops. Sampling is a pure function of `(seed, counter)`, so a run draws the same
minibatches each time it is replayed inside Lean.

Prioritized replay needs a priority tree and importance-sampling weights, and is not implemented
here.

We built the buffer by hand after the replay memory in the Methods of Mnih et al., which keeps the
last $N$ transitions and samples uniformly from them. `push` drops the oldest transition at
capacity, and `sampleRandom` draws with replacement. A draw reduces a 64-bit SplitMix64 word
modulo the size, so over all $2^{64}$ words each index has probability within $2^{-64}$ of
$1/\mathrm{size}$. The FIFO facts are proved in `TorchLean.NN.RL.Replay.Capacity`
(`Buffer.push_size_of_full`, `Buffer.push_contents_of_full` and the rest). Sampling has no
theorem or test.

References:
- Lin, "Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching"
  (1992), Machine Learning 8(3-4):293-321, early experience replay:
  https://doi.org/10.1007/BF00992699
- Mnih et al., "Human-level control through deep reinforcement learning" (2015), Nature
  518(7540):529-533, replay memory in Methods: https://doi.org/10.1038/nature14236
- Schaul, Quan, Antonoglou and Silver, "Prioritized Experience Replay" (2015), ICLR 2016:
  https://arxiv.org/abs/1511.05952
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Replay

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]
variable {obsShape : Shape} {nActions : Nat}

/--
Replay transition from a Gym-style observed step, with `done` set from `terminated` alone.

A step cut off by a time limit has a successor state whose value the TD target should bootstrap
from, so it is stored with `done = false`. The PPO rollout makes the same split between its
bootstrap and continuation masks. A validated `TorchLean.RL.Boundary.Transition` is an
`ObservedTransition` over `Float`, so it converts as is.
-/
def ofObservedTransition
    (t : RL.ObservedTransition (Tensor α obsShape) (Fin nActions) α) :
    Transition α obsShape nActions :=
  { state := t.observation
    action := t.action
    reward := t.reward
    nextState := t.nextObservation
    done := t.terminated }

/--
Bounded FIFO replay buffer.

A buffer with `capacity = 0` is disabled: every push leaves it empty.
-/
structure Buffer (α : Type) [Storage α]
    (obsShape : Shape) (nActions : Nat) where
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
Append one transition, dropping the oldest one when the buffer is full. A buffer that satisfies
`items.size ≤ capacity` before the push satisfies it after.
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
`batchSize` transitions read in order from index `start`, wrapping around the end of the buffer
(entry `k` is `items[(start + k) % size]`). Tests and simple off-policy examples use it where
reproducibility matters more than random sampling. An empty buffer gives an empty batch.
-/
def sampleContiguous (b : Buffer α obsShape nActions) (start batchSize : Nat) :
    Array (Transition α obsShape nActions) :=
  if h : b.items.size = 0 then
    #[]
  else
    Id.run do
      let mut out := #[]
      for k in [0:batchSize] do
        out := out.push (b.items[(start + k) % b.items.size]'(Nat.mod_lt _ (Nat.pos_of_ne_zero h)))
      return out

/--
Pseudo-random sample of `batchSize` transitions, drawn with replacement and determined by
`(seed, counter)`. Draw `k` reads index
`Random.sampleNat (Random.keyOf seed (counter + k)) 0 size`, an integer computation
with no rounding.

The result pairs the batch with the next counter, `counter + batchSize`, which the caller passes
to the following call. An empty buffer gives an empty batch and the counter unchanged.
-/
def sampleRandom (b : Buffer α obsShape nActions) (seed counter batchSize : Nat) :
    Nat × Array (Transition α obsShape nActions) :=
  if h : b.items.size = 0 then
    (counter, #[])
  else
    Id.run do
      let mut out := #[]
      let mut c := counter
      for _ in [0:batchSize] do
        let idx := Random.sampleNat (Random.keyOf seed c) 0 b.items.size
        out := out.push (b.items[idx]'(Nat.mod_lt _ (Nat.pos_of_ne_zero h)))
        c := c + 1
      return (c, out)

end Buffer

end Replay
end RL
end TorchLean
