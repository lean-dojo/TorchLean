/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Random.Permutation

/-!
# Finite Sample Streams

A `SampleStream α` is the lower-level data source under the trainer API: a finite number of
samples, each computed only when its index is requested. `fromArray` wraps array-backed data
without copying, and generated data and tensor slices stay lazy.

Application code normally builds a `TorchLean.Trainer.Dataset`. Streams serve manual training loops
and data sources that already know their runtime scalar type.
-/

@[expose] public section

namespace TorchLean.Data

/-- A finite, lazily indexed source of samples. -/
structure SampleStream (α : Type) where
  /-- Number of samples available from the stream. -/
  size : Nat
  /-- Compute the sample at an in-bounds index. -/
  get : Fin size → α

namespace SampleStream

variable {α β : Type}

/-- Named halves of a stream split. -/
structure SplitResult (α : Type) where
  /-- Samples before the split point. -/
  selected : SampleStream α
  /-- Samples at and after the split point. -/
  remaining : SampleStream α

/-- State and stream produced by deterministic shuffling. -/
structure ShuffleResult (α : Type) where
  /-- Seed to use for the next deterministic shuffle. -/
  nextSeed : Nat
  /-- Stream in shuffled index order. -/
  stream : SampleStream α

/-- State and named partitions produced by a shuffled split. -/
structure RandomSplitResult (α : Type) where
  /-- Seed to use for the next deterministic shuffle. -/
  nextSeed : Nat
  /-- Samples selected before the split point. -/
  selected : SampleStream α
  /-- Samples selected at and after the split point. -/
  remaining : SampleStream α

/-- Wrap an array as a finite stream without copying it. -/
def fromArray (xs : Array α) : SampleStream α :=
  SampleStream.mk xs.size fun i => xs[i]

/-- Materialize the samples in index order. -/
def toArray (stream : SampleStream α) : Array α :=
  Array.ofFn stream.get

/-- A stream built from an array reports that array's length. -/
@[simp] theorem size_fromArray (xs : Array α) :
    (fromArray xs).size = xs.size := rfl

/-- Whether the stream has no samples. -/
def isEmpty (stream : SampleStream α) : Bool :=
  stream.size = 0

/-- The sample at index `i`, or `none` when `i` is out of range. -/
def get? (stream : SampleStream α) (i : Nat) : Option α :=
  if h : i < stream.size then some (stream.get ⟨i, h⟩) else none

/-- Transform samples when they are requested. -/
def map (f : α → β) (stream : SampleStream α) : SampleStream β :=
  SampleStream.mk stream.size fun i => f (stream.get i)

/-- Append two finite streams in index order. -/
def append (xs ys : SampleStream α) : SampleStream α :=
  SampleStream.mk (xs.size + ys.size) (Fin.append xs.get ys.get)

/-- Split a stream into its first `n` samples and the rest. When `n` is at least the size, the
first part is the whole stream and the rest is empty. -/
def splitAt (n : Nat) (stream : SampleStream α) : SplitResult α :=
  let prefixSize := min n stream.size
  let selected := SampleStream.mk prefixSize fun i =>
    stream.get ⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_right _ _)⟩
  let remaining := SampleStream.mk (stream.size - prefixSize) fun i =>
    have hlt : prefixSize + i.val < stream.size := by
      simpa [Nat.add_comm] using Nat.add_lt_of_lt_sub i.isLt
    stream.get ⟨prefixSize + i.val, hlt⟩
  { selected, remaining }

/--
Shuffle the indices deterministically with `Random.permutation` under the key of `seed`,
returning `Random.nextSeed seed 0` and a stream in the shuffled order.

Only the index permutation is stored, and requesting a shuffled sample evaluates the original
stream at the permuted index.
-/
def shuffle (seed : Nat) (stream : SampleStream α) : ShuffleResult α :=
  let indices := Random.permutation (Random.keyOf seed 0) stream.size
  { nextSeed := Random.nextSeed seed 0
    stream := SampleStream.mk indices.size fun i => stream.get indices[i] }

/-- Deterministically shuffle a stream and discard the next pseudo-random seed. -/
def shuffled (seed : Nat) (stream : SampleStream α) : SampleStream α :=
  (shuffle seed stream).stream

/-- Shuffle a stream once and split the result at `n`. -/
def randomSplitAt (seed n : Nat) (stream : SampleStream α) :
    RandomSplitResult α :=
  let shuffled := shuffle seed stream
  let split := splitAt n shuffled.stream
  { nextSeed := shuffled.nextSeed
    selected := split.selected
    remaining := split.remaining }

/-- Cycle through a nonempty stream indefinitely. -/
def cycle (stream : SampleStream α) (h : 0 < stream.size) (i : Nat) : α :=
  haveI : NeZero stream.size := ⟨Nat.pos_iff_ne_zero.mp h⟩
  stream.get (Fin.ofNat stream.size i)

/-- Build a cycling sample function, rejecting an empty stream once at construction time. -/
def cycleOrError (stream : SampleStream α) (error : String := "empty sample stream") :
    Except String (Nat → α) :=
  if h : 0 < stream.size then .ok (stream.cycle h) else .error error

end SampleStream

end TorchLean.Data
