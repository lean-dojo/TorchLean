/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Finite Sample Streams

This module provides the lower-level data source used by the trainer API. A `SampleStream α`
records a finite number of samples and computes a sample only when its index is requested. Array-
backed data is wrapped without copying, while generated data and tensor slices remain lazy.

Application code normally constructs `TorchLean.Trainer.Dataset`; this module is useful for manual
training loops and data-source implementations that already know their runtime scalar type.
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

/-- Construct a finite stream from an index function. -/
def fromFunction (size : Nat) (get : Fin size → α) : SampleStream α :=
  { size, get }

/-- Wrap an array as a finite stream without copying it. -/
def fromArray (xs : Array α) : SampleStream α :=
  fromFunction xs.size fun i => xs[i]

/-- Materialize the samples in index order. -/
def toArray (stream : SampleStream α) : Array α :=
  Array.ofFn stream.get

/-- A stream built from an array reports that array's length. -/
@[simp] theorem size_fromArray (xs : Array α) :
    (fromArray xs).size = xs.size := rfl

/-- Return `true` exactly when the stream contains no samples. -/
def isEmpty (stream : SampleStream α) : Bool :=
  stream.size = 0

/-- Safely request a sample by a natural-number index. -/
def get? (stream : SampleStream α) (i : Nat) : Option α :=
  if h : i < stream.size then some (stream.get ⟨i, h⟩) else none

/-- Transform samples when they are requested. -/
def map (f : α → β) (stream : SampleStream α) : SampleStream β :=
  fromFunction stream.size fun i => f (stream.get i)

/-- Append two finite streams in index order. -/
def append (xs ys : SampleStream α) : SampleStream α :=
  fromFunction (xs.size + ys.size) fun i =>
    if h : i.val < xs.size then
      xs.get ⟨i.val, h⟩
    else
      have hle : xs.size ≤ i.val := Nat.le_of_not_gt h
      have hlt : i.val - xs.size < ys.size :=
        (Nat.sub_lt_iff_lt_add' hle).2 i.isLt
      ys.get ⟨i.val - xs.size, hlt⟩

/-- Split a stream into its first `n` samples and the remaining suffix. -/
def splitAt (n : Nat) (stream : SampleStream α) : SplitResult α :=
  let prefixSize := min n stream.size
  let selected := fromFunction prefixSize fun i =>
    stream.get ⟨i.val, Nat.lt_of_lt_of_le i.isLt (Nat.min_le_right _ _)⟩
  let remaining := fromFunction (stream.size - prefixSize) fun i =>
    have hlt : prefixSize + i.val < stream.size := by
      simpa [Nat.add_comm] using Nat.add_lt_of_lt_sub i.isLt
    stream.get ⟨prefixSize + i.val, hlt⟩
  { selected, remaining }

/--
Shuffle the indices deterministically, returning the next seed and a stream with the new order.

Only the index permutation is stored; requesting a shuffled sample still evaluates the original
stream at that index.
-/
def shuffle (seed : Nat) (stream : SampleStream α) : ShuffleResult α :=
  let next (state : Nat) := (1103515245 * state + 12345) % 2147483648
  let sourceIndices : Array (Fin stream.size) := Array.ofFn id
  let (seed', keyedIndices) := sourceIndices.foldl (fun (state, acc) index =>
    let state' := next state
    (state', acc.push (state', index))) (seed, #[])
  let indices := (keyedIndices.qsort (fun x y => x.1 < y.1)).map Prod.snd
  { nextSeed := seed'
    stream := fromFunction indices.size fun i => stream.get indices[i] }

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

/-- Split a nonempty stream into consecutive nonempty batches of size at most `batchSize`. -/
def batches (name : String) (batchSize : Nat) (stream : SampleStream α) :
    Except String (Array (Array α)) :=
  if batchSize = 0 then
    .error s!"{name}: batchSize must be > 0"
  else if stream.isEmpty then
    .error s!"{name}: empty sample stream"
  else
    let count := (stream.size + batchSize - 1) / batchSize
    let samples := stream.toArray
    .ok <| (Array.range count).map fun i =>
      samples.extract (i * batchSize) (min ((i + 1) * batchSize) samples.size)

/-- Cycle through a nonempty stream indefinitely. -/
def cycle (stream : SampleStream α) (h : 0 < stream.size) (i : Nat) : α :=
  stream.get ⟨i % stream.size, Nat.mod_lt _ h⟩

/-- Build a cycling sample function, rejecting an empty stream once at construction time. -/
def cycleOrError (stream : SampleStream α) (error : String := "empty sample stream") :
    Except String (Nat → α) :=
  if h : 0 < stream.size then .ok (stream.cycle h) else .error error

end SampleStream

/-- Configuration and deterministic shuffle state for epoch traversal. -/
structure EpochLoader (α : Type) where
  /-- Samples traversed by each epoch. -/
  samples : SampleStream α
  /-- Maximum number of samples in each batch. -/
  batchSize : Nat
  /-- Whether to shuffle before each epoch. -/
  shuffle : Bool := false
  /-- Seed threaded through deterministic shuffles. -/
  seed : Nat := 0
  /-- Whether to discard a final batch shorter than `batchSize`. -/
  dropLast : Bool := false

/-- Batches produced for one epoch together with the loader state for the next epoch. -/
structure Epoch (Loader Batch : Type) where
  /-- Loader state to use for the next epoch. -/
  nextLoader : Loader
  /-- Batches produced by this epoch. -/
  batches : Array Batch

namespace EpochLoader

variable {α : Type}

/-- Construct an epoch loader for a finite stream. -/
def fromStream (samples : SampleStream α) (batchSize : Nat)
    (shuffle : Bool := false) (seed : Nat := 0)
    (dropLast : Bool := false) : EpochLoader α :=
  { samples, batchSize, shuffle, seed, dropLast }

/-- Produce one epoch of batches and the loader state for the next epoch.
Each shuffle applies the next seed to the original source. Keeping that source avoids retaining
an additional permutation closure for every completed epoch. -/
def nextEpoch (name : String) (loader : EpochLoader α) :
    Except String (Epoch (EpochLoader α) (Array α)) := do
  let shuffled :=
    if loader.shuffle then
      SampleStream.shuffle loader.seed loader.samples
    else
      { nextSeed := loader.seed, stream := loader.samples }
  let batches ← SampleStream.batches name loader.batchSize shuffled.stream
  let batches :=
    if loader.dropLast then
      batches.filter fun batch => batch.size = loader.batchSize
    else
      batches
  pure
    { nextLoader :=
        { loader with seed := shuffled.nextSeed }
      batches }

/-- Produce one epoch and map each raw batch through `collate`. -/
def mapNextEpoch {β : Type} (name : String) (loader : EpochLoader α)
    (collate : Array α → Except String β) :
    Except String (Epoch (EpochLoader α) β) := do
  let result ← nextEpoch name loader
  pure
    { nextLoader := result.nextLoader
      batches := ← result.batches.mapM collate }

end EpochLoader

end TorchLean.Data
