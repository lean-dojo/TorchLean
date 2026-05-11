/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core

/-!
# Dataset and data loader utilities

This module defines a small in-memory dataset wrapper together with a deterministic (seeded) loader.
The shuffling logic is pure and reproducible.

Design boundary:
- this is not a replacement for high-throughput framework data loaders;
- it is a small, auditable loader for examples, tests, and proof-adjacent training loops;
- large or streaming datasets should sit above this layer and feed already-collated tensors into
  the runtime.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

/-!
## Dataset
-/
/-- A lightweight in-memory dataset wrapper backed by an `Array`. -/
structure Dataset (a : Type) where
  /-- Samples stored in insertion order. -/
  data : Array a

namespace Dataset

variable {a b : Type}

/-- Construct a dataset from a list. -/
def ofList (xs : List a) : Dataset a :=
  { data := xs.toArray }

/-- Convert a dataset to a list. -/
def toList (ds : Dataset a) : List a :=
  ds.data.toList

/-- Round-tripping a list through `Dataset` preserves order. -/
@[simp] theorem toList_ofList (xs : List a) :
    (Dataset.ofList xs).toList = xs := by
  simp [toList, ofList]

/-- Number of samples in the dataset. -/
def size (ds : Dataset a) : Nat :=
  ds.data.size

/-- `ofList` preserves the list length as dataset size. -/
@[simp] theorem size_ofList (xs : List a) :
    (Dataset.ofList xs).size = xs.length := by
  simp [size, ofList]

/-- Return `true` iff the dataset has no samples. -/
def isEmpty (ds : Dataset a) : Bool :=
  ds.data.isEmpty

/-- A dataset built from a nonempty cons list is not empty. -/
@[simp] theorem isEmpty_ofList_cons (x : a) (xs : List a) :
    (Dataset.ofList (x :: xs)).isEmpty = false := by
  simp [isEmpty, ofList]

/-- Safe indexing into the dataset. -/
def get? (ds : Dataset a) (i : Nat) : Option a :=
  ds.data[i]?

/-- Map a function over all samples in the dataset. -/
def map (f : a -> b) (ds : Dataset a) : Dataset b :=
  { data := ds.data.map f }

/-- Append two datasets. -/
def append (x y : Dataset a) : Dataset a :=
  { data := x.data ++ y.data }

/-- Split a dataset at index `n` (preserving order). -/
def splitAt (n : Nat) (ds : Dataset a) : Prod (Dataset a) (Dataset a) :=
  let (l, r) := ds.toList.splitAt n
  (ofList l, ofList r)

/-!
## Deterministic shuffle

We use a small LCG-based shuffle for reproducibility without IO.
-/
/-- One step of the simple LCG used to generate a deterministic pseudo-random stream. -/
def lcgNext (s : Nat) : Nat :=
  (1103515245 * s + 12345) % 2147483648

/--
Pair each element with a deterministic pseudo-random key.

We then sort by the key to obtain a deterministic permutation.
-/
def shufflePairs (seed : Nat) (xs : List a) : Prod Nat (Array (Prod Nat a)) :=
  xs.foldl (fun (s, acc) x =>
    let s' := lcgNext s
    (s', acc.push (s', x))) (seed, #[])

/--
Deterministically shuffle a dataset.

Returns the next seed along with the shuffled dataset, so repeated epochs can thread the seed and
remain pure/replayable.
-/
def shuffle (seed : Nat) (ds : Dataset a) : Prod Nat (Dataset a) :=
  let (seed', pairs) := shufflePairs seed ds.toList
  let pairs' := pairs.qsort (fun a b => a.fst < b.fst)
  let ys := pairs'.toList.map (fun p => p.snd)
  (seed', ofList ys)

/-!
## Batching

These helpers return batches as lists for easy use with TapeM utilities.
-/
/--
Split a dataset into non-empty batches of size at most `batchSize`.

Errors if `batchSize = 0` or the dataset is empty.
-/
def batches (tag : String) (batchSize : Nat) (ds : Dataset a) : Result (List (List a)) := by
  if batchSize = 0 then
    exact .error (tagError tag "batchSize must be > 0")
  else
    let xs := ds.toList
    if xs.isEmpty then
      exact .error (tagError tag "empty dataset")
    else
      let n := xs.length
      let numBatches := (n + batchSize - 1) / batchSize
      let bs := (List.range numBatches).map (fun i =>
        (xs.drop (i * batchSize)).take batchSize)
      exact .ok bs

/-- Like `batches`, but return each batch as an `Array`. -/
def batchesArray (tag : String) (batchSize : Nat) (ds : Dataset a) : Result (List (Array a)) := do
  let bs <- batches (tag := tag) batchSize ds
  pure (bs.map (fun b => b.toArray))

end Dataset

/-!
## DataLoader

`DataLoader.epoch` optionally shuffles and returns a list of batches.
-/
/--
Deterministic data loader configuration.

`epoch` threads the seed and (optionally) shuffles to produce a list of batches. This is a
deliberately small, pure analogue of a PyTorch `DataLoader`.
-/
structure DataLoader (a : Type) where
  /-- Dataset to batch. The loader keeps this value pure and explicit. -/
  dataset : Dataset a
  /-- Batch size. -/
  batchSize : Nat
  /-- If true, run the deterministic seeded shuffle before each epoch. -/
  shuffle : Bool := false
  /-- Seed threaded through deterministic shuffles. -/
  seed : Nat := 0
  /-- Drop the final partial batch when it is shorter than `batchSize`. -/
  dropLast : Bool := false

namespace DataLoader

variable {a : Type}

/--
Run one epoch: optionally shuffle and return the list of batches.

The returned `DataLoader` has its seed updated so you can call `epoch` repeatedly to get different
but reproducible shuffles.
-/
def epoch (tag : String) (dl : DataLoader a) :
  Result (Prod (DataLoader a) (List (List a))) := do
  let (seed', ds') :=
    if dl.shuffle then
      Dataset.shuffle dl.seed dl.dataset
    else
      (dl.seed, dl.dataset)
  let batches <- Dataset.batches (tag := tag) dl.batchSize ds'
  let batches :=
    if dl.dropLast then
      batches.filter (fun b => b.length = dl.batchSize)
    else
      batches
  let dl' : DataLoader a := { dl with seed := seed', dataset := ds' }
  pure (dl', batches)

/-- Like `epoch`, but return each batch as an `Array`. -/
def epochArray (tag : String) (dl : DataLoader a) :
  Result (Prod (DataLoader a) (List (Array a))) := do
  let (dl', batches) <- epoch (tag := tag) dl
  pure (dl', batches.map (fun b => b.toArray))

/--
Run one epoch and collate each batch into a single value.

This is the main building block for minibatch training where your model expects tensors with a
leading batch axis, but your dataset is stored as individual samples.
-/
def epochCollate {b : Type} (tag : String) (dl : DataLoader a) (collate : List a → Result b) :
    Result (Prod (DataLoader a) (List b)) := do
  let (dl', batches) <- epoch (tag := tag) dl
  let ys <- batches.mapM collate
  pure (dl', ys)

end DataLoader

end Train
end Autograd
end Runtime
