/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Data.Source -- shake: keep

/-!
# Typed Data Loaders

Minibatch collation for manual training loops, with the batch size in the type. `collate` stacks
an array of samples, `collateStream` batches a stream lazily, and a `Loader` yields one epoch at a
time. Most applications build a `Trainer.Dataset` and call `trainer.train`.
-/

@[expose] public section

namespace TorchLean
namespace Data

/--
Shape-typed loader for supervised samples.

The batch size `batch` is part of the type, so every emitted batch has leading dimension `batch`
and a final partial batch, which would have a different shape, is omitted.
-/
structure Loader (α : Type) [TorchLean.Storage α] (batch : Nat)
    (input target : Shape) where
  /-- Uncollated supervised samples. -/
  samples : SampleStream (TorchLean.Sample.Supervised α input target)
  /-- Whether to shuffle before each epoch. -/
  shuffle : Bool := false
  /-- Seed threaded through deterministic epoch shuffles. -/
  seed : Nat := 0

/-- Batches produced for one epoch together with the loader state for the next epoch. -/
structure Epoch (Loader Batch : Type) where
  /-- Loader state to use for the next epoch. -/
  nextLoader : Loader
  /-- Batches produced by this epoch. -/
  batches : Array Batch

/-- Collate a statically bounded sample function without an intermediate array. -/
def collateFn {α : Type} [TorchLean.Storage α]
    {σ τ : Shape} {n : Nat}
    (sample : Fin n → TorchLean.Sample.Supervised α σ τ) :
    TorchLean.Sample.Batch α n σ τ :=
  { input := Tensor.dim fun i => (sample i).input
    target := Tensor.dim fun i => (sample i).target }

/--
Collate `n` supervised samples into one sample with a leading batch axis. The array must hold
exactly `n` samples, and for sample shapes `σ` and `τ` the result has
`input : Tensor α (Shape.dim n σ)` and `target : Tensor α (Shape.dim n τ)`.

Example:
```lean
-- Three samples in, one batched sample out, with the batch size checked against the array length.
def batched (samples : Array (Sample.Supervised Float [2] [1])) :
    Except String (Sample.Batch Float 3 [2] [1]) :=
  Data.collate 3 samples
```
-/
def collate {α : Type} [TorchLean.Storage α]
    {σ τ : Shape} (n : Nat)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) :
    Except String (TorchLean.Sample.Batch α n σ τ) := do
  if h : batch.size = n then
    pure <| collateFn fun i =>
      batch[i.val]'(by simp [h])
  else
    throw s!"collate: expected batch size {n}, got {batch.size}"

/--
Turn a per-sample supervised stream into a stream of `n`-sample minibatches, for metrics and
manual loops whose model expects a leading batch axis.

Batches follow stream order (shuffling is the loader's job), and the final partial batch is
dropped, as with PyTorch's `drop_last=True`. A batch's samples are read and collated only when
that batch is requested. A zero `n` or an empty stream is an error.
-/
def collateStream {α : Type} [TorchLean.Storage α]
    {σ τ : Shape} (n : Nat)
    (stream : TorchLean.Data.SampleStream
      (TorchLean.Sample.Supervised α σ τ)) :
    Except String (TorchLean.Data.SampleStream
      (TorchLean.Sample.Batch α n σ τ)) := do
  if n = 0 then
    throw "batch size must be > 0"
  if stream.isEmpty then
    throw "empty sample stream"
  pure <| SampleStream.mk (stream.size / n) fun batchIndex =>
    let samples := Array.ofFn fun sampleIndex : Fin n =>
      have batchBound : (batchIndex.val + 1) * n ≤ stream.size :=
        Nat.le_trans
          (Nat.mul_le_mul_right n (Nat.succ_le_of_lt batchIndex.isLt))
          (Nat.div_mul_le_self stream.size n)
      have bound : batchIndex.val * n + sampleIndex.val < stream.size := by
        simp only [Nat.add_mul, Nat.one_mul] at batchBound
        omega
      stream.get ⟨batchIndex.val * n + sampleIndex.val, bound⟩
    collateFn fun i : Fin n =>
      samples[i.val]'(by simp [samples])

namespace Loader

/--
Return one epoch of full typed minibatches and the loader state for the next epoch.

An empty sample stream or a zero batch size is rejected whatever `requireBatch` is. A nonempty
stream with fewer rows than the batch size gives an epoch with no minibatch: the default returns
it, and `requireBatch := true` rejects it with a diagnostic that includes the batch size and row
count.
-/
def nextEpoch {α : Type} [TorchLean.Storage α]
    {n : Nat} {σ τ : Shape}
    (name : String) (loader : Loader α n σ τ) (requireBatch : Bool := false) :
    Except String (Epoch (Loader α n σ τ) (TorchLean.Sample.Batch α n σ τ)) := do
  let shuffled :=
    if loader.shuffle then SampleStream.shuffle loader.seed loader.samples
    else { nextSeed := loader.seed, stream := loader.samples }
  let batches ← (collateStream n shuffled.stream).mapError fun message => s!"{name}: {message}"
  if requireBatch && batches.isEmpty then
    throw s!"{name}: no full minibatch available (batch={n}, rows={loader.samples.size})"
  -- The next loader keeps the unshuffled samples, so epochs never stack permutation closures.
  pure { nextLoader := { loader with seed := shuffled.nextSeed }, batches := batches.toArray }

/-- Return the first full typed minibatch without materializing the rest of the epoch. -/
def firstBatch {α : Type} [TorchLean.Storage α]
    {n : Nat} {σ τ : Shape}
    (name : String) (loader : Loader α n σ τ) :
    Except String (TorchLean.Sample.Batch α n σ τ) := do
  let samples :=
    if loader.shuffle then SampleStream.shuffled loader.seed loader.samples
    else loader.samples
  let batches ← (collateStream n samples).mapError fun message => s!"{name}: {message}"
  match batches.get? 0 with
  | some b => pure b
  | none =>
      throw s!"{name}: no full minibatch available (batch={n}, rows={loader.samples.size})"

end Loader

/-- Build a `Loader` of `batchSize`-sample minibatches over a stream of individual samples. -/
def Loader.fromStream {α : Type} [TorchLean.Storage α]
    {σ τ : Shape}
    (samples : SampleStream (TorchLean.Sample.Supervised α σ τ))
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) :
    Loader α batchSize σ τ :=
  { samples, shuffle, seed }

end Data
end TorchLean
