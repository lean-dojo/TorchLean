/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Sample
public import NN.Data.SampleStream
public import NN.Tensor.Operations
public import NN.API.Data.Sources -- shake: keep

/-!
# Typed Data Loaders

Typed collation for low-level manual loops. Application code should usually construct a
`Trainer.Dataset` and use `trainer.train`.
-/

@[expose] public section

namespace TorchLean
namespace Data

/--
Shape-typed loader for supervised samples.

The batch size `batch` appears in the type. Every emitted sample therefore contains collated tensors
whose leading dimension is `batch`. Partial final batches are omitted because they have a different
shape.
-/
structure Loader (α : Type) [TorchLean.Storage α] (batch : Nat)
    (input target : Spec.Shape) where
  /-- Uncollated supervised samples. -/
  samples : SampleStream (TorchLean.Sample.Supervised α input target)
  /-- Whether to shuffle before each epoch. -/
  shuffle : Bool := false
  /-- Seed threaded through deterministic epoch shuffles. -/
  seed : Nat := 0

/-- Collate a statically bounded sample function without an intermediate array. -/
def Internal.collateSupervisedFn {α : Type} [TorchLean.Storage α]
    {σ τ : Spec.Shape} {n : Nat}
    (sample : Fin n → TorchLean.Sample.Supervised α σ τ) :
    TorchLean.Sample.Batch α n σ τ :=
  { input := Tensor.stackLeading fun i => (sample i).input
    target := Tensor.stackLeading fun i => (sample i).target }

/--
Collate `n` supervised samples into one sample with a leading batch axis.

For samples with shapes `input` and `target`, the result contains:

- `input : Tensor α (input.prependDim n)`;
- `target : Tensor α (target.prependDim n)`.

Example:
```lean
-- Three samples in, one batched sample out, with the batch size checked against the array length.
def batched (samples : Array (Sample.Supervised Float [2] [1])) :
    Except String (Sample.Batch Float 3 [2] [1]) :=
  Data.collateSupervised 3 samples
```
-/
def collateSupervised {α : Type} [TorchLean.Storage α]
    {σ τ : Spec.Shape} (n : Nat)
    (batch : Array (TorchLean.Sample.Supervised α σ τ)) :
    Except String (TorchLean.Sample.Batch α n σ τ) := do
  if h : batch.size = n then
    pure <| Internal.collateSupervisedFn fun i =>
      batch[i.val]'(by simp [h])
  else
    throw s!"collate: expected batch size {n}, got {batch.size}"

/--
Turn a per-sample supervised stream into a stream of fixed-size minibatches.

This is useful for metrics and manual loops when a model expects a leading
batch axis.

Notes:
- This drops the final partial batch (PyTorch `drop_last=True` behavior).
- Batches are formed in dataset order (shuffling is the loader's job).
- Samples are read and collated only when the corresponding batch is requested.
-/
def collateStream {α : Type} [TorchLean.Storage α]
    {σ τ : Spec.Shape} (n : Nat)
    (stream : TorchLean.Data.SampleStream
      (TorchLean.Sample.Supervised α σ τ)) :
    Except String (TorchLean.Data.SampleStream
      (TorchLean.Sample.Batch α n σ τ)) := do
  if n = 0 then
    throw "batched: batch size must be > 0"
  if stream.isEmpty then
    throw "batched: empty sample stream"
  pure <| SampleStream.fromFunction (stream.size / n) fun batchIndex =>
    let samples := Array.ofFn fun sampleIndex : Fin n =>
      have batchBound : (batchIndex.val + 1) * n ≤ stream.size :=
        Nat.le_trans
          (Nat.mul_le_mul_right n (Nat.succ_le_of_lt batchIndex.isLt))
          (Nat.div_mul_le_self stream.size n)
      have bound : batchIndex.val * n + sampleIndex.val < stream.size := by
        simp only [Nat.add_mul, Nat.one_mul] at batchBound
        omega
      stream.get ⟨batchIndex.val * n + sampleIndex.val, bound⟩
    Internal.collateSupervisedFn fun i : Fin n =>
      samples[i.val]'(by simp [samples])

namespace Loader

/-- Run one epoch: return the updated loader state and an array of typed minibatches. -/
def nextEpoch {α : Type} [TorchLean.Storage α]
    {n : Nat} {σ τ : Spec.Shape}
    (name : String) (loader : Loader α n σ τ) :
    Except String (TorchLean.Data.Epoch
      (Loader α n σ τ)
      (TorchLean.Sample.Batch α n σ τ)) := do
  let raw := EpochLoader.fromStream loader.samples n
    (shuffle := loader.shuffle) (seed := loader.seed) (dropLast := true)
  let result ←
    TorchLean.Data.EpochLoader.mapNextEpoch name raw
      (fun batch => collateSupervised (α := α) (σ := σ) (τ := τ) n batch)
  pure
    { nextLoader :=
        { samples := result.nextLoader.samples
          shuffle := loader.shuffle
          seed := result.nextLoader.seed }
      batches := result.batches }

/-- Produce the next typed epoch and map each minibatch through `f`. -/
def mapNextEpoch {α β : Type} [TorchLean.Storage α]
    {n : Nat} {σ τ : Spec.Shape}
    (name : String) (loader : Loader α n σ τ)
    (f : TorchLean.Sample.Batch α n σ τ → Except String β) :
    Except String (TorchLean.Data.Epoch
      (Loader α n σ τ) β) := do
  let result ← nextEpoch (α := α) (σ := σ) (τ := τ) name loader
  pure
    { nextLoader := result.nextLoader
      batches := ← result.batches.mapM f }

/--
Run one epoch and require at least one full typed minibatch.

This is the shared checked boundary for examples that need a nonempty array of full batches. It
keeps the "drop partial batches, but fail if nothing remains" policy with the loader API rather than
repeating it in each dataset-specific helper.
-/
def nextNonemptyEpoch {α : Type} [TorchLean.Storage α]
    {n : Nat} {σ τ : Spec.Shape}
    (name : String) (loader : Loader α n σ τ) :
    Except String (TorchLean.Data.Epoch
      (Loader α n σ τ)
      (TorchLean.Sample.Batch α n σ τ)) := do
  let result ← nextEpoch (α := α) (σ := σ) (τ := τ) name loader
  if result.batches.isEmpty then
    throw
      s!"{name}: no full minibatch available (batch={n}, rows={loader.samples.size})"
  else
    pure result

/-- Return the first full typed minibatch without materializing the rest of the epoch. -/
def firstFullBatch {α : Type} [TorchLean.Storage α]
    {n : Nat} {σ τ : Spec.Shape}
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

/--
Build a fixed-size supervised loader for a manual training loop.

The underlying stream stores individual samples. Each epoch emits tensors with a leading batch
axis, omitting any final partial batch.
-/
def Loader.fromStream {α : Type} [TorchLean.Storage α]
    {σ τ : Spec.Shape}
    (samples : SampleStream (TorchLean.Sample.Supervised α σ τ))
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) :
    Loader α batchSize σ τ :=
  { samples, shuffle, seed }

end Data
end TorchLean
