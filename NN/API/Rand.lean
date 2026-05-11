/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Core
public import NN.Runtime.Autograd.TorchLean.Random

/-!
# API Rand

Deterministic RNG helpers.

TorchLean treats randomness explicitly (via seeds/keys) so examples are reproducible.

PyTorch mapping (rough):
- `torch.Generator` and seed management
- `torch.rand` / Bernoulli masks for dropout
-/

@[expose] public section

namespace NN
namespace API

namespace rand

export _root_.Runtime.Autograd.TorchLean.Random (keyOf nextSeed uniform mask)

/-!
Convenience wrappers for "PyTorch-like" dimension-first workflows.

`uniform` / `mask` are shape-indexed (`Tensor α s`) and are best used when `s` is already inferred.
When you start from a runtime dims list (e.g. CLI args), `uniformND`/`maskND` are the ergonomic
bridge.
-/

/-- Dimension-first wrapper: uniform random tensor at runtime `dims`. -/
def uniformND {α : Type} [Context α] (key : UInt64) (dims : List Nat) :
    Spec.Tensor α (NN.Tensor.shapeOfDims dims) :=
  uniform (α := α) key (s := NN.Tensor.shapeOfDims dims)

/--
Dimension-first wrapper: Bernoulli keep-mask at runtime `dims` (useful for dropout-style masks).
-/
def maskND {α : Type} [Context α] (key : UInt64) (keepProb : α) (dims : List Nat) :
    Spec.Tensor α (NN.Tensor.shapeOfDims dims) :=
  mask (α := α) key keepProb (s := NN.Tensor.shapeOfDims dims)

/-- Deterministic uniform tensor at runtime `dims`, using `(seed,counter)` to derive a key. -/
def randND {α : Type} [Context α] (seed counter : Nat) (dims : List Nat) :
    Spec.Tensor α (NN.Tensor.shapeOfDims dims) :=
  uniformND (α := α) (keyOf seed counter) dims

/-
Seed management note:

TorchLean’s core is pure/seed-threaded (JAX-style). For ergonomic model-building (more PyTorch-like),
we provide a compact “seed stream” abstraction so you can pass *one* base seed and allocate per-layer
seeds deterministically.
-/

/--
Deterministic seed stream (seed + monotone counter).

This is intended for *model construction* (parameter init keys, dropout keys, etc.) where you want
PyTorch-like ergonomics but reproducible results.
-/
structure SeedStream where
  /-- Base seed (think `torch.manual_seed`). -/
  seed : Nat
  /-- Monotone counter (ensures each draw is distinct). -/
  counter : Nat := 0
deriving Repr, DecidableEq, Inhabited

namespace SeedStream

/-- Create a fresh stream from a base seed. -/
abbrev init (seed : Nat) : SeedStream :=
  { seed := seed, counter := 0 }

/--
Draw a fresh seed and advance the stream.

Implementation: we reuse `TorchLean.Random.nextSeed` as a small deterministic mixing function.
-/
def next (s : SeedStream) : Nat × SeedStream :=
  let out := nextSeed s.seed s.counter
  (out, { s with counter := s.counter + 1 })

/-- Draw `n` fresh seeds. -/
def nextN (n : Nat) (s : SeedStream) : List Nat × SeedStream :=
  let rec go : Nat → SeedStream → List Nat → (List Nat × SeedStream)
    | 0, st, acc => (acc.reverse, st)
    | k + 1, st, acc =>
        let (x, st') := next st
        go k st' (x :: acc)
  go n s []

end SeedStream

universe u

/--
State monad for deterministic seed allocation.

Lean's `StateT/StateM` ties the state/result universes together, while TorchLean model definitions
(e.g. `nn.Sequential`) live above `Type 0`.

So we define this seed builder directly (as a pure state monad), and run it in `Type 2`, which is
sufficient for the public API.
-/
abbrev SeedM (α : Type u) : Type u :=
  SeedStream → (α × SeedStream)

namespace SeedM

instance : Monad SeedM where
  pure x := fun st => (x, st)
  bind x f := fun st =>
    let (a, st') := x st
    f a st'

end SeedM

/--
Global seed stream used by `rand.runGlobal` / `nn.runGlobal`.

This is a convenience for script-like code that wants PyTorch-style "set the seed once" ergonomics.
In proofs and reproducibility-sensitive code, prefer the pure interfaces (`nn.run`) and pass the
base seed explicitly.
-/
initialize globalSeedStream : IO.Ref SeedStream ← IO.mkRef (SeedStream.init 0)

/--
Reset the global seed stream.

PyTorch analogue: `torch.manual_seed`.
-/
def manualSeed (seed : Nat) : IO Unit := do
  globalSeedStream.set (SeedStream.init seed)

/--
Run a seeded builder using the global seed stream and advance it.

This lets you build multiple models/layers in `IO` without explicitly threading seeds, while still
remaining deterministic.
-/
def runGlobal {α : Type} (x : SeedM α) : IO α := do
  let st ← globalSeedStream.get
  let (a, st') := x st
  globalSeedStream.set st'
  pure a

/-- Draw one fresh seed from the global seed stream. -/
def nextSeedGlobal : IO Nat :=
  runGlobal SeedStream.next

/-- Draw `n` fresh seeds from the global seed stream. -/
def nextSeedsGlobal (n : Nat) : IO (List Nat) :=
  runGlobal (SeedStream.nextN n)

end rand

end API
end NN
