/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

-- shake: keep-all

public import NN.API.Arithmetic -- shake: keep
public import NN.Spec.Core.Random -- shake: keep
public import NN.Tensor -- shake: keep

/-!
# Random Seeds

Deterministic RNG helpers.

TorchLean treats randomness explicitly (via seeds/keys) so examples are reproducible.

PyTorch mapping:
- `torch.Generator` and seed management
- `torch.rand` / Bernoulli masks for dropout
-/

@[expose] public section

namespace TorchLean.rand

export Spec.Random (keyOf nextSeed uniform mask)

/-
Seed management note:

TorchLean’s core is pure/seed-threaded (JAX-style). For ergonomic model-building (more
PyTorch-like), we provide a compact “seed stream” abstraction so you can pass *one* base seed and
allocate per-layer seeds deterministically.
-/

/--
Deterministic seed stream (seed + monotone counter).

This is intended for *model construction* (parameter init keys, dropout keys, etc.) where you want
PyTorch-like ergonomics but reproducible results.
-/
structure SeedStream where
  /-- Base seed (think `torch.manual_seed`). -/
  seed : Nat
  /-- Monotone counter mixed with the base seed for each draw. -/
  counter : Nat := 0
deriving Repr, DecidableEq, Inhabited

namespace SeedStream

/-- Create a fresh stream from a base seed. -/
abbrev init (seed : Nat) : SeedStream :=
  { seed := seed, counter := 0 }

/--
Draw a fresh seed and advance the stream.

Implementation: we reuse `Spec.Random.nextSeed` as a small deterministic mixing function.
-/
def next (stream : SeedStream) : Nat × SeedStream :=
  let seed := nextSeed stream.seed stream.counter
  (seed, { stream with counter := stream.counter + 1 })

end SeedStream

universe u

/--
State monad for deterministic seed allocation.

Lean's `StateT/StateM` ties the state/result universes together, while TorchLean model definitions
(e.g. `nn.Sequential`) live above `Type 0`.

This pure state-function representation preserves the result universe, including the higher
universe used by model definitions.
-/
abbrev SeedM (α : Type u) : Type u :=
  SeedStream → (α × SeedStream)

namespace SeedM

instance : Monad SeedM where
  pure value := fun stream => (value, stream)
  bind computation continuation := fun stream =>
    let (value, nextStream) := computation stream
    continuation value nextStream

end SeedM

/--
Global seed stream used by `rand.runGlobal` and `nn.withModel`.

This is a convenience for script-like code that wants PyTorch-style "set the seed once" ergonomics.
In proofs and reproducibility-sensitive code, prefer the pure interfaces (`nn.build`) and pass the
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
def runGlobal {α : Type} (computation : SeedM α) : IO α := do
  let stream ← globalSeedStream.get
  let (value, nextStream) := computation stream
  globalSeedStream.set nextStream
  pure value

/-- Draw one fresh seed from the global seed stream. -/
def nextSeedGlobal : IO Nat :=
  runGlobal SeedStream.next

end TorchLean.rand
