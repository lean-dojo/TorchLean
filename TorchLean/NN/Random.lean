/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Random

/-!
# Random Seeds

Deterministic seeds for model construction and dropout. The core treats randomness JAX-style, as
a pure function of a seed and a counter, so the same seeds give the same parameters and masks on
every run. `SeedStream` hands out per-layer seeds from one base seed, `nn.Builder` threads a
stream through model construction, and `manualSeed` and `runGlobal` give script-style code one
global stream.

PyTorch mapping:
- `torch.Generator` and seed management: `SeedStream` and `manualSeed`
- `torch.rand` and Bernoulli dropout masks: `uniform` and `mask`
-/

@[expose] public section

namespace TorchLean.Random

/--
A seed stream for model construction: draw `counter` of the base seed `seed` hands out
`Random.nextSeed seed counter`.
-/
structure SeedStream where
  /-- The base seed (the argument of `torch.manual_seed`). -/
  seed : Nat
  /-- The counter of the next draw, which `next` advances by one. -/
  counter : Nat := 0
deriving Repr, DecidableEq, Inhabited

namespace SeedStream

/-- Return `Random.nextSeed stream.seed stream.counter` and the stream with its counter
advanced by one. -/
def next (stream : SeedStream) : Nat × SeedStream :=
  let seed := nextSeed stream.seed stream.counter
  (seed, { stream with counter := stream.counter + 1 })

end SeedStream

end TorchLean.Random

namespace TorchLean.nn

universe u

/--
Deterministic model builder that threads the initialization seed stream, written as a plain
function `SeedStream → α × SeedStream`. `StateM` would put the result in the universe of the
state, while model types such as `nn.Sequential` live in `Type 1`. The function form keeps the
result in universe `u`.
-/
abbrev Builder (α : Type u) : Type u :=
  Random.SeedStream → (α × Random.SeedStream)

namespace Builder

instance : Monad Builder where
  pure value := fun stream => (value, stream)
  bind computation continuation := fun stream =>
    let (value, nextStream) := computation stream
    continuation value nextStream

end Builder

end TorchLean.nn

namespace TorchLean.Random

/--
The process-wide seed stream behind `Random.runGlobal` and `nn.buildIO`, starting at base seed `0`.
It gives scripts PyTorch's "set the seed once" workflow. Proofs, and code whose results must not
depend on call order, use `nn.build` with a base seed instead.
-/
initialize globalSeedStream : IO.Ref SeedStream ← IO.mkRef ({ seed := 0 } : SeedStream)

/-- Restart the global seed stream at base seed `seed` (PyTorch: `torch.manual_seed`). -/
def manualSeed (seed : Nat) : IO Unit := do
  globalSeedStream.set { seed := seed }

/--
Run an `nn.Builder` computation on the global seed stream and store the advanced stream. Successive
calls build several models or layers in `IO` without passing seeds around, and after the same
`manualSeed` they draw the same seeds in the same order.
-/
def runGlobal {α : Type} (computation : nn.Builder α) : IO α := do
  let stream ← globalSeedStream.get
  let (value, nextStream) := computation stream
  globalSeedStream.set nextStream
  pure value

/-- Draw one fresh seed from the global seed stream. -/
def nextSeedGlobal : IO Nat :=
  runGlobal SeedStream.next

end TorchLean.Random
