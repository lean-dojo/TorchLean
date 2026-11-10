/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.Datatype

/-!
# Random

Deterministic pseudorandom draws: each one is a pure function of a seed, a counter and a linear
index, in the functional style of JAX's keyed generators.

## Seeded draws and `IO`

Lean can generate random numbers in `IO`, but the results depend on hidden runtime state, which is
a poor fit for TorchLean. Lowering and verification share one semantics, and a graph stops being a
pure mathematical object unless the RNG state is modeled. Replays and certificate checks would
depend on that state, and typed graph recording would have to handle effects and mixed dtypes.
Seeded draws keep the semantic core pure.

## Main definitions

- `splitmix64`, one SplitMix64 step on 64-bit words, the odd counter step `counterStep`, and the
  keys and seeds built on them (`keyOf`, `nextSeed`)
- `sampleNat`, scalar draws keyed by `(seed, counter, linearIndex)`, and `sampleUnit` and
  `keepBit`, which turn a draw into a fraction of its range and into a keep bit
- `uniform`, `normal` and `mask`, shape-indexed tensors of uniform draws, approximate standard
  normal draws (Box-Muller) and dropout-style `{0,1}` masks

`keyOf` applies `splitmix64` to the Weyl sequence `seed + counter * counterStep` in `UInt64`, and
`nextSeed` to the same sequence offset by `0x9e3779b97f4a7c15`. With the odd step `counterStep`,
two distinct seeds at most `10 ^ 5` apart give `splitmix64` the same input only at counters at
least `87435955166711` apart. That bound is a computation over the seed offsets `1` to `10 ^ 5`
with the inverse of `counterStep` modulo `2 ^ 64`. No theorem in this library states it.

We copied `splitmix64` by hand from Sebastiano Vigna's `splitmix.c`, "a fixed-increment version
of Java 8's SplittableRandom generator" in his words, the generator of Steele, Lea and Flood
(2014). The mixer, shifts `30`, `27` and `31` with its two multipliers, is the paper's
`mix64variant13` (Figure 16), David Stafford's Mix13. So `splitmix64 0` is `0xe220a8397b1dcdaf`,
the first output of Vigna's generator from the state `0`. Our keys don't split the way the paper's
generators do: a key is `splitmix64` of the Weyl sequence above. The capture test
`test_random_operators_use_keyed_streams` in `scripts/checks/test_ir_capture.py` recomputes the
uniform draws with its own copy of the mixer.

The module lives in the spec layer because the registered operators `Op.randUniform`, `Op.randn`
and `Op.bernoulliMask` (`TorchLean.Tensor.Op.Table`) denote pure tensors through it. Runtime
code calls these same definitions: an eager session counts the random tensors it draws
(`rngCounter`) and derives each one's key from that count, so its masks are reproducible.

For PyTorch-like randomness at the boundary, sample an initial seed in `IO` once and pass it as the
runtime `seed`. Everything after that point is seeded.

## References

- Guy L. Steele Jr., Doug Lea and Christine H. Flood (2014), "Fast splittable pseudorandom number
  generators", Proceedings of the 2014 ACM International Conference on Object Oriented Programming
  Systems Languages & Applications (OOPSLA), 453-472, `https://doi.org/10.1145/2660193.2660195`.
- Sebastiano Vigna (2015), `splitmix.c`: `https://prng.di.unimi.it/splitmix.c`
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape.Coord (linearize)

namespace TorchLean.Random

open TorchLean.Tensor
open Shape (Coord)

/-! ### SplitMix64-style mixing -/

/--
One step of SplitMix64 from the state `x`: add the increment `0x9e3779b97f4a7c15`, then mix the
sum with two xor-shift-multiply rounds and a final xor-shift. It is the mixing core of every key,
seed and draw below.
-/
def splitmix64 (x : UInt64) : UInt64 :=
  let z1 := x + 0x9e3779b97f4a7c15
  let z2 := (z1 ^^^ (z1 >>> 30)) * 0xbf58476d1ce4e5b9
  let z3 := (z2 ^^^ (z2 >>> 27)) * 0x94d049bb133111eb
  z3 ^^^ (z3 >>> 31)

/-- The odd constant by which the counter steps the Weyl sequence of `keyOf` and `nextSeed`, as
SplitMix64 steps its state by `0x9e3779b97f4a7c15`. -/
def counterStep : UInt64 := 0xd1b54a32d192ed03

/-- The key of draw `counter` under `seed`: `splitmix64` of `seed + counter * counterStep` in
`UInt64`. As `counterStep` is odd, one seed gives `splitmix64` distinct inputs at counters that
differ modulo `2 ^ 64`, and two distinct seeds at most `10 ^ 5` apart give it the same input only
at counters at least `87435955166711` apart. Neither fact is a theorem here: the first holds as
an odd number is a unit modulo `2 ^ 64`, the second is the computation of the module doc. -/
def keyOf (seed counter : Nat) : UInt64 :=
  splitmix64 (UInt64.ofNat seed + UInt64.ofNat counter * counterStep)

/-- A fresh seed from `seed` and `counter`, as a natural number: the SplitMix64 output one step
after `keyOf seed counter`. -/
def nextSeed (seed counter : Nat) : Nat :=
  (splitmix64 (UInt64.ofNat seed + UInt64.ofNat counter * counterStep + 0x9e3779b97f4a7c15)).toNat

/-! ### Sampling helpers -/

/-- A pseudorandom natural number in `[0, denom)` for `denom > 0`: `splitmix64` of `key` plus the
linear index, reduced modulo `denom`. At the default `2 ^ 32` each residue has `2 ^ 32` preimages
among the 64-bit words. For another `denom` the residues below `2 ^ 64 % denom` get one more, the
usual modulo bias, which we accept: at `denom = 3`, `0` has `6148914691236517206` preimages and
`1` and `2` one fewer. -/
def sampleNat (key : UInt64) (linearIndex : Nat) (denom : Nat := (2:Nat) ^ 32) : Nat :=
  (splitmix64 (key + UInt64.ofNat linearIndex)).toNat % denom

/-- The fraction `u / denom`, formed exactly and rounded into the scalar type.

Forming the fraction before the cast keeps either integer from overflowing a low-precision format.
Rounding can send a draw below `1` to the endpoint `1`. At `denom = 0` the result is the
backend's own `u / 0`, and the generators always pass a positive denominator. -/
def sampleUnit {α : Type} [Context α] (u denom : Nat) : α :=
  if denom = 0 then (u : α) / (denom : α)
  else Rat.cast ((u : Rat) / (denom : Rat))

/-- The keep bit of a draw `u ∈ [0, denom)`: `1` when the converted draw `sampleUnit u denom` is
below `keepProb`, and `0` otherwise.

The endpoints come first: a probability that is not positive drops every element and one of `1`
or more keeps every element. Binary32 can round the largest 32-bit draw to `1`, so comparing that
draw with a keep probability of `1` would drop it. -/
def keepBit {α : Type} [Context α] (keepProb : α) (u denom : Nat) : α :=
  if keepProb > 0 then
    if (1 : α) > keepProb then
      if keepProb > sampleUnit (α := α) u denom then 1 else 0
    else 1
  else 0

/-! ### Uniform tensors -/

/-- A tensor of deterministic unit draws: the entry at a coordinate is the draw of `key` at its
row-major index, its exact fraction `u / 2 ^ 32` cast to the backend, which can round it to `1`. -/
def uniform {α : Type} [Storage α] [Context α] (key : UInt64) {s : Shape} :
    Tensor α s :=
  Tensor.Rep.ofFn fun coordinate =>
    let denom : Nat := (2:Nat) ^ 32
    let u := sampleNat key (Coord.linearize coordinate).val denom
    sampleUnit (α := α) u denom

/-! ### Mask construction -/

/-- A dropout-style mask: the entry at a coordinate is `keepBit keepProb u (2 ^ 32)` for the draw
`u` of `key` at its row-major index, one for a kept coordinate and zero for a dropped one. -/
def mask {α : Type} [Storage α] [Context α]
    (key : UInt64) (keepProb : α) {s : Shape} : Tensor α s :=
  Tensor.Rep.ofFn fun coordinate =>
    let denom : Nat := (2:Nat) ^ 32
    let u := sampleNat key (Coord.linearize coordinate).val denom
    keepBit (α := α) keepProb u denom

/-! ### Standard normal tensors -/

/--
The cosine branch of the Box-Muller transform, `z = sqrt(-2 * log u1) * cos(2π * u2)`, which
turns two independent uniforms `u1,u2 ∈ (0,1)` into a standard normal sample.

We clamp `u1` below by the backend's default safeguard (`Context.defaultEpsilon`). That avoids
`log 0` only when the safeguard is positive and the backend's `max` selects it, a law `Context`
alone does not give. The sampler is deterministic and meant for examples and benchmarks. It is
not a cryptographic generator. Box and Muller give a pair of normals from `u1` and `u2`, the cosine
one and a sine one, and we keep the cosine one, so `normalScalar` spends two draws per sample.
The capture test `test_randn_uses_a_keyed_normal_sampler` (`scripts/checks/test_ir_capture.py`)
compares `normal` with a binary64 evaluation of this formula on the same rounded draws, to within
a tolerance of the dtype.

Reference: G. E. P. Box and Mervin E. Muller (1958), "A Note on the Generation of Random Normal
Deviates", The Annals of Mathematical Statistics 29(2), 610-611,
`https://doi.org/10.1214/aoms/1177706645`.
-/
def boxMullerCos {α : Type} [Context α] (u1 u2 : α) : α :=
  let u1' := Max.max u1 Context.defaultEpsilon
  let r : α := MathFunctions.sqrt (-2 * MathFunctions.log u1')
  let theta : α := (2 * MathFunctions.pi) * u2
  r * MathFunctions.cos theta

/--
An approximate standard normal sample: `boxMullerCos` of the 32-bit uniform draws of `key` at the
indices `2 * linearIndex` and `2 * linearIndex + 1`. The finite grid, the clamp and the backend's
arithmetic keep the law from being exactly `N(0,1)`.
-/
def normalScalar {α : Type} [Context α] (key : UInt64) (linearIndex : Nat) :
    α :=
  let denom : Nat := (2:Nat) ^ 32
  let u1n := sampleNat key (2 * linearIndex) denom
  let u2n := sampleNat key (2 * linearIndex + 1) denom
  let u1 : α := sampleUnit (α := α) u1n denom
  let u2 : α := sampleUnit (α := α) u2n denom
  boxMullerCos (α := α) u1 u2

/-- A tensor of approximate standard-normal draws: the entry at a coordinate is
`normalScalar key` at its row-major index. -/
def normal {α : Type} [Storage α] [Context α]
    (key : UInt64) {s : Shape} : Tensor α s :=
  Tensor.Rep.ofFn fun coordinate =>
    normalScalar (α := α) key (Coord.linearize coordinate).val

end TorchLean.Random
