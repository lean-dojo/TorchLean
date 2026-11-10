/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Random

/-!
# Keyed Permutations

A keyed partial Fisher-Yates permutation built on `TorchLean.Random.sampleNat`, the one generator
for data shuffles and feature sampling.

We wrote `permutation` by hand after the in-place shuffle of Durstenfeld's Algorithm 235. Ours
walks up from position `0`, and step `i` swaps position `i` with one of the `n - i` positions at
or after it. Stopping after `count` steps leaves `count` distinct indices, a sample without
replacement, which is what feature sampling wants. Step `i` draws `sampleNat key i (n - i)` and so
carries the modulo bias of `sampleNat`. No theorem says the result is a permutation, and no test
runs it.

## References

- Richard Durstenfeld (1964), "Algorithm 235: Random permutation", Communications of the ACM 7(7),
  420, `https://doi.org/10.1145/364520.364540`.
-/

@[expose] public section

namespace TorchLean.Random

/--
A keyed partial Fisher-Yates permutation of `0, ..., n - 1`: for `i < count` (at most `n`),
position `i` is swapped with position `i + sampleNat key i (n - i)`, and the first `count`
positions are returned. The default `count := n` gives a full permutation.
-/
def permutation (key : UInt64) (n : Nat) (count : Nat := n) : Array (Fin n) := Id.run do
  let count := min count n
  let mut indices := Array.finRange n
  for i in [:count] do
    indices := indices.swapIfInBounds i (i + sampleNat key i (n - i))
  return indices.extract 0 count

end TorchLean.Random
