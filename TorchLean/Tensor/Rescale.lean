/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Elementwise

/-!
# Recomputing a value that is not finite from inputs scaled by a power of two

`Context.rescale old x f` keeps `old` when it is finite (`Context.isFinite`). Otherwise it tries
`k = 1, ..., Nat.log2 n + 2` in order and returns the first finite
`f (fun i => x i / 2 ^ k) (1 / 2 ^ k)`, each quotient rounded once by `Context.divNat`, or `old`
when none is finite. In the binary formats of FloatLib an overflow rounds to an infinity in the
encoding `ieee` and to a NaN in `finiteMaxNaN` and `finiteUnsignedZero`, so either starts the
search. In `finite` it saturates at the largest finite value, so `old` is kept. The caller's `f`
evaluates its formula at the scaled inputs and undoes the scale from `c = 1 / 2 ^ k`. Scaling by a
power of two avoids rounding a binary significand unless the result enters the subnormal range.
Trying smaller scales first avoids unnecessary underflow of the inputs. The search is bounded:
it does not guarantee that a candidate is finite or more accurate. Those properties depend on
the inputs, the scalar format, and the formula supplied by the caller.

The variance `Tensor.var` and the inner sum `CrossEntropy.log1pSum` of the two-level cross-entropy
normalizer recompute their sums this way. `rescale_eq_self` reads `rescale` at a finite `old`.
-/

@[expose] public section

namespace TorchLean.Context

variable {α : Type} [Context α]

/-- `old` when it is finite (`Context.isFinite`). Otherwise the first finite value of
`f (fun i => divNat (x i) (2 ^ k)) (divNat 1 (2 ^ k))` for `k = 1, ..., Nat.log2 n + 2`, or `old`
when none is finite. -/
def rescale {n : Nat} (old : α) (x : Fin n → α) (f : (Fin n → α) → α → α) : α :=
  if isFinite old then old
  else
    ((List.range (Nat.log2 n + 2)).findSome? fun k =>
      let scaled := f (fun i => divNat (x i) (2 ^ (k + 1))) (divNat 1 (2 ^ (k + 1)))
      if isFinite scaled then some scaled else none).getD old

/-- `rescale old x f` is `old` when `old` is finite. -/
theorem rescale_eq_self {n : Nat} {old : α} (x : Fin n → α) (f : (Fin n → α) → α → α)
    (h : isFinite old = true) : rescale old x f = old := by
  simp [rescale, h]

end TorchLean.Context
