/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.OfFn

/-!
# Shape-prefix and sequence operations

`Sequence.mapAccum` and `Sequence.mapAccumRight` thread a state through the indices of `Fin n` and
return a `Tensor Result [n]`, so the traversal records its length in the result type.
`Tensor.mapLeading` and `Tensor.zipEach` apply a function at every index of an arbitrary leading
shape.
-/

@[expose] public section


namespace TorchLean

namespace Sequence

/-- Forward loop of `mapAccum`: visit `i, ..., n - 1` and append each result. -/
def mapAccumLoop {State Result : Type} (n : Nat)
    (step : (i : Fin n) → State → State × Result) :
    (i : Nat) → i ≤ n → State → (acc : Array Result) → acc.size = i →
      State × { results : Array Result // results.size = n }
  | i, hi, state, acc, hSize =>
      if hLt : i < n then
        let (next, result) := step ⟨i, hLt⟩ state
        mapAccumLoop n step (i + 1) hLt next (acc.push result) (by rw [Array.size_push, hSize])
      else
        (state, ⟨acc, by grind⟩)
  termination_by i => n - i

/--
Backward loop of `mapAccumRight`: visit `i - 1, ..., 0` and push each result.

The accumulator holds the results in visiting order, so entry `k` belongs to index `n - 1 - k`.
Pushing keeps the loop linear: prepending would copy the accumulator at every step.
-/
def mapAccumRightLoop {State Result : Type} (n : Nat)
    (step : (i : Fin n) → State → State × Result) :
    (i : Nat) → i ≤ n → State → (acc : Array Result) → acc.size + i = n →
      State × { results : Array Result // results.size = n }
  | 0, _, state, acc, hSize => (state, ⟨acc, by grind⟩)
  | i + 1, hi, state, acc, hSize =>
      let (previous, result) := step ⟨i, hi⟩ state
      mapAccumRightLoop n step i (Nat.le_of_succ_le hi) previous (acc.push result)
        (by rw [Array.size_push]; grind)

/--
Traverse the indices `0, ..., n - 1`, threading a state and collecting one result per index.

The result type `Tensor Result [n]` records that the traversal produced `n` values, and the
indexed step suits a source that is already a function `Fin n → α`. Results go into one array that
is packed into the tensor once, so the traversal is linear in `n`.
-/
def mapAccum {State Result : Type} [Storage Result]
    (n : Nat) (state : State) (step : (i : Fin n) → State → State × Result) :
    State × Tensor Result [n] :=
  let (final, results) := mapAccumLoop n step 0 (Nat.zero_le n) state #[] rfl
  (final, Tensor.ofFn fun i => results.val[i.val]'(by rw [results.property]; exact i.isLt))

/--
Traverse `Fin n` from right to left, threading a state, and return the results in index order.
Reverse-mode passes through a fixed-length sequence, such as backpropagation through time, thread
their state this way.
-/
def mapAccumRight {State Result : Type} [Storage Result]
    (n : Nat) (state : State) (step : (i : Fin n) → State → State × Result) :
    State × Tensor Result [n] :=
  let (initial, results) :=
    mapAccumRightLoop n step n (Nat.le_refl n) state #[] (by simp)
  (initial, Tensor.ofFn fun i =>
    results.val[n - 1 - i.val]'(by rw [results.property]; have := i.isLt; omega))

end Sequence

end TorchLean

namespace TorchLean.Tensor

/--
Apply a function independently at every index of a leading shape.

Example:
```lean
-- Write the per-sample function and let the leading axis take care of itself: `[5, 3]` in,
-- `[5, 1]` out, with no loop over the batch.
def firstFeature (batch : Tensor Float [5, 3]) : Tensor Float [5, 1] :=
  Tensor.mapLeading [5] (fun row => Tensor.take row 0 1) batch
```
-/
def mapLeading {α : Type} [TorchLean.Storage α]
    (leading : Shape) {inShape outShape : Shape}
    (f : Tensor α inShape → Tensor α outShape)
    (x : Tensor α (leading ++ inShape)) : Tensor α (leading ++ outShape) :=
  match leading with
  | .scalar => f x
  | .dim _ rest =>
      Tensor.dim (fun i => mapLeading rest f (Tensor.unstack x i))

/-- Apply `f` to the two blocks at every index of a shared leading shape, the two-argument
`mapLeading`. -/
def zipEach {α : Type} [TorchLean.Storage α]
    (leading : Shape) {leftShape rightShape : Shape} (outShape : Shape)
    (f : Tensor α leftShape → Tensor α rightShape → Tensor α outShape)
    (left : Tensor α (leading ++ leftShape))
    (right : Tensor α (leading ++ rightShape)) : Tensor α (leading ++ outShape) :=
  match leading with
  | .scalar => f left right
  | .dim _ rest =>
      Tensor.dim (fun i =>
        zipEach rest outShape f (Tensor.unstack left i) (Tensor.unstack right i))

end TorchLean.Tensor
