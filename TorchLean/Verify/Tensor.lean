/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Fold
public import TorchLean.Verify.CROWN.FlatBox

/-!
# Tensor helpers for verification artifacts

Verification tools sit at the boundary between external JSON artifacts and Lean tensors. A flat
payload loads only when its length equals the size of the requested shape, and a nested matrix
payload also has each row length checked. Past this point the checkers work with tensors. The
Boolean tests (ordered bounds, box containment, threshold refutation) take tensors over any scalar
type with a `Context` and compare entries by `Context.leBool` and `Context.ltBool`. On `Float`
these are the IEEE 754 comparisons, which are false whenever an operand is NaN.
-/

@[expose] public section

namespace TorchLean.Tensor


section Order

variable {α : Type} [Storage α] [Context α]

/-- Whether `lo ≤ hi` holds at every coordinate by `Context.leBool`, which is false at a NaN. -/
def boundsOrdered {n : Nat} (lo hi : Tensor α [n]) : Bool :=
  Tensor.all id (Tensor.le lo hi)

/-- `boxWithin rootLo rootHi lo hi` is `true` when `rootLo ≤ lo ≤ hi ≤ rootHi` coordinatewise: the
leaf box is nonempty and lies inside the root box. -/
def boxWithin {n : Nat} (rootLo rootHi lo hi : Tensor α [n]) : Bool :=
  boundsOrdered rootLo lo && boundsOrdered lo hi && boundsOrdered hi rootHi

/-- `refutesThreshold lowerBound threshold` is `true` when some coordinate has a lower bound
strictly above its threshold, `threshold[i] < lowerBound[i]`. A NaN coordinate never counts. -/
def refutesThreshold {n : Nat} (lowerBound threshold : Tensor α [n]) : Bool :=
  Tensor.foldl (· || ·) false (Tensor.lt threshold lowerBound)

/-- `refutesThresholdAt lowerBound threshold k` checks `threshold[k] < lowerBound[k]` at the one
coordinate `k`. It is `false` when `k` is out of range or either entry is NaN. -/
def refutesThresholdAt {n : Nat} (lowerBound threshold : Tensor α [n])
    (witnessIdx : Nat) : Bool :=
  if h : witnessIdx < n then
    Context.ltBool (threshold.getScalar ⟨witnessIdx, h⟩) (lowerBound.getScalar ⟨witnessIdx, h⟩)
  else false

end Order

/-- Read a row-major array into a tensor of shape `shape`, or `none` when the array length differs
from `shape.size`. -/
def ofArray? {α : Type} [Storage α] (shape : Shape) (xs : Array α) :
    Option (Tensor α shape) :=
  if hSize : xs.size = shape.size then
    some <| Tensor.Rep.ofArray xs hSize
  else
    none

/--
Read an array of rows into a tensor of shape `[rows, cols]`.

The result is `none` unless `xs` has `rows` rows and each of them has length `cols`.
-/
def ofRows? {α : Type} [Storage α] (rows cols : Nat) (xs : Array (Array α)) :
    Option (Tensor α [rows, cols]) :=
  if hRows : xs.size = rows then
    if hCols : ∀ i : Fin rows,
        (xs[i.val]'(by simp [hRows, i.isLt])).size = cols then
      some <|
        Tensor.dim (fun i =>
          let row := xs[i.val]'(by simp [hRows, i.isLt])
          Tensor.dim (fun j =>
            let h : j.val < row.size := by
              simp [row, hCols i, j.isLt]
            Tensor.scalar (row[j.val]'h)))
    else
      none
  else
    none

/-- Load a decoded array of values of type `α` into a tensor of shape `shape`. A length mismatch
throws an `IO` error that names `ctx` and both lengths. -/
def requireArray {α : Type} [Storage α] (ctx : String) (shape : Shape) (xs : Array α) :
    IO (Tensor α shape) := do
  match ofArray? shape xs with
  | some x => pure x
  | none =>
      throw <| IO.userError s!"{ctx}: expected {shape.size} values, got {xs.size}"

end TorchLean.Tensor
