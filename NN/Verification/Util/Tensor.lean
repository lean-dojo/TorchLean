/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Reductions
public import NN.MLTheory.CROWN.Flatbox

/-!
# Tensor helpers for verification artifacts

Verification tools often sit at the boundary between Lean tensors and external JSON artifacts.
Flat payloads are checked against the requested tensor shape; nested matrix payloads also check
each row. Once loaded, the checkers work with tensors rather than serialized arrays.
-/

@[expose] public section

namespace NN.Verification.Util.Tensor

open Spec TorchLean
open NN.MLTheory.CROWN

/-- Pointwise ordering of equally shaped bounds. NaN comparisons fail. -/
def boundsOrdered {n : Nat} (lo hi : TorchLean.Tensor Float [n]) : Bool :=
  TorchLean.Tensor.allSpec id (TorchLean.Tensor.map2Spec (fun x y => decide (x ≤ y)) lo hi)

/-- Check ordered leaf bounds contained in an equally shaped root box. -/
def boxWithin {n : Nat} (rootLo rootHi lo hi : TorchLean.Tensor Float [n]) : Bool :=
  boundsOrdered rootLo lo && boundsOrdered lo hi && boundsOrdered hi rootHi

/-- A strict lower-bound witness refutes its corresponding threshold. -/
def refutesThreshold {n : Nat} (lowerBound threshold : TorchLean.Tensor Float [n]) : Bool :=
  TorchLean.Tensor.foldl (· || ·) false
    (TorchLean.Tensor.map2Spec (fun lb thr => decide (thr < lb)) lowerBound threshold)

/-- Check the supplied coordinate; out-of-range witnesses and NaN comparisons fail. -/
def refutesThresholdAt {n : Nat} (lowerBound threshold : TorchLean.Tensor Float [n])
    (witnessIdx : Nat) : Bool :=
  if h : witnessIdx < n then
    decide (threshold.getScalar ⟨witnessIdx, h⟩ < lowerBound.getScalar ⟨witnessIdx, h⟩)
  else false

/-- Read a row-major artifact into the requested tensor shape; reject a length mismatch. -/
def ofArray? {α : Type} [Storage α] (shape : Shape) (xs : Array α) :
    Option (Tensor α shape) :=
  if hSize : xs.size = shape.size then
    some <| TorchLean.Tensor.Internal.Rep.ofArray xs (by
      simpa only [Spec.Shape.internalSize_eq] using hSize)
  else
    none

/--
Read a nested row-major artifact into a matrix tensor.

Both the row count and every row length are checked before the tensor is built.
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

/-- Read a JSON float array into the requested shape, or raise a schema error. -/
def requireArray (ctx : String) (shape : Shape) (xs : Array Float) :
    IO (Tensor Float shape) := do
  match ofArray? shape xs with
  | some x => pure x
  | none =>
      throw <| IO.userError s!"{ctx}: expected {shape.size} floats, got {xs.size}"

end NN.Verification.Util.Tensor
