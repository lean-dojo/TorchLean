/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Conversion
public import NN.Tensor.Reductions
public import NN.MLTheory.CROWN.Flatbox

/-!
# Tensor helpers for verification artifacts

Verification tools often sit at the boundary between Lean tensors and external JSON artifacts.
This module keeps those conversions in one place instead of reimplementing vector/matrix unpacking
inside each checker.
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

/-- Convert a float array into a length-`n` vector tensor, returning `none` on length mismatch. -/
def vecOfArray (n : Nat) (xs : Array Float) : Option (Tensor Float [n]) :=
  if hSize : xs.size = n then
    some <| Tensor.dim (fun i =>
      let h : i.val < xs.size := by
        simp [hSize, i.isLt]
      Tensor.scalar (xs[i.val]'h))
  else
    none

/--
Convert a row-major flat array into a `rows × cols` matrix tensor.

This is the common JSON-artifact shape: external tools often serialize matrices as one flat float
array plus schema-level dimensions.
-/
def matOfFlatArray (rows cols : Nat) (xs : Array Float) :
    Option (Tensor Float [rows, cols]) :=
  if hSize : xs.size = rows * cols then
    some <|
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          let h : i.val * cols + j.val < xs.size := by
            rw [hSize]
            have hIdxLtNext : i.val * cols + j.val < (i.val + 1) * cols := by
              rw [Nat.add_mul, one_mul]
              exact Nat.add_lt_add_left j.isLt (i.val * cols)
            exact lt_of_lt_of_le hIdxLtNext
              (Nat.mul_le_mul_right cols (Nat.succ_le_of_lt i.isLt))
          Tensor.scalar (xs[i.val * cols + j.val]'h)))
  else
    none

/--
Convert a `rows × cols` float matrix payload into a matrix tensor.

Both the row count and every row length are checked before the tensor is built.
-/
def matOfArray (rows cols : Nat) (xs : Array (Array Float)) :
    Option (Tensor Float [rows, cols]) :=
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

/-- Convert a vector tensor to a float array. -/
def vecToArray {n : Nat} {storage : TorchLean.Storage Float}
    (x : @Tensor Float [n] storage) : Array Float :=
  Tensor.to x (Array Float)

/-- Load a length-checked vector tensor from a JSON float array, or raise a schema error. -/
def requireVecOfArray (ctx : String) (n : Nat) (xs : Array Float) :
    IO (Tensor Float [n]) := do
  match vecOfArray n xs with
  | some x => pure x
  | none =>
      throw <| IO.userError s!"{ctx}: expected {n} floats, got {xs.size}"

/-- Load a row-major matrix tensor from a JSON float array, or raise a schema error. -/
def requireMatOfFlatArray (ctx : String) (rows cols : Nat) (xs : Array Float) :
    IO (Tensor Float [rows, cols]) := do
  match matOfFlatArray rows cols xs with
  | some x => pure x
  | none =>
      throw <| IO.userError s!"{ctx}: expected {rows * cols} floats, got {xs.size}"

end NN.Verification.Util.Tensor
