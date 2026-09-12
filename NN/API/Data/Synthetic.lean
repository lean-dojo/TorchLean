/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Algebra.Order.Algebra
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public import NN.Tensor.Internal.Elab.TensorLiteral
public import NN.Tensor.Operations
public import NN.Tensor -- shake: keep

/-!
# Synthetic Data

This module provides deterministic tabular grids used by examples and tests. Domain-specific
datasets and sample packing belong in their respective data modules.
-/

@[expose] public section


namespace TorchLean.Data
namespace Synthetic

open Spec TorchLean

/-! ## Tabular grids -/

/--
Cartesian product of two vectors (batched tensor of points).

`cartesianGrid xCoordinates yCoordinates` produces a tensor `X : (m*n, 2)` containing all pairs
`(x, y)` with:
- `x` taken from `xCoordinates : (m,)`
- `y` taken from `yCoordinates : (n,)`

Ordering is row-major: for each `x` in `xCoordinates` (outer loop), we sweep all `y` in
`yCoordinates` (inner loop).

PyTorch analogue: `torch.cartesian_prod(xCoordinates, yCoordinates)` (up to shape).
-/
def cartesianGrid {α : Type} [TorchLean.Storage α] [Zero α] {m n : Nat}
    (xCoordinates : Tensor α [m]) (yCoordinates : Tensor α [n]) :
    Tensor α [m * n, 2] :=
  Tensor.stack 0 (fun ij =>
    let i : Fin m := ij.divNat (m := m) (n := n)
    let j : Fin n := ij.modNat (m := m) (n := n)
    let x : α := xCoordinates[i]
    let y : α := yCoordinates[j]
    Tensor.stack 0 (fun k =>
      Tensor.full [] <|
        match k.val with
        | 0 => x
        | 1 => y
        | _ => 0))

/--
Linearly spaced points including endpoints.

`linspace lo hi count` returns a vector tensor of shape `(count,)`:
- empty if `count = 0`
- `[lo]` if `count = 1`
- otherwise `count` points from `lo` to `hi` (inclusive).

PyTorch analogue: `torch.linspace`.
-/
def linspace {α : Type} [TorchLean.Storage α] [Context α]
    (lower upper : α) (count : Nat) :
    Tensor α [count] :=
  match count with
  | 0 => by
      simpa [Shape.insertAxis] using
        Tensor.stack (α := α) (count := 0) (shape := []) 0 Fin.elim0
  | 1 => Tensor.repeatAxis 0 1 (Tensor.full [] lower)
  | n + 2 =>
      let denom : α := (n + 1 : Nat)
      Tensor.stack 0 (fun i =>
        let t : α := i.val / denom
        Tensor.full [] (lower + t * (upper - lower)))

/-- Square grid over `[lower, upper] x [lower, upper]`. -/
def squareGrid {α : Type} [TorchLean.Storage α] [Context α]
    (lower upper : α) (count : Nat) :
    Tensor α [count * count, 2] :=
  let axis := linspace lower upper count
  cartesianGrid axis axis

end Synthetic
end TorchLean.Data
