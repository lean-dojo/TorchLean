/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Operations

/-!
# Synthetic Data

Deterministic tabular grids for examples and tests: `cartesianGrid`, `linspace` and
`squareGrid`. Domain-specific datasets and sample packing live in their own data modules.
-/

@[expose] public section


namespace TorchLean.Data
namespace Synthetic

/-! ## Tabular grids -/

/--
Cartesian product of two coordinate vectors, as a batch of points.

`cartesianGrid xCoordinates yCoordinates` has the `m * n` pairs `(x, y)` as rows in row-major
order: row `i * n + j` holds `xCoordinates[i]` and `yCoordinates[j]`, so `x` varies slowest.

PyTorch analogue: `torch.cartesian_prod(xCoordinates, yCoordinates)`, with the same shape and
row order.
-/
def cartesianGrid {α : Type} [TorchLean.Storage α] {m n : Nat}
    (xCoordinates : Tensor α [m]) (yCoordinates : Tensor α [n]) :
    Tensor α [m * n, 2] :=
  Tensor.stack 0 (fun ij =>
    let i : Fin m := ij.divNat (m := m) (n := n)
    let j : Fin n := ij.modNat (m := m) (n := n)
    let x : α := xCoordinates[i]
    let y : α := yCoordinates[j]
    Tensor.stack 0 (fun k =>
      Tensor.full [] (if k.val = 0 then x else y)))

/--
Linearly spaced points including endpoints.

`linspace lower upper count` returns a vector tensor of shape `(count,)`:
- empty if `count = 0`
- `[lower]` if `count = 1`
- otherwise `count` points from `lower` to `upper`, with both endpoints returned unchanged.

Interior point `i` of `n + 2` uses `t = i / (n + 1)` and `s = (n + 1 - i) / (n + 1)`, each
rounded once from the exact rational. Finite endpoints of one strict sign give
`lower + t * (upper - lower)`, replaced by `upper` where it rounds past `upper`, and all other
endpoints give `s * lower + t * upper`. The points other than NaNs are monotone, and finite
endpoints give finite points: `upper - lower` in the first form and `s * lower + t * upper` in the
second are at most the larger endpoint in magnitude. An infinite endpoint gives the second form's
limit, such as `-∞` inside `linspace (-∞) 0 count`, while opposite infinities and a NaN endpoint
give NaN inside.

PyTorch analogue: `torch.linspace`, which steps from each end by the rounded `step`, so interior
points can differ in the last bits. Its ends are `start + step * 0` and `end - step * 0`: a `-0`
end can come back as `+0`, and a `step` that overflows gives NaN ends.
-/
def linspace {α : Type} [TorchLean.Storage α] [Context α]
    (lower upper : α) (count : Nat) :
    Tensor α [count] :=
  match count with
  | 0 => by
      simpa using
        Tensor.stack (α := α) (count := 0) (shape := []) 0 Fin.elim0
  | 1 => Tensor.repeatAxis 0 1 (Tensor.full [] lower)
  | n + 2 =>
      let sameSign := Context.isFinite lower && Context.isFinite upper &&
        decide (lower > 0 ∧ upper > 0 ∨ 0 > lower ∧ 0 > upper)
      Tensor.stack 0 (fun i =>
        let t : α := ((i.val / (n + 1) : Rat) : α)
        let s : α := (((n + 1 - i.val : Nat) / (n + 1) : Rat) : α)
        let point :=
          if i.val = 0 then lower
          else if i.val = n + 1 then upper
          else if sameSign then
            let x := lower + t * (upper - lower)
            if upper > lower ∧ x > upper ∨ lower > upper ∧ upper > x then upper else x
          else s * lower + t * upper
        Tensor.full [] point)

/--
Square grid of `count * count` points over `[lower, upper] x [lower, upper]`, with
`linspace lower upper count` along each axis.
-/
def squareGrid {α : Type} [TorchLean.Storage α] [Context α]
    (lower upper : α) (count : Nat) :
    Tensor α [count * count, 2] :=
  let axis := linspace lower upper count
  cartesianGrid axis axis

end Synthetic
end TorchLean.Data
