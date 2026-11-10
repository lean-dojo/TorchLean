/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Op.Table.Rule
public meta import TorchLean.Tensor.ConfiguredBinary

/-!
# Gradients of `scatter_reduce` against torch autograd

The Op rules composed as the PyTorch capture composes `scatter_reduce` with `prod`, `amax` and
`amin` give torch autograd's gradients of the same vectors in `Float` and `Float32`.
-/

@[expose] public section

namespace TorchLean.Test.OpTable

/-- The gradients at `g` of `base.scatter_reduce(0, index, updates, rule, includeSelf)` as the
PyTorch capture imports it when `includeSelf` holds or `rule` is `prod`: `Op.scatter rule` at the
index, after, for a product without `includeSelf`, a replacing scatter of `1` at the index, whose
base pullback is zero where it wrote. The base gradient, then the update gradient. -/
def scatterReduceGradients {α : Type} [Storage α] [Context α] [Op.Capabilities α]
    (rule : Tensor.ScatterReduce) (includeSelf : Bool) {n m : Nat} (base : Tensor α [n])
    (index : Tensor (Fin n) [m]) (updates : Tensor α [m]) (g : Tensor α [n]) :
    Option (Array α × Array α) := do
  let reduced ← (Op.scatter (s := [n]) (t := [m]) rule [⟨0, Nat.one_pos⟩] (by simp)).rule?
    (α := α)
  let replaced ← (Op.scatter (s := [n]) (t := [m]) .replace [⟨0, Nat.one_pos⟩] (by simp)).rule?
    (α := α)
  let seeds : Tensor α [m] := Tensor.full _ 1
  let seeded := if includeSelf then base else (replaced.forward (base, index, seeds, ())).1
  let gradients := reduced.backward (seeded, index, updates, ()) (g, ())
  let baseGradient := if includeSelf then gradients.1 else
    (replaced.backward (base, index, seeds, ()) (gradients.1, ())).1
  pure (baseGradient.to (Array α), gradients.2.2.1.to (Array α))

/-- The gradients at `g` of `base.scatter_reduce(0, index, updates, rule, include_self=False)`
with `rule` `max` or `min` as the PyTorch capture imports it on a vector of `n`: the base padded by
one entry, a replacing scatter of each target's first update (`seed`, the later updates sent to the
padding), then `rule` with the other updates (`fold`, the first ones sent to the padding), and the
padding cropped, so the base entry at a written target takes no part. `base` and `g` come padded,
and the base gradient drops the padding. -/
def firstSeededGradients {α : Type} [Storage α] [Context α] [Op.Capabilities α]
    (rule : Tensor.ScatterReduce) {n m : Nat} (base : Tensor α [n + 1])
    (seed fold : Tensor (Fin (n + 1)) [m]) (updates : Tensor α [m]) (g : Tensor α [n + 1]) :
    Option (Array α × Array α) := do
  let reduced ← (Op.scatter (s := [n + 1]) (t := [m]) rule [⟨0, Nat.one_pos⟩] (by simp)).rule?
    (α := α)
  let replaced ← (Op.scatter (s := [n + 1]) (t := [m]) .replace [⟨0, Nat.one_pos⟩]
    (by simp)).rule? (α := α)
  let seeded := (replaced.forward (base, seed, updates, ())).1
  let folded := reduced.backward (seeded, fold, updates, ()) (g, ())
  let seededGradients := replaced.backward (base, seed, updates, ()) (folded.1, ())
  pure ((seededGradients.1.to (Array α)).pop,
    Array.zipWith (· + ·) (folded.2.2.1.to (Array α)) (seededGradients.2.2.1.to (Array α)))

-- Gradient parity with torch autograd (torch 2.13, CPU, binary64 for `Float` and binary32 for
-- `Float32`) of the Op rules composed as the capture composes them, evaluated here in `Float` and
-- `Float32`, not in the IR's binary formats and not by running a captured graph's backward. `amax`
-- and `amin` are the rules `max` and `min`, and ties share evenly.
/-- The index sending both updates to entry `0` of a vector of two. -/
def pair0 : Tensor (Fin 2) [2] := [0, 0]
/-- The index sending one update to entry `0` of a vector of two. -/
def first0 : Tensor (Fin 2) [1] := [0]
/-- The cotangent of ones on a vector of two, in `Float`. -/
def ones2 : Tensor Float [2] := [1, 1]
/-- `ones2` in `Float32`. -/
def ones2f : Tensor Float32 [2] := [1, 1]
/-- The index sending three updates to entries `0, 1, 1` of a vector of three. -/
def spread : Tensor (Fin 3) [3] := [0, 1, 1]
/-- The index sending three updates to entries `2, 2, 0` of a vector of three. -/
def reversed : Tensor (Fin 3) [3] := [2, 2, 0]
/-- `spread` as the `seed` index of `firstSeededGradients`: the first update of each target, with
the later update sent to the padding entry `3`. -/
def seedSpread : Tensor (Fin 4) [3] := [0, 1, 3]
/-- `spread` as the `fold` index of `firstSeededGradients`: the later update, with the first ones
sent to the padding entry `3`. -/
def foldSpread : Tensor (Fin 4) [3] := [3, 3, 1]
/-- The updates of the `spread` cases, in `Float`. -/
def spreadUpdates : Tensor Float [3] := [4, 2, 1 / 2]
/-- `spreadUpdates` in `Float32`. -/
def spreadUpdatesf : Tensor Float32 [3] := [4, 2, 1 / 2]
/-- A cotangent on the padded vector of four, zero at the padding, in `Float`. -/
def spreadGradient : Tensor Float [4] := [7, 8, 9, 0]
/-- `spreadGradient` in `Float32`. -/
def spreadGradientf : Tensor Float32 [4] := [7, 8, 9, 0]
/-- The base of the product case, in `Float`. -/
def productBase : Tensor Float [3] := [3 / 2, 2, 3]
/-- `productBase` in `Float32`. -/
def productBasef : Tensor Float32 [3] := [3 / 2, 2, 3]
/-- The updates of the product case, in `Float`. -/
def productUpdates : Tensor Float [3] := [2, 1 / 2, 4]
/-- `productUpdates` in `Float32`. -/
def productUpdatesf : Tensor Float32 [3] := [2, 1 / 2, 4]
/-- A cotangent on a vector of three, in `Float`. -/
def upstream3 : Tensor Float [3] := [7, 8, 9]
/-- `upstream3` in `Float32`. -/
def upstream3f : Tensor Float32 [3] := [7, 8, 9]
#guard scatterReduceGradients .max true ([1, 0] : Tensor Float [2]) pair0
    ([1, 1] : Tensor Float [2]) ones2 == some (#[1 / 3, 1], #[1 / 3, 1 / 3])
#guard scatterReduceGradients .min true ([-0.0, 0] : Tensor Float [2]) first0
    ([0] : Tensor Float [1]) ones2 == some (#[1 / 2, 1], #[1 / 2])
#guard scatterReduceGradients .prod true ([2, 1] : Tensor Float [2]) pair0
    ([4, 3] : Tensor Float [2]) ones2 == some (#[12, 1], #[6, 8])
#guard scatterReduceGradients .prod false ([2, 1] : Tensor Float [2]) pair0
    ([4, 3] : Tensor Float [2]) ones2 == some (#[0, 1], #[3, 4])
#guard scatterReduceGradients .prod true ([2, 1] : Tensor Float [2]) pair0
    ([0, 3] : Tensor Float [2]) ones2 == some (#[0, 1], #[6, 0])
#guard scatterReduceGradients .prod false ([0, 1] : Tensor Float [2]) pair0
    ([0, 3] : Tensor Float [2]) ones2 == some (#[0, 1], #[3, 0])
#guard scatterReduceGradients .max true ([1, 2, 3] : Tensor Float [3]) spread
    spreadUpdates upstream3 == some (#[0, 4, 9], #[7, 4, 0])
#guard firstSeededGradients .min ([1, 2, 3, 0] : Tensor Float [4]) seedSpread foldSpread
    spreadUpdates spreadGradient == some (#[0, 0, 9], #[7, 0, 8])
#guard scatterReduceGradients .prod true productBase reversed productUpdates upstream3 ==
  some (#[28, 8, 9], #[27 / 2, 54, 21 / 2])
#guard scatterReduceGradients .max true ([1, 0] : Tensor Float32 [2]) pair0
    ([1, 1] : Tensor Float32 [2]) ones2f == some (#[1 / 3, 1], #[1 / 3, 1 / 3])
#guard firstSeededGradients .min ([1, 2, 3, 0] : Tensor Float32 [4]) seedSpread foldSpread
    spreadUpdatesf spreadGradientf == some (#[0, 0, 9], #[7, 0, 8])
#guard scatterReduceGradients .prod true productBasef reversed productUpdatesf upstream3f ==
  some (#[28, 8, 9], #[27 / 2, 54, 21 / 2])

end TorchLean.Test.OpTable
