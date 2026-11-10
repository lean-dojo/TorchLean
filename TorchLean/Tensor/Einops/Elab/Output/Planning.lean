/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep
public meta import TorchLean.Tensor.Einops.Elab.Einsum.Index -- shake: keep
public meta import TorchLean.Tensor.Native.Tile.Registers
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Data.Rat.Cast.Order

/-!
# Einsum output-block planning

The output compiler evaluates neighboring row-major entries together so they
share contraction-coordinate work and keep a small family of scalar totals
live. The static policy for that choice lives here, and the tiling semantics and
concrete register implementations do not depend on it.

Because einsum is polymorphic over the element representation, the planner
measures a logical working set in element reads and assumes no byte width. It
never changes the order of one output's contraction.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Tensor (Rep)

/-- Count tensor element reads in a generated expression. -/
def generatedScalarReadCount : Expr → Nat
  | expression@(.app function argument) =>
      let current :=
        if expression.isAppOfArity ``Rep.getFlat 5 ||
            expression.isAppOfArity ``Rep.getFlatUSize 6 then
          1
        else
          0
      current +
        generatedScalarReadCount function +
        generatedScalarReadCount argument
  | .lam _ type body _ | .forallE _ type body _ =>
      generatedScalarReadCount type + generatedScalarReadCount body
  | .letE _ type value body _ =>
      generatedScalarReadCount type +
        generatedScalarReadCount value +
        generatedScalarReadCount body
  | .mdata _ body | .proj _ _ body =>
      generatedScalarReadCount body
  | _ => 0

/--
Choose the register width for one contiguous output block among the widths of `outputTiles`.

The narrowest tile amortizes coordinate work once a contraction has more than eight
terms. With eight terms the register callback setup costs more than the coordinate
sharing saves, so the block stays scalar. A single narrowest block also stays scalar when
every term reads at least three tensors. The widest tile whose width fits in the output and
whose live reads per step (reads per term times lanes) stay within 32 replaces it. Unknown
contraction lengths use the narrowest tile. The number of complete blocks and the tail are
derived separately and may be arbitrary natural numbers.
-/
def einsumOutputTileWidth?
    (outputLength : Nat) (contractionEntries : Option Nat)
    (scalarReadsPerTerm : Nat) : Option Nat :=
  match Native.outputTiles with
  | [] => none
  | narrowest :: wider =>
      if outputLength < narrowest.width ||
          (outputLength < 2 * narrowest.width && 3 ≤ scalarReadsPerTerm) then
        none
      else
        match contractionEntries with
        | some entries =>
            if entries ≤ 8 then
              none
            else
              some <| wider.foldl (init := narrowest.width) fun chosen tile =>
                if tile.width ≤ outputLength &&
                    max 1 scalarReadsPerTerm * tile.width ≤ 32 then
                  tile.width
                else
                  chosen
        | none =>
            some narrowest.width

end TorchLean.Einops.Elab
