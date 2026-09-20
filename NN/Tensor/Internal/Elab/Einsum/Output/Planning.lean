/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public import NN.Tensor.Internal.Representation.Basic.Core
public meta import NN.Tensor.Internal.Elab.Einsum.Index -- shake: keep

/-!
# Einsum output-block planning

The output compiler evaluates neighboring row-major entries together so they
share contraction-coordinate work and keep a small family of scalar totals
live. This module owns the static policy; tiling semantics and concrete
register implementations remain independent of it.

Because TorchLean.Tensor.Internal is polymorphic over element representation, the planner
measures a logical working set in element reads rather than assuming a byte
width. It never changes the order of one output's contraction.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean

/-- Count tensor element reads in a generated expression. -/
partial def generatedScalarReadCount : Expr → Nat
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
Choose the concrete register width for one contiguous output block.

Four lanes amortize coordinate work once a contraction has more than eight
terms. Exactly eight terms remain scalar: the register callback setup costs
more than the coordinate sharing saves at that boundary. A single four-lane
block also remains scalar when every term reads at least three tensors.
Eight lanes are used whenever at least eight output positions are available
and the per-step live read family remains modest. Unknown contraction lengths
use the conservative four-lane implementation. The number of complete blocks
and the tail are derived separately and may be arbitrary natural numbers.
-/
def einsumOutputTileWidth?
    (outputLength : Nat) (contractionEntries : Option Nat)
    (scalarReadsPerTerm : Nat) : Option Nat :=
  if outputLength < 4 ||
      (outputLength < 8 && 3 ≤ scalarReadsPerTerm) then
    none
  else
    match contractionEntries with
    | some entries =>
        if entries ≤ 8 then
          none
        else if 8 ≤ outputLength &&
            max 1 scalarReadsPerTerm * 8 ≤ 32 then
          some 8
        else
          some 4
    | none =>
        some 4

end TorchLean.Tensor.Internal.Elab.Impl
