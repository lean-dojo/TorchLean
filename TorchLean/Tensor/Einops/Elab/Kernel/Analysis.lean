/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Check.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Contraction.Loop -- shake: keep
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Data.Rat.Cast.Order

/-!
# Einsum contraction-invariance analysis

Contraction-invariance analysis recognizes operands whose physical reads stay
constant across all contracted coordinates, singleton-broadcast axes included,
and decides which of them a semiring factors out of the contraction sum. The
kernel builder and the `einops?` report share that decision.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Report whether an operand is constant across every contracted coordinate.

Output axes never vary inside the contraction. A contracted physical axis is
also constant when its dimension reduces to one, because broadcasting always
reads its sole entry.
-/
def operandIsContractionInvariant
    (axes : List EinsumAxis) (dimensions : List Expr)
    (outputAxes : List EinsumAxis) : MetaM Bool := do
  match axes, dimensions with
  | [], [] => pure true
  | axis :: axes, dimension :: dimensions => do
      unless ← operandIsContractionInvariant axes dimensions outputAxes do
        return false
      if outputAxes.contains axis then
        return true
      match ← getNatValue? dimension with
      | some 1 => pure true
      | some _ => pure false
      | none =>
          withTransparency .reducible <|
            isDefEq dimension (mkNatLit 1)
  | _, _ =>
      throwError
        "internal error: an einsum operand's axes and dimensions have \
          different lengths"

/--
The operand range `(prefixCount, operandCount - suffixCount)` that stays inside each contraction
sum when a `Semiring` on `scalarType` moves the maximal source-order prefix and suffix of
contraction-invariant operands outside it. It is `none` unless an axis is contracted, `scalarType`
is a `Semiring`, and at least one operand moves while at least one stays.
-/
def einsumFactorBounds? (scalarType : Expr) (inputAxes : List (List EinsumAxis))
    (inputDimensions : List (List Expr))
    (outputAxes contractedAxes : List EinsumAxis) : MetaM (Option (Nat × Nat)) := do
  if contractedAxes.isEmpty then
    return none
  let .some _ ← trySynthInstance (← mkAppM ``Semiring #[scalarType])
    | return none
  let invariantOperands ←
    (inputAxes.zip inputDimensions).mapM fun (axes, dimensions) =>
      operandIsContractionInvariant axes dimensions outputAxes
  let prefixCount := (invariantOperands.takeWhile id).length
  let suffixCount := (invariantOperands.reverse.takeWhile id).length
  if prefixCount + suffixCount < invariantOperands.length &&
      (0 < prefixCount || 0 < suffixCount) then
    return some (prefixCount, invariantOperands.length - suffixCount)
  return none

end TorchLean.Einops.Elab
