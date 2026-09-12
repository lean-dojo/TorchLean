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
public meta import NN.Tensor.Internal.Check.Einsum
public import NN.Tensor.Internal.Check.Einsum
public meta import NN.Tensor.Internal.Elab.Einsum.Contraction.Loop -- shake: keep

/-!
# Einsum contraction-invariance analysis

This module recognizes operands whose physical reads are constant across all
contracted coordinates, including singleton-broadcast axes.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

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
    (axes : List Check.EinsumAxis) (dimensions : List Expr)
    (outputAxes : List Check.EinsumAxis) : MetaM Bool := do
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

end TorchLean.Tensor.Internal.Elab.Impl
