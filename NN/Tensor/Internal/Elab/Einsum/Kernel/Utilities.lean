/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import Aesop.BuiltinRules
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
meta import Mathlib.Tactic.ToAdditive
public meta import NN.Tensor.Internal.Elab.Einsum.Contraction.Loop -- shake: keep

/-!
# Generated kernel utilities

This module closes independently generated results over only the lets they
use and constructs direct correctness certificates for compile-time-expanded
finite folds.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Introduce generated lets around several results independently.

`mkLetFVars` drops unused let declarations from each result. A
contraction-invariant operand can therefore remain in a scalar factor while
coordinate-dependent reads stay only in the middle product.
-/
def withGeneratedLetResults (bindings : List (Name × Expr))
    (body : List Expr → TermElabM (Array Expr)) :
    TermElabM (Array Expr) := do
  let rec
    /-- Preserve source order while closing each result over only the lets it uses. -/
    visit (remaining : List (Name × Expr))
        (values : List Expr) : TermElabM (Array Expr) := do
      match remaining with
      | [] => body values
      | (name, value) :: remaining =>
          withLetDecl name (← inferType value) value fun localValue => do
            let results ←
              visit remaining (values.concat localValue)
            let mut closedResults := #[]
            for result in results do
              closedResults := closedResults.push <|
                ← mkLetFVars
                  (generalizeNondepLet := false) #[localValue] result
            return closedResults
  visit bindings []

/--
Unroll a finite fold of known length and construct its equality certificate
from the standard successor and zero laws.

Building this proof directly avoids asking the simplifier to normalize the
dependent operand family generated for every einsum input.
-/
def unrollFiniteFold
    (length : Nat) (step initial : Expr) : MetaM (Expr × Expr) := do
  let fold ←
    mkAppM ``Fin.foldl #[mkNatLit length, step, initial]
  match length with
  | 0 =>
      let hFold ← mkAppM ``Fin.foldl_zero #[step, initial]
      let hUnrolled ← mkAppM ``Eq.symm #[hFold]
      let hUnrolled ←
        withTransparency .all <|
          mkExpectedTypeHint hUnrolled (← mkEq initial fold)
      return (initial, hUnrolled)
  | remainingLength + 1 =>
      let coordinateType ←
        mkAppM ``Fin #[mkNatLit (remainingLength + 1)]
      let firstCoordinate ← mkNumeral coordinateType 0
      let nextInitial := step.beta #[initial, firstCoordinate]
      let stateType ← inferType initial
      let remainingCoordinateType ←
        mkAppM ``Fin #[mkNatLit remainingLength]
      let remainingStep ←
        withLocalDeclD `state stateType fun state =>
          withLocalDeclD `operand remainingCoordinateType fun operand => do
            let successor ← mkAppM ``Fin.succ #[operand]
            mkLambdaFVars #[state, operand] <|
              step.beta #[state, successor]
      let (unrolled, hUnrolledRemaining) ←
        unrollFiniteFold remainingLength remainingStep nextInitial
      let remainingFold ←
        mkAppM ``Fin.foldl #[
          mkNatLit remainingLength, remainingStep, nextInitial]
      let hUnrolledRemaining ←
        withTransparency .all <|
          mkExpectedTypeHint hUnrolledRemaining
            (← mkEq unrolled remainingFold)
      let hFoldRemaining ←
        mkAppM ``Fin.foldl_succ #[step, initial]
      let hFoldRemaining ←
        withTransparency .all <|
          mkExpectedTypeHint hFoldRemaining
            (← mkEq fold remainingFold)
      let hRemainingFold ← mkAppM ``Eq.symm #[hFoldRemaining]
      let hUnrolled ←
        mkAppM ``Eq.trans #[hUnrolledRemaining, hRemainingFold]
      return (unrolled, hUnrolled)


end TorchLean.Tensor.Internal.Elab.Impl
