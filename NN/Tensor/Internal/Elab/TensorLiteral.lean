/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

meta import Mathlib.Tactic.Basic
meta import Mathlib.Tactic.ToAdditive
meta import Mathlib.Tactic.ToDual
public meta import NN.Spec.Core.Shape
public import NN.Tensor.Internal.Representation.Basic.Pointwise
public meta import Lean.Elab.Term -- shake: keep
public import NN.Tensor.Internal.Representation.Basic -- shake: keep

/-!
# Native tensor literals

Lean's ordinary bracket syntax remains list syntax unless its expected type is
`Spec.Shape` or `TorchLean.Tensor α shape`. Shape brackets elaborate through
`Shape.ofList`. For a tensor expected type, ordinary nested brackets construct
a tensor and verify every dimension during elaboration.

The elaborator delegates to the packed implementation, so there is one storage
invariant and no literal-only tensor representation.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- Internal syntax used to distinguish tensor literals from ordinary list literals. -/
syntax (name := tensorOrListLiteralStx)
  "tensor_literal%[" withoutPosition(term,*,?) "]" : term

macro_rules
  | `([$elements,*]) => `(tensor_literal%[$elements,*])

/--
Expand bracket elements to the ordinary list constructors without invoking
bracket syntax again. This is the fallback for every non-shape, non-tensor
expected type.
-/
private def expandListLiteral
    (elements : Array (TSyntax `term)) : TermElabM (TSyntax `term) := do
  let mut result ← `(List.nil)
  for element in elements.reverse do
    result ← `(List.cons $element $result)
  return result

/--
Elaborate intercepted bracket syntax as a shape or tensor when directed by the
expected type. Every other use expands to Lean's ordinary list constructors.
-/
@[term_elab tensorOrListLiteralStx]
def elabTensorLiteral : TermElab := fun stx expectedType? => withRef stx do
  let `(tensor_literal%[$elements,*]) := stx
    | throwUnsupportedSyntax
  let elements := elements.getElems
  if let some expectedType := expectedType? then
    let expectedType ←
      withTransparency .reducible <| whnf (← instantiateMVars expectedType)
    if expectedType.isConstOf ``Spec.Shape then
      let dimensions ← elements.mapM fun element =>
        elabTermEnsuringType element (mkConst ``Nat)
      let dimensionList ← mkListLit (mkConst ``Nat) dimensions.toList
      let result ← mkAppM ``Spec.Shape.ofList (Array.singleton dimensionList)
      return ← ensureHasType (some expectedType) result
    if expectedType.isAppOfArity ``TorchLean.Tensor.Internal.Rep 3 then
      let tensorArguments := expectedType.getAppArgs
      let expectedShape ← withTransparency .reducible <| whnf tensorArguments[1]!
      unless expectedShape.isAppOfArity ``List.cons 3 do
        throwError
          "tensor bracket syntax requires a positive-rank tensor, but the expected \
            tensor shape is {expectedShape}"
      let shapeArguments := expectedShape.getAppArgs
      let expectedLength := shapeArguments[1]!
      let innerShape ← withTransparency .reducible <| whnf shapeArguments[2]!
      let literalLength := mkNatLit elements.size
      unless ← withTransparency .default <|
          isDefEq expectedLength literalLength do
        throwError
          "tensor literal has leading dimension {literalLength}, but the expected \
            leading dimension is {expectedLength}"
      let scalarType := tensorArguments[0]!
      let storage := tensorArguments[2]!
      let elementType :=
        if innerShape.isAppOfArity ``List.nil 1 then
          scalarType
        else
          mkApp3 expectedType.getAppFn scalarType innerShape storage
      let values ← elements.mapM fun element =>
        elabTermEnsuringType element elementType
      let valueList ← mkListLit elementType values.toList
      let result ← if innerShape.isAppOfArity ``List.nil 1 then
        mkAppM ``TorchLean.Tensor.Internal.Rep.ofList (Array.singleton valueList)
      else
        mkAppM ``TorchLean.Tensor.Internal.Rep.stackList (Array.singleton valueList)
      return ← ensureHasType (some expectedType) result
  elabTerm (← expandListLiteral elements) expectedType?

end TorchLean.Tensor.Internal.Elab
