/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Spec.Core.Shape
public import NN.Tensor.Internal.Representation.Basic -- shake: keep
import NN.Spec.Core.Shape

/-!
# Native tensor literals

Lean's ordinary bracket syntax remains list syntax unless its expected type is
`Spec.Shape` or `TorchLean.Tensor α shape`. Shape brackets elaborate through
`Shape.ofList`. For a tensor expected type, ordinary nested brackets construct
a tensor and verify every dimension during elaboration.

List patterns retain their usual meaning, including empty and nested patterns.
The tensor elaborator delegates to the packed implementation, so there is one
storage invariant and no separate representation for literals.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- Brackets whose interpretation can be directed by an expected tensor type. -/
syntax (name := tensorOrListLiteralStx) (priority := high)
  "[" withoutPosition(term,*,?) "]" : term

/-- Internal marker carrying the expected type through bracket expansion. -/
syntax (name := tensorLiteralExpectedTypeStx) "tensor_literal_expected_type%" : term

/--
Keep the contents as Lean's builtin list syntax and mark the surrounding type
ascription for the tensor elaborator. Lean expands macros before collecting
pattern variables: a custom term node at that point would make even `[x]` an
invalid list pattern. A type ascription lets that collector see the ordinary
list constructors while leaving the expected-type decision to elaboration.

The inner node uses the original list parser directly. Its expansion therefore
keeps Lean's own handling of long lists and cannot call this macro again.
-/
@[macro tensorOrListLiteralStx]
def expandTensorLiteral : Macro := fun stx => do
  let literal : TSyntax `term := ⟨stx.setKind ``«term[_]»⟩
  `(($literal : tensor_literal_expected_type%))

/--
Interpret marked brackets using their expected type. Ordinary lists, including
the constructor expressions produced while elaborating patterns, go straight
to Lean's list elaboration.

For tensors, elaborate the list with the scalar or subtensor type required by
the trailing dimensions, then use the usual packed constructors. This also
works when a caller has already expanded the list macro before elaboration.
-/
@[term_elab Lean.Parser.Term.typeAscription]
def elabTensorLiteral : TermElab := fun stx expectedType? => withRef stx do
  let `(($literal : tensor_literal_expected_type%)) := stx
    | throwUnsupportedSyntax
  if let some expectedType := expectedType? then
    let expectedType ←
      withTransparency .reducible <| whnf (← instantiateMVars expectedType)
    if expectedType.isConstOf ``Spec.Shape then
      let listType ← mkAppM ``List #[mkConst ``Nat]
      let dimensions ← elabTermEnsuringType literal listType
      let result ← mkAppM ``Spec.Shape.ofList #[dimensions]
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
      if literal.raw.isOfKind ``«term[_]» then
        let literalLength := mkNatLit literal.raw[1].getSepArgs.size
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
      let listType ← mkAppM ``List #[elementType]
      let values ← elabTermEnsuringType literal listType
      let result ← if innerShape.isAppOfArity ``List.nil 1 then
        mkAppM ``TorchLean.Tensor.Internal.Rep.ofList #[values]
      else
        mkAppM ``TorchLean.Tensor.Internal.Rep.stackList #[values]
      return ← ensureHasType (some expectedType) result
  elabTerm literal expectedType?

end TorchLean.Tensor.Internal.Elab
