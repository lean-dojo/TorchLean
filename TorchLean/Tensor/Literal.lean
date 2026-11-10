/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Rep.Reindex -- shake: keep
public import TorchLean.Tensor.Rep.Traversal -- shake: keep

/-!
# Native tensor literals

Lean's bracket syntax builds a list unless its expected type is `TorchLean.Tensor α shape`.
Shape brackets are plain lists, since `TorchLean.Shape` is `List Nat`. For a tensor expected type,
nested brackets construct a tensor and check every dimension during elaboration.

List patterns keep their usual meaning, including empty and nested patterns. The tensor
elaborator builds each literal with `Rep.ofList` or `Rep.stackList`, so a literal is an
ordinary tensor in the `Storage` buffer of its element type.
-/

public meta section

namespace TorchLean.Einops.Elab

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

The inner node takes the kind of Lean's builtin list syntax, so its expansion keeps
Lean's own handling of long lists and cannot call this macro again.
-/
@[macro tensorOrListLiteralStx]
def expandTensorLiteral : Macro := fun stx => do
  let literal : TSyntax `term := ⟨stx.setKind ``«term[_]»⟩
  `(($literal : tensor_literal_expected_type%))

/--
Interpret marked brackets by their expected type. Ordinary lists, including the constructor
expressions produced while elaborating patterns, go straight to Lean's list elaboration.

For a tensor of positive rank, elaborate the list at the scalar or subtensor type that the trailing
dimensions require, then build the tensor with the usual packed constructors. Brackets that a caller
has already macro-expanded take the same path.
-/
@[term_elab Lean.Parser.Term.typeAscription]
def elabTensorLiteral : TermElab := fun stx expectedType? => withRef stx do
  let `(($literal : tensor_literal_expected_type%)) := stx
    | throwUnsupportedSyntax
  if let some expectedType := expectedType? then
    let expectedType ←
      withTransparency .reducible <| whnf (← instantiateMVars expectedType)
    if expectedType.isAppOfArity ``Tensor.Rep 3 then
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
      -- Retain the expected storage even while the scalar's format proofs are pending.
      let result ← if innerShape.isAppOfArity ``List.nil 1 then
        mkAppOptM ``Tensor.Rep.ofList
          #[some scalarType, some storage, some values]
      else
        mkAppOptM ``Tensor.Rep.stackList
          #[some scalarType, some storage, some innerShape, some values]
      return ← ensureHasType (some expectedType) result
  elabTerm literal expectedType?

end TorchLean.Einops.Elab
