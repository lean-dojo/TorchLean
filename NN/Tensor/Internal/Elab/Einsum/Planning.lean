/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum.Index
public import NN.Tensor.Internal.Elab.Common
public import NN.Tensor.Internal.Elab.Einsum.Index

/-!
# Automatic einsum contraction planning

The planner chooses a nesting order for contracted logical axes from the
physical row-major strides of every operand. High-stride axes remain outside
low-stride axes, improving locality without specializing to a tensor rank or
named operation.

Planning is enabled only when the scalar type has lawful commutative addition.
The returned permutation proof is consumed by the lowering correctness
theorem; ordered scalar types retain the source contraction order.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Insert an axis into a descending, stable stride-cost order.

Equal-cost axes retain their source order, making generated kernels
deterministic across compiler runs.
-/
private def insertContractionAxis
    (item : Check.EinsumAxis × Nat)
    (items : List (Check.EinsumAxis × Nat)) :
    List (Check.EinsumAxis × Nat) :=
  match items with
  | [] => [item]
  | current :: remaining =>
      if item.2 ≥ current.2 then
        item :: items
      else
        current :: insertContractionAxis item remaining

/-- Stable descending insertion sort for the small logical-axis lists. -/
private def sortContractionAxes :
    List (Check.EinsumAxis × Nat) →
      List (Check.EinsumAxis × Nat)
  | [] => []
  | item :: items =>
      insertContractionAxis item (sortContractionAxes items)

/-- Project one bounded component from a nested contraction coordinate. -/
private def contractionCoordinateComponent : Nat → Expr → MetaM Expr
  | 0, coordinate =>
      withTransparency .all <| mkAppM ``Prod.fst #[coordinate]
  | position + 1, coordinate =>
      withTransparency .all do
        contractionCoordinateComponent position <|
          ← mkAppM ``Prod.snd #[coordinate]

/--
Aggregate the physical row-major stride paid when one logical axis changes.

Repeated labels contribute every physical occurrence. Singleton occurrences
are ignored because broadcasting fixes their physical coordinate at zero.
-/
private def contractionAxisStrideCost
    (axis : Check.EinsumAxis)
    (inputAxes : List (List Check.EinsumAxis))
    (inputShapes : List (List Nat)) : Nat :=
  (inputAxes.zip inputShapes).foldl
    (fun total (axes, shape) =>
      total +
        ((axes.zip shape).zipIdx.foldl
          (fun operandTotal ((currentAxis, physicalLength), position) =>
            if currentAxis == axis && physicalLength != 1 then
              operandTotal + (shape.drop (position + 1)).prod
            else
              operandTotal)
          0))
    0

/--
Order contracted axes by descending aggregate physical row-major stride.

The function is pure cost analysis: it does not grant permission to use the
order. `planContraction?` separately requires lawful commutative addition and
constructs the permutation certificate.
-/
def contractionAxisOrder
    (inputAxes : List (List Check.EinsumAxis))
    (inputShapes : List (List Nat))
    (contractedAxes : List Check.EinsumAxis) :
    List Check.EinsumAxis :=
  let weighted :=
    contractedAxes.map fun axis =>
      (axis, contractionAxisStrideCost axis inputAxes inputShapes)
  (sortContractionAxes weighted).map fun item => item.1

/--
Choose a certified contraction-axis order when exact algebra permits it.

The result contains the planned axes and lengths, their coordinate
equivalence to the checked source order, and the permutation certificate used
by the final coordinate-sum theorem.
-/
def planContraction?
    (checked scalarType : Expr)
    (inputDimensions : List (List Expr))
    (inputAxes : List (List Check.EinsumAxis))
    (contractedAxes : List Check.EinsumAxis)
    (contractedLengths : List Expr) :
    TermElabM
      (Option
        (List Check.EinsumAxis × List Expr × Expr × Expr × Expr × Expr)) := do
  if contractedAxes.length < 2 ||
      contractedAxes.length != contractedLengths.length ||
      inputAxes.length != inputDimensions.length then
    return none
  let addCommMonoidType ← mkAppM ``AddCommMonoid #[scalarType]
  let .some _ ← trySynthInstance addCommMonoidType
    | return none
  let mut concreteInputShapes : List (List Nat) := []
  for dimensions in inputDimensions do
    let some concreteShape ← concreteNatExpressions? dimensions
      | return none
    concreteInputShapes := concreteInputShapes.concat concreteShape
  let plannedAxes :=
    contractionAxisOrder inputAxes concreteInputShapes contractedAxes
  if plannedAxes == contractedAxes then
    return none
  let mut plannedLengths : List Expr := []
  for axis in plannedAxes do
    let position := contractedAxes.idxOf axis
    if position == contractedLengths.length then
      throwError
        "internal error: a planned contraction axis has no source length"
    plannedLengths := plannedLengths.concat contractedLengths[position]!
  let plannedAxesExpr := Lean.toExpr plannedAxes
  let originalAxesExpr := Lean.toExpr contractedAxes
  let literalPermutationType ←
    mkAppM ``List.Perm #[plannedAxesExpr, originalAxesExpr]
  let decidePermutation ← `(tactic| decide)
  let hLiteralPermutation ←
    certifyWithTactic
      "that the selected contraction plan permutes its source axes"
      literalPermutationType decidePermutation
  let originalNodupType ←
    mkAppM ``List.Nodup #[originalAxesExpr]
  let hOriginal ←
    certifyWithTactic
      "that the source contraction axes are duplicate-free"
      originalNodupType decidePermutation
  let axisLength ←
    mkAppM ``Check.CheckedEinsum.axisLength #[checked]
  let coordinateEquiv ←
    mkAppM ``Lowering.coordinatePermutationEquiv #[
      axisLength, hOriginal, hLiteralPermutation]
  let plannedShape ← shapeExpr plannedLengths
  let plannedCoordinateType ← mkAppM ``Coord #[plannedShape]
  let (coordinateMap, hCoordinateMap) ←
    withLocalDeclD `contractionCoordinate plannedCoordinateType
        fun coordinate => do
      let mut semanticComponents : List Expr := []
      for axis in contractedAxes do
        let plannedPosition := plannedAxes.idxOf axis
        if plannedPosition == plannedAxes.length then
          throwError
            "internal error: a planned contraction omitted an axis"
        let component ←
          contractionCoordinateComponent plannedPosition coordinate
        semanticComponents := semanticComponents.concat component
      let semanticCoordinate ←
        coordinateFromComponents semanticComponents
      let abstractCoordinate ←
        mkAppM ``Equiv.toFun #[coordinateEquiv, coordinate]
      let pointwiseType ←
        mkEq semanticCoordinate abstractCoordinate
      let hPointwise ←
        withTransparency .all <|
          mkExpectedTypeHint (← mkEqRefl semanticCoordinate) pointwiseType
      let coordinateMap ←
        mkLambdaFVars #[coordinate] semanticCoordinate
      let hCoordinateMap ←
        mkLambdaFVars #[coordinate] hPointwise
      pure (coordinateMap, hCoordinateMap)
  return some
    (plannedAxes, plannedLengths, coordinateMap,
      hCoordinateMap, hOriginal, hLiteralPermutation)

end TorchLean.Tensor.Internal.Elab.Impl
