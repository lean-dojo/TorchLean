/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Einops.Elab.Einsum.Index
public import TorchLean.Tensor.Einops.Elab
public import TorchLean.Tensor.Einops.Elab.Einsum.Index

/-!
# Automatic einsum contraction planning

The planner chooses a nesting order for the contracted logical axes from the
physical row-major strides of every operand. High-stride axes stay outside
low-stride ones, which improves locality without specializing to a tensor rank
or a named operation.

Planning runs only when the scalar type has an `AddCommMonoid` instance, and
the lowering correctness theorem consumes the permutation proof it returns.
Every other scalar type keeps the source contraction order.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- Project one bounded component from a nested contraction coordinate. -/
private def contractionCoordinateComponent : Nat → Expr → MetaM Expr
  | 0, coordinate =>
      withTransparency .all <| mkAppM ``Prod.fst #[coordinate]
  | position + 1, coordinate =>
      withTransparency .all do
        contractionCoordinateComponent position <|
          ← mkAppM ``Prod.snd #[coordinate]

/--
Sum of the physical row-major strides that one step of a logical axis costs
across all operands.

Every physical occurrence of a repeated label counts. Occurrences of length
one are skipped, because broadcasting fixes their physical coordinate at zero.
-/
private def contractionAxisStrideCost
    (axis : EinsumAxis)
    (inputAxes : List (List EinsumAxis))
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
Order contracted axes by descending total physical row-major stride.

The order is a cost estimate only. Whether it may be used is decided by
`planContraction?`, which requires an `AddCommMonoid` instance and builds the
permutation certificate.
-/
def contractionAxisOrder
    (inputAxes : List (List EinsumAxis))
    (inputShapes : List (List Nat))
    (contractedAxes : List EinsumAxis) :
    List EinsumAxis :=
  let weighted :=
    contractedAxes.map fun axis =>
      (axis, contractionAxisStrideCost axis inputAxes inputShapes)
  -- Stable sorting preserves source order when stride costs agree.
  (weighted.mergeSort fun a b => a.2 ≥ b.2).map Prod.fst

/--
Choose a certified contraction-axis order when the scalar type has an
`AddCommMonoid` instance.

The result holds the planned axes and lengths, the coordinate map back to the
checked source order with its pointwise proof, and the duplicate-freedom and
permutation proofs that the final coordinate-sum theorem consumes. It is
`none` for fewer than two contracted axes, for symbolic input shapes and when
the planned order equals the source order.
-/
def planContraction?
    (checked scalarType : Expr)
    (inputDimensions : List (List Expr))
    (inputAxes : List (List EinsumAxis))
    (contractedAxes : List EinsumAxis)
    (contractedLengths : List Expr) :
    TermElabM
      (Option
        (List EinsumAxis × List Expr × Expr × Expr × Expr × Expr)) := do
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
    mkAppM ``Einsum.axisLength #[checked]
  let coordinateEquiv ←
    mkAppM ``Lowering.coordinatePermutationEquiv #[
      axisLength, hOriginal, hLiteralPermutation]
  let plannedShape ← shapeExpr plannedLengths
  let plannedCoordinateType ← mkAppM ``Shape.Coord #[plannedShape]
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

end TorchLean.Einops.Elab
