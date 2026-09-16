/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum.Index
public meta import NN.Tensor.Internal.Elab.Native.Index
public import NN.Tensor.Internal.Laws.Equivalence -- shake: keep
public import NN.Tensor.Internal.Elab.Einsum.Index
import NN.Tensor.Internal.Elab.Native.Index

/-!
# Certified transform index compilation

This module compiles the flat-index function of a checked rearrangement or
repeat while elaborating the surrounding expression. The generated program
contains only the resulting row-major arithmetic; parser data, axis lookup,
and checked-plan interpretation remain in erased correctness proofs.

Symbolic dimensions remain symbolic arithmetic in the generated expression;
the returned theorem still identifies that program with the independent
coordinate semantics.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Apply a generated index map while exposing only stored composition, lambdas,
and local lets.

This deliberately avoids `whnf`: checked maps contain large dependent proof
programs that should remain opaque while executable maps are composed.
-/
private partial def applyExplicitIndexMap
    (map index : Expr) : MetaM Expr := do
  let map := map.consumeMData
  if map.isAppOfArity ``Function.comp 5 then
    let arguments := map.getAppArgs
    let innerValue ← applyExplicitIndexMap arguments[4]! index
    applyExplicitIndexMap arguments[3]! innerValue
  else
    match map with
    | .lam _ _ body _ =>
        pure (body.instantiate1 index)
    | .letE _ _ assignment body _ =>
        applyExplicitIndexMap (body.instantiate1 assignment) index
    | _ =>
        pure (mkApp map index)

/--
Build function composition as an explicit lambda.

Generated index maps are executable programs. Lambda form exposes their
stored composition without unfolding checked-plan definitions.
-/
def explicitComposition
    (domain outer inner : Expr) : MetaM Expr := do
  withLocalDeclD `index domain fun index => do
    let innerValue ← applyExplicitIndexMap inner index
    let outerValue ← applyExplicitIndexMap outer innerValue
    mkLambdaFVars #[index] outerValue

/--
Estimate the runtime cost of a generated flat-index map.

Only value-level arithmetic contributes to the estimate. Dependent bounds,
types, and correctness proofs are deliberately ignored because they are
erased before execution. The estimate is used to choose between composing a
shape-only chain and materializing its already-certified native result.
-/
partial def flatIndexProgramCost (map : Expr) (fuel : Nat := 1024) : Nat :=
  if fuel = 0 then
    1024
  else
    let map := map.consumeMData
    match map with
    | .lam _ _ body _ =>
        flatIndexProgramCost body (fuel - 1)
    | .letE _ _ assignment body _ =>
        flatIndexProgramCost (body.instantiate1 assignment) (fuel - 1)
    | .proj ``Fin 0 source =>
        flatIndexProgramCost source (fuel - 1)
    | _ =>
        if map.isAppOfArity ``Fin.mk 3 then
          flatIndexProgramCost map.getAppArgs[1]! (fuel - 1)
        else if map.isAppOfArity ``Fin.val 2 then
          flatIndexProgramCost map.getAppArgs[1]! (fuel - 1)
        else if map.isAppOfArity ``Function.comp 5 then
          1 +
            flatIndexProgramCost map.getAppArgs[3]! (fuel - 1) +
            flatIndexProgramCost map.getAppArgs[4]! (fuel - 1)
        else if let some (left, right) :=
            natOperationOperands? ``Nat.add ``HAdd.hAdd map then
          1 +
            flatIndexProgramCost left (fuel - 1) +
            flatIndexProgramCost right (fuel - 1)
        else if let some (left, right) :=
            natOperationOperands? ``Nat.sub ``HSub.hSub map then
          1 +
            flatIndexProgramCost left (fuel - 1) +
            flatIndexProgramCost right (fuel - 1)
        else if let some (left, right) :=
            natOperationOperands? ``Nat.mul ``HMul.hMul map then
          2 +
            flatIndexProgramCost left (fuel - 1) +
            flatIndexProgramCost right (fuel - 1)
        else if let some (left, right) :=
            natOperationOperands? ``Nat.div ``HDiv.hDiv map then
          6 +
            flatIndexProgramCost left (fuel - 1) +
            flatIndexProgramCost right (fuel - 1)
        else if let some (left, right) :=
            natOperationOperands? ``Nat.mod ``HMod.hMod map then
          6 +
            flatIndexProgramCost left (fuel - 1) +
            flatIndexProgramCost right (fuel - 1)
        else
          0

/--
Maximum inherited flat-index cost that remains cheaper to fuse than to
materialize through the native transform kernel.

The weighted estimate reflects native arithmetic cost: quotient and remainder
are more expensive than addition or multiplication. The threshold is
independent of tensor rank and transformation kind.
-/
def maxFusedFlatIndexCost : Nat := 48

/-- Report whether an inherited flat-index map should remain fused. -/
def shouldFuseFlatIndex (map : Expr) : Bool :=
  flatIndexProgramCost map ≤ maxFusedFlatIndexCost

/-- Transport a natural-number bound through equality of two index values. -/
private def indexBoundFromEquality
    (bound hValue rightBound : Expr) : MetaM Expr := do
  let predicate ←
    withLocalDeclD `index (mkConst ``Nat) fun index => do
      let proposition ← mkLT index bound
      mkLambdaFVars #[index] proposition
  let hBound ← mkAppM ``congrArg #[predicate, hValue]
  mkAppM ``Eq.mpr #[hBound, rightBound]

/--
Partially evaluate one compact checked transform index.

The returned equality has the orientation needed to transport the certified
source bound back to the generated arithmetic expression.
-/
private def compileTransformIndex
    (checked outputIndex : Expr) : MetaM (Expr × Expr) := do
  let value ← mkAppM ``Check.CheckedTransform.value #[checked]
  let normalized ← mkAppM ``Check.TransformPlan.normalized #[value]
  let axisLength ← mkAppM ``Check.TransformPlan.axisLength #[value]
  let inputAxes ←
    mkAppM ``Check.NormalizedTransform.inputAxes #[normalized]
  let outputAxes ←
    mkAppM ``Check.NormalizedTransform.outputAxes #[normalized]
  let outputValue ← mkAppM ``Fin.val #[outputIndex]
  let compactIndex ←
    mkAppM ``rearrangeLinearIndex #[
      axisLength, inputAxes, outputAxes, outputValue]
  compileRowMajorIndex compactIndex

/--
Compile a checked output-to-input flat projection.

The executable result is specialized row-major arithmetic when the reflected
plan reduces. Its proof connects that arithmetic directly to linearization of
the plan's independent coordinate projection, without unfolding the checked
plan's proof fields.
-/
def checkedFlatProjection (checked hAxes : Expr) :
    MetaM (Expr × Expr) := do
  let value ← mkAppM ``Check.CheckedTransform.value #[checked]
  let normalized ← mkAppM ``Check.TransformPlan.normalized #[value]
  let inputShape ←
    mkAppM ``Check.NormalizedTransform.input #[normalized]
  let outputShape ←
    mkAppM ``Check.TransformPlan.output #[value]
  let inputSize ← mkAppM ``Shape.size #[inputShape]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let outputIndexType ← mkAppM ``Fin #[outputSize]
  withLocalDeclD `outputIndex outputIndexType fun outputIndex => do
    let (compiledValue, hCompiledCompact) ←
      compileTransformIndex checked outputIndex
    let outputCoordinate ← mkAppM ``Coord.unlinearize #[outputIndex]
    let sourceCoordinate ←
      mkAppM ``Check.CheckedTransform.inputCoordinateOfOutput #[
        checked, hAxes, outputCoordinate]
    let sourceIndex ← mkAppM ``Coord.linearize #[sourceCoordinate]
    let hSourceCompact ←
      mkAppM ``Check.CheckedTransform.inputCoordinateOfOutput_linearize #[
        checked, hAxes, outputIndex]
    let hCompactSource ← mkAppM ``Eq.symm #[hSourceCompact]
    let hCompiledSource ←
      mkAppM ``Eq.trans #[hCompiledCompact, hCompactSource]
    let sourceBound ← mkAppM ``Fin.isLt #[sourceIndex]
    let compiledBound ←
      indexBoundFromEquality inputSize hCompiledSource sourceBound
    let compiledIndex ←
      mkAppOptM ``Fin.mk #[
        some inputSize, some compiledValue, some compiledBound]
    let hCompiledCoordinate ←
      mkAppOptM ``Fin.ext #[
        some inputSize, some compiledIndex,
        some sourceIndex, some hCompiledSource]
    let flatMap ← mkLambdaFVars #[outputIndex] compiledIndex
    let hFlatMap ← mkLambdaFVars #[outputIndex] hCompiledCoordinate
    return (flatMap, hFlatMap)

end TorchLean.Tensor.Internal.Elab.Impl
