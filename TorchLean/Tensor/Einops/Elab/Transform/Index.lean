/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Einops.Elab.Einsum.Index
public meta import TorchLean.Tensor.Einops.Elab.Index
public import TorchLean.Tensor.Einops.Equivalence.Lowering -- shake: keep
public import TorchLean.Tensor.Einops.Elab.Einsum.Index
import TorchLean.Tensor.Einops.Elab.Index

/-!
# Certified transform index compilation

The flat-index function of a checked rearrangement or repeat is compiled while the surrounding
expression elaborates. Only the resulting row-major arithmetic reaches the generated program,
and parser data, axis lookup and checked-plan interpretation stay in erased correctness proofs.

With symbolic dimensions the generated expression keeps symbolic arithmetic, and the returned
theorem identifies that program with the independent coordinate semantics all the same.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open TorchLean.Shape (Coord)

/--
Apply a generated index map, unfolding only stored `Function.comp`, lambdas and local lets.

It avoids `whnf` on purpose: checked maps hold large dependent proof programs that should stay
opaque while executable maps are composed.
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
Build the composition of `outer` after `inner` as a lambda over `domain`.

The lambda form exposes the stored composition of generated index maps without unfolding
checked-plan definitions.
-/
def explicitComposition
    (domain outer inner : Expr) : MetaM Expr := do
  withLocalDeclD `index domain fun index => do
    let innerValue ← applyExplicitIndexMap inner index
    let outerValue ← applyExplicitIndexMap outer innerValue
    mkLambdaFVars #[index] outerValue

/--
Estimate the arithmetic cost of a generated flat-index map.

Addition, subtraction and composition cost 1, multiplication 2, and quotient and remainder 6.
Dependent bounds, types and correctness proofs cost nothing because they are erased before
execution, and so does any unrecognized expression, which limits the estimate to generated
row-major arithmetic. Running out of `fuel` gives 1024. The estimate decides between composing
a shape-only chain and materializing its certified native result.
-/
def flatIndexProgramCost (map : Expr) (fuel : Nat := 1024) : Nat :=
  match fuel with
  | 0 => 1024
  | fuel + 1 =>
    let map := map.consumeMData
    match map with
    | .lam _ _ body _ =>
        flatIndexProgramCost body fuel
    | .letE _ _ assignment body _ =>
        flatIndexProgramCost (body.instantiate1 assignment) fuel
    | .proj ``Fin 0 source =>
        flatIndexProgramCost source fuel
    | _ =>
        if map.isAppOfArity ``Fin.mk 3 then
          flatIndexProgramCost map.getAppArgs[1]! fuel
        else if map.isAppOfArity ``Fin.val 2 then
          flatIndexProgramCost map.getAppArgs[1]! fuel
        else if map.isAppOfArity ``Function.comp 5 then
          1 +
            flatIndexProgramCost map.getAppArgs[3]! fuel +
            flatIndexProgramCost map.getAppArgs[4]! fuel
        else if let some (left, right) :=
            natOperationOperands? ``Nat.add ``HAdd.hAdd map then
          1 +
            flatIndexProgramCost left fuel +
            flatIndexProgramCost right fuel
        else if let some (left, right) :=
            natOperationOperands? ``Nat.sub ``HSub.hSub map then
          1 +
            flatIndexProgramCost left fuel +
            flatIndexProgramCost right fuel
        else if let some (left, right) :=
            natOperationOperands? ``Nat.mul ``HMul.hMul map then
          2 +
            flatIndexProgramCost left fuel +
            flatIndexProgramCost right fuel
        else if let some (left, right) :=
            natOperationOperands? ``Nat.div ``HDiv.hDiv map then
          6 +
            flatIndexProgramCost left fuel +
            flatIndexProgramCost right fuel
        else if let some (left, right) :=
            natOperationOperands? ``Nat.mod ``HMod.hMod map then
          6 +
            flatIndexProgramCost left fuel +
            flatIndexProgramCost right fuel
        else
          0

/--
Heuristic cost limit for fusing an inherited flat-index map, in the weighted units of
`Einops.Elab.flatIndexProgramCost`.

Quotient and remainder weigh more than addition or multiplication because they cost more as
native instructions. The same limit applies to every tensor rank and transformation kind.
-/
def maxFusedFlatIndexCost : Nat := 48

/--
`Einops.Elab.shouldFuseFlatIndex map` is `true` when the estimated cost of `map` is at most the
fusion limit.
-/
def shouldFuseFlatIndex (map : Expr) : Bool :=
  flatIndexProgramCost map ≤ maxFusedFlatIndexCost

/--
Partially evaluate one compact checked transform index.

The returned equality has the orientation needed to transport the certified
source bound back to the generated arithmetic expression.
-/
private def compileTransformIndex
    (checked outputIndex : Expr) : MetaM (Expr × Expr) := do
  let value ← mkAppM ``Transform.value #[checked]
  let normalized ← mkAppM ``TransformPlan.normalized #[value]
  let axisLength ← mkAppM ``TransformPlan.axisLength #[value]
  let inputAxes ←
    mkAppM ``NormalForm.inputAxes #[normalized]
  let outputAxes ←
    mkAppM ``NormalForm.outputAxes #[normalized]
  let outputValue ← mkAppM ``Fin.val #[outputIndex]
  let compactIndex ←
    mkAppM ``Rearrangement.linearIndex #[
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
  let value ← mkAppM ``Transform.value #[checked]
  let normalized ← mkAppM ``TransformPlan.normalized #[value]
  let inputShape ←
    mkAppM ``NormalForm.input #[normalized]
  let outputShape ←
    mkAppM ``TransformPlan.output #[value]
  let inputSize ← mkAppM ``Shape.size #[inputShape]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let outputIndexType ← mkAppM ``Fin #[outputSize]
  withLocalDeclD `outputIndex outputIndexType fun outputIndex => do
    let (compiledValue, hCompiledCompact) ←
      compileTransformIndex checked outputIndex
    let outputCoordinate ← mkAppM ``Coord.unlinearize #[outputIndex]
    let sourceCoordinate ←
      mkAppM ``Transform.inputOfOutput #[
        checked, hAxes, outputCoordinate]
    let sourceIndex ← mkAppM ``Coord.linearize #[sourceCoordinate]
    let hSourceCompact ←
      mkAppM ``Transform.inputOfOutput_linearize #[
        checked, hAxes, outputIndex]
    let hCompactSource ← mkAppM ``Eq.symm #[hSourceCompact]
    let hCompiledSource ←
      mkAppM ``Eq.trans #[hCompiledCompact, hCompactSource]
    let sourceBound ← mkAppM ``Fin.isLt #[sourceIndex]
    let compiledBound ←
      mkAppM ``lt_of_eq_of_lt #[hCompiledSource, sourceBound]
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

end TorchLean.Einops.Elab
