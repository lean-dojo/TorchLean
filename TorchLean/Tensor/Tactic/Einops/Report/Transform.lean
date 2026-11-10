/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Tactic.Einops.Report.Decode
public import TorchLean.Tensor.Tactic.Einops.Report.Decode

/-!
# Transform reports

The `einops?` report for a concrete `rearrange`, `repeat` or `reduce` certificate.

Work estimates count the generated traversal and calls to supplied reducers, fold steps, and
finalizers. They exclude work and allocations performed inside those user functions.
-/

public meta section

namespace TorchLean.Einops.Report

open Lean Lean.Elab Tactic Meta

/--
Decode a concrete rearrange, repeat, or reduce certificate into its normalized
axes, verified logical stages, and native execution strategy.
-/
def concreteTransform (operation : String)
    (checked inputScalarType outputScalarType : Expr)
    (supplementaryTypeEntries : List (String × String))
    (nativeExecution correctnessTheorem denotation : String)
    (nonemptyReduction := false) (foldReduction := false) :
    MetaM (Option String) := do
  let some checkedFields :=
      constructorFields? checked ``Transform.mk 2
    | return none
  let some planFields :=
      constructorFields? checkedFields[0]! ``TransformPlan.mk 3
    | return none
  let some normalizationFields :=
      constructorFields? planFields[0]! ``CheckedNormalization.mk 2
    | return none
  let some normalizedFields :=
      constructorFields? normalizationFields[0]!
        ``NormalForm.mk 7
    | return none
  let some inputShape ← decodeNatList normalizedFields[2]!
    | return none
  let some outputShape ← decodeNatList planFields[2]!
    | return none
  let inputScalarTypeDescription ← formatType inputScalarType
  let outputScalarTypeDescription ← formatType outputScalarType
  let inputTensorType ← concreteTensorType inputScalarType inputShape
  let outputTensorType ← concreteTensorType outputScalarType outputShape
  let scalarTypeEntries :=
    if operation = "rearrange" ∨ operation = "repeat" then
      [("Scalar type", inputScalarTypeDescription)]
    else
      [("Input scalar type", inputScalarTypeDescription),
       ("Output scalar type", outputScalarTypeDescription)]
  let typeEntries :=
    scalarTypeEntries ++ supplementaryTypeEntries ++
      [("Input tensor", inputTensorType),
       ("Output tensor", outputTensorType)]
  let some ellipsisRank ← decodeNatLiteral normalizedFields[3]!
    | return none
  let some inputGroups ←
      decodeExprList normalizedFields[5]! fun group =>
        decodeExprList group decodeTransformAxis
    | return none
  let some outputGroups ←
      decodeExprList normalizedFields[6]! fun group =>
        decodeExprList group decodeTransformAxis
    | return none
  let inputAxes := inputGroups.flatten
  let outputAxes := outputGroups.flatten
  let allAxes := uniqueAxes (inputAxes ++ outputAxes)
  let mut lengths : List (Expr × Nat) := []
  for axis in allAxes do
    let some length ←
        decodeNatLiteral (mkApp planFields[1]! axis.1)
      | return none
    lengths := lengths.concat (axis.1, length)
  let some elementaryInputShape := axisShape? inputAxes lengths
    | return none
  let some elementaryOutputShape := axisShape? outputAxes lengths
    | return none
  let reducedAxes := axesNotIn inputAxes outputAxes
  let reducedShape := (axisShape? reducedAxes lengths).getD []
  let lengthDescriptions :=
    allAxes.map fun axis =>
      let length := (lengths.lookup axis.1).getD 0
      s!"{axis.2} = {length}"
  let literalAxes :=
    allAxes.filterMap fun axis =>
      if axis.1.isAppOfArity ``AxisId.anonymous 2 then
        some axis.2
      else
        none
  let literalSummary :=
    if literalAxes.isEmpty then
      "none"
    else
      String.intercalate ", " literalAxes
  let axisRelation :=
    if operation = "rearrange" then
      "Axis relation: input and output contain exactly the same elementary axes."
    else if operation = "repeat" then
      "Axis relation: every input axis is preserved; only output axes may be introduced."
    else
      "Axis relation: every retained output axis comes from the input."
  let mut obligations :=
    ["Grammar: both sides have at most one ellipsis, every output ellipsis \
        is sourced by the input, and grouping is normalized.",
     s!"Rank: {inputShape.length} input dimensions reconstruct \
        {inputGroups.length} normalized groups; the ellipsis expands to \
        {ellipsisRank} axes.",
     s!"Anonymous literal axes: {literalSummary}; every literal length agrees \
        with its inferred logical length.",
     s!"Input reconstruction: grouped elementary lengths give exactly \
        {inputShape}.",
     s!"Output derivation: grouped elementary lengths give exactly \
        {outputShape} at rank {outputShape.length}.",
     axisRelation]
  if nonemptyReduction then
    obligations := obligations ++
      [s!"Nonempty reduction: fiber shape {reducedShape} has \
          certified positive size {reducedShape.prod}."]
  let logicalStages :=
    if operation = "rearrange" then
      [s!"Rep.reshape: physical input {inputShape} to \
          elementary {elementaryInputShape} on \
          {formatAxes inputAxes}.",
       s!"Rep.reindex: permute {formatAxes inputAxes} to \
          {formatAxes outputAxes}.",
       s!"Rep.reshape: elementary output \
          {elementaryOutputShape} to physical output \
          {outputShape}."]
    else if operation = "repeat" then
      let introducedAxes := axesNotIn outputAxes inputAxes
      let introducedShape :=
        (axisShape? introducedAxes lengths).getD []
      let singletonShape :=
        elementaryInputShape ++ List.replicate introducedAxes.length 1
      let expandedShape := elementaryInputShape ++ introducedShape
      [s!"Rep.reshape: physical input {inputShape} to \
          elementary {elementaryInputShape}.",
       s!"Rep.reshape: append singleton dimensions for introduced axes \
          {formatAxes introducedAxes}, producing \
          {singletonShape}.",
       s!"Rep.broadcast: expand those singleton dimensions to \
          {expandedShape}.",
       s!"Rep.reindex: permute {formatAxes (inputAxes ++ introducedAxes)} \
          to {formatAxes outputAxes}.",
       s!"Rep.reshape: elementary output \
          {elementaryOutputShape} to physical output \
          {outputShape}."]
    else
      let canonicalShape := elementaryOutputShape ++ reducedShape
      let reductionStage :=
        if nonemptyReduction then
          s!"Rep.reduceNonempty: aggregate each certified nonempty fiber \
            over reduced axes {formatAxes reducedAxes}."
        else
          s!"Rep.reduce: aggregate each finite fiber over reduced axes \
            {formatAxes reducedAxes}."
      [s!"Rep.reshape: physical input {inputShape} to \
          elementary {elementaryInputShape}.",
       s!"Rep.reindex: put retained axes {formatAxes outputAxes} first \
          and reduced axes {formatAxes reducedAxes} last, producing \
          {canonicalShape}.",
       reductionStage,
       s!"Rep.reshape: retained elementary shape \
          {elementaryOutputShape} to physical output \
          {outputShape}."]
  let inputEntries := inputShape.prod
  let outputEntries := outputShape.prod
  let workEstimate :=
    if operation = "rearrange" ∨ operation = "repeat" then
      [s!"Output entries: {outputEntries}.",
       s!"Scalar reads / writes: {outputEntries} / {outputEntries}.",
       "Result buffers: 1.",
       "Materialized intermediate tensors: 0."]
    else
      let fiberEntries := reducedShape.prod
      let visits := outputEntries * fiberEntries
      let reductionCalls :=
        if foldReduction && nonemptyReduction then
          [s!"First-value initializations: {outputEntries}.",
           s!"Fold-step calls: {outputEntries * (fiberEntries - 1)}."]
        else if foldReduction then
          [s!"Fold-step calls: {visits}.",
           s!"Finalizer calls: {outputEntries}."]
        else
          [s!"Aggregate calls: {outputEntries}."]
      [s!"Input / output entries: {inputEntries} / {outputEntries}.",
       s!"Reduction fiber entries per output: {fiberEntries}.",
       s!"Input visits: {visits}.",
       s!"Output writes: {outputEntries}."] ++ reductionCalls ++
      ["Result buffers: 1.",
       "Materialized reshape, permutation, and fiber tensors: 0."]
  return some <| String.intercalate "\n" <|
    [operation] ++
      typeCheckSection typeEntries ++
    [
     s!"  Physical shape: {inputShape} -> \
        {outputShape}",
     s!"  Normalized axes: {formatAxisGroups inputGroups} -> \
        {formatAxisGroups outputGroups}",
     s!"  Inferred lengths: \
        {String.intercalate ", " lengthDescriptions}",
     ] ++
      formatSection "Discharged obligations" obligations ++
      ["  Verified logical stages:"] ++
      numberedStageLines 1 logicalStages ++
      formatSection "Generated execution strategy" [nativeExecution] ++
      formatSection "Shape-derived work estimate" workEstimate ++
      formatSection "Correctness"
        [s!"Theorem: {correctnessTheorem}.",
         s!"Proven result: the native program equals {denotation}."] ++
      performanceFooter

end TorchLean.Einops.Report
