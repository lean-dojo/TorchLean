/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Tactic.Einops.Report.Decode
public import TorchLean.Tensor.Tactic.Einops.Report.Decode

/-!
# Pack reports

The `einops?` report for a concrete `pack` or `unpack` certificate: segment metadata, native
execution and the correctness theorem.

Unpack returns a component-indexed function, and its reported totals count one evaluation of each
component. Each component's buffer is built when that component is evaluated.
-/

public meta section

namespace TorchLean.Einops.Report

open Lean Lean.Elab Tactic Meta

/-- Decode concrete pack or unpack metadata into its segment-level report. -/
def concretePack (operation : String)
    (checked scalarType : Expr) :
    MetaM (Option String) := do
  let inputShapesExpression ←
    mkAppM ``Pack.inputShapes #[checked]
  let leadingShapeExpression ←
    mkAppM ``Pack.leadingShape #[checked]
  let trailingShapeExpression ←
    mkAppM ``Pack.trailingShape #[checked]
  let metadataExpression ←
    mkAppM ``Pack.metadata #[checked]
  let segmentLengthsExpression ←
    mkAppM ``Pack.segmentLengths #[checked]
  let outputShapeExpression ←
    mkAppM ``Pack.output #[checked]
  let some inputShapes ←
      decodeExprList inputShapesExpression decodeNatList
    | return none
  let some leadingShape ← decodeNatList leadingShapeExpression
    | return none
  let some trailingShape ← decodeNatList trailingShapeExpression
    | return none
  let some metadata ← decodeExprList metadataExpression decodeNatList
    | return none
  let some segmentLengths ← decodeNatList segmentLengthsExpression
    | return none
  let some outputShape ← decodeNatList outputShapeExpression
    | return none
  let scalarTypeDescription ← formatType scalarType
  let componentTypes ←
    inputShapes.mapM fun shape => concreteTensorType scalarType shape
  let packedType ← concreteTensorType scalarType outputShape
  let componentTypeEntries :=
    componentTypes.mapIdx fun index type => (s!"Component {index}", type)
  let typeEntries :=
    if operation = "pack" then
      [("Scalar type", scalarTypeDescription)] ++
        componentTypeEntries ++
        [("Packed output tensor", packedType)]
    else
      [("Scalar type", scalarTypeDescription),
       ("Packed input tensor", packedType)] ++
        componentTypeEntries
  let componentDetails :=
    (inputShapes.zip (metadata.zip segmentLengths)).mapIdx
      fun index (shape, starShape, segmentLength) =>
        s!"Component {index}: {shape} = \
          {leadingShape} ++ {starShape} ++ \
          {trailingShape}; packed segment length \
          {segmentLength}."
  let mut segmentStart := 0
  let mut partitionDetails : List String := []
  for (segmentLength, component) in segmentLengths.zipIdx do
    let segmentEnd := segmentStart + segmentLength
    partitionDetails := partitionDetails.concat <|
      s!"Component {component}: packed-axis interval \
        [{segmentStart}, {segmentEnd})."
    segmentStart := segmentEnd
  let decompositionDetails :=
    if componentDetails.isEmpty then
      ["Component decomposition: the checked family is empty."]
    else
      componentDetails
  let obligations :=
    [s!"Grammar: exactly one `*` separates {leadingShape.length} fixed \
        leading axes from {trailingShape.length} fixed trailing axes.",
     s!"Fixed ranks: leading shape {leadingShape} has rank \
        {leadingShape.length}; trailing shape {trailingShape} \
        has rank {trailingShape.length}.",
     s!"Component count: {inputShapes.length} component shapes, metadata \
        entries, and segment lengths agree."] ++
      decompositionDetails ++
      partitionDetails ++
    [s!"Packed-axis partition: segment lengths \
        {segmentLengths} sum exactly to {segmentLengths.sum}.",
     s!"Output derivation: prefix ++ packed axis ++ suffix gives \
        {outputShape}."]
  let logicalStages :=
    if operation = "pack" then
      [s!"Rep.reshape: flatten each star shape \
          {metadata} to segment lengths \
          {segmentLengths}, preserving fixed prefix \
          {leadingShape} and suffix \
          {trailingShape}.",
       s!"Rep.concatenateAxes: concatenate those segments along the \
          packed axis, producing {outputShape}."]
    else
      [s!"Rep.splitAxis: split packed input {outputShape} at \
          segment lengths {segmentLengths}.",
       s!"Rep.reshape: restore component star shapes \
          {metadata}, producing component shapes \
          {inputShapes}."]
  let nativeExecution :=
    if operation = "pack" then
      "allocate one packed output buffer and map each flat index directly to \
        its certified component coordinate."
    else
      "allocate each component buffer and map every component flat index \
        directly to its certified packed coordinate."
  let (correctnessTheorem, denotation) :=
    if operation = "pack" then
      (Lowering.reportName Lowering.packTensor.correctness,
        Lowering.reportName Lowering.packTensor.denotation)
    else
      (Lowering.reportName Lowering.unpackTensor.correctness,
        Lowering.reportName Lowering.unpackTensor.denotation)
  let shapeLine :=
    if operation = "pack" then
      s!"  Component shapes: {inputShapes}"
    else
      s!"  Packed shape: {outputShape}"
  let resultLine :=
    if operation = "pack" then
      s!"  Packed shape: {outputShape}"
    else
      s!"  Component shapes: {inputShapes}"
  let componentEntries := inputShapes.map List.prod
  let totalComponentEntries := componentEntries.sum
  let packedEntries := outputShape.prod
  let workEstimate :=
    if operation = "pack" then
      [s!"Component entries: {componentEntries}; total \
          {totalComponentEntries}.",
       s!"Packed output entries: {packedEntries}.",
       s!"Scalar reads / writes: {totalComponentEntries} / {packedEntries}.",
       "Result buffers: 1 packed tensor.",
       "Materialized reshape and concatenation tensors: 0."]
    else
      [s!"Packed input entries: {packedEntries}.",
       s!"Component entries: {componentEntries}; total \
          {totalComponentEntries}.",
       s!"Scalar reads / writes: {totalComponentEntries} / \
          {totalComponentEntries}.",
       s!"Result buffers: {inputShapes.length}, one per component.",
       "Materialized split and reshape tensors: 0."]
  return some <| String.intercalate "\n" <|
    [operation] ++
      typeCheckSection typeEntries ++
      [shapeLine, resultLine,
       s!"  Fixed prefix / suffix: {leadingShape} / \
          {trailingShape}",
       s!"  Star metadata: {metadata}",
       s!"  Segment lengths: {segmentLengths}"] ++
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
