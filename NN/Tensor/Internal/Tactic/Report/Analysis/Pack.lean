/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import Mathlib.Algebra.GroupWithZero.Nat
public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Common

/-!
# Pack reports

This module reports segment metadata and native execution for concrete `pack`
and `unpack` certificates.
-/

public meta section

namespace TorchLean.Tensor.Internal.Report.Impl

open Lean Elab Tactic Meta

/-- Decode concrete pack or unpack metadata into its segment-level report. -/
def concretePackReport (operation : String)
    (checked scalarType : Expr) :
    MetaM (Option String) := do
  let inputShapesExpression ←
    mkAppM ``Check.CheckedPack.inputShapes #[checked]
  let leadingShapeExpression ←
    mkAppM ``Check.CheckedPack.leadingShape #[checked]
  let trailingShapeExpression ←
    mkAppM ``Check.CheckedPack.trailingShape #[checked]
  let metadataExpression ←
    mkAppM ``Check.CheckedPack.metadata #[checked]
  let segmentLengthsExpression ←
    mkAppM ``Check.CheckedPack.segmentLengths #[checked]
  let outputShapeExpression ←
    mkAppM ``Check.CheckedPack.output #[checked]
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
    ((List.range componentTypes.length).zip componentTypes).map
      fun component => (s!"Component {component.1}", component.2)
  let typeEntries :=
    if operation = "pack" then
      [("Scalar type", scalarTypeDescription)] ++
        componentTypeEntries ++
        [("Rep output tensor", packedType)]
    else
      [("Scalar type", scalarTypeDescription),
       ("Rep input tensor", packedType)] ++
        componentTypeEntries
  let componentDetails :=
    ((List.range inputShapes.length).zip
      (inputShapes.zip (metadata.zip segmentLengths))).map
      fun component =>
        s!"Component {component.1}: {formatShape component.2.1} = \
          {formatShape leadingShape} ++ {formatShape component.2.2.1} ++ \
          {formatShape trailingShape}; packed segment length \
          {component.2.2.2}."
  let mut segmentStart := 0
  let mut partitionDetails : List String := []
  for (component, segmentLength) in
      (List.range segmentLengths.length).zip segmentLengths do
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
     s!"Fixed ranks: leading shape {formatShape leadingShape} has rank \
        {leadingShape.length}; trailing shape {formatShape trailingShape} \
        has rank {trailingShape.length}.",
     s!"Component count: {inputShapes.length} component shapes, metadata \
        entries, and segment lengths agree."] ++
      decompositionDetails ++
      partitionDetails ++
    [s!"Rep-axis partition: segment lengths \
        {formatShape segmentLengths} sum exactly to {segmentLengths.sum}.",
     s!"Output derivation: prefix ++ packed axis ++ suffix gives \
        {formatShape outputShape}."]
  let logicalStages :=
    if operation = "pack" then
      [s!"Rep.reshape: flatten each star shape \
          {formatShapes metadata} to segment lengths \
          {formatShape segmentLengths}, preserving fixed prefix \
          {formatShape leadingShape} and suffix \
          {formatShape trailingShape}.",
       s!"Rep.concatenateAxes: concatenate those segments along the \
          packed axis, producing {formatShape outputShape}."]
    else
      [s!"Rep.splitAxis: split packed input {formatShape outputShape} at \
          segment lengths {formatShape segmentLengths}.",
       s!"Rep.reshape: restore component star shapes \
          {formatShapes metadata}, producing component shapes \
          {formatShapes inputShapes}."]
  let nativeExecution :=
    if operation = "pack" then
      "allocate one packed output buffer and map each flat index directly to \
        its certified component coordinate."
    else
      "allocate each component buffer and map every component flat index \
        directly to its certified packed coordinate."
  let (correctnessTheorem, denotation) :=
    if operation = "pack" then
      ("Lowering.packTensor_correct", "Semantics.denotePack")
    else
      ("Lowering.unpackTensor_correct", "Semantics.denoteUnpack")
  let shapeLine :=
    if operation = "pack" then
      s!"  Component shapes: {formatShapes inputShapes}"
    else
      s!"  Rep shape: {formatShape outputShape}"
  let resultLine :=
    if operation = "pack" then
      s!"  Rep shape: {formatShape outputShape}"
    else
      s!"  Component shapes: {formatShapes inputShapes}"
  let componentEntries := inputShapes.map shapeSize
  let totalComponentEntries := componentEntries.sum
  let packedEntries := shapeSize outputShape
  let workEstimate :=
    if operation = "pack" then
      [s!"Component entries: {formatShape componentEntries}; total \
          {totalComponentEntries}.",
       s!"Rep output entries: {packedEntries}.",
       s!"Scalar reads / writes: {totalComponentEntries} / {packedEntries}.",
       "Result buffers: 1 packed tensor.",
       "Materialized reshape and concatenation tensors: 0."]
    else
      [s!"Rep input entries: {packedEntries}.",
       s!"Component entries: {formatShape componentEntries}; total \
          {totalComponentEntries}.",
       s!"Scalar reads / writes: {totalComponentEntries} / \
          {totalComponentEntries}.",
       s!"Result buffers: {inputShapes.length}, one per component.",
       "Materialized split and reshape tensors: 0."]
  return some <| String.intercalate "\n" <|
    [operation] ++
      typeCheckSection typeEntries ++
      [shapeLine, resultLine,
       s!"  Fixed prefix / suffix: {formatShape leadingShape} / \
          {formatShape trailingShape}",
       s!"  Star metadata: {formatShapes metadata}",
       s!"  Segment lengths: {formatShape segmentLengths}"] ++
      reportSection "Discharged obligations" obligations ++
      ["  Verified logical stages:"] ++
      numberedStageLines 1 logicalStages ++
      reportSection "Generated execution strategy" [nativeExecution] ++
      reportSection "Shape-derived work estimate" workEstimate ++
      reportSection "Correctness"
        [s!"Theorem: {correctnessTheorem}.",
         s!"Proven result: the native program equals {denotation}."] ++
      performanceFooter

end TorchLean.Tensor.Internal.Report.Impl
