/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum.Output.Planning
public import NN.Tensor.Internal.Elab.Einsum.ParallelOutput
public meta import NN.Tensor.Internal.Elab.Einsum.Planning
public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Common
public import NN.Tensor.Internal.Elab.Einsum.Output.Planning
public import NN.Tensor.Internal.Elab.Einsum.Planning
public import NN.Tensor.Internal.Tactic.Report.Analysis.Common

/-!
# Einsum reports

This module reports checked axes, contraction work, generated loops, and the
correctness chain for concrete `einsum` certificates.
-/

public meta section

namespace TorchLean.Tensor.Internal.Report.Impl

open Lean Elab Tactic Meta

/--
Decode a concrete einsum certificate into its axis analysis, semantic stages,
generated loops, and correctness chain.
-/
def concreteEinsumReport
    (checked scalarType : Expr) (correctnessTheorem : String) :
    MetaM (Option String) := do
  let checked ← reportWhnf checked
  let some checkedFields :=
      constructorFields? checked ``Check.CheckedEinsum.mk 10
    | return none
  let pattern := checkedFields[0]!
  let inputShapesExpression := checkedFields[1]!
  let axisLength := checkedFields[2]!
  let inputAxesExpression ←
    mkAppM ``Check.einsumInputAxes #[pattern, inputShapesExpression]
  let outputAxesExpression ←
    mkAppM ``Check.einsumOutputAxes #[pattern, inputShapesExpression]
  let globalAxesExpression ←
    mkAppM ``Check.einsumGlobalAxes #[pattern, inputShapesExpression]
  let outputShapeExpression ←
    mkAppM ``Check.CheckedEinsum.output #[checked]
  let some inputShapes ←
      decodeExprList inputShapesExpression decodeNatList
    | return none
  let some inputAxes ←
      decodeExprList inputAxesExpression fun axes =>
        decodeExprList axes decodeEinsumAxis
    | return none
  let some inputAxisValues ←
      decodeExprList inputAxesExpression fun axes =>
        decodeExprList axes decodeEinsumAxisValue
    | return none
  let some outputAxes ←
      decodeExprList outputAxesExpression decodeEinsumAxis
    | return none
  let some outputAxisValues ←
      decodeExprList outputAxesExpression decodeEinsumAxisValue
    | return none
  let some globalAxes ←
      decodeExprList globalAxesExpression decodeEinsumAxis
    | return none
  let some globalAxisValues ←
      decodeExprList globalAxesExpression decodeEinsumAxisValue
    | return none
  let some outputShape ← decodeNatList outputShapeExpression
    | return none
  let scalarTypeDescription ← formatType scalarType
  let operandTypes ←
    inputShapes.mapM fun shape => concreteTensorType scalarType shape
  let outputType ← concreteTensorType scalarType outputShape
  let typeEntries :=
    [("Scalar type", scalarTypeDescription)] ++
      (((List.range operandTypes.length).zip operandTypes).map
        fun operand => (s!"Operand {operand.1}", operand.2)) ++
      [("Output tensor", outputType)]
  let mut lengths : List (Expr × Nat) := []
  for axis in globalAxes do
    let some length ← decodeNatLiteral (mkApp axisLength axis.1)
      | return none
    lengths := lengths.concat (axis.1, length)
  let some globalShape := axisShape? globalAxes lengths
    | return none
  let contractedAxes := axesNotIn globalAxes outputAxes
  let contractedAxisValues :=
    globalAxisValues.filter fun axis => !outputAxisValues.contains axis
  let some contractedShape := axisShape? contractedAxes lengths
    | return none
  let operandLines :=
    ((List.range inputShapes.length).zip (inputShapes.zip inputAxes)).map
      fun operand =>
        s!"  Operand {operand.1}: shape {formatShape operand.2.1}, axes \
          {formatAxes operand.2.2}"
  let lengthDescriptions :=
    globalAxes.map fun axis =>
      s!"{axis.2} = {(axisLength? lengths axis.1).getD 0}"
  let diagonalDescriptions :=
    ((List.range inputAxes.length).zip inputAxes).filterMap fun operand =>
      let repeated := repeatedAxes operand.2
      if repeated.isEmpty then
        none
      else
        some s!"operand {operand.1} on {formatAxes repeated}"
  let broadcastDetails :=
    ((List.range inputShapes.length).zip (inputShapes.zip inputAxes)).filterMap
      fun operand =>
        let broadcasts :=
          broadcastDescriptions operand.2.1 operand.2.2 lengths
        if broadcasts.isEmpty then
          none
        else
          some s!"operand {operand.1}: \
            {String.intercalate ", " broadcasts}"
  let diagonalSummary :=
    if diagonalDescriptions.isEmpty then
      "none"
    else
      String.intercalate "; " diagonalDescriptions
  let broadcastSummary :=
    if broadcastDetails.isEmpty then
      "none"
    else
      String.intercalate "; " broadcastDetails
  let ellipsisAxes :=
    globalAxes.filter fun axis =>
      axis.1.isAppOfArity ``Check.EinsumAxis.ellipsis 1
  let ellipsisSummary :=
    if ellipsisAxes.isEmpty then
      "none"
    else
      formatAxes ellipsisAxes
  let rankObligations :=
    ((List.range inputShapes.length).zip (inputShapes.zip inputAxes)).map
      fun operand =>
        s!"Operand {operand.1}: physical rank {operand.2.1.length} equals \
          its {operand.2.2.length} expanded logical labels."
  let obligations :=
    ["Grammar: every operand and the output use supported singleton logical \
        axes, with at most one ellipsis per expression.",
     s!"Operand count: {inputShapes.length} tensor operands match \
        {inputAxes.length} checked input expressions."] ++
      rankObligations ++
    [s!"Expanded ellipsis axes: {ellipsisSummary}.",
     s!"Repeated-label dimensions: {diagonalSummary}; every repeated \
        occurrence within an operand has the same physical length.",
     s!"Broadcast compatibility: {broadcastSummary}; every physical \
        dimension is either 1 or its resolved global logical length.",
     "Global lengths: every non-singleton logical length is witnessed by an \
        operand dimension.",
     "Output axes: labels are duplicate-free and every output label is \
        supplied by an input operand.",
     s!"Output derivation: {formatAxes outputAxes} has exact physical shape \
        {formatShape outputShape}."]
  let canonicalAxes := outputAxes ++ contractedAxes
  let logicalStages :=
    [s!"Rep.pull: align every operand with global axes \
        {formatAxes globalAxes} and shape {formatShape globalShape}.",
     "Rep.zipWith: multiply aligned operand values in source order.",
     s!"Rep.reindex: place output axes before contracted axes, changing \
        {formatAxes globalAxes} to {formatAxes canonicalAxes}.",
     s!"Rep.push: sum each fiber over {formatAxes contractedAxes}, \
        producing output shape {formatShape outputShape}."]
  let contractionLoop :=
    if contractedAxes.isEmpty then
      "Use the single empty contraction coordinate; no reduction loop remains."
    else
      s!"Traverse contraction shape {formatShape contractedShape} on axes \
        {formatAxes contractedAxes}; concrete portable bounds use native \
        USize counters, while symbolic or oversized bounds retain Fin.foldl."
  let outputEntries := shapeSize outputShape
  let contractionEntries := shapeSize contractedShape
  let contractionTerms := outputEntries * contractionEntries
  let operandReads := contractionTerms * inputShapes.length
  let multiplications :=
    contractionTerms * (inputShapes.length - 1)
  let additions := contractionTerms
  let invariantOperands :=
    (inputShapes.zip inputAxes).map fun (shape, axes) =>
      (shape.zip axes).all fun (dimension, axis) =>
        (outputAxes.any fun outputAxis => outputAxis.1 == axis.1) ||
          dimension == 1
  let prefixCount := (invariantOperands.takeWhile id).length
  let suffixCount := (invariantOperands.reverse.takeWhile id).length
  let semiringType ← mkAppM ``Semiring #[scalarType]
  let hasSemiring ←
    match ← trySynthInstance semiringType with
    | .some _ => pure true
    | _ => pure false
  let factored :=
    !contractedAxes.isEmpty &&
      hasSemiring &&
      prefixCount + suffixCount < inputShapes.length &&
      (0 < prefixCount || 0 < suffixCount)
  let addCommMonoidType ← mkAppM ``AddCommMonoid #[scalarType]
  let hasAddCommMonoid ←
    match ← trySynthInstance addCommMonoidType with
    | .some _ => pure true
    | _ => pure false
  let plannedAxisValues :=
    if hasAddCommMonoid then
      Elab.Impl.contractionAxisOrder
        inputAxisValues inputShapes contractedAxisValues
    else
      contractedAxisValues
  let contractionPlanned :=
    plannedAxisValues != contractedAxisValues
  let axisName : Check.EinsumAxis → String
    | .named name => name
    | .ellipsis index => s!"ellipsis[{index}]"
  let formatAxisValues (axes : List Check.EinsumAxis) : String :=
    "[" ++ String.intercalate ", " (axes.map axisName) ++ "]"
  let planningLoop :=
    if contractionPlanned then
      [s!"Traverse contracted axes in planned order \
          {formatAxisValues plannedAxisValues} instead of source order \
          {formatAxisValues contractedAxisValues}. A lawful AddCommMonoid \
          certificate permits the coordinate-sum permutation."]
    else if 2 ≤ contractedAxisValues.length && !hasAddCommMonoid then
      [s!"Retain source contraction order \
          {formatAxisValues contractedAxisValues}; the scalar type supplies \
          no lawful AddCommMonoid certificate for reassociation."]
    else
      []
  let scalarReadsPerTerm :=
    if factored then
      inputShapes.length - prefixCount - suffixCount
    else
      inputShapes.length
  let outputTileWidth? :=
    match outputShape.getLast? with
    | some outputLength =>
        Elab.Impl.einsumOutputTileWidth?
          outputLength (some contractionEntries) scalarReadsPerTerm
    | none => none
  let outputBlockingLoop :=
    match outputShape.getLast?, outputTileWidth? with
    | some outputLength, some tileWidth =>
        let completeBlocks := outputLength / tileWidth
        let tail := outputLength % tileWidth
        [s!"Evaluate the final output axis as {completeBlocks} contiguous \
            {tileWidth}-lane blocks plus a {tail}-entry tail. Lanes share \
            contraction-coordinate work while each scalar fold keeps its \
            original order."]
    | _, _ => []
  let outputTaskCount :=
    Elab.Impl.einsumOutputTaskCount
      factored outputShape contractionEntries
  let parallelOutput := 1 < outputTaskCount
  let resultBuffers :=
    if parallelOutput then
      s!"Result storage: {outputTaskCount} ordered chunk arrays; chunk 0 \
        reserves final capacity."
    else
      "Result buffers: 1."
  let factorizationLoop :=
    if factored then
      [s!"Use semiring distributivity to move {prefixCount} source-prefix and \
          {suffixCount} source-suffix contraction-invariant operands outside \
          each inner sum without permuting operand order."]
    else
      []
  let outputLoop :=
    if parallelOutput then
      s!"Partition outer output axis of length {outputShape.headD 0} into \
        {outputTaskCount} balanced ranges. Run chunk 0 on the calling thread \
        and {outputTaskCount - 1} with Task.spawn, then append in row-major \
        order."
    else
      s!"Traverse output shape {formatShape outputShape} on axes \
        {formatAxes outputAxes} with nested row-major loops; concrete \
        portable axes use native USize counters, while symbolic or oversized \
        axes retain Fin.foldl. Each completed scalar is pushed into one \
        preallocated output buffer."
  let multiplicationEstimate :=
    if factored then
      let middleCount :=
        inputShapes.length - prefixCount - suffixCount
      let generatedMultiplications :=
        outputEntries *
          (contractionEntries * (middleCount - 1) +
            prefixCount + suffixCount)
      s!"Scalar multiplications after semiring factorization: \
        {generatedMultiplications} (unfactored source products: \
        {multiplications})."
    else
      s!"Ordered scalar multiplications: {multiplications} \
        (operand count minus one per term; the identity seed is eliminated)."
  let workEstimate :=
    [s!"Output entries: {outputEntries}.",
     s!"Contraction terms per output: {contractionEntries}.",
     s!"Total contraction terms: {contractionTerms}.",
     s!"Operand scalar reads: {operandReads}.",
     multiplicationEstimate,
     s!"Additive accumulator steps: {additions}.",
     s!"Output writes: {outputEntries}.",
     resultBuffers,
     "Materialized aligned, broadcast, product, and contraction tensors: 0."]
  let generatedLoops :=
    [outputLoop] ++
    outputBlockingLoop ++
    planningLoop ++
    [contractionLoop] ++
    factorizationLoop ++
    [
     "Decode row-major output and contraction coordinates once, then reuse \
        hoisted strides and index bases for each operand read.",
     "Multiply the operand scalars and accumulate directly into the current \
        output entry; no broadcast, product, or contraction tensor is \
        materialized."]
  let parallelCorrectness :=
    if parallelOutput then
      ["The chunk assembly theorem proves equality with sequential Array.ofFn; \
          scalar contraction order is unchanged."]
    else
      []
  let planningCorrectness :=
    if contractionPlanned then
      ["Lowering.coordinateSum_permute_of_eq proves that the planned \
          coordinate traversal equals the checked source-order sum."]
    else
      []
  let blockingCorrectness :=
    if outputTileWidth?.isSome then
      ["The arbitrary-width tile semantics and concrete lane refinement \
          identify each output block with the same row-major scalar results; \
          no scalar contraction is reordered."]
    else
      []
  return some <| String.intercalate "\n" <|
    ["einsum"] ++
      typeCheckSection typeEntries ++
      operandLines ++
    [s!"  Output: shape {formatShape outputShape}, axes \
        {formatAxes outputAxes}",
     s!"  Contracted axes: {formatAxes contractedAxes}, shape \
        {formatShape contractedShape}",
     s!"  Global logical axes: {formatAxes globalAxes}, shape \
        {formatShape globalShape}",
     s!"  Inferred lengths: \
        {String.intercalate ", " lengthDescriptions}",
     s!"  Repeated-label diagonals: {diagonalSummary}",
     s!"  Singleton broadcasts: {broadcastSummary}"] ++
      reportSection "Discharged obligations" obligations ++
      ["  Verified logical stages:"] ++
      numberedStageLines 1 logicalStages ++
      ["  Generated execution strategy:"] ++
      numberedStageLines 1 generatedLoops ++
      reportSection "Shape-derived work estimate" workEstimate ++
      reportSection "Correctness"
        ["CheckedEinsum certifies ranks, axis sources, repeated dimensions, \
            broadcasting, global lengths, and output shape.",
         "Native loop theorems identify every USize traversal with \
            Fin.foldl; the row-major output theorem identifies the completed \
            buffer with Array.ofFn."] ++
        planningCorrectness ++
        blockingCorrectness ++
        parallelCorrectness ++
        [
         s!"{correctnessTheorem} identifies the executable kernel with \
            Semantics.denoteEinsum.",
         "Proven result: the executable tensor equals the independent einsum \
            denotation."] ++
      performanceFooter

end TorchLean.Tensor.Internal.Report.Impl
