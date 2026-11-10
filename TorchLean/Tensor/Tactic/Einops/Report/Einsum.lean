/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Einops.Elab.Kernel.Analysis
public meta import TorchLean.Tensor.Einops.Elab.Output.Planning
public import TorchLean.Tensor.Einops.Elab.Output.Parallel
public meta import TorchLean.Tensor.Einops.Elab.Einsum.Planning
public meta import TorchLean.Tensor.Tactic.Einops.Report.Decode
public import TorchLean.Tensor.Einops.Elab.Kernel.Analysis
public import TorchLean.Tensor.Einops.Elab.Output.Planning
public import TorchLean.Tensor.Einops.Elab.Einsum.Planning
public import TorchLean.Tensor.Tactic.Einops.Report.Decode

/-!
# Einsum reports

The `einops?` report for a concrete `einsum` certificate: checked axes, contraction work,
generated loops and the correctness chain.

The optimization descriptions are reconstructed from the current shapes, scalar instances and
planning policies, without reading the supplied kernel body or recording which optimizations its
elaborator selected.
-/

public meta section

namespace TorchLean.Einops.Report

open Lean Lean.Elab Tactic Meta

/--
Decode a concrete einsum certificate into its axis analysis, semantic stages,
generated loops, and correctness chain.
-/
def concreteEinsum
    (checked scalarType : Expr) (correctnessTheorem : String) :
    MetaM (Option String) := do
  let checked ← whnfAll checked
  let some checkedFields :=
      constructorFields? checked ``Einsum.mk 10
    | return none
  let pattern := checkedFields[0]!
  let inputShapesExpression := checkedFields[1]!
  let axisLength := checkedFields[2]!
  let inputAxesExpression ←
    mkAppM ``EinsumPattern.inputAxes #[pattern, inputShapesExpression]
  let outputAxesExpression ←
    mkAppM ``EinsumPattern.outputAxes #[pattern, inputShapesExpression]
  let globalAxesExpression ←
    mkAppM ``EinsumPattern.globalAxes #[pattern, inputShapesExpression]
  let outputShapeExpression ←
    mkAppM ``Einsum.output #[checked]
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
      (operandTypes.mapIdx fun index type => (s!"Operand {index}", type)) ++
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
    (inputShapes.zip inputAxes).mapIdx fun index (shape, axes) =>
      s!"  Operand {index}: shape {shape}, axes {formatAxes axes}"
  let lengthDescriptions :=
    globalAxes.map fun axis =>
      s!"{axis.2} = {(lengths.lookup axis.1).getD 0}"
  let diagonalDescriptions :=
    inputAxes.zipIdx.filterMap fun (axes, index) =>
      let repeated := repeatedAxes axes
      if repeated.isEmpty then
        none
      else
        some s!"operand {index} on {formatAxes repeated}"
  let broadcastDetails :=
    (inputShapes.zip inputAxes).zipIdx.filterMap fun ((shape, axes), index) =>
      let broadcasts := broadcastDescriptions shape axes lengths
      if broadcasts.isEmpty then
        none
      else
        some s!"operand {index}: {String.intercalate ", " broadcasts}"
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
      axis.1.isAppOfArity ``EinsumAxis.ellipsis 1
  let ellipsisSummary :=
    if ellipsisAxes.isEmpty then
      "none"
    else
      formatAxes ellipsisAxes
  let rankObligations :=
    (inputShapes.zip inputAxes).mapIdx fun index (shape, axes) =>
      s!"Operand {index}: physical rank {shape.length} equals \
        its {axes.length} expanded logical labels."
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
        {outputShape}."]
  let canonicalAxes := outputAxes ++ contractedAxes
  let logicalStages :=
    [s!"Rep.pull: align every operand with global axes \
        {formatAxes globalAxes} and shape {globalShape}.",
     "Tensor.zipWith: multiply aligned operand values in source order.",
     s!"Rep.reindex: place output axes before contracted axes, changing \
        {formatAxes globalAxes} to {formatAxes canonicalAxes}.",
     s!"Rep.push: sum each fiber over {formatAxes contractedAxes}, \
        producing output shape {outputShape}."]
  let contractionLoop :=
    if contractedAxes.isEmpty then
      "Use the single empty contraction coordinate; no reduction loop remains."
    else
      s!"Traverse contraction shape {contractedShape} on axes \
        {formatAxes contractedAxes}; concrete portable bounds use native \
        USize counters, while symbolic or oversized bounds retain Fin.foldl."
  let outputEntries := outputShape.prod
  let contractionEntries := contractedShape.prod
  let contractionTerms := outputEntries * contractionEntries
  let operandReads := contractionTerms * inputShapes.length
  let multiplications :=
    contractionTerms * (inputShapes.length - 1)
  let additions := if contractedAxes.isEmpty then 0 else contractionTerms
  let factorBounds? ←
    Elab.einsumFactorBounds? scalarType inputAxisValues
      (inputShapes.map fun shape => shape.map mkNatLit) outputAxisValues contractedAxisValues
  let factored := factorBounds?.isSome
  let (prefixCount, suffixCount) :=
    match factorBounds? with
    | some (middleStart, middleStop) => (middleStart, inputShapes.length - middleStop)
    | none => (0, 0)
  let addCommMonoidType ← mkAppM ``AddCommMonoid #[scalarType]
  let hasAddCommMonoid ←
    match ← trySynthInstance addCommMonoidType with
    | .some _ => pure true
    | _ => pure false
  let plannedAxisValues :=
    if hasAddCommMonoid then
      Elab.contractionAxisOrder
        inputAxisValues inputShapes contractedAxisValues
    else
      contractedAxisValues
  let contractionPlanned :=
    plannedAxisValues != contractedAxisValues
  let some plannedAxes :=
      plannedAxisValues.mapM fun axis => (globalAxisValues.zip globalAxes).lookup axis
    | return none
  let planningLoop :=
    if contractionPlanned then
      [s!"Traverse contracted axes in planned order \
          {formatAxes plannedAxes} instead of source order \
          {formatAxes contractedAxes}. A lawful AddCommMonoid \
          certificate permits the coordinate-sum permutation."]
    else if 2 ≤ contractedAxisValues.length && !hasAddCommMonoid then
      [s!"Retain source contraction order \
          {formatAxes contractedAxes}, since the scalar type supplies \
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
        Elab.einsumOutputTileWidth?
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
    Elab.einsumOutputTaskCount
      outputShape contractionEntries
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
      s!"Traverse output shape {outputShape} on axes \
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
     s!"Operand scalar reads before factorization: {operandReads}.",
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
     if contractedAxes.isEmpty then
       "Multiply the operand scalars in source order and store the product \
          directly as the output entry; nothing is added, and no broadcast, \
          product, or contraction tensor is materialized."
     else
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
    [s!"  Output: shape {outputShape}, axes \
        {formatAxes outputAxes}",
     s!"  Contracted axes: {formatAxes contractedAxes}, shape \
        {contractedShape}",
     s!"  Global logical axes: {formatAxes globalAxes}, shape \
        {globalShape}",
     s!"  Inferred lengths: \
        {String.intercalate ", " lengthDescriptions}",
     s!"  Repeated-label diagonals: {diagonalSummary}",
     s!"  Singleton broadcasts: {broadcastSummary}"] ++
      formatSection "Discharged obligations" obligations ++
      ["  Verified logical stages:"] ++
      numberedStageLines 1 logicalStages ++
      ["  Generated execution strategy:"] ++
      numberedStageLines 1 generatedLoops ++
      formatSection "Shape-derived work estimate" workEstimate ++
      formatSection "Correctness"
        [s!"{Lowering.reportName ``Einsum} certifies ranks, axis sources, repeated \
            dimensions, broadcasting, global lengths, and output shape.",
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

end TorchLean.Einops.Report
