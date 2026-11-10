/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Tactic.Einops.Report.Einsum
public meta import TorchLean.Tensor.Tactic.Einops.Report.Pack
public meta import TorchLean.Tensor.Tactic.Einops.Report.ParseShape
public meta import TorchLean.Tensor.Tactic.Einops.Report.Symbolic
public meta import TorchLean.Tensor.Tactic.Einops.Report.Transform
public import TorchLean.Tensor.Tactic.Einops.Report.Einsum
public import TorchLean.Tensor.Tactic.Einops.Report.Pack
public import TorchLean.Tensor.Tactic.Einops.Report.ParseShape
public import TorchLean.Tensor.Tactic.Einops.Report.Symbolic
public import TorchLean.Tensor.Tactic.Einops.Report.Transform

/-!
# Report rendering

`renderApplication` sends each reflected application to its operation-specific report. When the
dimensions are not concrete, the concrete decoder returns `none` and a symbolic report takes its
place.
-/

public meta section

namespace TorchLean.Einops

open Lean Lean.Elab Tactic Meta

namespace Report

/-- Read the final public operation from a fused checked transformation. -/
private def fusedTransformOperation? (checked : Expr) :
    MetaM (Option String) := do
  let value ← mkAppM ``Transform.value #[checked]
  let normalized ← mkAppM ``TransformPlan.normalized #[value]
  let kind ←
    withTransparency .reducible <|
      whnf (← mkAppM ``NormalForm.kind #[normalized])
  if kind.isConstOf ``TransformKind.rearrange then
    return some "rearrange"
  if kind.isConstOf ``TransformKind.expand then
    return some "repeat"
  return none

/--
Render the checked types, obligations, lowering stages, execution strategy,
static work estimates, and correctness theorem for a recognized operation.
-/
def renderApplication (expression : Expr) : MetaM String := do
  let arguments := expression.getAppArgs
  if Lowering.transformTensorFused.isApp expression then
    let operation :=
      (← fusedTransformOperation? arguments[3]!).getD "shape transform"
    let scalarType ← formatType arguments[0]!
    let sourceShape ← formatType arguments[2]!
    let coordinateMapType ← inferredType arguments[5]!
    let flatMapType ← inferredType arguments[6]!
    let sourceTensorType ← inferredType arguments[8]!
    let supplementaryTypeEntries := [
      ("Original source shape", sourceShape),
      ("Coordinate semantics", coordinateMapType),
      ("Native flat-index map", flatMapType)]
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Original source tensor", sourceTensorType),
      ("Output tensor", s!"checked.OutputTensor {scalarType}")] ++
      supplementaryTypeEntries
    let nativeExecution :=
      "compose every checked source-index map at elaboration time, then fill \
        one final row-major output buffer by reading the original source \
        buffer directly; multidimensional coordinates and intermediate \
        tensors are absent from the scalar loop."
    let correctnessTheorem := Lowering.reportName <|
      (Lowering.transformTensorFused.operationCorrectness.lookup operation).getD
        Lowering.transformTensorFused.correctness
    let denotation :=
      if operation = "rearrange" then
        Lowering.reportName Lowering.rearrangeTensor.denotation ++
          " after the preceding coordinate pullback"
      else if operation = "repeat" then
        Lowering.reportName Lowering.repeatTensor.denotation ++
          " after the preceding coordinate pullback"
      else
        "the checked coordinate pullback"
    return (← concreteTransform operation arguments[3]!
        arguments[0]! arguments[0]! supplementaryTypeEntries
        nativeExecution correctnessTheorem denotation).getD <|
      symbolic operation symbolicTypeEntries
        ["The final checked plan certifies grammar, ellipsis expansion, \
            axis lengths, and grouped shape equations.",
         "The coordinate map records the independent transformation \
            semantics.",
         "The flat-index certificate proves that every native source index \
            is the linearization of the corresponding source coordinate."]
        ["compose all output-to-source flat-index maps",
         "fill one final output buffer directly from the original source buffer"]
        ["Scalar reads and writes: Shape.size outputShape.",
         "Result buffers: 1.",
         "Materialized intermediate shape transformations: 0.",
         "Multidimensional coordinates constructed in the scalar loop: 0."]
        nativeExecution correctnessTheorem denotation
  else if Lowering.rearrangeTensor.isApp expression then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input tensor", s!"checked.InputTensor {scalarType}"),
      ("Output tensor", s!"checked.OutputTensor {scalarType}")]
    return (← concreteTransform "rearrange" arguments[2]!
        arguments[0]! arguments[0]! []
        "reuse storage for both reshapes and allocate one reindexed output buffer."
        (Lowering.reportName Lowering.rearrangeTensor.correctness)
        (Lowering.reportName Lowering.rearrangeTensor.denotation)).getD <|
      symbolic "rearrange" symbolicTypeEntries
        ["Grammar, ellipsis expansion, literal-axis lengths, and grouped \
            shape equations are certified by the checked plan.",
         "Axis relation: input and output contain exactly the same \
            elementary axes.",
         "The symbolic output shape is derived from the certified output \
            groups and axis-length function."]
        ["reshape physical input into elementary axes",
         "permute elementary axes", "reshape into physical output"]
        ["Scalar reads and writes: Shape.size outputShape.",
         "Result buffers: 1.",
         "Materialized intermediate tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "zero-copy reshapes around one allocated reindex buffer"
        (Lowering.reportName Lowering.rearrangeTensor.correctness)
        (Lowering.reportName Lowering.rearrangeTensor.denotation)
  else if Lowering.repeatTensor.isApp expression then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input tensor", s!"checked.InputTensor {scalarType}"),
      ("Output tensor", s!"checked.OutputTensor {scalarType}")]
    return (← concreteTransform "repeat" arguments[2]!
        arguments[0]! arguments[0]! []
        "allocate one output buffer and read each entry through the certified \
          output-to-input coordinate map."
        (Lowering.reportName Lowering.repeatTensor.correctness)
        (Lowering.reportName Lowering.repeatTensor.denotation)).getD <|
      symbolic "repeat" symbolicTypeEntries
        ["Grammar, ellipsis expansion, literal-axis lengths, and grouped \
            shape equations are certified by the checked plan.",
         "Axis relation: every input elementary axis is preserved; only \
            output axes may be introduced.",
         "Every introduced axis has a certified symbolic length."]
        ["reshape into elementary axes and append singleton dimensions",
         "broadcast introduced dimensions", "permute elementary axes",
         "reshape into physical output"]
        ["Scalar reads and writes: Shape.size outputShape.",
         "Result buffers: 1.",
         "Materialized broadcast and permutation tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one output fill through the certified output-to-input coordinate map"
        (Lowering.reportName Lowering.repeatTensor.correctness)
        (Lowering.reportName Lowering.repeatTensor.denotation)
  else if Lowering.reduceFoldTensor.isApp expression ||
      Lowering.reduceFoldViewTensor.isApp expression then
    let entry :=
      if Lowering.reduceFoldTensor.isApp expression then Lowering.reduceFoldTensor
      else Lowering.reduceFoldViewTensor
    let operation := "reduce (ordered fold)"
    let nativeExecution :=
      "allocate one output buffer, fold each removed-axis fiber directly \
        from left to right in row-major order, and finalize it once."
    let correctnessTheorem :=
      Lowering.reportName entry.correctness
    let denotation :=
      Lowering.reportName entry.denotation ++
        " with the same row-major left-fold order"
    let logicalStages :=
      ["reshape into elementary axes", "move retained axes first",
       "aggregate removed-axis fibers", "reshape into physical output"]
    let inputScalarType ← formatType arguments[0]!
    let accumulatorType ← formatType arguments[1]!
    let outputScalarType ← formatType arguments[2]!
    let stepType ← inferredType arguments[5]!
    let initialType ← inferredType arguments[6]!
    let finishType ← inferredType arguments[7]!
    let supplementaryTypeEntries := [
      ("Accumulator type", accumulatorType),
      ("Fold step", stepType),
      ("Initial accumulator", initialType),
      ("Finalizer", finishType)]
    let symbolicTypeEntries := [
      ("Input scalar type", inputScalarType),
      ("Output scalar type", outputScalarType)] ++
      supplementaryTypeEntries ++
      [("Input tensor", s!"checked.InputTensor {inputScalarType}"),
       ("Output tensor", s!"checked.OutputTensor {outputScalarType}")]
    let symbolicObligations :=
      ["Grammar, ellipsis expansion, literal-axis lengths, and grouped shape \
          equations are certified by the checked plan.",
       "Axis relation: every retained output axis comes from the input.",
       "The reduction-fiber shape and cardinality are derived from exactly \
          the removed elementary axes."]
    let symbolicWork :=
      ["Output entries: Shape.size outputShape.",
       "Input visits and fold-step calls: Shape.size outputShape × \
          reductionFiberSize.",
       "Finalizer calls and output writes: Shape.size outputShape.",
       "Result buffers: 1.",
       "Materialized fiber tensors: 0.",
       "Exact numeric counts require concrete local dimensions."]
    return (← concreteTransform operation arguments[8]!
        arguments[0]! arguments[2]! supplementaryTypeEntries
        nativeExecution correctnessTheorem denotation
        false true).getD <|
      symbolic operation symbolicTypeEntries symbolicObligations
        logicalStages symbolicWork nativeExecution correctnessTheorem
        denotation
  else if Lowering.reduceNonemptyFoldTensor.isApp expression ||
      Lowering.reduceNonemptyFoldViewTensor.isApp expression then
    let entry :=
      if Lowering.reduceNonemptyFoldTensor.isApp expression then
        Lowering.reduceNonemptyFoldTensor
      else Lowering.reduceNonemptyFoldViewTensor
    let step := arguments[2]!
    let operation := "reduce (ordered nonempty fold)"
    let inputScalarType ← formatType arguments[0]!
    let stepType ← inferredType step
    let supplementaryTypeEntries := [
      ("First-value fold step", stepType)]
    let symbolicTypeEntries := [
      ("Scalar type", inputScalarType),
      ("First-value fold step", stepType),
      ("Input tensor", s!"checked.InputTensor {inputScalarType}"),
      ("Output tensor", s!"checked.OutputTensor {inputScalarType}")]
    let nativeExecution :=
      "allocate one output buffer, initialize each result from the first \
        value of its certified nonempty row-major fiber, then fold the \
        remaining values from left to right."
    let correctnessTheorem :=
      Lowering.reportName entry.correctness
    let denotation :=
      Lowering.reportName entry.denotation ++
        " with the same row-major first-value fold order"
    return (← concreteTransform operation arguments[3]!
        arguments[0]! arguments[0]! supplementaryTypeEntries
        nativeExecution correctnessTheorem denotation true true).getD <|
      symbolic operation symbolicTypeEntries
        ["Grammar, ellipsis expansion, literal-axis lengths, and grouped \
            shape equations are certified by the checked plan.",
         "Axis relation: every retained output axis comes from the input.",
         "The reduction-fiber cardinality is certified positive, so its \
            first row-major value exists."]
        ["reshape into elementary axes", "move retained axes first",
         "initialize from the first value and fold the remaining fiber in \
            row-major order",
         "reshape into physical output"]
        ["Output entries: Shape.size outputShape.",
         "Input visits: Shape.size outputShape × reductionFiberSize.",
         "First-value initializations and output writes: \
            Shape.size outputShape.",
         "Fold-step calls: Shape.size outputShape × \
            (reductionFiberSize - 1).",
         "Result buffers: 1.",
         "Materialized fiber tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        nativeExecution correctnessTheorem denotation
  else if Lowering.reduceTensor.isApp expression ||
      Lowering.reduceViewTensor.isApp expression then
    let entry :=
      if Lowering.reduceTensor.isApp expression then Lowering.reduceTensor
      else Lowering.reduceViewTensor
    let inputScalarType ← formatType arguments[0]!
    let outputScalarType ← formatType arguments[1]!
    let aggregateType ← inferredType arguments[4]!
    let supplementaryTypeEntries := [
      ("Aggregate", aggregateType)]
    let symbolicTypeEntries := [
      ("Input scalar type", inputScalarType),
      ("Output scalar type", outputScalarType),
      ("Aggregate", aggregateType),
      ("Input tensor", s!"checked.InputTensor {inputScalarType}"),
      ("Output tensor", s!"checked.OutputTensor {outputScalarType}")]
    return (← concreteTransform "reduce" arguments[5]!
        arguments[0]! arguments[1]! supplementaryTypeEntries
        "allocate one output buffer, enumerate each removed-axis fiber, and \
          aggregate it directly."
        (Lowering.reportName entry.correctness)
        (Lowering.reportName entry.denotation)).getD <|
      symbolic "reduce" symbolicTypeEntries
        ["Grammar, ellipsis expansion, literal-axis lengths, and grouped \
            shape equations are certified by the checked plan.",
         "Axis relation: every retained output axis comes from the input.",
         "The reduction-fiber shape and cardinality are derived from exactly \
            the removed elementary axes."]
        ["reshape into elementary axes", "move retained axes first",
         "aggregate removed-axis fibers", "reshape into physical output"]
        ["Output entries and aggregate calls: Shape.size outputShape.",
         "Input visits: Shape.size outputShape × reductionFiberSize.",
         "Result buffers: 1.",
         "Materialized fiber tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one output fill that directly enumerates and aggregates each fiber"
        (Lowering.reportName entry.correctness)
        (Lowering.reportName entry.denotation)
  else if Lowering.reduceNonemptyTensor.isApp expression then
    let inputScalarType ← formatType arguments[0]!
    let outputScalarType ← formatType arguments[1]!
    let aggregateType ← inferredType arguments[4]!
    let supplementaryTypeEntries := [
      ("Nonempty aggregate", aggregateType)]
    let symbolicTypeEntries := [
      ("Input scalar type", inputScalarType),
      ("Output scalar type", outputScalarType),
      ("Nonempty aggregate", aggregateType),
      ("Input tensor", s!"checked.InputTensor {inputScalarType}"),
      ("Output tensor", s!"checked.OutputTensor {outputScalarType}")]
    return (← concreteTransform "reduce (nonempty aggregate)"
        arguments[5]! arguments[0]! arguments[1]!
        supplementaryTypeEntries
        "allocate one output buffer, enumerate each certified nonempty \
          removed-axis fiber, and aggregate it directly."
        (Lowering.reportName Lowering.reduceNonemptyTensor.correctness)
        (Lowering.reportName Lowering.reduceNonemptyTensor.denotation) true).getD <|
      symbolic "reduce (nonempty aggregate)" symbolicTypeEntries
        ["Grammar, ellipsis expansion, literal-axis lengths, and grouped \
            shape equations are certified by the checked plan.",
         "Axis relation: every retained output axis comes from the input.",
         "The reduction-fiber cardinality is certified positive."]
        ["reshape into elementary axes", "move retained axes first",
         "aggregate provably nonempty fibers",
         "reshape into physical output"]
        ["Output entries and aggregate calls: Shape.size outputShape.",
         "Input visits: Shape.size outputShape × reductionFiberSize.",
         "Result buffers: 1.",
         "Materialized fiber tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one output fill that directly enumerates each certified nonempty fiber"
        (Lowering.reportName Lowering.reduceNonemptyTensor.correctness)
        (Lowering.reportName Lowering.reduceNonemptyTensor.denotation)
  else if Lowering.einsumTensorKernel.isApp expression then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input tensor family", s!"checked.InputTensors {scalarType}"),
      ("Generated scalar function",
        s!"Fin (Shape.size checked.output) → {scalarType}"),
      ("Generated output tensor", s!"checked.OutputTensor {scalarType}"),
      ("Kernel result", s!"checked.OutputTensor {scalarType}")]
    return (← concreteEinsum arguments[6]! arguments[0]!
        (Lowering.reportName Lowering.einsumTensorKernel.correctness)).getD <|
      symbolic "einsum" symbolicTypeEntries
        ["Operand count, supported grammar, expanded ranks, repeated-label \
            dimensions, and singleton broadcasting are certified.",
         "Every global logical length is witnessed by an input operand.",
         "Output labels are duplicate-free, input-sourced, and determine the \
            symbolic output shape."]
        ["align operands on global logical axes",
         "multiply operands pointwise in source order",
         "put output axes before contracted axes",
         "sum contracted-axis fibers"]
        ["Contraction terms: Shape.size outputShape × \
            Shape.size contractedShape.",
         "Operand reads: contraction terms × operand count.",
         "Ordered multiplications: contraction terms × \
            (operand count - 1).",
         "Additive accumulator steps: contraction terms, or 0 without a contracted axis.",
         "Result buffers: 1; materialized aligned/product tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one generated output fill with certified row-major input-index plans"
        (Lowering.reportName Lowering.einsumTensorKernel.correctness)
        (Lowering.reportName Lowering.einsumTensorKernel.denotation)
  else if Lowering.einsumTensor.isApp expression then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input tensor family", s!"checked.InputTensors {scalarType}"),
      ("Output tensor", s!"checked.OutputTensor {scalarType}")]
    return (← concreteEinsum arguments[6]! arguments[0]!
        (Lowering.reportName Lowering.einsumTensor.correctness)).getD <|
      symbolic "einsum" symbolicTypeEntries
        ["Operand count, supported grammar, expanded ranks, repeated-label \
            dimensions, and singleton broadcasting are certified.",
         "Every global logical length is witnessed by an input operand.",
         "Output labels are duplicate-free, input-sourced, and determine the \
            symbolic output shape."]
        ["align operands on global logical axes",
         "multiply operands pointwise in source order",
         "put output axes before contracted axes",
         "sum contracted-axis fibers"]
        ["Contraction terms: Shape.size outputShape × \
            Shape.size contractedShape.",
         "Operand reads: contraction terms × operand count.",
         "Ordered multiplications: contraction terms × \
            (operand count - 1).",
         "Additive accumulator steps: contraction terms, or 0 without a contracted axis.",
         "Result buffers: 1; materialized aligned/product tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one fused output fill with certified row-major input-index plans"
        (Lowering.reportName Lowering.einsumTensor.correctness)
        (Lowering.reportName Lowering.einsumTensor.denotation)
  else if Lowering.packTensor.isApp expression then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input component family", s!"checked.InputTensors {scalarType}"),
      ("Packed output tensor", s!"checked.OutputTensor {scalarType}")]
    return (← concretePack "pack" arguments[2]! arguments[0]!).getD <|
      symbolic "pack" symbolicTypeEntries
        ["The fixed leading and trailing ranks match the pattern.",
         "Every component shape decomposes into the shared prefix, its star \
            region, and the shared suffix.",
         "Star-region sizes form an exact ordered partition of the packed \
            axis."]
        ["reshape each star region to one segment axis",
         "concatenate component segments"]
        ["Scalar reads and writes: Shape.size packedOutputShape.",
         "Result buffers: 1 packed tensor.",
         "Materialized reshape and concatenation tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one packed output fill through the certified component equivalence"
        (Lowering.reportName Lowering.packTensor.correctness)
        (Lowering.reportName Lowering.packTensor.denotation)
  else if Lowering.unpackTensor.isApp expression then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Packed input tensor", s!"checked.OutputTensor {scalarType}"),
      ("Output component family", s!"checked.InputTensors {scalarType}")]
    return (← concretePack "unpack" arguments[2]! arguments[0]!).getD <|
      symbolic "unpack" symbolicTypeEntries
        ["The fixed leading and trailing ranks match the pattern.",
         "Every component shape decomposes into the shared prefix, its star \
            region, and the shared suffix.",
         "Star-region sizes form an exact ordered partition of the packed \
            axis."]
        ["split the packed axis at certified segment boundaries",
         "reshape each segment to its component shape"]
        ["Scalar reads and writes: the sum of component shape sizes.",
         "Result buffers: one per component.",
         "Materialized split and reshape tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one direct output fill per component through the certified packed \
          coordinate map"
        (Lowering.reportName Lowering.unpackTensor.correctness)
        (Lowering.reportName Lowering.unpackTensor.denotation)
  else if Lowering.checkParseShape.isApp expression then
    if let some report ← concreteParseShape arguments[0]! arguments[1]! then
      return report
    let result ← whnfAll expression
    let succeeded := result.isAppOfArity ``Except.ok 3
    let rejected := result.isAppOfArity ``Except.error 3
    let operation :=
      if succeeded then "parse_shape (symbolic metadata check)"
      else if rejected then "parse_shape (rejected metadata check)"
      else "parse_shape (unverified metadata check)"
    let shapeType ← inferredType arguments[1]!
    let checkerResultType ← inferredType expression
    let obligations ←
      if succeeded then
        pure <| formatSection "Discharged obligations"
          ["The checker reduces to success; its correctness theorems \
              certify grammar, rank, ellipsis, literal, unit-axis, and \
              unique-name requirements.",
           "Returned dimension expressions remain symbolic in the local context."]
      else if rejected then
        pure <| formatSection "Check status"
          [s!"The checker rejects this input: {← ppExpr result.getAppArgs[2]!}"]
      else
        pure <| formatSection "Check status"
          ["The checker result remains symbolic. Success has not been established.",
           "Correctness theorems require a proof that checking returns `Except.ok`."]
    return String.intercalate "\n" <|
        [operation] ++
          typeCheckSection [
            ("Structural shape metadata", shapeType),
            ("Checker result", checkerResultType),
            ("Scalar type",
              "arbitrary and erased before this metadata check")] ++
          obligations ++
          formatSection "Shape-derived work estimate"
            ["Scalar reads: 0.",
             "Output buffers: 0.",
             "Metadata-list size remains symbolic."] ++
          formatSection "Correctness"
            ["Successful checking yields ordered, duplicate-free bindings \
                whose lengths agree with their physical dimensions."]
  else
    return "Unsupported einops application"


end Report

end TorchLean.Einops
