/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Lowering.Reduce
public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Einsum
public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Pack
public meta import NN.Tensor.Internal.Tactic.Report.Analysis.ParseShape
public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Symbolic
public meta import NN.Tensor.Internal.Tactic.Report.Analysis.Transform
public meta import NN.Tensor.Internal.Lowering.TransformFusion -- shake: keep

/-!
# Report rendering

This module dispatches reflected applications to their operation-specific
reports and preserves a symbolic account when concrete dimensions are not
available.
-/

public meta section

namespace TorchLean.Tensor.Internal

open Lean Elab Tactic Meta

namespace Report

open Report.Impl

/-- Read the final public operation from a fused checked transformation. -/
private def fusedTransformOperation? (checked : Expr) :
    MetaM (Option String) := do
  let value ← mkAppM ``Check.CheckedTransform.value #[checked]
  let normalized ← mkAppM ``Check.TransformPlan.normalized #[value]
  let kind ←
    withTransparency .reducible <|
      whnf (← mkAppM ``Check.NormalizedTransform.kind #[normalized])
  if kind.isConstOf ``Check.TransformKind.rearrange then
    return some "rearrange"
  if kind.isConstOf ``Check.TransformKind.repeat then
    return some "repeat"
  return none

/--
Render the checked types, obligations, lowering stages, execution strategy,
static work estimates, and correctness theorem for a recognized operation.
-/
def renderApplication (expression : Expr) : MetaM String := do
  let arguments := expression.getAppArgs
  if expression.isAppOfArity ``Lowering.transformTensorFused 9 then
    let operation :=
      (← fusedTransformOperation? arguments[3]!).getD "shape transform"
    let scalarType ← formatType arguments[0]!
    let sourceShape ← formatType arguments[2]!
    let coordinateMapType ← inferredType arguments[5]!
    let flatMapType ← inferredType arguments[6]!
    let supplementaryTypeEntries := [
      ("Original source shape", sourceShape),
      ("Coordinate semantics", coordinateMapType),
      ("Native flat-index map", flatMapType)]
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Original source tensor", s!"Rep {scalarType} {sourceShape}"),
      ("Output tensor", s!"checked.OutputTensor {scalarType}")] ++
      supplementaryTypeEntries
    let nativeExecution :=
      "compose every checked source-index map at elaboration time, then fill \
        one final row-major output buffer by reading the original source \
        buffer directly; multidimensional coordinates and intermediate \
        tensors are absent from the scalar loop."
    let correctnessTheorem :=
      if operation = "rearrange" then
        "Lowering.transformTensorFused_rearrange_correct"
      else if operation = "repeat" then
        "Lowering.transformTensorFused_repeat_correct"
      else
        "Lowering.transformTensorFused_correct"
    let denotation :=
      if operation = "rearrange" then
        "Semantics.denoteRearrange after the preceding coordinate pullback"
      else if operation = "repeat" then
        "Semantics.denoteRepeat after the preceding coordinate pullback"
      else
        "the checked coordinate pullback"
    return (← concreteTransformReport operation arguments[3]!
        arguments[0]! arguments[0]! supplementaryTypeEntries
        nativeExecution correctnessTheorem denotation).getD <|
      symbolicReport operation symbolicTypeEntries
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
  else if expression.isAppOfArity ``Lowering.rearrangeTensor 5 then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input tensor", s!"checked.InputTensor {scalarType}"),
      ("Output tensor", s!"checked.OutputTensor {scalarType}")]
    return (← concreteTransformReport "rearrange" arguments[2]!
        arguments[0]! arguments[0]! []
        "reuse storage for both reshapes and allocate one reindexed output buffer."
        "Lowering.rearrangeTensor_correct"
        "Semantics.denoteRearrange").getD <|
      symbolicReport "rearrange" symbolicTypeEntries
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
        "Lowering.rearrangeTensor_correct" "Semantics.denoteRearrange"
  else if expression.isAppOfArity ``Lowering.repeatTensor 5 then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input tensor", s!"checked.InputTensor {scalarType}"),
      ("Output tensor", s!"checked.OutputTensor {scalarType}")]
    return (← concreteTransformReport "repeat" arguments[2]!
        arguments[0]! arguments[0]! []
        "allocate one output buffer and read each entry through the certified \
          output-to-input coordinate map."
        "Lowering.repeatTensor_correct" "Semantics.denoteRepeat").getD <|
      symbolicReport "repeat" symbolicTypeEntries
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
        "Lowering.repeatTensor_correct" "Semantics.denoteRepeat"
  else if expression.isAppOfArity ``Lowering.reduceFoldTensor 11 then
    let step := arguments[5]!
    let finish := arguments[7]!
    let directFoldExecution :=
      "allocate one output buffer, fold each removed-axis fiber directly \
        from left to right in row-major order, and finalize it once."
    let (operation, nativeExecution, nonemptyReduction) :=
      if containsConstant finish ``HDiv.hDiv then
        ("reduce (nonempty aggregate)",
          "allocate one output buffer, fold each certified nonempty \
            removed-axis fiber from left to right in row-major order, and \
            divide once by its certified cardinality.", true)
      else if containsConstant step ``Bool.or then
        ("reduce", directFoldExecution, false)
      else if containsConstant step ``Bool.and then
        ("reduce", directFoldExecution, false)
      else if containsConstant step ``HMul.hMul then
        ("reduce", directFoldExecution, false)
      else if containsConstant step ``HAdd.hAdd then
        ("reduce", directFoldExecution, false)
      else
        ("reduce (ordered fold)", directFoldExecution, false)
    let correctnessTheorem :=
      "Lowering.reduceFoldTensor_ordered_correct"
    let denotation :=
      "Semantics.denoteOrderedReduce with the same row-major left-fold order"
    let logicalStages :=
      if nonemptyReduction then
        ["reshape into elementary axes", "move retained axes first",
         "aggregate provably nonempty fibers",
         "reshape into physical output"]
      else
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
          the removed elementary axes."] ++
        if nonemptyReduction then
          ["The symbolic reduction-fiber cardinality is certified positive."]
        else
          []
    let symbolicWork :=
      ["Output entries: Shape.size outputShape.",
       "Input visits and fold-step calls: Shape.size outputShape × \
          reductionFiberSize.",
       "Finalizer calls and output writes: Shape.size outputShape.",
       "Result buffers: 1.",
       "Materialized fiber tensors: 0.",
       "Exact numeric counts require concrete local dimensions."]
    return (← concreteTransformReport operation arguments[8]!
        arguments[0]! arguments[2]! supplementaryTypeEntries
        nativeExecution correctnessTheorem denotation
        nonemptyReduction true).getD <|
      symbolicReport operation symbolicTypeEntries symbolicObligations
        logicalStages symbolicWork nativeExecution correctnessTheorem
        denotation
  else if expression.isAppOfArity
      ``Lowering.reduceNonemptyFoldTensor 7 then
    let step := arguments[2]!
    let operation :=
      if containsConstant step ``min then
        "reduce (minimum)"
      else if containsConstant step ``max then
        "reduce (maximum)"
      else
        "reduce (ordered nonempty fold)"
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
      "Lowering.reduceNonemptyFoldTensor_ordered_correct"
    let denotation :=
      "Semantics.denoteOrderedReduceNonempty with the same row-major \
        first-value fold order"
    return (← concreteTransformReport operation arguments[3]!
        arguments[0]! arguments[0]! supplementaryTypeEntries
        nativeExecution correctnessTheorem denotation true true).getD <|
      symbolicReport operation symbolicTypeEntries
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
  else if expression.isAppOfArity ``Lowering.reduceTensor 8 then
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
    return (← concreteTransformReport "reduce" arguments[5]!
        arguments[0]! arguments[1]! supplementaryTypeEntries
        "allocate one output buffer, enumerate each removed-axis fiber, and \
          aggregate it directly."
        "Lowering.reduceTensor_correct" "Semantics.denoteReduce").getD <|
      symbolicReport "reduce" symbolicTypeEntries
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
        "Lowering.reduceTensor_correct" "Semantics.denoteReduce"
  else if expression.isAppOfArity ``Lowering.reduceNonemptyTensor 9 then
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
    return (← concreteTransformReport "reduce (nonempty aggregate)"
        arguments[5]! arguments[0]! arguments[1]!
        supplementaryTypeEntries
        "allocate one output buffer, enumerate each certified nonempty \
          removed-axis fiber, and aggregate it directly."
        "Lowering.reduceNonemptyTensor_correct"
        "Semantics.denoteReduceNonempty" true).getD <|
      symbolicReport "reduce (nonempty aggregate)" symbolicTypeEntries
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
        "Lowering.reduceNonemptyTensor_correct"
        "Semantics.denoteReduceNonempty"
  else if expression.isAppOfArity ``Lowering.einsumTensorKernel 12 then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input tensor family", s!"checked.InputTensors {scalarType}"),
      ("Generated scalar function",
        s!"Fin (Shape.size checked.output) → {scalarType}"),
      ("Generated output tensor", s!"checked.OutputTensor {scalarType}"),
      ("Kernel result", s!"checked.OutputTensor {scalarType}")]
    return (← concreteEinsumReport arguments[6]! arguments[0]!
        "Lowering.einsumTensorKernel_correct").getD <|
      symbolicReport "einsum" symbolicTypeEntries
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
         "Additive accumulator steps: contraction terms.",
         "Result buffers: 1; materialized aligned/product tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one generated output fill with certified row-major input-index plans"
        "Lowering.einsumTensorKernel_correct" "Semantics.denoteEinsum"
  else if expression.isAppOfArity ``Lowering.einsumTensor 8 then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input tensor family", s!"checked.InputTensors {scalarType}"),
      ("Output tensor", s!"checked.OutputTensor {scalarType}")]
    return (← concreteEinsumReport arguments[6]! arguments[0]!
        "Lowering.einsumTensor_correct").getD <|
      symbolicReport "einsum" symbolicTypeEntries
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
         "Additive accumulator steps: contraction terms.",
         "Result buffers: 1; materialized aligned/product tensors: 0.",
         "Exact numeric counts require concrete local dimensions."]
        "one fused output fill with certified row-major input-index plans"
        "Lowering.einsumTensor_correct" "Semantics.denoteEinsum"
  else if expression.isAppOfArity ``Lowering.packTensor 4 then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Input component family", s!"checked.InputTensors {scalarType}"),
      ("Rep output tensor", s!"checked.OutputTensor {scalarType}")]
    return (← concretePackReport "pack" arguments[2]! arguments[0]!).getD <|
      symbolicReport "pack" symbolicTypeEntries
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
        "Lowering.packTensor_correct" "Semantics.denotePack"
  else if expression.isAppOfArity ``Lowering.unpackTensor 4 then
    let scalarType ← formatType arguments[0]!
    let symbolicTypeEntries := [
      ("Scalar type", scalarType),
      ("Rep input tensor", s!"checked.OutputTensor {scalarType}"),
      ("Output component family", s!"checked.InputTensors {scalarType}")]
    return (← concretePackReport "unpack" arguments[2]! arguments[0]!).getD <|
      symbolicReport "unpack" symbolicTypeEntries
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
        "Lowering.unpackTensor_correct" "Semantics.denoteUnpack"
  else
    let shapeType ← inferredType arguments[1]!
    let checkerResultType ← inferredType expression
    return (← concreteParseShapeReport arguments[0]! arguments[1]!).getD <|
      String.intercalate "\n" <|
        ["parse_shape (symbolic metadata check)"] ++
          typeCheckSection [
            ("Structural shape metadata", shapeType),
            ("Checker result", checkerResultType),
            ("Rep scalar type",
              "arbitrary and erased before this metadata check")] ++
          reportSection "Discharged obligations"
            ["A generated kernel proof certifies grammar, rank, ellipsis, \
                literal, unit-axis, and unique-name requirements.",
             "Actual dimension expressions remain symbolic in the local \
                context."] ++
          reportSection "Shape-derived work estimate"
            ["Rep scalar reads: 0.",
             "Rep output buffers: 0.",
             "Metadata-list size remains symbolic."] ++
          reportSection "Correctness"
            ["Successful checking yields ordered, duplicate-free bindings \
                whose lengths agree with their physical dimensions."]


end Report

end TorchLean.Tensor.Internal
