/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Kernel.Product
public meta import NN.Tensor.Internal.Elab.Einsum.Output
public meta import NN.Tensor.Internal.Elab.Einsum.Planning
public meta import NN.Tensor.Internal.Elab.Transform.View
public import NN.Tensor.Internal.Elab.Einsum.Output
public import NN.Tensor.Internal.Elab.Einsum.Planning
public import NN.Tensor.Internal.Elab.Syntax
public import NN.Tensor.Internal.Elab.Transform.View

/-!
# Elaboration of einsum

This module implements arbitrary positive-arity `einsum` over heterogeneous
tensor shapes. Concrete shapes use the executable checker; symbolic shapes
retain their original `Nat` expressions while proving repeated-label,
broadcasting, source-occurrence, and output-shape invariants.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab

open TorchLean.Tensor
open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

open Impl

/--
Elaborate arbitrary positive-arity einsum syntax to the generic verified
lowering, reflecting concrete checks or proving symbolic shape invariants.
-/
@[term_elab TorchLean.Tensor.einsumStx]
def elabEinsum : TermElab := fun stx expectedType? => withRef stx do
  let `(einsum $tensorSyntax:term,* $sourceSyntax:str) := stx
    | throwUnsupportedSyntax
  withCommonScalarFamily "einsum" "operand" tensorSyntax.getElems fun
      inputTensors inputShapes scalarType storage => do
    synthesizeSyntheticMVarsNoPostponing
    let scalarLevel ← getDecLevel scalarType
    let source := sourceSyntax.getString
    let parsedPattern ←
      match Syntax.parseEinsumPattern source with
      | .error diagnostic =>
          throwPatternDiagnostic "einsum" "pattern" source
            diagnostic.message diagnostic.span
      | .ok parsedPattern => pure parsedPattern
    let structuralInputShapes :=
      inputShapes.map fun dimensions =>
        List.replicate dimensions.length 1
    let inputAxes :=
      Check.einsumInputAxes parsedPattern structuralInputShapes
    let outputAxes :=
      Check.einsumOutputAxes parsedPattern structuralInputShapes
    let globalAxes :=
      Check.einsumGlobalAxes parsedPattern structuralInputShapes
    let contractedAxes :=
      globalAxes.filter fun axis => !outputAxes.contains axis
    let concreteInputShapes? ← concreteShapes? inputShapes
    let (checked, compactOutputShape, outputShapeAgreement,
        kernelInputShapes, kernelAxisExpressions) ←
      match concreteInputShapes? with
      | some concreteInputShapes => do
          let checkedValue ←
            match Check.checkEinsum parsedPattern concreteInputShapes with
            | .error diagnostic =>
                throwPatternDiagnostic "einsum" "shape" source
                  diagnostic.message diagnostic.span
            | .ok checked => pure checked
          let (checked, _, compactOutputShape, outputShapeAgreement) ←
            symbolicCheckedEinsumExpr source parsedPattern inputShapes
              expectedType?
          let kernelInputShapes :=
            concreteInputShapes.map fun shape =>
              shape.map Lean.toExpr
          let kernelAxisExpressions :=
            globalAxes.map fun axis =>
              (axis, Lean.toExpr (checkedValue.axisLength axis))
          pure
            (checked, compactOutputShape, outputShapeAgreement,
              kernelInputShapes, kernelAxisExpressions)
      | none =>
          let (checked, axisExpressions, compactOutputShape,
              outputShapeAgreement) ←
            symbolicCheckedEinsumExpr source parsedPattern inputShapes
              expectedType?
          pure
            (checked, compactOutputShape, outputShapeAgreement,
              inputShapes, axisExpressions)
    let compactResultType :=
      mkAppN (mkConst ``Rep [scalarLevel]) #[
        scalarType, compactOutputShape, storage]
    if let some expectedType := expectedType? then
      unless ←
          withTransparency .reducible <|
            isDefEq compactResultType expectedType do
        throwError
          "einsum result has type {compactResultType}, but the expected type is\
            {indentExpr expectedType}"
    let axisLengths
        (description : String)
        (axes : List Check.EinsumAxis)
        (assignments : List (Check.EinsumAxis × Expr)) :
        TermElabM (List Expr) :=
      axes.mapM fun axis =>
        match einsumAxisExpression? assignments axis with
        | some length => pure length
        | none =>
            throwError
              "internal error: an {description} einsum axis has no resolved length"
    let kernelOutputLengths ←
      axisLengths "output" outputAxes kernelAxisExpressions
    let kernelContractedLengths ←
      axisLengths "contracted" contractedAxes kernelAxisExpressions
    let checkedOutputShape :=
      mkApp (mkConst ``Check.CheckedEinsum.output) checked
    let finishResult (result : Expr) : TermElabM Expr := do
      let result ← mkAppM ``Rep.castShape #[outputShapeAgreement, result]
      synthesizeSyntheticMVarsNoPostponing
      let result ← instantiateMVars result
      if result.hasMVar then
        throwError "internal error: generated einsum term contains unresolved metavariables"
      exposePublicTensorType result compactOutputShape expectedType?
    if ← hasConcreteZeroDimension kernelOutputLengths then
      -- Empty output has no scalar reads to certify and no output axes to traverse.
      let shapes ← inputShapes.mapM fun dimensions => shapeExpr dimensions
      let inputs ← buildTensorFamily scalarType storage shapes inputTensors
      return ← finishResult (← mkAppM ``Lowering.einsumTensor #[checked, inputs])
    let contractionPlan? ←
      planContraction? checked scalarType kernelInputShapes inputAxes
        contractedAxes kernelContractedLengths
    let (executionContractedAxes, executionContractedLengths,
        contractionCoordinateMap?, hContractionCoordinateMap?,
        hContractionOriginalNodup?, hContractionPermutation?) :=
      match contractionPlan? with
      | none =>
          (contractedAxes, kernelContractedLengths,
            none, none, none, none)
      | some (plannedAxes, plannedLengths, coordinateMap,
          hCoordinateMap, hOriginal, hPermutation) =>
          (plannedAxes, plannedLengths, some coordinateMap,
            some hCoordinateMap, some hOriginal, some hPermutation)
    let inputShapeExpressions ←
      inputShapes.mapM fun dimensions => shapeExpr dimensions
    let inputViews ←
      (inputTensors.zip inputShapeExpressions).mapM fun (tensor, inputShape) => do
        match ← fusedTransformInput? inputShape tensor with
        | none => pure none
        | some (_, inputFlatMap, _, sourceTensor, hTensor, _, _) => do
            let hRead ←
              mkAppM ``Lowering.source_getFlat_eq_of_eq_pullFlat #[
                tensor, sourceTensor, inputFlatMap, hTensor]
            pure <| some (sourceTensor, inputFlatMap, hRead)
    let inputs ←
      buildTensorFamily scalarType storage inputShapeExpressions inputTensors
    let (inputProduct, hInputProduct, inputFactorization?) ←
      compileEinsumInputProduct checked inputs scalarType
        inputTensors inputViews kernelInputShapes inputAxes outputAxes
        contractedAxes executionContractedAxes
        kernelOutputLengths kernelContractedLengths executionContractedLengths
        contractionCoordinateMap?
    let mut generatedInputValues := [
        ("compiled input product", inputProduct),
        ("compiled input-product certificate", hInputProduct)]
    if let some (leftFactor, middleProduct, rightFactor, hFactoredProduct) :=
        inputFactorization? then
      generatedInputValues := generatedInputValues ++ [
        ("compiled invariant left factor", leftFactor),
        ("compiled varying input product", middleProduct),
        ("compiled invariant right factor", rightFactor),
        ("compiled factorization certificate", hFactoredProduct)]
    for (description, value) in generatedInputValues do
      if value.hasLooseBVars then
        throwError
          "internal error: {description} contains loose bound variables:\
            {indentExpr value}"
    let (outputValues, hOutputValues, outputArray, hOutputArray) ←
      if inputViews.any fun view => view.isSome then
        let outputResults ←
          withGeneratedLetResults
              [(`inputProductCorrect, hInputProduct)] fun localProofs => do
            let (outputValues, hOutputValues, outputArray, hOutputArray) ←
              compileEinsumOutput checked inputs scalarType storage
                kernelOutputLengths kernelContractedLengths
                executionContractedLengths
                contractionCoordinateMap? hContractionCoordinateMap?
                hContractionOriginalNodup?
                hContractionPermutation?
                inputProduct localProofs[0]! inputFactorization?
            pure #[outputValues, hOutputValues, outputArray, hOutputArray]
        pure
          (outputResults[0]!, outputResults[1]!,
            outputResults[2]!, outputResults[3]!)
      else
        compileEinsumOutput checked inputs scalarType storage
          kernelOutputLengths kernelContractedLengths
          executionContractedLengths
          contractionCoordinateMap? hContractionCoordinateMap?
          hContractionOriginalNodup?
          hContractionPermutation?
          inputProduct hInputProduct inputFactorization?
    for (description, value) in [
        ("compiled output function", outputValues),
        ("compiled output-function certificate", hOutputValues),
        ("compiled output array", outputArray),
        ("compiled output-array certificate", hOutputArray)] do
      if value.hasLooseBVars then
        throwError
          "internal error: {description} contains loose bound variables:\
            {indentExpr value}"
    let outputSize := mkApp (mkConst ``Shape.size) checkedOutputShape
    let referenceOutputArray :=
      mkAppN (mkConst ``Array.ofFn [scalarLevel]) #[
        scalarType, outputSize, outputValues]
    let observedOutputBuffer ←
      mkAppM ``Storage.toArray #[outputArray]
    let hOutputArray ←
      withTransparency .all <|
        mkExpectedTypeHint hOutputArray
          (← mkEq observedOutputBuffer referenceOutputArray)
    let hOutputArraySize ←
      mkAppM ``Storage.size_eq_of_toArray_eq_ofFn #[
        outputArray, outputValues, hOutputArray]
    let rawOutputTensor ←
      mkAppM ``Rep.mk #[outputArray, hOutputArraySize]
    let rawOutputTensorCorrectness ←
      mkAppM ``Rep.mk_eq_ofFlatFn #[
        outputValues, outputArray, hOutputArraySize, hOutputArray]
    -- Output generation already normalizes the complete native loop nest and
    -- seals its array-level certificate. Wrapping that array directly avoids a
    -- second normalization pass over the same generated kernel and its proofs.
    let outputTensor := rawOutputTensor
    let hOutputTensor := rawOutputTensorCorrectness
    let result ←
      mkAppM ``Lowering.einsumTensorKernel #[
        checked, inputs, outputValues, hOutputValues,
        outputTensor, hOutputTensor]
    finishResult result

end TorchLean.Tensor.Internal.Elab
