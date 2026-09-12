/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Analysis
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Utilities
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.View

/-!
# Verified einsum scalar product generation

The scalar kernel compiler is organized as certified operand indexing,
generated-term utilities, contraction-invariance analysis, and ordered product
assembly; this module is the final assembly stage and is what importers of the
kernel compiler name.

This module assembles direct operand reads into the ordered scalar product
used by a checked einsum. It also extracts semiring factors that are invariant
across the contraction while retaining a pointwise semantic certificate.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Generate the direct scalar product used by a literal einsum and an erased
pointwise proof that it equals the generic verified implementation.

Output coordinates are computed before the contraction lambda. Eligible
multi-term output-index bases are hoisted from the contraction loop. With
multiple contracted axes, every operand's outer-axis contributions are
combined into a base that the loop compiler floats outside the innermost
contraction loop. Over a semiring, maximal source-order prefixes and suffixes
that do not use a contracted axis are also returned as optional factors.
-/
def compileEinsumInputProduct
    (checked inputTensorFamily scalarType : Expr)
    (inputTensors : List Expr)
    (inputViews : List (Option (Expr × Expr × Expr)))
    (inputDimensions : List (List Expr))
    (inputAxes : List (List Check.EinsumAxis))
    (outputAxes semanticContractedAxes contractedAxes :
      List Check.EinsumAxis)
    (outputLengths semanticContractedLengths contractedLengths : List Expr)
    (contractionCoordinateMap? : Option Expr) :
    TermElabM
      (Expr × Expr × Option (Expr × Expr × Expr × Expr)) := do
  if inputTensors.length != inputDimensions.length ||
      inputTensors.length != inputAxes.length ||
      inputTensors.length != inputViews.length then
    throwError
      "internal error: einsum tensors, views, dimensions, and axes have \
        different lengths"
  let hasInputView := inputViews.any fun view => view.isSome
  let factorBounds? ←
    if contractedAxes.isEmpty then
      pure none
    else
      let semiringType ← mkAppM ``Semiring #[scalarType]
      let .some _ ← trySynthInstance semiringType
        | pure none
      let invariantOperands ←
        (inputAxes.zip inputDimensions).mapM fun (axes, dimensions) =>
          operandIsContractionInvariant axes dimensions outputAxes
      let prefixCount :=
        (invariantOperands.takeWhile id).length
      let suffixCount :=
        (invariantOperands.reverse.takeWhile id).length
      if prefixCount + suffixCount < inputTensors.length &&
          (0 < prefixCount || 0 < suffixCount) then
        pure <| some (prefixCount, inputTensors.length - suffixCount)
      else
        pure none
  let axisLengthAssignments :=
    outputAxes.zip outputLengths ++ contractedAxes.zip contractedLengths
  let mut inputLogicalDimensions : List (List Expr) := []
  for operandAxes in inputAxes do
    let mut logicalDimensions : List Expr := []
    for axis in operandAxes do
      let some logicalLength :=
          einsumAxisExpression? axisLengthAssignments axis
        | throwError
            "internal error: an einsum input axis has no resolved length"
      logicalDimensions := logicalDimensions.concat logicalLength
    inputLogicalDimensions :=
      inputLogicalDimensions.concat logicalDimensions
  let reference ←
    mkAppM ``Lowering.einsumInputProduct #[checked, inputTensorFamily]
  let outputShape ← shapeExpr outputLengths
  let semanticContractedShape ← shapeExpr semanticContractedLengths
  let contractedShape ← shapeExpr contractedLengths
  let outputCoordinateType ← mkAppM ``Coord #[outputShape]
  let contractionCoordinateType ← mkAppM ``Coord #[contractedShape]
  let semanticContractionCoordinate (coordinate : Expr) : MetaM Expr :=
    match contractionCoordinateMap? with
    | none => pure coordinate
    | some coordinateMap => pure <| mkApp coordinateMap coordinate
  let contractionIsEmpty ←
    hasConcreteZeroDimension contractedLengths
  if contractionIsEmpty then
    -- No contraction coordinate exists, so the output fold never evaluates
    -- its scalar-product function. Keeping the semantic function itself
    -- avoids generating impossible bounds for reads from zero-sized inputs.
    let correctness ←
      withLocalDeclD `outputCoordinate outputCoordinateType
          fun outputCoordinate => do
        withLocalDeclD `contractionCoordinate contractionCoordinateType
            fun contractionCoordinate => do
          let semanticCoordinate ←
            semanticContractionCoordinate contractionCoordinate
          let value :=
            mkApp (mkApp reference outputCoordinate)
              semanticCoordinate
          let hValue ← mkEqRefl value
          let hPointwise ←
            mkLambdaFVars #[contractionCoordinate] hValue
          mkLambdaFVars #[outputCoordinate] hPointwise
    return (reference, correctness, none)
  let compiledResults ←
    withLocalDeclD `outputCoordinate outputCoordinateType
        fun outputCoordinate => do
      let mut outputCoordinateBindings : List (Name × Expr) := []
      for axisPosition in [:outputAxes.length] do
        let coordinate ← coordinateAxisExpr axisPosition outputCoordinate
        outputCoordinateBindings :=
          outputCoordinateBindings.concat
            (Name.mkSimple s!"outputAxis{axisPosition}", coordinate)
      let outputResults ←
        withGeneratedLetResults outputCoordinateBindings
            fun outputCoordinates => do
        let outputCoordinateAssignments :=
          outputAxes.zip outputCoordinates
        let axesArray := inputAxes.toArray
        let dimensionsArray := inputDimensions.toArray
        let logicalDimensionsArray := inputLogicalDimensions.toArray
        let mut operandOutputCoordinates :
            List (List (Option Expr)) := []
        let mut outputBaseSlots : List Bool := []
        let mut outputBaseBindings : List (Name × Expr) := []
        for operandIndex in [:inputTensors.length] do
          let operandAxes := axesArray[operandIndex]!
          let operandDimensions := dimensionsArray[operandIndex]!
          let operandLogicalDimensions :=
            logicalDimensionsArray[operandIndex]!
          if operandAxes.length != operandDimensions.length then
            throwError
              "internal error: an einsum operand's axes and dimensions \
                have different lengths"
          let hoistOutputTerms ←
            shouldHoistOutputIndexTerms operandAxes operandDimensions
              outputAxes
          let mut outputCoordinates : List (Option Expr) := []
          for axis in operandAxes do
            if outputAxes.contains axis then
              let some outputCoordinate :=
                  einsumAxisExpression? outputCoordinateAssignments axis
                | throwError
                    "internal error: an output einsum axis has no \
                      generated coordinate"
              outputCoordinates :=
                outputCoordinates.concat (some outputCoordinate)
            else
              outputCoordinates := outputCoordinates.concat none
          operandOutputCoordinates :=
            operandOutputCoordinates.concat outputCoordinates
          outputBaseSlots := outputBaseSlots.concat hoistOutputTerms
          if hoistOutputTerms then
            let outputBase ←
              hoistedOutputIndexBase operandDimensions
                operandLogicalDimensions outputCoordinates
            outputBaseBindings :=
              outputBaseBindings.concat
                (Name.mkSimple s!"input{operandIndex}OutputBase", outputBase)
        withGeneratedLetResults outputBaseBindings fun outputBaseValues => do
          let outputBaseArray := outputBaseValues.toArray
          let mut nextOutputBase := 0
          let mut operandOutputBases : List (Option Expr) := []
          for hasOutputBase in outputBaseSlots do
            if hasOutputBase then
              if hBound : nextOutputBase < outputBaseArray.size then
                operandOutputBases :=
                  operandOutputBases.concat
                    (some outputBaseArray[nextOutputBase])
                nextOutputBase := nextOutputBase + 1
              else
                throwError
                  "internal error: generated einsum output bases were exhausted"
            else
              operandOutputBases := operandOutputBases.concat none
          if nextOutputBase != outputBaseArray.size then
            throwError
              "internal error: generated einsum output bases were not all \
                consumed"
          withLocalDeclD `contractionCoordinate contractionCoordinateType
              fun contractionCoordinate => do
            let semanticCoordinate ←
              semanticContractionCoordinate contractionCoordinate
            let mut contractionCoordinateBindings :
                List (Name × Expr) := []
            let mut contractionCoordinates : List Expr := []
            for axisPosition in [:contractedAxes.length] do
              let coordinate ←
                coordinateAxisExpr axisPosition contractionCoordinate
              contractionCoordinates :=
                contractionCoordinates.concat coordinate
              contractionCoordinateBindings :=
                contractionCoordinateBindings.concat
                  (Name.mkSimple s!"contractedAxis{axisPosition}",
                    coordinate)
            let contractionCoordinateAssignments :=
              contractedAxes.zip contractionCoordinates
            let outerContractionAssignments :=
              contractionCoordinateAssignments.dropLast
            let outputCoordinatesArray :=
              operandOutputCoordinates.toArray
            let outputBasesArray := operandOutputBases.toArray
            let mut contractionBaseSlots : List Bool := []
            let mut contractionBaseAxes :
                List (List Check.EinsumAxis) := []
            let mut contractionBaseBindings : List (Name × Expr) := []
            for operandIndex in [:inputTensors.length] do
              let operandAxes := axesArray[operandIndex]!
              let operandDimensions := dimensionsArray[operandIndex]!
              let operandLogicalDimensions :=
                logicalDimensionsArray[operandIndex]!
              let mut contribution := mkNatLit 0
              let mut includedAxes : List Check.EinsumAxis := []
              for (axis, coordinate) in outerContractionAssignments do
                let axisContribution ←
                  contractedAxisIndexContribution axis coordinate
                    operandAxes operandDimensions operandLogicalDimensions
                contribution ← natAddExpr contribution axisContribution
                unless (← getNatValue? axisContribution) == some 0 do
                  includedAxes := includedAxes.concat axis
              if includedAxes.isEmpty then
                contractionBaseSlots :=
                  contractionBaseSlots.concat false
                contractionBaseAxes :=
                  contractionBaseAxes.concat []
              else
                let outputBase ←
                  match outputBasesArray[operandIndex]! with
                  | some outputBase => pure outputBase
                  | none =>
                      hoistedOutputIndexBase operandDimensions
                        operandLogicalDimensions
                        outputCoordinatesArray[operandIndex]!
                let contractionBase ←
                  natAddExpr contribution outputBase
                contractionBaseSlots :=
                  contractionBaseSlots.concat true
                contractionBaseAxes :=
                  contractionBaseAxes.concat includedAxes
                contractionBaseBindings :=
                  contractionBaseBindings.concat
                    (Name.mkSimple
                      s!"input{operandIndex}ContractionBase",
                      contractionBase)
            let contractionResults ←
              withGeneratedLetResults contractionBaseBindings
                  fun contractionBaseValues => do
              let contractionBaseArray :=
                contractionBaseValues.toArray
              let mut nextContractionBase := 0
              let mut operandIndexBases : List (Option Expr) := []
              for operandIndex in [:inputTensors.length] do
                if contractionBaseSlots[operandIndex]! then
                  if hBound :
                      nextContractionBase <
                        contractionBaseArray.size then
                    operandIndexBases :=
                      operandIndexBases.concat
                        (some
                          contractionBaseArray[nextContractionBase])
                    nextContractionBase :=
                      nextContractionBase + 1
                  else
                    throwError
                      "internal error: generated einsum contraction bases \
                        were exhausted"
                else
                  operandIndexBases :=
                    operandIndexBases.concat
                      outputBasesArray[operandIndex]!
              if nextContractionBase != contractionBaseArray.size then
                throwError
                  "internal error: generated einsum contraction bases were \
                    not all consumed"
              withGeneratedLetResults contractionCoordinateBindings
                  fun contractionCoordinates => do
                let contractionCoordinateAssignments :=
                  contractedAxes.zip contractionCoordinates
                let tensorArray := inputTensors.toArray
                let viewArray := inputViews.toArray
                let indexBasesArray := operandIndexBases.toArray
                let contractionBaseAxesArray :=
                  contractionBaseAxes.toArray
                let mut inputReadBindings : List (Name × Expr) := []
                let mut inputReadCorrectness : List Expr := []
                let mut certifiedInputValues : List Expr := []
                for operandIndex in [:inputTensors.length] do
                  let operandAxes := axesArray[operandIndex]!
                  let operandDimensions :=
                    dimensionsArray[operandIndex]!
                  let operandLogicalDimensions :=
                    logicalDimensionsArray[operandIndex]!
                  let inputIndexValue ←
                    compileInputFlatIndexValue operandAxes
                      operandDimensions operandLogicalDimensions
                      outputCoordinatesArray[operandIndex]!
                      indexBasesArray[operandIndex]!
                      contractionBaseAxesArray[operandIndex]!
                      contractionCoordinateAssignments
                  let operand ←
                    foldOperandIndexExpr inputTensors.length operandIndex
                  let certifiedInputIndex :=
                    mkAppN (mkConst ``Lowering.einsumInputFlatIndex) #[
                      checked, operand, outputCoordinate,
                      semanticCoordinate]
                  let inputShape ← shapeExpr operandDimensions
                  let inputSize ← mkAppM ``Shape.size #[inputShape]
                  let inputIndexType ← mkAppM ``Fin #[inputSize]
                  let inputPlan ←
                    mkAppM ``Lowering.inputFlatIndexPlan #[
                      Lean.toExpr outputAxes,
                      Lean.toExpr semanticContractedAxes,
                      Lean.toExpr operandAxes,
                      inputShape]
                  let plannedInputIndexValue ←
                    mkAppM ``Lowering.evaluateInputFlatIndexPlan #[
                      outputShape, semanticContractedShape,
                      outputCoordinate, semanticCoordinate, inputPlan]
                  let plannedInputIndexValue ←
                    zetaReduce plannedInputIndexValue
                  let normalizedInputIndexValue ←
                    zetaReduce inputIndexValue
                  let inputIndexValueEquality ←
                    mkEq normalizedInputIndexValue plannedInputIndexValue
                  let hInputIndexValue ←
                    certifyCompiledKernel
                      "that a compiled einsum input index agrees with \
                        its verified reference"
                      inputIndexValueEquality
                  let certifiedInputIndexValue ←
                    mkAppM ``Fin.val #[certifiedInputIndex]
                  let certificationEquality ←
                    mkEq plannedInputIndexValue certifiedInputIndexValue
                  let hCertifiedInputIndexValue ←
                    withTransparency .all <|
                      mkExpectedTypeHint
                        (← mkEqRefl plannedInputIndexValue)
                        certificationEquality
                  let hInputIndexValue ←
                    mkAppM ``Eq.trans #[
                      hInputIndexValue, hCertifiedInputIndexValue]
                  let expectedInputIndexEquality ←
                    mkEq inputIndexValue certifiedInputIndexValue
                  let hInputIndexValue ←
                    withTransparency .all <|
                      mkExpectedTypeHint hInputIndexValue
                        expectedInputIndexEquality
                  let useNativeIndex ←
                    supportsNativeInputIndex operandDimensions
                  let (inputValue, inputValueCorrect) ←
                    compileInputViewRead
                      tensorArray[operandIndex]! inputSize
                      inputIndexValue normalizedInputIndexValue
                      certifiedInputIndex hInputIndexValue useNativeIndex
                      viewArray[operandIndex]!
                  let certifiedInputValue ←
                    withTransparency .all <|
                      mkAppM ``Rep.getFlat #[
                        tensorArray[operandIndex]!, certifiedInputIndex]
                  inputReadBindings :=
                    inputReadBindings.concat
                      (Name.mkSimple s!"input{operandIndex}Value", inputValue)
                  inputReadCorrectness :=
                    inputReadCorrectness.concat inputValueCorrect
                  certifiedInputValues :=
                    certifiedInputValues.concat certifiedInputValue
                withGeneratedLetResults inputReadBindings fun inputValues => do
                  let inputValueArray := inputValues.toArray
                  let inputReadCorrectnessArray :=
                    inputReadCorrectness.toArray
                  let certifiedInputValueArray :=
                    certifiedInputValues.toArray
                  let mut product ← mkNumeral scalarType 1
                  let mut certifiedProduct := product
                  let mut productCorrect ← mkEqRefl product
                  for operandIndex in [:inputTensors.length] do
                    let inputValue := inputValueArray[operandIndex]!
                    let certifiedInputValue :=
                      certifiedInputValueArray[operandIndex]!
                    let expectedInputValueEquality ←
                      mkEq inputValue certifiedInputValue
                    let inputValueCorrect ←
                      withTransparency .all <|
                        mkExpectedTypeHint
                          inputReadCorrectnessArray[operandIndex]!
                          expectedInputValueEquality
                    let nextProduct ← mkMul product inputValue
                    let nextCertifiedProduct ←
                      mkMul certifiedProduct certifiedInputValue
                    let preserveLeft ←
                      withLocalDeclD `left scalarType fun left => do
                        let value ← mkMul left inputValue
                        mkLambdaFVars #[left] value
                    let leftCorrect ←
                      mkAppM ``congrArg #[preserveLeft, productCorrect]
                    let preserveRight ←
                      withLocalDeclD `right scalarType fun right => do
                        let value ← mkMul certifiedProduct right
                        mkLambdaFVars #[right] value
                    let rightCorrect ←
                      mkAppM ``congrArg #[
                        preserveRight, inputValueCorrect]
                    productCorrect ←
                      mkAppM ``Eq.trans #[leftCorrect, rightCorrect]
                    product := nextProduct
                    certifiedProduct := nextCertifiedProduct
                  let hReferenceFold ←
                    mkAppM ``Lowering.einsumInputProduct_eq_foldl #[
                      checked, inputTensorFamily, outputCoordinate,
                      semanticCoordinate]
                  let hFoldReference ←
                    mkAppM ``Eq.symm #[hReferenceFold]
                  let hFoldReferenceType ←
                    withTransparency .reducible <|
                      whnf (← inferType hFoldReference)
                  let some (_, foldedProduct, _) :=
                      hFoldReferenceType.eq?
                    | throwError
                        "internal error: the einsum input-product bridge did \
                          not produce an equality"
                  let productFoldEquality ←
                    mkEq certifiedProduct foldedProduct
                  let hCertifiedFold ←
                    if hasInputView then
                      let foldedProduct := foldedProduct.consumeMData
                      unless foldedProduct.isAppOfArity ``Fin.foldl 4 do
                        throwError
                          "internal error: the einsum input-product bridge did \
                            not expose a finite fold"
                      let foldArguments := foldedProduct.getAppArgs
                      let foldStep := foldArguments[2]!
                      let foldInitial := foldArguments[3]!
                      let (_, hUnrolledFold) ←
                        unrollFiniteFold inputTensors.length
                          foldStep foldInitial
                      withTransparency .all <|
                        mkExpectedTypeHint hUnrolledFold
                          productFoldEquality
                    else
                      withTransparency .all <|
                        mkExpectedTypeHint (← mkEqRefl certifiedProduct)
                          productFoldEquality
                  let hCertifiedProduct ←
                    mkAppM ``Eq.trans #[
                      hCertifiedFold, hFoldReference]
                  let correctness ←
                    mkAppM ``Eq.trans #[
                      productCorrect, hCertifiedProduct]
                  let mut identityTheorems : SimpTheorems := {}
                  for theoremName in #[``one_mul, ``mul_one] do
                    identityTheorems ← identityTheorems.addConst theoremName
                  let identityContext ←
                    Simp.mkContext
                      (config := {
                        zeta := false
                        failIfUnchanged := false
                      })
                      (simpTheorems := #[identityTheorems])
                      (congrTheorems := ← getSimpCongrTheorems)
                  let (simplifiedProduct, _) ←
                    simp product identityContext
                  let hProductSimplified ←
                    match simplifiedProduct.proof? with
                    | none => mkEqRefl product
                    | some proof => pure proof
                  let hSimplifiedProduct ←
                    mkAppM ``Eq.symm #[hProductSimplified]
                  let correctness ←
                    mkAppM ``Eq.trans #[
                      hSimplifiedProduct, correctness]
                  match factorBounds? with
                  | none =>
                      pure #[simplifiedProduct.expr, correctness]
                  | some (prefixCount, suffixStart) =>
                      let buildProduct
                          (start stop : Nat) :
                          MetaM (Expr × Expr × Expr) := do
                        let mut rawProduct ←
                          mkNumeral scalarType 1
                        for operandIndex in [start:stop] do
                          rawProduct ←
                            mkMul rawProduct
                              inputValueArray[operandIndex]!
                        let (simplified, _) ←
                          simp rawProduct identityContext
                        let hRawSimplified ←
                          match simplified.proof? with
                          | none => mkEqRefl rawProduct
                          | some proof => pure proof
                        pure
                          (rawProduct, simplified.expr,
                            hRawSimplified)
                      let (rawLeftFactor, leftFactor,
                          hLeftFactor) ←
                        buildProduct 0 prefixCount
                      let (rawMiddleProduct, middleProduct,
                          hMiddleProduct) ←
                        buildProduct prefixCount suffixStart
                      let (rawRightFactor, rightFactor,
                          hRightFactor) ←
                        buildProduct suffixStart inputTensors.length
                      let leftValues :=
                        inputValues.take prefixCount
                      let middleValues :=
                        (inputValues.drop prefixCount).take
                          (suffixStart - prefixCount)
                      let rightValues :=
                        inputValues.drop suffixStart
                      let leftList ←
                        mkListLit scalarType leftValues
                      let middleList ←
                        mkListLit scalarType middleValues
                      let rightList ←
                        mkListLit scalarType rightValues
                      let rawFactoredProduct ←
                        mkMul
                          (← mkMul rawLeftFactor rawMiddleProduct)
                          rawRightFactor
                      let hRawFactoredProduct ←
                        mkAppM ``Lowering.foldl_mul_split_three #[
                          leftList, middleList, rightList]
                      let hRawFactoredProduct ←
                        withTransparency .all <|
                          mkExpectedTypeHint hRawFactoredProduct
                            (← mkEq rawFactoredProduct product)
                      let factoredProduct ←
                        mkMul
                          (← mkMul leftFactor middleProduct)
                          rightFactor
                      let hLeftFactor ←
                        mkAppM ``Eq.symm #[hLeftFactor]
                      let hMiddleProduct ←
                        mkAppM ``Eq.symm #[hMiddleProduct]
                      let hRightFactor ←
                        mkAppM ``Eq.symm #[hRightFactor]
                      let multiply ←
                        withLocalDeclD `left scalarType fun left =>
                          withLocalDeclD `right scalarType fun right => do
                            let product ← mkMul left right
                            mkLambdaFVars #[left, right] product
                      let hLeftMiddle ←
                        mkAppM ``congrArg₂ #[
                          multiply, hLeftFactor, hMiddleProduct]
                      let hFactoredRaw ←
                        mkAppM ``congrArg₂ #[
                          multiply, hLeftMiddle, hRightFactor]
                      let hFactoredProduct ←
                        mkAppM ``Eq.trans #[
                          hFactoredRaw, hRawFactoredProduct]
                      let hFactoredProduct ←
                        mkAppM ``Eq.trans #[
                          hFactoredProduct, hProductSimplified]
                      let hFactoredProduct ←
                        withTransparency .all <|
                          mkExpectedTypeHint hFactoredProduct
                            (← mkEq factoredProduct
                              simplifiedProduct.expr)
                      pure #[
                        simplifiedProduct.expr, correctness,
                        leftFactor, middleProduct, rightFactor,
                        hFactoredProduct]
            let contractionFunction ←
              mkLambdaPreservingLets
                contractionCoordinate contractionResults[0]!
            let contractionCorrectness ←
              mkLambdaPreservingLets
                contractionCoordinate contractionResults[1]!
            match factorBounds? with
            | none =>
                pure #[contractionFunction, contractionCorrectness]
            | some _ =>
                let leftFactor := contractionResults[2]!
                let middleProduct := contractionResults[3]!
                let rightFactor := contractionResults[4]!
                let hFactoredProduct ←
                  mkLambdaPreservingLets
                    contractionCoordinate contractionResults[5]!
                for (description, factor) in [
                    ("left", leftFactor), ("right", rightFactor)] do
                  if factor.containsFVar
                      contractionCoordinate.fvarId! then
                    throwError
                      "internal error: the {description} einsum factor \
                        depends on a contracted coordinate"
                let middleProduct ←
                  mkLambdaPreservingLets
                    contractionCoordinate middleProduct
                pure #[
                  contractionFunction, contractionCorrectness,
                  leftFactor, middleProduct, rightFactor,
                  hFactoredProduct]
      let compiled ←
        mkLambdaPreservingLets outputCoordinate outputResults[0]!
      let correctness ←
        mkLambdaPreservingLets outputCoordinate outputResults[1]!
      match factorBounds? with
      | none => pure #[compiled, correctness]
      | some _ =>
          let leftFactor ←
            mkLambdaPreservingLets outputCoordinate outputResults[2]!
          let middleProduct ←
            mkLambdaPreservingLets outputCoordinate outputResults[3]!
          let rightFactor ←
            mkLambdaPreservingLets outputCoordinate outputResults[4]!
          let hFactoredProduct ←
            mkLambdaPreservingLets outputCoordinate outputResults[5]!
          pure #[
            compiled, correctness, leftFactor, middleProduct,
            rightFactor, hFactoredProduct]
  match factorBounds? with
  | none =>
      return (compiledResults[0]!, compiledResults[1]!, none)
  | some _ =>
      return (compiledResults[0]!, compiledResults[1]!,
        some
          (compiledResults[2]!, compiledResults[3]!,
            compiledResults[4]!, compiledResults[5]!))

end TorchLean.Tensor.Internal.Elab.Impl
