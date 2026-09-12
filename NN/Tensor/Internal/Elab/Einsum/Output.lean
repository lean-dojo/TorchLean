/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Batteries.Data.Vector.Lemmas
public import NN.Tensor.Internal.Elab.Einsum.Contraction.Loop
public meta import NN.Tensor.Internal.Elab.Einsum.Output.Fusion
public meta import NN.Tensor.Internal.Elab.Einsum.Output.Planning
public meta import NN.Tensor.Internal.Elab.Einsum.OutputIndex
public meta import NN.Tensor.Internal.Elab.Einsum.ParallelOutput
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Product -- shake: keep
public meta import NN.Tensor.Internal.Lowering.Einsum.Planning -- shake: keep

/-!
# Verified einsum output generation

This module generates arbitrary-rank output traversals and proves that
their native loops implement the independent einsum semantics.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Name one complete generated output buffer and its semantic certificate.

The auxiliary record keeps the executable native loop and its proof behind one
constant, so final tensor assembly does not re-normalize the complete buffer
construction.
-/
private def sealSequentialOutput (output : Expr) : TermElabM Expr := do
  let name ← mkAuxName `_einops_sequential_output
  mkAuxDefinitionFor name output (zetaDelta := true)

/--
Compile the contraction into one arbitrary-rank output loop nest.

Every output axis follows the same recursive lowering. Concrete portable
lengths use native counters; symbolic and oversized lengths use `Fin.foldl`.
Small concrete contractions evaluate one output at a time. A concrete final
axis uses four lanes from eight contraction terms when fewer than eight
output positions are available, and otherwise uses eight lanes when the live
read family is modest. Multi-axis tiled contractions flatten into one native
loop from 32 terms for four lanes and 128 terms for eight lanes. Untiled
scalar contractions retain the nested arbitrary-rank fold. Each lane keeps
the scalar reduction order, and both concrete widths advance one contraction
coordinate per recursive step so their callbacks inline consistently. When
the contraction is one native loop, completed tiles are appended directly to
the output buffer; nested contractions retain the vector result required by
their outer folds. Any incomplete tile is emitted by the scalar path. The
returned certificate identifies the complete loop nest with the general
`Array.ofFn` executor.
-/
def compileEinsumOutput
    (checked inputTensorFamily scalarType storage : Expr)
    (outputLengths semanticContractedLengths contractedLengths : List Expr)
    (contractionCoordinateMap? hContractionCoordinateMap?
      hContractionOriginalNodup? hContractionPermutation? : Option Expr)
    (inputProduct hInputProduct : Expr)
    (inputFactorization? : Option (Expr × Expr × Expr × Expr)) :
    TermElabM (Expr × Expr × Expr × Expr) := do
  let reference ←
    mkAppM ``Lowering.einsumOutput #[checked, inputTensorFamily]
  let outputShape ← shapeExpr outputLengths
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let checkedOutputShape ←
    mkAppM ``Check.CheckedEinsum.output #[checked]
  let checkedOutputSize ←
    mkAppM ``Shape.size #[checkedOutputShape]
  let checkedOutputIndexType ← mkAppM ``Fin #[checkedOutputSize]
  let outputCoordinateType ← mkAppM ``Coord #[outputShape]
  let semanticContractedShape ← shapeExpr semanticContractedLengths
  let contractedShape ← shapeExpr contractedLengths
  let contractionCoordinateType ← mkAppM ``Coord #[contractedShape]
  let contractionIsEmpty ←
    hasConcreteZeroDimension contractedLengths
  let contractionEntries? :=
    (← concreteNatExpressions? contractedLengths).map Shape.size
  let tileProduct :=
    match inputFactorization? with
    | some (_, middleProduct, _, _) => middleProduct
    | none => inputProduct
  let scalarReadsPerTerm :=
    generatedScalarReadCount tileProduct
  let scalarLevel ← getDecLevel scalarType
  let add ←
    synthInstance (← mkAppM ``Add #[scalarType])
  let ofNatZero ←
    synthInstance (← mkAppM ``OfNat #[scalarType, mkNatLit 0])
  -- Meta-level application builders do not insert `optParam` defaults.
  let zero ← Expr.ofNat scalarType 0
  let referenceInputProduct ←
    mkAppM ``Lowering.einsumInputProduct #[checked, inputTensorFamily]
  let semanticContractionCoordinate (coordinate : Expr) : MetaM Expr :=
    match contractionCoordinateMap? with
    | none => pure coordinate
    | some coordinateMap => pure <| mkApp coordinateMap coordinate
  let referenceProductAt (outputCoordinate : Expr) : MetaM Expr :=
    match contractionCoordinateMap? with
    | none => pure <| mkApp referenceInputProduct outputCoordinate
    | some _ =>
        withLocalDeclD `contractionCoordinate contractionCoordinateType
            fun contractionCoordinate => do
          let semanticCoordinate ←
            semanticContractionCoordinate contractionCoordinate
          let value :=
            mkApp (mkApp referenceInputProduct outputCoordinate)
              semanticCoordinate
          mkLambdaFVars #[contractionCoordinate] value
  let contractionPlanCorrect ←
    withLocalDeclD `outputCoordinate outputCoordinateType
        fun outputCoordinate => do
      let plannedProduct ← referenceProductAt outputCoordinate
      let plannedSum ←
        mkAppM ``Semantics.coordinateSum #[
          contractedShape, plannedProduct, zero]
      let semanticProduct :=
        mkApp referenceInputProduct outputCoordinate
      let semanticSum ←
        mkAppM ``Semantics.coordinateSum #[
          semanticContractedShape, semanticProduct, zero]
      let correctness ←
        match contractionCoordinateMap?, hContractionCoordinateMap?,
            hContractionOriginalNodup?, hContractionPermutation? with
        | none, none, none, none =>
            withTransparency .all <|
              mkExpectedTypeHint (← mkEqRefl plannedSum)
                (← mkEq plannedSum semanticSum)
        | some coordinateMap, some hCoordinateMap,
            some hOriginal, some hPermutation => do
            let axisLength ←
              mkAppM ``Check.CheckedEinsum.axisLength #[checked]
            let proof ←
              mkAppM ``Lowering.coordinateSum_permute_of_eq #[
                axisLength, hOriginal, hPermutation,
                coordinateMap, hCoordinateMap, semanticProduct, zero]
            withTransparency .all <|
              mkExpectedTypeHint proof
                (← mkEq plannedSum semanticSum)
        | _, _, _, _ =>
            throwError
              "internal error: an einsum contraction plan has incomplete \
                permutation evidence"
      mkLambdaFVars #[outputCoordinate] correctness
  -- The focused contexts cancel native-coordinate conversion round trips
  -- without simplifying arbitrary user scalar expressions.
  let (nativeIndexSimpContext, nativeCoordinateSimpContext) ←
    nativeOutputIndexSimpContexts
  let withRawAtCoordinate (outputIndex : Expr)
      (outputCoordinates semanticOutputCoordinates
        hOutputCoordinates : List Expr)
      (hSemanticOutputIndex : Expr)
      (body : Expr → Expr → Expr → Expr →
        TermElabM (Expr × Expr)) :
      TermElabM (Expr × Expr) := do
    let outputCoordinate ←
      coordinateFromComponents outputCoordinates
    let semanticOutputCoordinate ←
      coordinateFromComponents semanticOutputCoordinates
    let hOutputCoordinate ←
      coordinateFromComponentsEquality outputCoordinates
        semanticOutputCoordinates hOutputCoordinates
    let unlinearizedOutputCoordinate ←
      mkAppOptM ``Coord.unlinearize #[
        some outputShape, some outputIndex]
    let hSemanticOutputIndex ←
      withTransparency .all <|
        mkExpectedTypeHint hSemanticOutputIndex
          (← mkEq semanticOutputCoordinate unlinearizedOutputCoordinate)
    let applyAtOutput (function : Expr) : Expr :=
      match function.consumeMData with
      | .lam _ _ functionBody _ =>
          functionBody.instantiate1 outputCoordinate
      | _ => mkApp function outputCoordinate
    let finish
        (productAtOutput rawOutput finalizeOutput
          finalizedOutput hFinalizedReference : Expr) :
        TermElabM (Expr × Expr) := do
      let referenceAtCoordinate ←
        withLocalDeclD `outputCoordinate outputCoordinateType
            fun coordinate => do
          let product := mkApp referenceInputProduct coordinate
          let value ←
            mkAppM ``Semantics.coordinateSum #[
              semanticContractedShape, product, zero]
          mkLambdaFVars #[coordinate] value
      let hReferenceCoordinate ←
        mkAppM ``congrArg #[
          referenceAtCoordinate, hOutputCoordinate]
      let hPlannedSemantic :=
        mkApp contractionPlanCorrect outputCoordinate
      let hFinalizedSemantic ←
        mkAppM ``Eq.trans #[
          hFinalizedReference, hPlannedSemantic]
      let hRawSemantic ←
        mkAppM ``Eq.trans #[
          hFinalizedSemantic, hReferenceCoordinate]
      let hSemanticIndex ←
        mkAppM ``congrArg #[
          referenceAtCoordinate, hSemanticOutputIndex]
      let hRawIndex ←
        mkAppM ``Eq.trans #[
          hRawSemantic, hSemanticIndex]
      let referenceValue := mkApp reference outputIndex
      let expectedEquality ←
        mkEq finalizedOutput referenceValue
      let correctness ←
        withTransparency .all <|
          mkExpectedTypeHint hRawIndex expectedEquality
      body productAtOutput rawOutput finalizeOutput correctness
    match inputFactorization? with
    | none =>
        let appliedInputProduct := applyAtOutput inputProduct
        withLeadingLetPair appliedInputProduct
            fun leadingLocals productAtOutput => do
          let (productAtOutput, hProductSimplified?) ←
            simplifyNativeProductIndices nativeIndexSimpContext
              leadingLocals productAtOutput
          let rawOutput ←
            mkAppM ``Semantics.coordinateSum #[
              contractedShape, productAtOutput, zero]
          let referenceProductAtOutput ←
            referenceProductAt outputCoordinate
          let hRawReference ←
            if contractionIsEmpty then
              mkAppM ``Lowering.coordinateSum_congr_of_isEmpty #[
                contractedShape, productAtOutput,
                referenceProductAtOutput, zero]
            else
              let hProductAtOutput :=
                mkApp hInputProduct outputCoordinate
              let hProductAtOutput ←
                match hProductSimplified? with
                | none => pure hProductAtOutput
                | some hProductSimplified => do
                    let hSimplifiedProduct ←
                      mkAppM ``Eq.symm #[hProductSimplified]
                    withLocalDeclD `contractionCoordinate
                        contractionCoordinateType
                        fun contractionCoordinate => do
                      let hSimplifiedAtCoordinate ←
                        mkAppM ``congrFun #[
                          hSimplifiedProduct, contractionCoordinate]
                      let hReferenceAtCoordinate :=
                        mkApp hProductAtOutput contractionCoordinate
                      let hProductAtCoordinate ←
                        mkAppM ``Eq.trans #[
                          hSimplifiedAtCoordinate,
                          hReferenceAtCoordinate]
                      mkLambdaFVars #[
                        contractionCoordinate] hProductAtCoordinate
                pure <|
                  mkAppN
                    (mkConst ``Lowering.coordinateSum_congr [scalarLevel]) #[
                  scalarType, add, ofNatZero, contractedShape,
                  productAtOutput, referenceProductAtOutput, zero,
                  hProductAtOutput]
          let finalizeOutput ←
            withLocalDeclD `value scalarType fun value =>
              mkLambdaFVars #[value] value
          finish productAtOutput rawOutput finalizeOutput
            rawOutput hRawReference
    | some (leftFactor, middleProduct, rightFactor, hFactoredProduct) =>
        let leftAtOutput := applyAtOutput leftFactor
        let appliedMiddleProduct := applyAtOutput middleProduct
        let rightAtOutput := applyAtOutput rightFactor
        withLeadingLetPair appliedMiddleProduct
            fun leadingLocals middleAtOutput => do
          let (middleAtOutput, hMiddleSimplified?) ←
            simplifyNativeProductIndices nativeIndexSimpContext
              leadingLocals middleAtOutput
          let rawOutput ←
            mkAppM ``Semantics.coordinateSum #[
              contractedShape, middleAtOutput, zero]
          let finalizeOutput ←
            withLocalDeclD `value scalarType fun value => do
              let finalized ←
                mkMul (← mkMul leftAtOutput value) rightAtOutput
              mkLambdaFVars #[value] finalized
          let finalizedOutput := finalizeOutput.beta #[rawOutput]
          let hFactoredSum ←
            mkAppM ``Lowering.mul_coordinateSum_mul #[
              contractedShape, leftAtOutput, rightAtOutput, middleAtOutput]
          let referenceProductAtOutput ←
            referenceProductAt outputCoordinate
          let hInputAtOutput := mkApp hInputProduct outputCoordinate
          let hFactoredAtOutput :=
            mkApp hFactoredProduct outputCoordinate
          let hPointwise ←
            withLocalDeclD `contractionCoordinate
                contractionCoordinateType fun contractionCoordinate => do
              let middleValue :=
                mkApp middleAtOutput contractionCoordinate
              let hMiddleAtCoordinate ←
                match hMiddleSimplified? with
                | none => mkEqRefl middleValue
                | some hMiddleSimplified => do
                    let hSimplifiedMiddle ←
                      mkAppM ``Eq.symm #[hMiddleSimplified]
                    mkAppM ``congrFun #[
                      hSimplifiedMiddle, contractionCoordinate]
              let preserveMiddle ←
                withLocalDeclD `middle scalarType fun middle => do
                  let factored ←
                    mkMul (← mkMul leftAtOutput middle) rightAtOutput
                  mkLambdaFVars #[middle] factored
              let hSimplifiedFactored ←
                mkAppM ``congrArg #[
                  preserveMiddle, hMiddleAtCoordinate]
              let hFactoredAtCoordinate :=
                mkApp hFactoredAtOutput contractionCoordinate
              let hInputAtCoordinate :=
                mkApp hInputAtOutput contractionCoordinate
              let hFactoredInput ←
                mkAppM ``Eq.trans #[
                  hSimplifiedFactored, hFactoredAtCoordinate]
              let hFactoredReference ←
                mkAppM ``Eq.trans #[
                  hFactoredInput, hInputAtCoordinate]
              mkLambdaFVars #[
                contractionCoordinate] hFactoredReference
          let hFactoredReferenceSum :=
            mkAppN
              (mkConst ``Lowering.coordinateSum_congr [scalarLevel]) #[
              scalarType, add, ofNatZero, contractedShape,
              ← withLocalDeclD `contractionCoordinate
                  contractionCoordinateType fun contractionCoordinate => do
                    let middleValue :=
                      mkApp middleAtOutput contractionCoordinate
                    let factored ←
                      mkMul
                        (← mkMul leftAtOutput middleValue)
                        rightAtOutput
                    mkLambdaFVars #[contractionCoordinate] factored,
              referenceProductAtOutput, zero, hPointwise]
          let hFinalizedReference ←
            mkAppM ``Eq.trans #[
              hFactoredSum, hFactoredReferenceSum]
          finish middleAtOutput rawOutput finalizeOutput
            finalizedOutput hFinalizedReference
  let withRawBoundCoordinates (outputIndex : Expr)
      (coordinateValues semanticCoordinateValues
        hCoordinateValues : List Expr)
      (hSemanticOutputIndex : Expr)
      (body : Expr → Expr → Expr → Expr →
        TermElabM (Expr × Expr)) :
      TermElabM (Expr × Expr) := do
    unless coordinateValues.length = semanticCoordinateValues.length &&
        coordinateValues.length = hCoordinateValues.length do
      throwError
        "internal error: generated output coordinates and equalities have \
          different lengths"
    let mut outputCoordinateBindings : List (Name × Expr) := []
    for axisPosition in [:coordinateValues.length] do
      outputCoordinateBindings :=
        outputCoordinateBindings.concat
          (Name.mkSimple s!"outputAxis{axisPosition}",
            coordinateValues[axisPosition]!)
    withGeneratedLetPair outputCoordinateBindings
        fun outputCoordinates => do
      let mut hOutputCoordinates : List Expr := []
      for axisPosition in [:outputCoordinates.length] do
        let expectedEquality ←
          mkEq outputCoordinates[axisPosition]!
            semanticCoordinateValues[axisPosition]!
        let hOutputCoordinate ←
          withTransparency .all <|
            mkExpectedTypeHint hCoordinateValues[axisPosition]!
              expectedEquality
        hOutputCoordinates :=
          hOutputCoordinates.concat hOutputCoordinate
      withRawAtCoordinate outputIndex outputCoordinates
        semanticCoordinateValues hOutputCoordinates hSemanticOutputIndex body
  let compileBoundCoordinates (outputIndex : Expr)
      (coordinateValues semanticCoordinateValues
        hCoordinateValues : List Expr)
      (hSemanticOutputIndex : Expr) :
      TermElabM (Expr × Expr) := do
    withRawBoundCoordinates outputIndex coordinateValues
        semanticCoordinateValues hCoordinateValues hSemanticOutputIndex
        fun productAtOutput rawOutput finalizeOutput hRawReference => do
      let (optimizedOutput, hRawOptimized) ←
        compileCoordinateSum contractedLengths productAtOutput zero rawOutput
      let hOptimizedRaw ←
        mkAppM ``Eq.symm #[hRawOptimized]
      let finalizedOutput := finalizeOutput.beta #[optimizedOutput]
      let hFinalizedRaw ←
        mkAppM ``congrArg #[finalizeOutput, hOptimizedRaw]
      let hOptimizedReference ←
        mkAppM ``Eq.trans #[hFinalizedRaw, hRawReference]
      pure (finalizedOutput, hOptimizedReference)
  -- `einsumTensorKernel` erases this function and its certificate at runtime.
  -- Supplying the independent semantics directly avoids constructing a second
  -- scalar executor solely to serve as a proof index.
  let outputCorrectness ←
    withLocalDeclD `outputIndex checkedOutputIndexType fun outputIndex => do
      let value := mkApp reference outputIndex
      let hValue ← mkEqRefl value
      mkLambdaFVars #[outputIndex] hValue
  let outputBufferType :=
    mkAppN (mkConst ``Storage.Buffer [scalarLevel]) #[
      scalarType, storage]
  -- Build a native output fold together with its equality to the corresponding
  -- finite fold.
  let compileNativeOutputFold
      (length nativeBound hBound generatedStep initial generatedFinLoop : Expr)
      (nativeCoordinateName : Name) : TermElabM (Expr × Expr) := do
    let (nativeCallback, rawNativeCallback, hNativeCallback) ←
      withLocalDeclD `output outputBufferType fun nativeOutput =>
        withLocalDeclD nativeCoordinateName (mkConst ``USize)
            fun nativeCoordinate => do
          let nativeCoordinateNat ←
            mkAppM ``USize.toNat #[nativeCoordinate]
          let nativeCoordinateBound ←
            mkLT nativeCoordinateNat length
          withLocalDeclD `hCoordinate nativeCoordinateBound
              fun hCoordinate => do
            let semanticCoordinate ←
              mkAppOptM ``Fin.mk #[
                some length, some nativeCoordinateNat, some hCoordinate]
            let body :=
              generatedStep.beta #[nativeOutput, semanticCoordinate]
            let (simplifiedBody, hSimplifiedBody) ←
              simplifyNativeOutputIndices nativeCoordinateSimpContext body
            let callbackLocals :=
              #[nativeOutput, nativeCoordinate, hCoordinate]
            let nativeCallback ←
              mkLambdaFVars callbackLocals simplifiedBody
            let rawNativeCallback ←
              mkLambdaFVars callbackLocals body
            let mut hNativeCallback := hSimplifiedBody
            for localValue in callbackLocals.reverse do
              let hFunction ←
                mkLambdaFVars #[localValue] hNativeCallback
              hNativeCallback ← mkAppM ``funext #[hFunction]
            pure
              (nativeCallback, rawNativeCallback, hNativeCallback)
    let nativeLoop ←
      mkAppM ``nativeFinFoldl #[
        length, nativeBound, hBound, nativeCallback, initial]
    let rawNativeLoop ←
      mkAppM ``nativeFinFoldl #[
        length, nativeBound, hBound, rawNativeCallback, initial]
    let hNativeCallbackLoop ←
      withLocalDeclD `step (← inferType nativeCallback) fun step => do
        let loop ←
          mkAppM ``nativeFinFoldl #[
            length, nativeBound, hBound, step, initial]
        let preserveCallback ← mkLambdaFVars #[step] loop
        mkAppM ``congrArg #[preserveCallback, hNativeCallback]
    let hRawNativeFin ←
      mkAppM ``nativeFinFoldl_eq_fin_foldl #[
        length, nativeBound, hBound, generatedStep, initial]
    let hRawNativeFin ←
      withTransparency .all <|
        mkExpectedTypeHint hRawNativeFin
          (← mkEq rawNativeLoop generatedFinLoop)
    let hNativeFin ←
      mkAppM ``Eq.trans #[hNativeCallbackLoop, hRawNativeFin]
    pure (nativeLoop, hNativeFin)
  let buildOutputLeaf
      (coordinates : List Expr) (output : Expr) :
      TermElabM (Expr × Expr × Expr) := do
    let outputCoordinate ← coordinateFromComponents coordinates
    let outputIndex ←
      mkAppOptM ``Coord.linearize #[
        some outputShape, some outputCoordinate]
    let hUnlinearize ←
      mkAppOptM ``Coord.unlinearize_linearize #[
        some outputShape, some outputCoordinate]
    let hSemanticOutputIndex ←
      mkAppM ``Eq.symm #[hUnlinearize]
    let mut hCoordinates : List Expr := []
    for coordinate in coordinates do
      hCoordinates := hCoordinates.concat (← mkEqRefl coordinate)
    let (value, hValueReference) ←
      compileBoundCoordinates outputIndex coordinates coordinates
        hCoordinates hSemanticOutputIndex
    let referenceValue := mkApp reference outputIndex
    let generatedOutput :=
      mkAppN (mkConst ``Storage.push [scalarLevel]) #[
        scalarType, storage, output, value]
    let referenceOutput :=
      mkAppN (mkConst ``Storage.push [scalarLevel]) #[
        scalarType, storage, output, referenceValue]
    let pushValue ←
      withLocalDeclD `value scalarType fun scalar => do
        let pushed :=
          mkAppN (mkConst ``Storage.push [scalarLevel]) #[
            scalarType, storage, output, scalar]
        mkLambdaFVars #[scalar] pushed
    let hGeneratedReference ←
      mkAppM ``congrArg #[pushValue, hValueReference]
    return (generatedOutput, referenceOutput, hGeneratedReference)
  -- Compile neighboring output coordinates through one shared contraction.
  -- The returned pointwise certificates let each caller choose how to store
  -- the completed tile without changing scalar reduction order.
  let compileOutputTile
      (coordinateSets : List (List Expr))
      (finishTile : Expr → List Expr → Expr →
        Option (Expr × List Expr) →
        TermElabM (Expr × Expr)) :
      TermElabM (Expr × Expr) := do
    let tileWidth := coordinateSets.length
    let (updateTileName, updateTileRefinementName,
        selectTileName, coordinateSumSelectName,
        tileCongruenceName) ←
      match tileWidth with
      | 4 =>
          pure
            (``updateTile4, ``updateTile4_eq_updateTile,
              ``selectTile4, ``coordinateSum_selectTile4,
              ``ofFn_selectTile4_congr)
      | 8 =>
          pure
            (``updateTile8, ``updateTile8_eq_updateTile,
              ``selectTile8, ``coordinateSum_selectTile8,
              ``ofFn_selectTile8_congr)
      | _ =>
          throwError
            "internal error: unsupported generated output tile width \
              {tileWidth}"
    let tileWidthExpr := mkNatLit tileWidth
    let tileStateType ←
      mkAppM ``Vector #[scalarType, tileWidthExpr]
    let laneType ← mkAppM ``Fin #[tileWidthExpr]
    let initialTile ←
      mkAppM ``Vector.replicate #[tileWidthExpr, zero]
    let mut laneData :
        List (Expr × List Expr × List Expr × Expr) := []
    let mut outputIndices : List Expr := []
    for coordinateValues in coordinateSets do
      let outputCoordinate ←
        coordinateFromComponents coordinateValues
      let outputIndex ←
        mkAppOptM ``Coord.linearize #[
          some outputShape, some outputCoordinate]
      let hUnlinearize ←
        mkAppOptM ``Coord.unlinearize_linearize #[
          some outputShape, some outputCoordinate]
      let hSemanticOutputIndex ←
        mkAppM ``Eq.symm #[hUnlinearize]
      let mut hCoordinateValues : List Expr := []
      for coordinateValue in coordinateValues do
        hCoordinateValues :=
          hCoordinateValues.concat (← mkEqRefl coordinateValue)
      laneData :=
        laneData.concat
          (outputIndex, coordinateValues,
            hCoordinateValues, hSemanticOutputIndex)
      outputIndices := outputIndices.concat outputIndex
    let rec
      /-- Open every lane's staged lets before compiling the shared fold. -/
      visitLanes
        (remaining :
          List (Expr × List Expr × List Expr × Expr))
        (products rawOutputs finalizeOutputs rawCorrectness : List Expr) :
        TermElabM (Expr × Expr) := do
      match remaining with
      | (outputIndex, coordinateValues,
          hCoordinateValues, hSemanticOutputIndex) :: remaining => do
          withRawBoundCoordinates outputIndex
              coordinateValues coordinateValues
              hCoordinateValues hSemanticOutputIndex
              fun product rawOutput finalizeOutput hRawCorrect => do
            visitLanes remaining
              (products.concat product)
              (rawOutputs.concat rawOutput)
              (finalizeOutputs.concat finalizeOutput)
              (rawCorrectness.concat hRawCorrect)
      | [] => do
          unless products.length = tileWidth &&
              rawOutputs.length = tileWidth &&
              finalizeOutputs.length = tileWidth &&
              rawCorrectness.length = tileWidth do
            throwError
              "internal error: an output tile produced inconsistent lane data"
          let tileValues ←
            withLocalDeclD `lane laneType fun lane =>
              withLocalDeclD `contractionCoordinate
                  contractionCoordinateType
                  fun contractionCoordinate => do
                let mut laneValues := #[]
                for product in products do
                  laneValues :=
                    laneValues.push <|
                      mkApp product contractionCoordinate
                let selectedValues ←
                  mkAppM selectTileName laneValues
                mkLambdaFVars #[lane, contractionCoordinate] <|
                  mkApp selectedValues lane
          let tileStep ←
            withLocalDeclD `totals tileStateType fun totals =>
              withLocalDeclD `contractionCoordinate
                  contractionCoordinateType
                  fun contractionCoordinate => do
                let mut updateArguments := #[totals]
                for product in products do
                  updateArguments :=
                    updateArguments.push <|
                      mkApp product contractionCoordinate
                let updated ←
                  mkAppM updateTileName updateArguments
                mkLambdaFVars #[
                  totals, contractionCoordinate] updated
          let hTileStep ←
            withLocalDeclD `totals tileStateType fun totals =>
              withLocalDeclD `contractionCoordinate
                  contractionCoordinateType
                  fun contractionCoordinate => do
                let mut updateArguments := #[totals]
                for product in products do
                  updateArguments :=
                    updateArguments.push <|
                      mkApp product contractionCoordinate
                let concreteUpdate ←
                  mkAppM updateTileName updateArguments
                let laneValues ←
                  withLocalDeclD `lane laneType fun lane =>
                    mkLambdaFVars #[lane] <|
                      mkAppN tileValues #[lane, contractionCoordinate]
                let genericUpdate ←
                  mkAppM ``updateTile #[totals, laneValues]
                let hUpdate ←
                  mkAppM updateTileRefinementName updateArguments
                let hUpdate ←
                  withTransparency .all <|
                    mkExpectedTypeHint hUpdate
                      (← mkEq concreteUpdate genericUpdate)
                mkLambdaFVars #[totals, contractionCoordinate] hUpdate
          let foldReference ←
            mkAppM ``coordinateFoldl #[
              contractedShape, tileStep, initialTile]
          let (optimizedTile, hFoldOptimized) ←
            compileCoordinateFold contractedLengths
              tileStep initialTile foldReference
          let rawTileFunction ←
            mkAppM selectTileName rawOutputs.toArray
          let rawTile ←
            mkAppM ``Vector.ofFn #[rawTileFunction]
          let genericRawTileFunction ←
            withLocalDeclD `lane laneType fun lane => do
              let laneValues := mkApp tileValues lane
              let laneSum ←
                mkAppM ``Semantics.coordinateSum #[
                  contractedShape, laneValues, zero]
              mkLambdaFVars #[lane] laneSum
          let genericRawTile ←
            mkAppM ``Vector.ofFn #[genericRawTileFunction]
          let hFoldGeneric ←
            mkAppM ``coordinateFoldl_updateTile_of_eq #[
              contractedShape, tileStep, tileValues, zero, hTileStep]
          let hFoldGeneric ←
            withTransparency .all <|
              mkExpectedTypeHint hFoldGeneric
                (← mkEq foldReference genericRawTile)
          let mut coordinateSumSelectArguments :=
            #[contractedShape]
          for product in products do
            coordinateSumSelectArguments :=
              coordinateSumSelectArguments.push product
          coordinateSumSelectArguments :=
            coordinateSumSelectArguments.push zero
          let hCoordinateSumSelect ←
            mkAppM coordinateSumSelectName
              coordinateSumSelectArguments
          let hCoordinateSumSelect ←
            withTransparency .all <|
              mkExpectedTypeHint hCoordinateSumSelect
                (← mkEq genericRawTileFunction rawTileFunction)
          let hGenericRaw ←
            mkAppM ``ofFn_congr #[hCoordinateSumSelect]
          let hGenericRaw ←
            withTransparency .all <|
              mkExpectedTypeHint hGenericRaw
                (← mkEq genericRawTile rawTile)
          let hFoldRaw ←
            mkAppM ``Eq.trans #[hFoldGeneric, hGenericRaw]
          let hOptimizedFold ←
            mkAppM ``Eq.symm #[hFoldOptimized]
          let hOptimizedRaw ←
            mkAppM ``Eq.trans #[hOptimizedFold, hFoldRaw]
          let mut referenceValues : List Expr := []
          let mut finalizedRawValues : List Expr := []
          let mut hFinalizedRawValues : List Expr := []
          for laneIndex in [:tileWidth] do
            let finalizeOutput := finalizeOutputs[laneIndex]!
            let finalizedRawValue :=
              finalizeOutput.beta #[rawOutputs[laneIndex]!]
            let referenceValue :=
              mkApp reference outputIndices[laneIndex]!
            let hFinalizedRawValue ←
              withTransparency .all <|
                mkExpectedTypeHint rawCorrectness[laneIndex]!
                  (← mkEq finalizedRawValue referenceValue)
            referenceValues :=
              referenceValues.concat referenceValue
            finalizedRawValues :=
              finalizedRawValues.concat finalizedRawValue
            hFinalizedRawValues :=
              hFinalizedRawValues.concat hFinalizedRawValue
          let finalizedRawTileFunction ←
            mkAppM selectTileName finalizedRawValues.toArray
          let finalizedRawTile ←
            mkAppM ``Vector.ofFn #[finalizedRawTileFunction]
          let finishEmittedTile
              (emittedTile hEmittedFinalizedRaw : Expr)
              (fusedSource? : Option (Expr × List Expr)) :
              TermElabM (Expr × Expr) := do
            let mut tileCongruenceArguments :=
              finalizedRawValues.toArray
            for referenceValue in referenceValues do
              tileCongruenceArguments :=
                tileCongruenceArguments.push referenceValue
            for correctness in hFinalizedRawValues do
              tileCongruenceArguments :=
                tileCongruenceArguments.push correctness
            let hFinalizedRawReference ←
              mkAppM tileCongruenceName tileCongruenceArguments
            let hEmittedReference ←
              mkAppM ``Eq.trans #[
                hEmittedFinalizedRaw, hFinalizedRawReference]
            finishTile emittedTile referenceValues hEmittedReference
              fusedSource?
          match inputFactorization? with
          | none =>
              let hEmittedFinalizedRaw ←
                withTransparency .all <|
                  mkExpectedTypeHint hOptimizedRaw
                    (← mkEq optimizedTile finalizedRawTile)
              finishEmittedTile optimizedTile hEmittedFinalizedRaw none
          | some _ =>
              -- Retain one shared contraction result. Substituting it into
              -- every finalized lane would duplicate the native tile loop.
              withGeneratedLetPair
                  [(`contractionTotals, optimizedTile)] fun values => do
                let contractionTotals := values[0]!
                let hContractionTotalsRaw ←
                  withTransparency .all <|
                    mkExpectedTypeHint hOptimizedRaw
                      (← mkEq contractionTotals rawTile)
                let finalizeTile ←
                  withLocalDeclD `values tileStateType fun values => do
                    let mut finalizedValues := #[]
                    for laneIndex in [:tileWidth] do
                      let lane ← mkNumeral laneType laneIndex
                      let value ←
                        mkAppM ``Vector.get #[values, lane]
                      finalizedValues := finalizedValues.push <|
                        finalizeOutputs[laneIndex]!.beta #[value]
                    let finalizedFunction ←
                      mkAppM selectTileName finalizedValues
                    let finalizedTile ←
                      mkAppM ``Vector.ofFn #[finalizedFunction]
                    mkLambdaFVars #[values] finalizedTile
                let emittedTile :=
                  finalizeTile.beta #[contractionTotals]
                let hEmittedFinalizedTile ←
                  mkAppM ``congrArg #[
                    finalizeTile, hContractionTotalsRaw]
                let mut finalizedTileValues := #[]
                let mut hFinalizedTileValues := #[]
                for laneIndex in [:tileWidth] do
                  let lane ← mkNumeral laneType laneIndex
                  let rawTileValue ←
                    mkAppM ``Vector.get #[rawTile, lane]
                  finalizedTileValues := finalizedTileValues.push <|
                    finalizeOutputs[laneIndex]!.beta #[rawTileValue]
                  let hRawTileValue ←
                    mkAppM ``Vector.get_ofFn #[rawTileFunction, lane]
                  let hRawTileValue ←
                    withTransparency .all <|
                      mkExpectedTypeHint hRawTileValue
                        (← mkEq rawTileValue rawOutputs[laneIndex]!)
                  hFinalizedTileValues :=
                    hFinalizedTileValues.push <|
                      ← mkAppM ``congrArg #[
                        finalizeOutputs[laneIndex]!, hRawTileValue]
                let mut finalizeCongruenceArguments :=
                  finalizedTileValues
                for finalizedRawValue in finalizedRawValues do
                  finalizeCongruenceArguments :=
                    finalizeCongruenceArguments.push finalizedRawValue
                for correctness in hFinalizedTileValues do
                  finalizeCongruenceArguments :=
                    finalizeCongruenceArguments.push correctness
                let hFinalizedTileRaw ←
                  mkAppM tileCongruenceName
                    finalizeCongruenceArguments
                let hEmittedFinalizedRaw ←
                  mkAppM ``Eq.trans #[
                    hEmittedFinalizedTile, hFinalizedTileRaw]
                let hEmittedFinalizedRaw ←
                  withTransparency .all <|
                    mkExpectedTypeHint hEmittedFinalizedRaw
                      (← mkEq emittedTile finalizedRawTile)
                finishEmittedTile emittedTile hEmittedFinalizedRaw <|
                  some (optimizedTile, finalizeOutputs)
    visitLanes laneData [] [] [] []
  let rec
    /--
    Generate nested output folds and their equality to the semantic coordinate
    traversal. The coordinate list is kept in source-axis order.
    -/
    buildOutputLoops
      (remainingLengths : List Expr)
      (coordinates : List Expr)
      (output : Expr) : TermElabM (Expr × Expr × Expr) := do
    match remainingLengths with
    | [] => buildOutputLeaf coordinates output
    | length :: remainingLengths => do
        let tiling? : Option (Nat × Nat) ←
          match remainingLengths with
          | [] => do
              let some lengthValue ← getNatValue? length
                | pure none
              if length.hasFVar then
                pure none
              else if ← withTransparency .all <|
                  isDefEq length (mkNatLit lengthValue) then
                let tileWidth? :=
                  einsumOutputTileWidth?
                    lengthValue contractionEntries? scalarReadsPerTerm
                let some tileWidth := tileWidth?
                  | pure none
                -- Lane coordinates are derived from a native block counter,
                -- so the complete axis must fit the portable native range.
                match ← nativeLoopBound? length with
                | some _ => pure (some (lengthValue, tileWidth))
                | none => pure none
              else
                pure none
          | _ => pure none
        if let some (lengthValue, tileWidth) := tiling? then
          let tileDeclarations ←
            match tileWidth with
            | 4 =>
                pure
                  (``pushTile4, ``pushTile4_ofFn,
                    ``fin_foldl_four)
            | 8 =>
                pure
                  (``pushTile8, ``pushTile8_ofFn,
                    ``fin_foldl_eight)
            | _ =>
                throwError
                  "internal error: unsupported generated output tile width \
                    {tileWidth}"
          let (pushTileName, pushTileOfFnName, finFoldlTileName) :=
            tileDeclarations
          let blockCount := lengthValue / tileWidth
          let tailCount := lengthValue % tileWidth
          let tileWidthExpr := mkNatLit tileWidth
          let blockCountExpr := mkNatLit blockCount
          let tailCountExpr := mkNatLit tailCount
          let tiledPrefixExpr := mkNatLit (blockCount * tileWidth)
          let coordinateType ← mkAppM ``Fin #[length]
          let blockType ← mkAppM ``Fin #[blockCountExpr]
          let laneType ← mkAppM ``Fin #[tileWidthExpr]
          let tailType ← mkAppM ``Fin #[tailCountExpr]
          let hTiledPrefix ←
            mkAppM ``Nat.le_add_right #[
              tiledPrefixExpr, tailCountExpr]
          let makeTileCoordinate (block lane : Expr) : MetaM Expr := do
            let prefixCoordinate ←
              mkAppM ``Fin.mkDivMod #[block, lane]
            let coordinate ←
              mkAppM ``Fin.castLE #[hTiledPrefix, prefixCoordinate]
            withTransparency .all <|
              mkExpectedTypeHint coordinate coordinateType
          let makeTailCoordinate (tail : Expr) : MetaM Expr := do
            let coordinate ←
              mkAppM ``Fin.natAdd #[tiledPrefixExpr, tail]
            withTransparency .all <|
              mkExpectedTypeHint coordinate coordinateType
          withLocalDeclD `output outputBufferType fun loopOutput =>
            withLocalDeclD
                (Name.mkSimple s!"outputAxis{coordinates.length}")
                coordinateType fun coordinate => do
              let (generatedBody, referenceBody, hBody) ←
                buildOutputLeaf
                  (coordinates.concat coordinate) loopOutput
              let generatedStep ←
                mkLambdaFVars #[loopOutput, coordinate] generatedBody
              let referenceStep ←
                mkLambdaFVars #[loopOutput, coordinate] referenceBody
              let hCoordinateFunction ←
                mkLambdaFVars #[coordinate] hBody
              let hForOutput ←
                mkAppM ``funext #[hCoordinateFunction]
              let hOutputFunction ←
                mkLambdaFVars #[loopOutput] hForOutput
              let hStep ← mkAppM ``funext #[hOutputFunction]
              withLocalDeclD
                  (Name.mkSimple s!"outputTile{coordinates.length}")
                  blockType fun block => do
                let referenceLaneStep ←
                  withLocalDeclD `output outputBufferType fun tileOutput =>
                    withLocalDeclD `lane laneType fun lane => do
                      let tileCoordinate ←
                        makeTileCoordinate block lane
                      let body :=
                        referenceStep.beta #[
                          tileOutput, tileCoordinate]
                      mkLambdaFVars #[tileOutput, lane] body
                let referenceBlockBody ←
                  mkAppM ``Fin.foldl #[
                    tileWidthExpr, referenceLaneStep, loopOutput]
                let mut tileCoordinates : List Expr := []
                let mut coordinateSets : List (List Expr) := []
                for laneIndex in [:tileWidth] do
                  let lane ← mkNumeral laneType laneIndex
                  let tileCoordinate ←
                    makeTileCoordinate block lane
                  let coordinateValues :=
                    coordinates.concat tileCoordinate
                  tileCoordinates :=
                    tileCoordinates.concat tileCoordinate
                  coordinateSets :=
                    coordinateSets.concat coordinateValues
                let (generatedBlockBody, hBlockBody) ←
                  compileOutputTile coordinateSets
                    fun emittedTile referenceValues hTileReference
                        fusedSource? => do
                      let ordinaryGeneratedBlock ←
                        mkAppM pushTileName #[loopOutput, emittedTile]
                      let (rawTile, finalizers) :=
                        match fusedSource? with
                        | some source => source
                        | none => (emittedTile, [])
                      let (generatedBlockBody, hGeneratedOrdinary) ←
                        match ← fuseNativeFinSumPush tileWidth loopOutput
                            rawTile emittedTile finalizers with
                        | some fused => pure fused
                        | none =>
                            pure
                              (ordinaryGeneratedBlock,
                                ← mkEqRefl ordinaryGeneratedBlock)
                      let pushTileFunction ←
                        withLocalDeclD `values
                            (← inferType emittedTile) fun values => do
                          let pushed ←
                            mkAppM pushTileName #[loopOutput, values]
                          mkLambdaFVars #[values] pushed
                      let hOrdinaryReferenceTile ←
                        mkAppM ``congrArg #[
                          pushTileFunction, hTileReference]
                      let mut pushReferenceArguments := #[loopOutput]
                      for value in referenceValues do
                        pushReferenceArguments :=
                          pushReferenceArguments.push value
                      let hReferenceTileUnrolled ←
                        mkAppM pushTileOfFnName
                          pushReferenceArguments
                      let hOrdinaryUnrolled ←
                        mkAppM ``Eq.trans #[
                          hOrdinaryReferenceTile,
                          hReferenceTileUnrolled]
                      let mut unrolledReference := loopOutput
                      for tileCoordinate in tileCoordinates do
                        unrolledReference :=
                          referenceStep.beta #[
                            unrolledReference, tileCoordinate]
                      let hOrdinaryUnrolled ←
                        withTransparency .all <|
                          mkExpectedTypeHint hOrdinaryUnrolled
                            (← mkEq ordinaryGeneratedBlock
                              unrolledReference)
                      let hGeneratedUnrolled ←
                        mkAppM ``Eq.trans #[
                          hGeneratedOrdinary, hOrdinaryUnrolled]
                      let hReferenceUnrolled ←
                        mkAppM finFoldlTileName #[
                          referenceLaneStep, loopOutput]
                      let hUnrolledReference ←
                        mkAppM ``Eq.symm #[hReferenceUnrolled]
                      let hUnrolledReference ←
                        withTransparency .all <|
                          mkExpectedTypeHint hUnrolledReference
                            (← mkEq unrolledReference referenceBlockBody)
                      let hBlockBody ←
                        mkAppM ``Eq.trans #[
                          hGeneratedUnrolled, hUnrolledReference]
                      pure (generatedBlockBody, hBlockBody)
                let generatedBlockStep ←
                  mkLambdaFVars #[loopOutput, block]
                    generatedBlockBody
                let referenceBlockStep ←
                  mkLambdaFVars #[loopOutput, block]
                    referenceBlockBody
                let hBlockFunction ←
                  mkLambdaFVars #[block] hBlockBody
                let hForOutput ←
                  mkAppM ``funext #[hBlockFunction]
                let hOutputFunction ←
                  mkLambdaFVars #[loopOutput] hForOutput
                let hBlockStep ←
                  mkAppM ``funext #[hOutputFunction]
                let generatedBlockFinLoop ←
                  mkAppM ``Fin.foldl #[
                    blockCountExpr, generatedBlockStep, output]
                let referenceBlockLoop ←
                  mkAppM ``Fin.foldl #[
                    blockCountExpr, referenceBlockStep, output]
                let hBlockFinLoops ←
                  withLocalDeclD `step
                      (← inferType generatedBlockStep) fun step => do
                    let fold ←
                      mkAppM ``Fin.foldl #[
                        blockCountExpr, step, output]
                    let preserveStep ←
                      mkLambdaFVars #[step] fold
                    mkAppM ``congrArg #[
                      preserveStep, hBlockStep]
                let some (nativeBlockBound, hBlockBound) ←
                    nativeLoopBound? blockCountExpr
                  | throwError
                      "internal error: a portable tiled output length \
                        produced a non-portable block count"
                let (nativeBlockLoop, hNativeGeneratedBlocks) ←
                  compileNativeOutputFold
                    blockCountExpr nativeBlockBound hBlockBound
                    generatedBlockStep output generatedBlockFinLoop
                    (Name.mkSimple
                      s!"nativeOutputTile{coordinates.length}")
                let hNativeReferenceBlocks ←
                  mkAppM ``Eq.trans #[
                    hNativeGeneratedBlocks, hBlockFinLoops]
                let tailStepData ←
                  withLocalDeclD `output outputBufferType
                      fun tailOutput =>
                    withLocalDeclD `tail tailType fun tail => do
                      let tailCoordinate ←
                        makeTailCoordinate tail
                      let generatedTailBody :=
                        generatedStep.beta #[
                          tailOutput, tailCoordinate]
                      let referenceTailBody :=
                        referenceStep.beta #[
                          tailOutput, tailCoordinate]
                      let hAtOutput ←
                        mkAppM ``congrFun #[
                          hStep, tailOutput]
                      let hTailBody ←
                        mkAppM ``congrFun #[
                          hAtOutput, tailCoordinate]
                      let generatedTailStep ←
                        mkLambdaFVars #[
                          tailOutput, tail] generatedTailBody
                      let referenceTailStep ←
                        mkLambdaFVars #[
                          tailOutput, tail] referenceTailBody
                      let hTailFunction ←
                        mkLambdaFVars #[tail] hTailBody
                      let hForOutput ←
                        mkAppM ``funext #[hTailFunction]
                      let hOutputFunction ←
                        mkLambdaFVars #[
                          tailOutput] hForOutput
                      let hTailStep ←
                        mkAppM ``funext #[hOutputFunction]
                      pure
                        (generatedTailStep,
                          referenceTailStep, hTailStep)
                let (generatedTailStep,
                    referenceTailStep, hTailStep) := tailStepData
                let generatedTailLoop ←
                  mkAppM ``Fin.foldl #[
                    tailCountExpr, generatedTailStep,
                    nativeBlockLoop]
                let referenceTailLoop ←
                  mkAppM ``Fin.foldl #[
                    tailCountExpr, referenceTailStep,
                    referenceBlockLoop]
                let mut unrolledTail := nativeBlockLoop
                for tailIndex in [:tailCount] do
                  let tail ← mkNumeral tailType tailIndex
                  unrolledTail :=
                    generatedTailStep.beta #[unrolledTail, tail]
                let hUnrolledTail ←
                  if tailCount = 0 then
                    mkAppM ``Eq.symm #[
                      ← mkAppM ``Fin.foldl_zero #[
                        generatedTailStep, nativeBlockLoop]]
                  else
                    certifyWithTactic
                      "that the short tiled-output tail equals its \
                        finite-fold semantics"
                      (← mkEq unrolledTail generatedTailLoop)
                      (← `(tactic|
                        simp only [
                          Fin.foldl_succ,
                          Fin.foldl_zero] <;>
                        congr))
                let hTailLoops ←
                  withLocalDeclD `step
                      (← inferType generatedTailStep) fun step => do
                    let fold ←
                      mkAppM ``Fin.foldl #[
                        tailCountExpr, step, nativeBlockLoop]
                    let preserveStep ←
                      mkLambdaFVars #[step] fold
                    mkAppM ``congrArg #[
                      preserveStep, hTailStep]
                let hReferenceTailInitial ←
                  withLocalDeclD `initial outputBufferType
                      fun initial => do
                    let fold ←
                      mkAppM ``Fin.foldl #[
                        tailCountExpr, referenceTailStep, initial]
                    let preserveInitial ←
                      mkLambdaFVars #[initial] fold
                    mkAppM ``congrArg #[
                      preserveInitial, hNativeReferenceBlocks]
                let hGeneratedReferenceTail ←
                  mkAppM ``Eq.trans #[
                    hUnrolledTail, hTailLoops]
                let hGeneratedTiledReference ←
                  mkAppM ``Eq.trans #[
                    hGeneratedReferenceTail,
                    hReferenceTailInitial]
                let referenceLoop ←
                  mkAppM ``Fin.foldl #[
                    length, referenceStep, output]
                let hTiledReference ←
                  mkAppM ``fin_foldl_tiles #[
                    blockCountExpr, tileWidthExpr,
                    tailCountExpr, referenceStep, output]
                let hTiledReference ←
                  withTransparency .all <|
                    mkExpectedTypeHint hTiledReference
                      (← mkEq referenceTailLoop referenceLoop)
                let hGeneratedReference ←
                  mkAppM ``Eq.trans #[
                    hGeneratedTiledReference, hTiledReference]
                pure
                  (unrolledTail, referenceLoop,
                    hGeneratedReference)
        else
          let coordinateType ← mkAppM ``Fin #[length]
          withLocalDeclD `output outputBufferType fun loopOutput =>
            withLocalDeclD
                (Name.mkSimple s!"outputAxis{coordinates.length}")
                coordinateType fun coordinate => do
              let (generatedBody, referenceBody, hBody) ←
                buildOutputLoops remainingLengths
                  (coordinates.concat coordinate) loopOutput
              let generatedStep ←
                mkLambdaFVars #[loopOutput, coordinate] generatedBody
              let referenceStep ←
                mkLambdaFVars #[loopOutput, coordinate] referenceBody
              let hCoordinateFunction ←
                mkLambdaFVars #[coordinate] hBody
              let hForOutput ←
                mkAppM ``funext #[hCoordinateFunction]
              let hOutputFunction ←
                mkLambdaFVars #[loopOutput] hForOutput
              let hStep ← mkAppM ``funext #[hOutputFunction]
              let generatedFinLoop ←
                mkAppM ``Fin.foldl #[length, generatedStep, output]
              let referenceLoop ←
                mkAppM ``Fin.foldl #[length, referenceStep, output]
              let hFinLoops ←
                withLocalDeclD `step (← inferType generatedStep) fun step => do
                  let fold ←
                    mkAppM ``Fin.foldl #[length, step, output]
                  let preserveStep ←
                    mkLambdaFVars #[step] fold
                  mkAppM ``congrArg #[preserveStep, hStep]
              let some (nativeBound, hBound) ← nativeLoopBound? length
                | return (generatedFinLoop, referenceLoop, hFinLoops)
              let (nativeLoop, hNativeFin) ←
                compileNativeOutputFold
                  length nativeBound hBound
                  generatedStep output generatedFinLoop
                  (Name.mkSimple
                    s!"nativeOutputAxis{coordinates.length}")
              let hNativeReference ←
                mkAppM ``Eq.trans #[hNativeFin, hFinLoops]
              return (nativeLoop, referenceLoop, hNativeReference)
  let compileSequentialOutput :
      TermElabM (Expr × Expr) := do
    let emptyOutput :=
      mkAppN (mkConst ``Storage.emptyWithCapacity [scalarLevel]) #[
        scalarType, storage, outputSize]
    let (rawOutputBuffer, _, hRawOutputReference) ←
      buildOutputLoops outputLengths [] emptyOutput
    -- Only the complete loop nest exposes every outer native coordinate inside
    -- the staged input indices. Normalize here so those conversions disappear
    -- from the executable kernel while the simplifier supplies the equality
    -- needed by the semantic certificate.
    let (outputBuffer, hOutputRaw) ←
      simplifyNativeOutputIndices
        nativeCoordinateSimpContext rawOutputBuffer
    let hOutputReference ←
      mkAppM ``Eq.trans #[hOutputRaw, hRawOutputReference]
    let hOutputReferenceType ← inferType hOutputReference
    let hOutputReference ←
      sealCertificate hOutputReferenceType hOutputReference
    let hOutputArray ←
      match outputLengths with
      | [length] =>
          mkAppM ``storage_toArray_eq_array_ofFn_of_rankOne_eq #[
            length, reference, outputBuffer, hOutputReference]
      | _ => do
          let hCoordinateArray ←
            mkAppM
              ``coordinateFoldl_storagePush_linearized_toArray_eq_array_ofFn #[
              outputShape, reference]
          let toArrayFunction ←
            withLocalDeclD `buffer outputBufferType fun buffer => do
              let observed :=
                mkAppN (mkConst ``Storage.toArray [scalarLevel]) #[
                  scalarType, storage, buffer]
              mkLambdaFVars #[buffer] observed
          let hOutputCoordinateFold ←
            mkAppM ``congrArg #[toArrayFunction, hOutputReference]
          let hOutputCoordinateFoldType ←
            inferType hOutputCoordinateFold
          let hOutputCoordinateFold ←
            sealCertificate hOutputCoordinateFoldType
              hOutputCoordinateFold
          mkAppM ``Eq.trans #[
            hOutputCoordinateFold, hCoordinateArray]
    let hOutputArrayType ← inferType hOutputArray
    let hOutputArray ←
      sealCertificate hOutputArrayType hOutputArray
    pure (outputBuffer, hOutputArray)
  let compileFlatSequentialOutput? :
      TermElabM (Option (Expr × Expr)) := do
    let some concreteOutputLengths ←
        concreteNatExpressions? outputLengths
      | return none
    if concreteOutputLengths.any (· == 0) ||
        Shape.size concreteOutputLengths < 1024 then
      return none
    let some (nativeBound, hBound) ← nativeLoopBound? outputSize
      | return none
    let (nativeValues, hValues) ←
      withLocalDeclD `nativeOutputIndex (mkConst ``USize)
          fun nativeOutputIndex => do
        let nativeOutputIndexNat ←
          mkAppM ``USize.toNat #[nativeOutputIndex]
        let nativeOutputIndexBound ←
          mkLT nativeOutputIndexNat outputSize
        withLocalDeclD `hOutputIndex nativeOutputIndexBound
            fun hOutputIndex => do
          let outputIndex ←
            mkAppOptM ``Fin.mk #[
              some outputSize, some nativeOutputIndexNat,
              some hOutputIndex]
          let (nativeComponents, semanticComponents,
              hComponents) ←
            nativeCoordinateComponents outputLengths
              nativeOutputIndex hOutputIndex
          let semanticCoordinate ←
            coordinateFromComponents semanticComponents
          let unlinearizedCoordinate ←
            mkAppOptM ``Coord.unlinearize #[
              some outputShape, some outputIndex]
          let hSemanticUnlinearized ←
            withTransparency .all <|
              mkExpectedTypeHint
                (← mkEqRefl semanticCoordinate)
                (← mkEq semanticCoordinate unlinearizedCoordinate)
          let (value, hValue) ←
            compileBoundCoordinates outputIndex nativeComponents
              semanticComponents hComponents hSemanticUnlinearized
          let locals := #[nativeOutputIndex, hOutputIndex]
          pure
            (← mkLambdaFVars locals value,
              ← mkLambdaFVars locals hValue)
    let outputBuffer ←
      mkAppM ``nativeBufferOfFn #[
        outputSize, nativeBound, hBound, nativeValues]
    let hOutputArray ←
      mkAppM ``nativeBufferOfFn_toArray #[
        outputSize, nativeBound, hBound,
        nativeValues, reference, hValues]
    let producer ←
      withLocalDeclD `unused (mkConst ``Unit) fun unused =>
        mkLambdaFVars #[unused] outputBuffer
    let certifiedOutput ←
      mkAppM ``CertifiedFlatBuffer.mk #[producer, hOutputArray]
    let certifiedOutput ← sealSequentialOutput certifiedOutput
    let outputBuffer ←
      mkAppM ``CertifiedFlatBuffer.produce #[
        certifiedOutput, mkConst ``Unit.unit]
    let hOutputArray ←
      mkAppM ``CertifiedFlatBuffer.toArray_produce #[certifiedOutput]
    return some (outputBuffer, hOutputArray)
  let (outputBuffer, hOutputArray) ←
    match ←
        compileParallelEinsumOutput?
          scalarType storage outputBufferType reference outputLengths
          contractionEntries? inputFactorization?.isSome buildOutputLoops
          compileNativeOutputFold with
    | some parallelOutput => pure parallelOutput
    | none =>
        match ← compileFlatSequentialOutput? with
        | some flatOutput => pure flatOutput
        | none => compileSequentialOutput
  return (reference, outputCorrectness, outputBuffer, hOutputArray)

end TorchLean.Tensor.Internal.Elab.Impl
