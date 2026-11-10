/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Elab.Contraction.Loop
public meta import TorchLean.Tensor.Einops.Elab.Output.Fusion
public meta import TorchLean.Tensor.Einops.Elab.Output.Planning
public meta import TorchLean.Tensor.Einops.Elab.Output.Parallel
public meta import TorchLean.Tensor.Einops.Elab.Kernel.Product -- shake: keep
public meta import TorchLean.Tensor.Native.Tile.Registers
public import TorchLean.Tensor.Einops.Elab.Output.Planning
public import TorchLean.Tensor.Einops.Elab.Output.Index
public import TorchLean.Tensor.Einops.Elab.Output.Parallel
import TorchLean.Tensor.Einops.Elab.Output.Fusion

/-!
# Verified einsum output generation

`compileEinsumOutput` builds the output buffer of an einsum at any rank,
together with a proof that its native loops compute the independent semantics
`Lowering.einsumOutput`.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open Native
open TorchLean.Shape (Coord)

/--
The operands, shapes, instances and contexts shared by the stages of `compileEinsumOutput`.
-/
private structure OutputSetup where
  /-- The scalar type of the output entries. -/
  scalarType : Expr
  /-- The `Storage` instance of the output buffer. -/
  storage : Expr
  /-- The output axis lengths. -/
  outputLengths : List Expr
  /-- The contracted axis lengths in loop order. -/
  contractedLengths : List Expr
  /-- The generated product of the input entries. -/
  inputProduct : Expr
  /-- The certificate of `inputProduct`. -/
  hInputProduct : Expr
  /-- The semiring factorization of `inputProduct`, when one applies. -/
  inputFactorization? : Option (Expr × Expr × Expr × Expr)
  /-- The reference output `Lowering.einsumOutput`. -/
  reference : Expr
  /-- The output shape. -/
  outputShape : Expr
  /-- The number of output entries. -/
  outputSize : Expr
  /-- The type of output coordinates. -/
  outputCoordinateType : Expr
  /-- The contracted shape of the checked einsum. -/
  semanticContractedShape : Expr
  /-- The contracted shape in loop order. -/
  contractedShape : Expr
  /-- The type of loop-order contraction coordinates. -/
  contractionCoordinateType : Expr
  /-- Whether a contracted axis has length zero. -/
  contractionIsEmpty : Bool
  /-- The number of contraction terms, when every contracted length is concrete. -/
  contractionEntries? : Option Nat
  /-- The scalar reads per contraction term of the product a tile evaluates. -/
  scalarReadsPerTerm : Nat
  /-- The universe level of the scalar type. -/
  scalarLevel : Level
  /-- The `Add` instance of the scalar type. -/
  add : Expr
  /-- The `OfNat` instance of the scalar zero. -/
  ofNatZero : Expr
  /-- The scalar zero. -/
  zero : Expr
  /-- The reference product `Lowering.einsumInputProduct`. -/
  referenceInputProduct : Expr
  /-- The reference product at an output coordinate, over loop-order contraction coordinates. -/
  referenceProductAt : Expr → MetaM Expr
  /-- The equality of the loop-order contraction with the contraction of the checked einsum. -/
  contractionPlanCorrect : Expr
  /-- The focused simp context for native output indices. -/
  nativeIndexSimpContext : Simp.Context
  /-- The focused simp context for native coordinates. -/
  nativeCoordinateSimpContext : Simp.Context
  /-- The type of output buffers. -/
  bufferType : Expr

/--
Run `body` on the product at the output coordinate with components `outputCoordinates`, the raw
output, its finalizer, and the certificate that the finalized raw output is the reference entry
at `outputIndex`.
-/
private def withRawAtCoordinate (setup : OutputSetup) (outputIndex : Expr)
    (outputCoordinates semanticOutputCoordinates
      hOutputCoordinates : List Expr)
    (hSemanticOutputIndex : Expr)
    (body : Expr → Expr → Expr → Expr →
      TermElabM (Expr × Expr)) :
    TermElabM (Expr × Expr) := do
  let scalarType := setup.scalarType
  let inputProduct := setup.inputProduct
  let hInputProduct := setup.hInputProduct
  let inputFactorization? := setup.inputFactorization?
  let reference := setup.reference
  let outputShape := setup.outputShape
  let outputCoordinateType := setup.outputCoordinateType
  let semanticContractedShape := setup.semanticContractedShape
  let contractedShape := setup.contractedShape
  let contractionCoordinateType := setup.contractionCoordinateType
  let contractionIsEmpty := setup.contractionIsEmpty
  let scalarLevel := setup.scalarLevel
  let add := setup.add
  let ofNatZero := setup.ofNatZero
  let referenceInputProduct := setup.referenceInputProduct
  let referenceProductAt := setup.referenceProductAt
  let contractionPlanCorrect := setup.contractionPlanCorrect
  let nativeIndexSimpContext := setup.nativeIndexSimpContext
  let outputCoordinate ←
    coordinateFromComponents outputCoordinates
  let semanticOutputCoordinate ←
    coordinateFromComponents semanticOutputCoordinates
  let hOutputCoordinate ←
    coordinateFromComponentsEq outputCoordinates
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
          mkAppM ``Semantics.contraction #[
            semanticContractedShape, product]
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
          mkAppM ``Semantics.contraction #[
            contractedShape, productAtOutput]
        let referenceProductAtOutput ←
          referenceProductAt outputCoordinate
        let hRawReference ←
          if contractionIsEmpty then
            mkAppM ``Semantics.contraction_congr_of_isEmpty #[
              contractedShape, productAtOutput,
              referenceProductAtOutput]
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
                  (mkConst ``Semantics.contraction_congr
                    [scalarLevel]) #[
                scalarType, add, ofNatZero, contractedShape,
                productAtOutput, referenceProductAtOutput,
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
          mkAppM ``Semantics.contraction #[
            contractedShape, middleAtOutput]
        let finalizeOutput ←
          withLocalDeclD `value scalarType fun value => do
            let finalized ←
              mkMul (← mkMul leftAtOutput value) rightAtOutput
            mkLambdaFVars #[value] finalized
        let finalizedOutput := finalizeOutput.beta #[rawOutput]
        let hFactoredSum ←
          mkAppM ``Semantics.mul_contraction_mul #[
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
            (mkConst ``Semantics.contraction_congr
              [scalarLevel]) #[
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
            referenceProductAtOutput, hPointwise]
        let hFinalizedReference ←
          mkAppM ``Eq.trans #[
            hFactoredSum, hFactoredReferenceSum]
        finish middleAtOutput rawOutput finalizeOutput
          finalizedOutput hFinalizedReference

/--
Run `withRawAtCoordinate` with the generated output coordinates `coordinateValues` bound once as
let results.
-/
private def withRawBoundCoordinates (setup : OutputSetup) (outputIndex : Expr)
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
        (outputAxisName axisPosition,
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
    withRawAtCoordinate setup outputIndex outputCoordinates
      semanticCoordinateValues hOutputCoordinates hSemanticOutputIndex body

/--
Compile the output entry at the generated output coordinates `coordinateValues` and its
certificate against the reference entry at `outputIndex`.
-/
private def compileBoundCoordinates (setup : OutputSetup) (outputIndex : Expr)
    (coordinateValues semanticCoordinateValues
      hCoordinateValues : List Expr)
    (hSemanticOutputIndex : Expr) :
    TermElabM (Expr × Expr) := do
  let contractedLengths := setup.contractedLengths
  let zero := setup.zero
  withRawBoundCoordinates setup outputIndex coordinateValues
      semanticCoordinateValues hCoordinateValues hSemanticOutputIndex
      fun productAtOutput rawOutput finalizeOutput hRawReference => do
    -- Without a contracted axis the value is the product itself: nothing is added.
    let (optimizedOutput, hRawOptimized) ←
      if contractedLengths.isEmpty then do
        let product :=
          (productAtOutput.beta #[mkConst ``PUnit.unit [Level.one]]).headBeta
        let hProduct ←
          withTransparency .all <|
            mkExpectedTypeHint (← mkEqRefl product) (← mkEq rawOutput product)
        pure (product, hProduct)
      else
        compileCoordinateSum contractedLengths productAtOutput zero rawOutput
    let hOptimizedRaw ←
      mkAppM ``Eq.symm #[hRawOptimized]
    let finalizedOutput := finalizeOutput.beta #[optimizedOutput]
    let hFinalizedRaw ←
      mkAppM ``congrArg #[finalizeOutput, hOptimizedRaw]
    let hOptimizedReference ←
      mkAppM ``Eq.trans #[hFinalizedRaw, hRawReference]
    pure (finalizedOutput, hOptimizedReference)

/--
Build a native output fold together with its equality to the corresponding
finite fold.
-/
private def compileNativeOutputFold (setup : OutputSetup)
    (length nativeBound hBound generatedStep initial generatedFinLoop : Expr)
    (nativeCoordinateName : Name) : TermElabM (Expr × Expr) := do
  let nativeCoordinateSimpContext := setup.nativeCoordinateSimpContext
  let bufferType := setup.bufferType
  let (nativeCallback, rawNativeCallback, hNativeCallback) ←
    withLocalDeclD `output bufferType fun nativeOutput =>
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
    mkAppM ``finFoldl #[
      length, nativeBound, hBound, nativeCallback, initial]
  let rawNativeLoop ←
    mkAppM ``finFoldl #[
      length, nativeBound, hBound, rawNativeCallback, initial]
  let hNativeCallbackLoop ←
    withLocalDeclD `step (← inferType nativeCallback) fun step => do
      let loop ←
        mkAppM ``finFoldl #[
          length, nativeBound, hBound, step, initial]
      let preserveCallback ← mkLambdaFVars #[step] loop
      mkAppM ``congrArg #[preserveCallback, hNativeCallback]
  let hRawNativeFin ←
    mkAppM ``finFoldl_eq_fin_foldl #[
      length, nativeBound, hBound, generatedStep, initial]
  let hRawNativeFin ←
    withTransparency .all <|
      mkExpectedTypeHint hRawNativeFin
        (← mkEq rawNativeLoop generatedFinLoop)
  let hNativeFin ←
    mkAppM ``Eq.trans #[hNativeCallbackLoop, hRawNativeFin]
  pure (nativeLoop, hNativeFin)

/--
Append the entry at the output coordinate `coordinates` to `output`, returning the generated
buffer, the reference buffer, and their equality.
-/
private def buildOutputLeaf (setup : OutputSetup)
    (coordinates : List Expr) (output : Expr) :
    TermElabM (Expr × Expr × Expr) := do
  let scalarType := setup.scalarType
  let storage := setup.storage
  let reference := setup.reference
  let outputShape := setup.outputShape
  let scalarLevel := setup.scalarLevel
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
    compileBoundCoordinates setup outputIndex coordinates coordinates
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

/--
Compile neighboring output coordinates through one shared contraction.
The returned pointwise certificates let each caller choose how to store
the completed tile without changing scalar reduction order.
-/
private def compileOutputTile (setup : OutputSetup)
    (coordinateSets : List (List Expr))
    (finishTile : Expr → List Expr → Expr →
      Option (Expr × List Expr) →
      TermElabM (Expr × Expr)) :
    TermElabM (Expr × Expr) := do
  let scalarType := setup.scalarType
  let contractedLengths := setup.contractedLengths
  let inputFactorization? := setup.inputFactorization?
  let reference := setup.reference
  let outputShape := setup.outputShape
  let contractedShape := setup.contractedShape
  let contractionCoordinateType := setup.contractionCoordinateType
  let zero := setup.zero
  let tileWidth := coordinateSets.length
  let (updateTileName, updateTileRefinementName,
      selectTileName, coordinateSumSelectName,
      tileCongruenceName) ←
    match outputTile? tileWidth with
    | some tile =>
        pure
          (tile.update, tile.updateEqUpdateTile,
            tile.select, tile.coordinateSumSelect,
            tile.ofFnSelectCongr)
    | none =>
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
        withRawBoundCoordinates setup outputIndex
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
          mkAppM ``Coord.foldl #[
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
            let laneStep ←
              withLocalDeclD `total scalarType fun total =>
                withLocalDeclD `contractionCoordinate
                    contractionCoordinateType
                    fun contractionCoordinate => do
                  let laneTotal ←
                    mkAdd total (mkApp laneValues contractionCoordinate)
                  mkLambdaFVars #[total, contractionCoordinate] laneTotal
            let laneSum ←
              mkAppM ``Coord.foldl #[
                contractedShape, laneStep, zero]
            mkLambdaFVars #[lane] laneSum
        let genericRawTile ←
          mkAppM ``Vector.ofFn #[genericRawTileFunction]
        let hFoldGeneric ←
          mkAppM ``Coord.foldl_updateTile_of_eq #[
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
          mkCongrArg genericRawTile.appFn! hCoordinateSumSelect
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

/--
Generate nested output folds and their equality to the semantic coordinate
traversal. The coordinate list is kept in source-axis order.
-/
private def buildOutputLoops (setup : OutputSetup)
    (remainingLengths : List Expr)
    (coordinates : List Expr)
    (output : Expr) : TermElabM (Expr × Expr × Expr) := do
  let contractionEntries? := setup.contractionEntries?
  let scalarReadsPerTerm := setup.scalarReadsPerTerm
  let bufferType := setup.bufferType
  match remainingLengths with
  | [] => buildOutputLeaf setup coordinates output
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
          match outputTile? tileWidth with
          | some tile =>
              pure
                (tile.push, tile.pushOfFn,
                  tile.finFoldl)
          | none =>
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
        withLocalDeclD `output bufferType fun loopOutput =>
          withLocalDeclD
              (outputAxisName coordinates.length)
              coordinateType fun coordinate => do
            let (generatedBody, referenceBody, hBody) ←
              buildOutputLeaf setup
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
                withLocalDeclD `output bufferType fun tileOutput =>
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
                compileOutputTile setup coordinateSets
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
                compileNativeOutputFold setup
                  blockCountExpr nativeBlockBound hBlockBound
                  generatedBlockStep output generatedBlockFinLoop
                  (Name.mkSimple
                    s!"nativeOutputTile{coordinates.length}")
              let hNativeReferenceBlocks ←
                mkAppM ``Eq.trans #[
                  hNativeGeneratedBlocks, hBlockFinLoops]
              let tailStepData ←
                withLocalDeclD `output bufferType
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
                withLocalDeclD `initial bufferType
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
        withLocalDeclD `output bufferType fun loopOutput =>
          withLocalDeclD
              (outputAxisName coordinates.length)
              coordinateType fun coordinate => do
            let (generatedBody, referenceBody, hBody) ←
              buildOutputLoops setup remainingLengths
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
              compileNativeOutputFold setup
                length nativeBound hBound
                generatedStep output generatedFinLoop
                (Name.mkSimple
                  s!"nativeOutputAxis{coordinates.length}")
            let hNativeReference ←
              mkAppM ``Eq.trans #[hNativeFin, hFinLoops]
            return (nativeLoop, referenceLoop, hNativeReference)

/--
Compile the output with nested output folds from an empty buffer, returning the buffer and its
certificate against the reference array.
-/
private def compileSequentialOutput (setup : OutputSetup) :
    TermElabM (Expr × Expr) := do
  let scalarType := setup.scalarType
  let storage := setup.storage
  let outputLengths := setup.outputLengths
  let reference := setup.reference
  let outputShape := setup.outputShape
  let outputSize := setup.outputSize
  let scalarLevel := setup.scalarLevel
  let nativeCoordinateSimpContext := setup.nativeCoordinateSimpContext
  let bufferType := setup.bufferType
  let emptyOutput :=
    mkAppN (mkConst ``Storage.emptyWithCapacity [scalarLevel]) #[
      scalarType, storage, outputSize]
  let (rawBuffer, _, hRawOutputReference) ←
    buildOutputLoops setup outputLengths [] emptyOutput
  -- Only the complete loop nest exposes every outer native coordinate inside
  -- the staged input indices. Normalize here so those conversions disappear
  -- from the executable kernel while the simplifier supplies the equality
  -- needed by the semantic certificate.
  let (buffer, hOutputRaw) ←
    simplifyNativeOutputIndices
      nativeCoordinateSimpContext rawBuffer
  let hOutputReference ←
    mkEqTrans hOutputRaw hRawOutputReference
  let hOutputReferenceType ← inferType hOutputReference
  let hOutputReference ←
    sealCertificate hOutputReferenceType hOutputReference
  let hOutputArray ←
    match outputLengths with
    | [length] =>
        mkAppM ``toArray_eq_ofFn_of_rankOne #[
          length, reference, buffer, hOutputReference]
    | _ => do
        let hCoordinateArray ←
          mkAppM
            ``Coord.toArray_foldl_push_eq_ofFn #[
            outputShape, reference]
        let toArrayFunction ←
          withLocalDeclD `buffer bufferType fun buffer => do
            let observed :=
              mkAppN (mkConst ``Storage.toArray [scalarLevel]) #[
                scalarType, storage, buffer]
            mkLambdaFVars #[buffer] observed
        let hOutputCoordinateFold ←
          mkCongrArg toArrayFunction hOutputReference
        let hOutputCoordinateFoldType ←
          inferType hOutputCoordinateFold
        let hOutputCoordinateFold ←
          sealCertificate hOutputCoordinateFoldType
            hOutputCoordinateFold
        mkEqTrans hOutputCoordinateFold hCoordinateArray
  let hOutputArrayType ← inferType hOutputArray
  let hOutputArray ←
    sealCertificate hOutputArrayType hOutputArray
  pure (buffer, hOutputArray)

/--
Compile a concrete output of at least 1,024 entries as one flat native traversal, or return
`none` when the output does not qualify.
-/
private def compileFlatSequentialOutput? (setup : OutputSetup) :
    TermElabM (Option (Expr × Expr)) := do
  let outputLengths := setup.outputLengths
  let reference := setup.reference
  let outputShape := setup.outputShape
  let outputSize := setup.outputSize
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
          compileBoundCoordinates setup outputIndex nativeComponents
            semanticComponents hComponents hSemanticUnlinearized
        let locals := #[nativeOutputIndex, hOutputIndex]
        pure
          (← mkLambdaFVars locals value,
            ← mkLambdaFVars locals hValue)
  let buffer ←
    mkAppM ``bufferOfFn #[
      outputSize, nativeBound, hBound, nativeValues]
  let hOutputArray ←
    mkAppM ``bufferOfFn_toArray #[
      outputSize, nativeBound, hBound,
      nativeValues, reference, hValues]
  let producer ←
    withLocalDeclD `unused (mkConst ``Unit) fun unused =>
      mkLambdaFVars #[unused] buffer
  let certifiedOutput ←
    mkAppM ``CertifiedFlatBuffer.mk #[producer, hOutputArray]
  -- One auxiliary record keeps the executable native loop and its proof behind one constant, so
  -- final tensor assembly does not normalize the complete buffer construction again.
  let certifiedOutput ← sealAuxDefinition `_einops_sequential_output certifiedOutput
  let buffer ←
    mkAppM ``CertifiedFlatBuffer.produce #[
      certifiedOutput, mkConst ``Unit.unit]
  let hOutputArray ←
    mkAppM ``CertifiedFlatBuffer.toArray_produce #[certifiedOutput]
  return some (buffer, hOutputArray)

/--
Compile the contraction into a certified arbitrary-rank output buffer.

The compiler first tries task-parallel outer-axis chunks. Otherwise, concrete
portable buffers with at least 1,024 entries use a flat native traversal. The
remaining cases use nested output folds. Native counters require portable
concrete bounds, while symbolic or oversized axes retain `Fin.foldl`.

Nested folds may tile the final axis according to `einsumOutputTileWidth?`.
Each lane preserves its scalar contraction order, and incomplete tiles use
the scalar path. A single native contraction may fuse its completed tile
directly into the output append. Every selected path supplies an equality
certificate against the general `Array.ofFn` reference.
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
    mkAppM ``Einsum.output #[checked]
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
        mkAppM ``Semantics.contraction #[
          contractedShape, plannedProduct]
      let semanticProduct :=
        mkApp referenceInputProduct outputCoordinate
      let semanticSum ←
        mkAppM ``Semantics.contraction #[
          semanticContractedShape, semanticProduct]
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
              mkAppM ``Einsum.axisLength #[checked]
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
  -- `einsumTensorKernel` erases this function and its certificate at runtime.
  -- Handing it the independent semantics saves building a second scalar
  -- executor used only as a proof index.
  let outputCorrectness ←
    withLocalDeclD `outputIndex checkedOutputIndexType fun outputIndex => do
      let value := mkApp reference outputIndex
      let hValue ← mkEqRefl value
      mkLambdaFVars #[outputIndex] hValue
  let bufferType :=
    mkAppN (mkConst ``Storage.Buffer [scalarLevel]) #[
      scalarType, storage]
  let setup : OutputSetup := {
    scalarType, storage, outputLengths, contractedLengths, inputProduct, hInputProduct,
    inputFactorization?, reference, outputShape, outputSize, outputCoordinateType,
    semanticContractedShape, contractedShape, contractionCoordinateType, contractionIsEmpty,
    contractionEntries?, scalarReadsPerTerm, scalarLevel, add, ofNatZero, zero,
    referenceInputProduct, referenceProductAt, contractionPlanCorrect, nativeIndexSimpContext,
    nativeCoordinateSimpContext, bufferType }
  let (buffer, hOutputArray) ←
    match ←
        compileParallelEinsumOutput?
          scalarType storage bufferType reference outputLengths
          contractionEntries? (buildOutputLoops setup)
          (compileNativeOutputFold setup) with
    | some parallelOutput => pure parallelOutput
    | none =>
        match ← compileFlatSequentialOutput? setup with
        | some flatOutput => pure flatOutput
        | none => compileSequentialOutput setup
  return (reference, outputCorrectness, buffer, hOutputArray)

end TorchLean.Einops.Elab
