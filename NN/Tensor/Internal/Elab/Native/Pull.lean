/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Native.Transpose
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Affine
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Index
public import NN.Tensor.Internal.Elab.Native.Slice
public meta import NN.Tensor.Internal.Elab.Transform.Index
public import NN.Tensor.Internal.Elab.Native.Tensor -- shake: keep
public import NN.Tensor.Internal.Lowering.TransformFusion -- shake: keep

/-!
# Certified native flat pullbacks

This module compiles a certified output-to-source flat map into one native
output loop and direct source-buffer reads. Rearrangement, repetition, and
other shape-only transformations use this operation-independent lowering.

The generated term carries its pointwise equality to `Rep.pullFlat`.
Concrete portable shapes use `USize` counters and indices; symbolic shapes
remain on the existing general lowering.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Close a generated executable callback into one compiled auxiliary definition.

Large affine callbacks otherwise remain inline arguments of the semantic
reference term, forcing downstream elaborators to traverse their complete
proof-oriented syntax even though code generation is their only consumer.
-/
private def sealNativeCallback (callback : Expr) : TermElabM Expr := do
  let name ← mkAuxName `_einops_native_callback
  mkAuxDefinitionFor name callback (zetaDelta := true)

/--
Compile the native and semantic callbacks shared by flat-pullback kernels.
-/
private def compileNativePullFlatData?
    (outputShape sourceTensor runtimeFlatMap semanticFlatMap
      hFlatMap : Expr)
    (nativeSourceDimensions? : Option (List Expr) := none) :
    TermElabM
      (Option
        (Expr × Expr × Expr × Expr × Expr × Option (Expr × Expr))) := do
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let some (outputBound, hOutputBound) ← nativeLoopBound? outputSize
    | return none
  let sourceTensorType ← whnf (← inferType sourceTensor)
  let sourceTensorType := sourceTensorType.consumeMData
  unless sourceTensorType.isAppOfArity ``Rep 3 do
    throwError
      "internal error: a native flat pullback has a non-tensor source"
  let sourceShape := sourceTensorType.getAppArgs[1]!
  let sourceSize ← mkAppM ``Shape.size #[sourceShape]
  let sourceDimensions? ←
    match nativeSourceDimensions? with
    | some sourceDimensions => pure (some sourceDimensions)
    | none => staticListElements? sourceShape
  let useNativeSourceIndex ←
    match sourceDimensions? with
    | some sourceDimensions =>
        supportsNativeInputIndex sourceDimensions
    | none => pure false
  let outputIndexType ← mkAppM ``Fin #[outputSize]
  let values ←
    withLocalDeclD `outputIndex outputIndexType fun outputIndex => do
      let sourceIndex := mkApp semanticFlatMap outputIndex
      let value ← mkAppM ``Rep.getFlat #[sourceTensor, sourceIndex]
      mkLambdaFVars #[outputIndex] value
  let (nativeValues, hValues, nativeGather?) ←
    withLocalDeclD `outputIndex (mkConst ``USize) fun outputIndex => do
      let outputIndexNat ← mkAppM ``USize.toNat #[outputIndex]
      let outputIndexBoundType ← mkLT outputIndexNat outputSize
      withLocalDeclD `hOutputIndex outputIndexBoundType fun hOutputIndex => do
        let outputFin ←
          mkAppOptM ``Fin.mk #[
            some outputSize, some outputIndexNat, some hOutputIndex]
        let runtimeSourceIndex := mkApp runtimeFlatMap outputFin
        let semanticSourceIndex := mkApp semanticFlatMap outputFin
        let runtimeSourceIndexValue ←
          mkAppM ``Fin.val #[runtimeSourceIndex]
        let normalizedSourceIndexValue ← zetaReduce runtimeSourceIndexValue
        let (optimizedSourceIndexValue, hSourceIndexOptimized) ←
          simplifyAffineIndex normalizedSourceIndexValue
        let hOptimizedRuntimeSourceIndex ←
          mkAppM ``Eq.symm #[hSourceIndexOptimized]
        let runtimeSourceIndexBound ←
          mkAppM ``Fin.isLt #[runtimeSourceIndex]
        let optimizedSourceIndexBound ←
          indexBoundFromValueEquality sourceSize
            hOptimizedRuntimeSourceIndex runtimeSourceIndexBound
        let sourceIndexType ← mkAppM ``Fin #[sourceSize]
        let sourceIndexValue ←
          withLocalDeclD `sourceIndex sourceIndexType fun sourceIndex => do
            let value ← mkAppM ``Fin.val #[sourceIndex]
            mkLambdaFVars #[sourceIndex] value
        let hRuntimeSemanticSourceIndex ←
          mkAppM ``congrArg #[
            sourceIndexValue, mkApp hFlatMap outputFin]
        let hOptimizedSourceIndex ←
          mkAppM ``Eq.trans #[
            hOptimizedRuntimeSourceIndex, hRuntimeSemanticSourceIndex]
        let hOptimizedSourceIndexType ← inferType hOptimizedSourceIndex
        let hOptimizedSourceIndex ←
          sealCertificate hOptimizedSourceIndexType hOptimizedSourceIndex
        let (sourceValue, hSourceValue) ←
          compileInputRead sourceTensor sourceSize
            optimizedSourceIndexValue optimizedSourceIndexValue
            semanticSourceIndex hOptimizedSourceIndex useNativeSourceIndex
            (some optimizedSourceIndexBound)
            #[hOutputIndex]
        let nativeValues ←
          mkLambdaFVars #[outputIndex, hOutputIndex] sourceValue
        let nativeGather? ←
          if useNativeSourceIndex then
            let sourceValue := sourceValue.consumeMData
            unless sourceValue.isAppOfArity ``Rep.getFlatUSize 6 do
              throwError
                "internal error: a native pullback read is not a native \
                  tensor read"
            let sourceArguments := sourceValue.getAppArgs
            let sourceIndex := sourceArguments[4]!
            let hSourceIndex := sourceArguments[5]!
            let sourceIndices ←
              mkLambdaFVars #[outputIndex, hOutputIndex] sourceIndex
            let hSourceIndices ←
              mkLambdaFVars #[outputIndex, hOutputIndex] hSourceIndex
            let sourceIndices ← sealNativeCallback sourceIndices
            let hSourceIndicesType ← inferType hSourceIndices
            let hSourceIndices ←
              sealCertificate hSourceIndicesType hSourceIndices
            pure (some (sourceIndices, hSourceIndices))
          else
            pure none
        let expectedSourceValue := mkApp values outputFin
        let hSourceValue ←
          withTransparency .all <|
            mkExpectedTypeHint hSourceValue
              (← mkEq sourceValue expectedSourceValue)
        let hValues ←
          mkLambdaFVars #[outputIndex, hOutputIndex] hSourceValue
        let hValuesType ← inferType hValues
        let hValues ← sealCertificate hValuesType hValues
        let nativeValues ← sealNativeCallback nativeValues
        return (nativeValues, hValues, nativeGather?)
  return some
    (outputBound, hOutputBound, nativeValues, values, hValues, nativeGather?)

/-- Build the physical gather when available, otherwise retain scalar fill. -/
private def buildNativePullImplementation
    (sourceTensor outputBound hOutputBound nativeValues values hValues : Expr)
    (nativeGather? : Option (Expr × Expr)) :
    TermElabM (Expr × Expr) := do
  match nativeGather? with
  | some (sourceIndices, hSourceIndices) =>
      let implementation ←
        mkAppM ``nativeTensorGather #[
          sourceTensor, outputBound, hOutputBound,
          sourceIndices, hSourceIndices, values, hValues]
      let hImplementation ←
        mkAppM ``nativeTensorGather_correct #[
          sourceTensor, outputBound, hOutputBound,
          sourceIndices, hSourceIndices, values, hValues]
      return (implementation, hImplementation)
  | none =>
      let implementation ←
        mkAppM ``nativeTensorOfFlatFn #[
          outputBound, hOutputBound, nativeValues, values, hValues]
      let hImplementation ←
        mkAppM ``nativeTensorOfFlatFn_correct #[
          outputBound, hOutputBound, nativeValues, values, hValues]
      return (implementation, hImplementation)

/--
Return the number of contiguous source-buffer copies represented by a repeat
whose original logical axes are the exact suffix of the output axes.

For example, `column -> row column` lays out each repeated row as one complete
copy of the source buffer. Interleaved repeats retain the general gather
kernel.
-/
private def leadingRepeatCount?
    (checked : Check.CheckedTransform) : Option Nat := do
  guard (checked.value.normalized.kind = .repeat)
  let inputAxes := checked.value.normalized.inputAxes
  let outputAxes := checked.value.normalized.outputAxes
  guard (inputAxes.length ≤ outputAxes.length)
  let introducedCount := outputAxes.length - inputAxes.length
  guard (outputAxes.drop introducedCount = inputAxes)
  let introducedAxes := outputAxes.take introducedCount
  return (introducedAxes.map checked.value.axisLength).prod

/--
Recognize the exact rank-two axis swap represented by `row column ->
column row`.

Composite axes and rank-changing patterns deliberately stay on the general
certified gather path.
-/
private def transpose2DDimensions?
    (checked : Check.CheckedTransform) : Option (Nat × Nat) := do
  guard (checked.value.normalized.kind = .rearrange)
  match checked.value.normalized.inputGroups,
      checked.value.normalized.outputGroups,
      checked.value.normalized.input,
      checked.value.output with
  | [[inputRow], [inputColumn]],
      [[outputColumn], [outputRow]],
      [rows, columns], [outputColumns, outputRows] =>
      guard (outputColumn = inputColumn)
      guard (outputRow = inputRow)
      guard (outputColumns = columns)
      guard (outputRows = rows)
      return (rows, columns)
  | _, _, _, _ => none

/--
Compile a concrete rank-two swap to the storage-specific tiled native
transpose.

The generated index equality is checked by the same row-major simplifier used
for other native tensor kernels. Packed `FloatArray` and ordinary polymorphic
`Array` storage have dedicated implementations. If either the storage or index
program is not the exact supported form, elaboration falls back to the generic
gather.
-/
private def compileNativeTranspose2D?
    (checkedValue : Check.CheckedTransform)
    (outputShape sourceTensor runtimeFlatMap : Expr) :
    TermElabM (Option (Expr × Expr)) := do
  let some (rows, columns) := transpose2DDimensions? checkedValue
    | return none
  let sourceTensorType ← whnf (← inferType sourceTensor)
  let sourceTensorType := sourceTensorType.consumeMData
  unless sourceTensorType.isAppOfArity ``Rep 3 do
    return none
  let sourceArguments := sourceTensorType.getAppArgs
  let scalarType := sourceArguments[0]!
  let storage := sourceArguments[2]!
  let isFloat ← isDefEq scalarType (mkConst ``Float)
  let isPackedFloatStorage ←
    if isFloat then
      isDefEq storage (mkConst ``instFloatStorage)
    else
      pure false
  let arrayStorage ← mkAppM ``instArrayStorage #[scalarType]
  let isArrayStorage ← isDefEq storage arrayStorage
  unless isPackedFloatStorage || isArrayStorage do
    return none
  let sourceShape := sourceArguments[1]!
  let expectedSourceShape := Lean.toExpr ([rows, columns] : Shape)
  let expectedOutputShape := Lean.toExpr ([columns, rows] : Shape)
  unless ← isDefEq sourceShape expectedSourceShape do
    return none
  unless ← isDefEq outputShape expectedOutputShape do
    return none
  let literalRows := mkNatLit rows
  let literalColumns := mkNatLit columns
  let literalElementCount := mkNatLit (rows * columns)
  let some _ ← nativeLoopBound? literalRows
    | return none
  let some _ ← nativeLoopBound? literalColumns
    | return none
  let some _ ← nativeLoopBound? literalElementCount
    | return none
  let canonicalFlatMap ←
    mkAppM ``transpose2DShapeIndex #[literalRows, literalColumns]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let outputIndexType ← mkAppM ``Fin #[outputSize]
  let hFlatMap? ←
    withLocalDeclD `outputIndex outputIndexType fun outputIndex => do
      let canonicalIndex := mkApp canonicalFlatMap outputIndex
      let runtimeIndex := mkApp runtimeFlatMap outputIndex
      let extraDeclarations := #[
        ``transpose2DShapeIndex,
        ``transpose2DIndex,
        ``rectangularIndex]
      let (canonicalNormalized, hCanonical) ←
        compileRowMajorIndex canonicalIndex extraDeclarations
      let (runtimeNormalized, hRuntime) ←
        compileRowMajorIndex runtimeIndex extraDeclarations
      unless ← isDefEq canonicalNormalized runtimeNormalized do
        return none
      let hNormalized ←
        withTransparency .all <|
          mkExpectedTypeHint (← mkEqRefl canonicalNormalized)
            (← mkEq canonicalNormalized runtimeNormalized)
      let hCanonical ← mkAppM ``Eq.symm #[hCanonical]
      let equality ←
        mkAppM ``Eq.trans #[
          hCanonical, ← mkAppM ``Eq.trans #[hNormalized, hRuntime]]
      return some (← mkLambdaFVars #[outputIndex] equality)
  let some hFlatMap := hFlatMap?
    | return none
  let (implementation, hNative) ←
    if isPackedFloatStorage then
      let implementation ←
        mkAppM ``nativeFloatTranspose2D #[
          literalRows, literalColumns, sourceTensor]
      let hNative ←
        mkAppM ``nativeFloatTranspose2D_correct #[
          literalRows, literalColumns, sourceTensor]
      pure (implementation, hNative)
    else
      let implementation ←
        mkAppM ``nativeArrayTranspose2D #[
          literalRows, literalColumns, sourceTensor]
      let hNative ←
        mkAppM ``nativeArrayTranspose2D_correct #[
          literalRows, literalColumns, sourceTensor]
      pure (implementation, hNative)
  let hPullback ←
    mkAppM ``Rep.pullFlat_congr #[
      canonicalFlatMap, runtimeFlatMap, hFlatMap, sourceTensor]
  let hImplementation ← mkAppM ``Eq.trans #[hNative, hPullback]
  return some (implementation, hImplementation)

/--
Compile a leading-axis repeat to whole-buffer slice copies.

This avoids one remainder, one bounds proof, and one scalar storage dispatch
per output value. The implementation remains generic over `Storage`; packed
storage classes may implement `appendSlice` as a native bulk copy.
-/
private def compileNativeLeadingRepeatSlices?
    (checkedValue : Check.CheckedTransform)
    (outputShape sourceTensor runtimeFlatMap : Expr) :
    TermElabM (Option (Expr × Expr)) := do
  let some rowCount := leadingRepeatCount? checkedValue
    | return none
  let rowLength := Shape.size checkedValue.value.normalized.input
  let outputLength := Shape.size checkedValue.value.output
  if rowCount = 0 || rowLength = 0 ||
      rowCount * rowLength != outputLength then
    return none
  let literalRowCount := mkNatLit rowCount
  let literalRowLength := mkNatLit rowLength
  let literalZero := mkNatLit 0
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let hShapeSize ←
    certifyGeneratedInvariant
      "that a leading-axis repeat is a rectangle of source buffers"
      (← mkEq
        (← mkAppM ``Nat.mul #[literalRowCount, literalRowLength])
        outputSize)
  let some (rowBound, hRowBound) ← nativeLoopBound? literalRowCount
    | return none
  let sourceTensorType ← whnf (← inferType sourceTensor)
  let sourceTensorType := sourceTensorType.consumeMData
  unless sourceTensorType.isAppOfArity ``Rep 3 do
    throwError
      "internal error: a native leading repeat has a non-tensor source"
  let sourceShape := sourceTensorType.getAppArgs[1]!
  let sourceSize ← mkAppM ``Shape.size #[sourceShape]
  let rowType ← mkAppM ``Fin #[literalRowCount]
  let columnType ← mkAppM ``Fin #[literalRowLength]
  let hRanges ←
    withLocalDeclD `row rowType fun row => do
      let rowValue ← mkAppM ``Fin.val #[row]
      let start ←
        mkAppM ``Nat.add #[
          ← mkAppM ``Nat.mul #[rowValue, literalZero], literalZero]
      let stop ← mkAppM ``Nat.add #[start, literalRowLength]
      let hRange ←
        certifyGeneratedInvariant
          "that a leading-repeat source slice covers the source buffer"
          (← mkLE stop sourceSize)
      mkLambdaFVars #[row] hRange
  let values ←
    withLocalDeclD `outputIndex (← mkAppM ``Fin #[outputSize])
        fun outputIndex => do
      let value ←
        mkAppM ``Rep.getFlat #[
          sourceTensor, mkApp runtimeFlatMap outputIndex]
      mkLambdaFVars #[outputIndex] value
  let hValues ←
    withLocalDeclD `row rowType fun row => do
      withLocalDeclD `column columnType fun column => do
        let rowValue ← mkAppM ``Fin.val #[row]
        let columnValue ← mkAppM ``Fin.val #[column]
        let sourceIndexValue ←
          mkAppM ``Nat.add #[
            ← mkAppM ``Nat.add #[
              ← mkAppM ``Nat.mul #[rowValue, literalZero],
              literalZero],
            columnValue]
        let hSourceIndex ←
          certifyGeneratedInvariant
            "that a leading-repeat column lies inside the source buffer"
            (← mkLT sourceIndexValue sourceSize)
        let sourceIndex ←
          mkAppOptM ``Fin.mk #[
            some sourceSize, some sourceIndexValue, some hSourceIndex]
        let rectangularIndex ←
          mkAppM ``rectangularIndex #[
            literalRowCount, literalRowLength, row, column]
        let outputIndex ←
          mkAppM ``Fin.cast #[hShapeSize, rectangularIndex]
        let runtimeSourceIndex := mkApp runtimeFlatMap outputIndex
        let runtimeSourceValue ← mkAppM ``Fin.val #[runtimeSourceIndex]
        let hIndexValue ←
          certifyWithTactic
            "that leading-repeat slices implement the checked flat map"
            (← mkEq sourceIndexValue runtimeSourceValue)
            (← `(tactic|
              simp_all (config := { zeta := true }) [
                rectangularIndex,
                Nat.add_mod,
                Nat.mul_mod,
                Nat.mod_eq_of_lt] <;>
              omega))
        let hIndex ←
          finIndexEquality sourceSize sourceIndex runtimeSourceIndex
            hIndexValue
        let sourceValue ←
          mkAppM ``Rep.getFlat #[sourceTensor, sourceIndex]
        let expectedValue := mkApp values outputIndex
        let getFlatFunction ←
          withLocalDeclD `index (← mkAppM ``Fin #[sourceSize])
              fun index => do
            let value ← mkAppM ``Rep.getFlat #[sourceTensor, index]
            mkLambdaFVars #[index] value
        let hValue ← mkAppM ``congrArg #[getFlatFunction, hIndex]
        let hValue ←
          withTransparency .all <|
            mkExpectedTypeHint hValue (← mkEq sourceValue expectedValue)
        mkLambdaFVars #[row, column] hValue
  let hRangesType ← inferType hRanges
  let hRanges ← sealCertificate hRangesType hRanges
  let hValuesType ← inferType hValues
  let hValues ← sealCertificate hValuesType hValues
  let implementation ←
    mkAppM ``nativeTensorOfSlices #[
      literalRowCount, literalRowLength, literalZero, literalZero,
      hShapeSize, rowBound, hRowBound, sourceTensor, values,
      hRanges, hValues]
  let hImplementation ←
    mkAppM ``nativeTensorOfSlices_correct #[
      literalRowCount, literalRowLength, literalZero, literalZero,
      hShapeSize, rowBound, hRowBound, sourceTensor, values,
      hRanges, hValues]
  let compiled ←
    mkAppM ``Rep.pullFlat #[runtimeFlatMap, sourceTensor]
  let hImplementation ←
    withTransparency .all <|
      mkExpectedTypeHint hImplementation
        (← mkEq implementation compiled)
  return some (implementation, hImplementation)

/--
Compile one flat pullback to a certified native tensor fill when the output
length is a portable concrete value.
-/
def compileNativePullFlat?
    (outputShape sourceTensor flatMap : Expr)
    (nativeSourceDimensions? : Option (List Expr) := none) :
    TermElabM (Option (Expr × Expr)) := do
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let outputIndexType ← mkAppM ``Fin #[outputSize]
  let hFlatMap ←
    withLocalDeclD `outputIndex outputIndexType fun outputIndex => do
      let sourceIndex := mkApp flatMap outputIndex
      let hSourceIndex ← mkEqRefl sourceIndex
      mkLambdaFVars #[outputIndex] hSourceIndex
  let some
      (outputBound, hOutputBound, nativeValues, values, hValues,
        nativeGather?) ←
      compileNativePullFlatData? outputShape sourceTensor flatMap flatMap
        hFlatMap nativeSourceDimensions?
    | return none
  let reference ←
    mkAppM ``Rep.pullFlat #[flatMap, sourceTensor]
  let (implementation, hImplementation) ←
    buildNativePullImplementation sourceTensor outputBound hOutputBound
      nativeValues values hValues nativeGather?
  let hCompiled ← mkEqRefl reference
  let result ←
    mkAppM ``nativeTensorKernel #[
      reference, reference, implementation, hCompiled, hImplementation]
  let correctness ←
    mkAppM ``nativeTensorKernel_correct #[
      reference, reference, implementation, hCompiled, hImplementation]
  let correctness ←
    withTransparency .all <|
      mkExpectedTypeHint correctness
        (← mkEq result reference)
  return some (result, correctness)

/--
Compile a checked rearrangement or repeat, including any preceding shape-only
chain, to one certified native output loop.
-/
def compileNativeTransform?
    (checked hAxes inputFlatMap sourceTensor reference
      hSemanticReference : Expr)
    (hasPrecedingTransform : Bool)
    (concreteChecked? : Option Check.CheckedTransform := none) :
    TermElabM (Option Expr) := do
  let specializedChecked ← zetaReduce checked
  let specializedHAxes ← zetaReduce hAxes
  let value ←
    mkAppM ``Check.CheckedTransform.value #[specializedChecked]
  let outputShape ← mkAppM ``Check.TransformPlan.output #[value]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let outputIndexType ← mkAppM ``Fin #[outputSize]
  let checkedFlatMap ←
    mkAppM ``Check.CheckedTransform.inputFlatIndexOfOutput #[
      specializedChecked, specializedHAxes]
  let (compiledCheckedFlatMap, hCompiledCheckedFlatMap) ←
    checkedFlatProjection specializedChecked specializedHAxes
  let runtimeFlatMap ←
    if hasPrecedingTransform then
      explicitComposition
        outputIndexType inputFlatMap compiledCheckedFlatMap
    else
      pure compiledCheckedFlatMap
  let semanticFlatMap ←
    if hasPrecedingTransform then
      mkAppM ``Function.comp #[inputFlatMap, checkedFlatMap]
    else
      pure checkedFlatMap
  let hFlatMap ←
    withLocalDeclD `outputIndex outputIndexType fun outputIndex => do
      let hCompiledCoordinate :=
        mkApp hCompiledCheckedFlatMap outputIndex
      let hCheckedCoordinate ←
        mkAppM ``Check.CheckedTransform.inputFlatIndexOfOutput_eq #[
          specializedChecked, specializedHAxes, outputIndex]
      let hCheckedFlatMap ←
        mkAppM ``Eq.trans #[
          hCompiledCoordinate, ← mkAppM ``Eq.symm #[hCheckedCoordinate]]
      let hRuntimeSemantic ←
        if hasPrecedingTransform then
          mkAppM ``congrArg #[inputFlatMap, hCheckedFlatMap]
        else
          pure hCheckedFlatMap
      mkLambdaFVars #[outputIndex] hRuntimeSemantic
  let hRuntimeFlatMap ←
    withLocalDeclD `outputIndex outputIndexType fun outputIndex => do
      let sourceIndex := mkApp runtimeFlatMap outputIndex
      mkLambdaFVars #[outputIndex] (← mkEqRefl sourceIndex)
  let compiled ←
    mkAppM ``Rep.pullFlat #[runtimeFlatMap, sourceTensor]
  let hCompiledSemantic ←
    mkAppM ``Rep.pullFlat_congr #[
      runtimeFlatMap, semanticFlatMap, hFlatMap, sourceTensor]
  let semantic ←
    mkAppM ``Rep.pullFlat #[semanticFlatMap, sourceTensor]
  let hSemanticReference ←
    withTransparency .all <|
      mkExpectedTypeHint hSemanticReference
        (← mkEq semantic reference)
  let hCompiled ←
    mkAppM ``Eq.trans #[hCompiledSemantic, hSemanticReference]
  let specializedImplementation? ←
    if hasPrecedingTransform then
      pure none
    else
      match concreteChecked? with
      | some checkedValue =>
          match ← compileNativeTranspose2D? checkedValue outputShape
              sourceTensor runtimeFlatMap with
          | some result => pure (some result)
          | none =>
              compileNativeLeadingRepeatSlices? checkedValue outputShape
                sourceTensor runtimeFlatMap
      | none => pure none
  let (implementation, hImplementation) ←
    match specializedImplementation? with
    | some result => pure result
    | none => do
        let some
            (outputBound, hOutputBound, nativeValues, values, hValues,
              nativeGather?) ←
            compileNativePullFlatData? outputShape sourceTensor runtimeFlatMap
              runtimeFlatMap hRuntimeFlatMap
          | return none
        buildNativePullImplementation sourceTensor outputBound hOutputBound
          nativeValues values hValues nativeGather?
  let hImplementation ←
    withTransparency .all <|
      mkExpectedTypeHint hImplementation
        (← mkEq implementation compiled)
  let result ←
    mkAppM ``nativeTensorKernel #[
      reference, compiled, implementation, hCompiled, hImplementation]
  return some result

end TorchLean.Tensor.Internal.Elab.Impl
