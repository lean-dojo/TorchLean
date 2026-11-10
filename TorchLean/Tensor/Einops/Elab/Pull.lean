/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Transpose
public meta import TorchLean.Tensor.Einops.Elab.Kernel.Affine
public meta import TorchLean.Tensor.Einops.Elab.Kernel.Index
public meta import TorchLean.Tensor.Einops.Elab.Transform.Index
public import TorchLean.Tensor.Native.Construction -- shake: keep
public import TorchLean.Tensor.Einops.Lowering.TransformFusion -- shake: keep
public import TorchLean.Tensor.Einops.Elab.Transform.Index
import TorchLean.Tensor.Einops.Elab.Kernel.Affine
import TorchLean.Tensor.Einops.Elab.Kernel.Index
import TorchLean.Tensor.Einops.Elab.Index

/-!
# Certified native flat pullbacks

A certified output-to-source flat map compiles into one native output loop that
reads the source buffer, and rearrangement, repetition and other shape-only
transformations share this operation-independent lowering. Concrete rank-two
swaps and leading-axis repeats have specialized transpose and slice-copy
kernels.

The generated term carries its pointwise equality to `Rep.pullFlat`. Output
sizes that are concrete and portable use `USize` counters, and all other output
sizes use the general lowering. Source reads use native indices when the source
buffer has a concrete size below `2^32`, and bounded `Fin` reads otherwise.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open Tensor (Rep)
open Native (transpose2DIndex)

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
          mkAppM ``lt_of_eq_of_lt #[
            hOptimizedRuntimeSourceIndex, runtimeSourceIndexBound]
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
            -- Each callback becomes one compiled auxiliary definition, so downstream
            -- elaborators do not traverse large affine callbacks inside the semantic reference.
            -- A let-bound local stays a parameter: the source tensor is computed once.
            let sourceIndices ←
              sealAuxDefinition `_einops_native_callback sourceIndices
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
        let nativeValues ← sealAuxDefinition `_einops_native_callback nativeValues
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
        mkAppM ``Native.tensorGather #[
          sourceTensor, outputBound, hOutputBound,
          sourceIndices, hSourceIndices, values, hValues]
      let hImplementation ←
        mkAppM ``Native.tensorGather_correct #[
          sourceTensor, outputBound, hOutputBound,
          sourceIndices, hSourceIndices, values, hValues]
      return (implementation, hImplementation)
  | none =>
      let implementation ←
        mkAppM ``Native.tensorOfFlatFn #[
          outputBound, hOutputBound, nativeValues, values, hValues]
      let hImplementation ←
        mkAppM ``Native.tensorOfFlatFn_correct #[
          outputBound, hOutputBound, nativeValues, values, hValues]
      return (implementation, hImplementation)

/--
Return the number of contiguous source-buffer copies represented by a repeat
whose original logical axes are the exact suffix of the output axes.

For example, `column -> row column` lays out each repeated row as one complete
copy of the source buffer. Interleaved repeats retain the general gather
kernel.
-/
private def leadingExpandCount?
    (checked : Transform) : Option Nat := do
  guard (checked.value.normalized.kind = .expand)
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
    (checked : Transform) : Option (Nat × Nat) := do
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
Compile a concrete rank-two swap to `Native.transpose2D`, which runs the storage's own transpose.

We return `none`, and elaboration falls back to the generic gather, when the shapes or the
row-major-normalized index program differ from the exact swap.
-/
private def compileNativeTranspose2D?
    (checkedValue : Transform)
    (outputShape sourceTensor runtimeFlatMap : Expr) :
    TermElabM (Option (Expr × Expr)) := do
  let some (rows, columns) := transpose2DDimensions? checkedValue
    | return none
  let sourceTensorType ← whnf (← inferType sourceTensor)
  let sourceTensorType := sourceTensorType.consumeMData
  unless sourceTensorType.isAppOfArity ``Rep 3 do
    return none
  let sourceArguments := sourceTensorType.getAppArgs
  let sourceShape := sourceArguments[1]!
  let expectedSourceShape := Lean.toExpr ([rows, columns] : Shape)
  let expectedOutputShape := Lean.toExpr ([columns, rows] : Shape)
  unless ← isDefEq sourceShape expectedSourceShape do
    return none
  unless ← isDefEq outputShape expectedOutputShape do
    return none
  let literalRows := mkNatLit rows
  let literalColumns := mkNatLit columns
  let some _ ← nativeLoopBound? literalRows
    | return none
  let some _ ← nativeLoopBound? literalColumns
    | return none
  let some (bound, hBound) ← nativeLoopBound? (mkNatMul literalColumns literalRows)
    | return none
  let canonicalFlatMap ←
    mkAppM ``transpose2DIndex #[literalRows, literalColumns]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let outputIndexType ← mkAppM ``Fin #[outputSize]
  let hFlatMap? ←
    withLocalDeclD `outputIndex outputIndexType fun outputIndex => do
      let canonicalIndex := mkApp canonicalFlatMap outputIndex
      let runtimeIndex := mkApp runtimeFlatMap outputIndex
      let extraDeclarations := #[``transpose2DIndex]
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
  let implementation ←
    mkAppM ``Native.transpose2D #[
      literalRows, literalColumns, bound, hBound, sourceTensor]
  let hNative ←
    mkAppM ``Native.transpose2D_correct #[
      literalRows, literalColumns, bound, hBound, sourceTensor]
  let hPullback ←
    mkAppM ``Rep.pullFlat_congr #[
      canonicalFlatMap, runtimeFlatMap, hFlatMap, sourceTensor]
  let hImplementation ← mkAppM ``Eq.trans #[hNative, hPullback]
  return some (implementation, hImplementation)

/--
Compile a flat pullback made of `rowCount` source slices of length `rowLength`, row `r` starting
at `r * sourceStride + sourceOffset`, to `Native.tensorOfSlices`, or return `none` for an empty
geometry or a row count without a portable native loop bound.

`certifyIndex sourceSize position sourceIndex mappedIndex` proves `sourceIndex = mappedIndex`,
where `sourceIndex : Fin sourceSize` has value `position` and `mappedIndex` is `flatMap` at the
output index of the same row and column.
-/
def compileNativeSlices?
    (outputShape sourceTensor flatMap : Expr)
    (rowCount rowLength sourceStride sourceOffset : Nat)
    (certifyIndex : Expr → Expr → Expr → Expr → TermElabM Expr) :
    TermElabM (Option (Expr × Expr)) := do
  if rowCount = 0 || rowLength = 0 then
    return none
  let literalRowCount := mkNatLit rowCount
  let literalRowLength := mkNatLit rowLength
  let literalSourceStride := mkNatLit sourceStride
  let literalSourceOffset := mkNatLit sourceOffset
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let hShapeSize ←
    certifyGeneratedInvariant
      "that a native slice copy is a rectangle of source rows"
      (← mkEq
        (← mkAppM ``Nat.mul #[literalRowCount, literalRowLength])
        outputSize)
  let some (rowBound, hRowBound) ← nativeLoopBound? literalRowCount
    | return none
  let sourceTensorType ← whnf (← inferType sourceTensor)
  let sourceTensorType := sourceTensorType.consumeMData
  unless sourceTensorType.isAppOfArity ``Rep 3 do
    throwError
      "internal error: a native slice copy received a non-tensor source"
  let scalarType := sourceTensorType.getAppArgs[0]!
  let sourceShape := sourceTensorType.getAppArgs[1]!
  let storage := sourceTensorType.getAppArgs[2]!
  let sourceSize ← mkAppM ``Shape.size #[sourceShape]
  let rowType ← mkAppM ``Fin #[literalRowCount]
  let columnType ← mkAppM ``Fin #[literalRowLength]
  let hRanges ←
    withLocalDeclD `row rowType fun row => do
      let rowValue ← mkAppM ``Fin.val #[row]
      let start ←
        mkAppM ``Nat.add #[
          ← mkAppM ``Nat.mul #[rowValue, literalSourceStride],
          literalSourceOffset]
      let stop ← mkAppM ``Nat.add #[start, literalRowLength]
      let hRange ←
        certifyGeneratedInvariant
          "that a native slice copy row lies inside the source buffer"
          (← mkLE stop sourceSize)
      mkLambdaFVars #[row] hRange
  let values ←
    withLocalDeclD `outputIndex (← mkAppM ``Fin #[outputSize])
        fun outputIndex => do
      let value ←
        mkAppM ``Rep.getFlat #[
          sourceTensor, mkApp flatMap outputIndex]
      mkLambdaFVars #[outputIndex] value
  let hValues ←
    withLocalDeclD `row rowType fun row => do
      withLocalDeclD `column columnType fun column => do
        let rowValue ← mkAppM ``Fin.val #[row]
        let columnValue ← mkAppM ``Fin.val #[column]
        let sourceIndexValue ←
          mkAppM ``Nat.add #[
            ← mkAppM ``Nat.add #[
              ← mkAppM ``Nat.mul #[rowValue, literalSourceStride],
              literalSourceOffset],
            columnValue]
        let hSourceIndex ←
          certifyGeneratedInvariant
            "that a native slice copy index lies inside the source buffer"
            (← mkLT sourceIndexValue sourceSize)
        let sourceIndex ←
          mkAppOptM ``Fin.mk #[
            some sourceSize, some sourceIndexValue, some hSourceIndex]
        let rectangularIndex ←
          mkAppM ``Equiv.toFun #[
            ← mkAppOptM ``finProdFinEquiv #[
              some literalRowCount, some literalRowLength],
            ← mkAppM ``Prod.mk #[row, column]]
        let outputIndex ←
          mkAppM ``Fin.cast #[hShapeSize, rectangularIndex]
        let mappedIndex := mkApp flatMap outputIndex
        let hIndex ←
          certifyIndex sourceSize sourceIndexValue sourceIndex mappedIndex
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
  let arguments := #[
    literalRowCount, literalRowLength, literalSourceStride,
    literalSourceOffset, hShapeSize, rowBound, hRowBound, sourceTensor,
    values, hRanges, hValues]
  let implementation ← mkAppM ``Native.tensorOfSlices arguments
  let hImplementation ← mkAppM ``Native.tensorOfSlices_correct arguments
  let compiled ←
    mkAppOptM ``Rep.pullFlat #[
      some scalarType, some storage, none, none,
      some flatMap, some sourceTensor]
  let hImplementation ←
    withTransparency .all <|
      mkExpectedTypeHint hImplementation
        (← mkEq implementation compiled)
  return some (implementation, hImplementation)

/--
Compile a leading-axis repeat to `compileNativeSlices?` with one full copy of the source buffer
per repeated row, so the slice stride and offset are zero.

`simp` and `omega` prove that each slice position is the source index the checked flat map selects.
-/
private def compileExpandSlices?
    (checkedValue : Transform)
    (outputShape sourceTensor runtimeFlatMap : Expr) :
    TermElabM (Option (Expr × Expr)) := do
  let some rowCount := leadingExpandCount? checkedValue
    | return none
  let rowLength := Shape.size checkedValue.value.normalized.input
  if rowCount * rowLength != Shape.size checkedValue.value.output then
    return none
  compileNativeSlices? outputShape sourceTensor runtimeFlatMap
      rowCount rowLength 0 0
    fun sourceSize position sourceIndex mappedIndex => do
      let mappedValue ← mkAppM ``Fin.val #[mappedIndex]
      -- Rewrite the `finProdFinEquiv` value first: once `simp_all` evaluates the
      -- literal product in its type, the value lemma does not match.
      let hIndexValue ←
        certifyWithTactic
          "that leading-repeat slices implement the checked flat map"
          (← mkEq position mappedValue)
          (← `(tactic|
            simp only [Equiv.toFun_as_coe, Fin.val_cast,
              finProdFinEquiv_apply_val] <;>
            simp_all (config := { zeta := true }) [
              Nat.add_mod,
              Nat.mul_mod,
              Nat.mod_eq_of_lt] <;>
            omega))
      finIndexEquality sourceSize sourceIndex mappedIndex hIndexValue

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
    mkAppM ``Native.tensorKernel #[
      reference, reference, implementation, hCompiled, hImplementation]
  let correctness ←
    mkAppM ``Native.tensorKernel_correct #[
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
    (concreteChecked? : Option Transform := none) :
    TermElabM (Option Expr) := do
  let specializedChecked ← zetaReduce checked
  let specializedHAxes ← zetaReduce hAxes
  let value ←
    mkAppM ``Transform.value #[specializedChecked]
  let outputShape ← mkAppM ``TransformPlan.output #[value]
  let outputSize ← mkAppM ``Shape.size #[outputShape]
  let outputIndexType ← mkAppM ``Fin #[outputSize]
  let checkedFlatMap ←
    mkAppM ``Transform.inputFlatIndexOfOutput #[
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
        mkAppM ``Transform.inputFlatIndexOfOutput_eq #[
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
              compileExpandSlices? checkedValue outputShape
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
    mkAppM ``Native.tensorKernel #[
      reference, compiled, implementation, hCompiled, hImplementation]
  return some result

end TorchLean.Einops.Elab
