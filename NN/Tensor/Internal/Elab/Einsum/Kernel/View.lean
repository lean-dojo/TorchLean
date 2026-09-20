/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Affine
public meta import NN.Tensor.Internal.Elab.Einsum.Kernel.Index
public import NN.Tensor.Internal.Elab.Einsum.Kernel.Affine
public import NN.Tensor.Internal.Elab.Einsum.Kernel.Index

/-!
# Certified einsum operand views

This module composes an optimized einsum operand index with a certified flat
input view. Generated code reads the original source tensor directly, while
the returned equality still targets the logical operand stored in the
independent einsum semantics.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Compile one logical operand read, optionally through a certified source view.

The optional tuple contains the source tensor, its logical-input-to-source
flat map, and the pointwise theorem that source reads equal reads from the
logical operand.
-/
def compileInputViewRead
    (logicalTensor logicalSize inputIndexValue normalizedInputIndexValue
      certifiedInputIndex hInputIndexValue : Expr)
    (useNativeLogicalIndex : Bool)
    (view? : Option (Expr × Expr × Expr)) :
    TermElabM (Expr × Expr) := do
  let none := view?
    | let (sourceTensor, inputFlatMap, hRead) := view?.get!
      let certifiedInputIndexBound ←
        mkAppM ``Fin.isLt #[certifiedInputIndex]
      let inputIndexBound ←
        indexBoundFromValueEquality logicalSize hInputIndexValue
          certifiedInputIndexBound
      let inputIndex ←
        mkAppOptM ``Fin.mk #[
          some logicalSize, some inputIndexValue, some inputIndexBound]
      let hInputIndex ←
        finIndexEquality logicalSize inputIndex certifiedInputIndex
          hInputIndexValue
      let sourceIndex := mkApp inputFlatMap inputIndex
      let certifiedSourceIndex := mkApp inputFlatMap certifiedInputIndex
      let hSourceIndex ←
        mkAppM ``congrArg #[inputFlatMap, hInputIndex]
      let sourceIndexValue ← mkAppM ``Fin.val #[sourceIndex]
      let normalizedSourceIndexValue ← zetaReduce sourceIndexValue
      let (optimizedSourceIndexValue, hSourceIndexOptimized) ←
        simplifyAffineIndex normalizedSourceIndexValue
      let certifiedSourceIndexValue ←
        mkAppM ``Fin.val #[certifiedSourceIndex]
      let sourceValueFunction ←
        withLocalDeclD `sourceIndex (← inferType sourceIndex)
            fun sourceIndex => do
          let value ← mkAppM ``Fin.val #[sourceIndex]
          mkLambdaFVars #[sourceIndex] value
      let hSourceIndexValue ←
        mkAppM ``congrArg #[sourceValueFunction, hSourceIndex]
      let hSourceIndexValue ←
        withTransparency .all <|
          mkExpectedTypeHint hSourceIndexValue
            (← mkEq sourceIndexValue certifiedSourceIndexValue)
      let hOptimizedSourceIndexValue ←
        mkAppM ``Eq.trans #[
          ← mkAppM ``Eq.symm #[hSourceIndexOptimized],
          hSourceIndexValue]
      let hOptimizedSourceIndexValue ←
        withTransparency .all <|
          mkExpectedTypeHint hOptimizedSourceIndexValue
            (← mkEq optimizedSourceIndexValue certifiedSourceIndexValue)
      let sourceTensorType ← inferType sourceTensor
      let sourceTensorType := sourceTensorType.consumeMData
      unless sourceTensorType.isAppOfArity ``Rep 3 do
        throwError
          "internal error: a fused einsum input view has a non-tensor source"
      let sourceShape := sourceTensorType.getAppArgs[1]!
      let sourceSize ← mkAppM ``Shape.size #[sourceShape]
      let sourceDimensions? ← staticListElements? sourceShape
      let useNativeSourceIndex ←
        match sourceDimensions? with
        | some sourceDimensions =>
            supportsNativeInputIndex sourceDimensions
        | none => pure false
      let (sourceValue, hSourceValue) ←
        compileInputRead sourceTensor sourceSize optimizedSourceIndexValue
          optimizedSourceIndexValue certifiedSourceIndex
          hOptimizedSourceIndexValue useNativeSourceIndex
      let hLogicalValue := mkApp hRead certifiedInputIndex
      let hValue ← mkAppM ``Eq.trans #[hSourceValue, hLogicalValue]
      let logicalValue ←
        mkAppM ``Rep.getFlat #[logicalTensor, certifiedInputIndex]
      let hValue ←
        withTransparency .all <|
          mkExpectedTypeHint hValue (← mkEq sourceValue logicalValue)
      return (sourceValue, hValue)
  compileInputRead logicalTensor logicalSize inputIndexValue
    normalizedInputIndexValue certifiedInputIndex hInputIndexValue
    useNativeLogicalIndex

end TorchLean.Tensor.Internal.Elab.Impl
