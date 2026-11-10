/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Einops.Elab.Contraction.Scalarize
public meta import TorchLean.Tensor.Native.Tile.Registers
public import TorchLean.Tensor.Einops.Elab.Einsum.Index
public import TorchLean.Tensor.Einops.Elab
public import TorchLean.Tensor.Einops.Elab.Contraction.Scalarize

/-!
# Flat contraction lowering

Flat lowering decodes a native row-major counter into certified tensor
coordinates and runs a large tiled contraction over several axes as one native
loop.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta
open TorchLean.Shape (Coord)

/--
Decode a concrete native flat counter into row-major coordinate components.

Each component uses native quotient or remainder arithmetic. The accompanying
equalities identify those executable components with the nested `Fin.divNat`
and `Fin.modNat` coordinates used by `Coord.unlinearize`.
-/
def nativeCoordinateComponents
    (lengths : List Expr) (flatIndex hFlatIndex : Expr) :
    TermElabM (List Expr × List Expr × List Expr) := do
  let flatSize ← mkAppM ``Shape.size #[← shapeExpr lengths]
  let flatIndexNat ← mkAppM ``USize.toNat #[flatIndex]
  let nativeFlatIndex ←
    mkAppOptM ``Fin.mk #[
      some flatSize, some flatIndexNat, some hFlatIndex]
  let rec
    /-- Decode one product coordinate and continue with its row remainder. -/
    visit
      (remainingLengths : List Expr)
      (nativeIndex hNativeIndex semanticIndex hNativeSemantic : Expr) :
      TermElabM (List Expr × List Expr × List Expr) := do
    match remainingLengths with
    | [] => return ([], [], [])
    | [length] => do
        let nativeHeadValue ←
          mkAppM ``USize.toNat #[nativeIndex]
        let nativeHead ←
          mkAppOptM ``Fin.mk #[
            some length, some nativeHeadValue, some hNativeIndex]
        let tailSize := mkNatLit 1
        let some (tailBound, hTailBound) ← nativeLoopBound? tailSize
          | throwError
              "internal error: unit row-major tail is not portable"
        let nativeQuotient ←
          mkAppM ``HDiv.hDiv #[nativeIndex, tailBound]
        let nativeQuotientValue ←
          mkAppM ``USize.toNat #[nativeQuotient]
        let hNativeQuotient ←
          mkAppM ``Native.toNat_div_lt_of_lt_mul #[
            length, tailSize, nativeIndex, tailBound,
            hTailBound, hNativeIndex]
        let quotientHead ←
          mkAppOptM ``Fin.mk #[
            some length, some nativeQuotientValue, some hNativeQuotient]
        let semanticHead ←
          mkAppOptM ``Fin.divNat #[
            some length, some tailSize, some semanticIndex]
        let hNativeHeadDiv ←
          mkAppM ``Native.fin_div_eq_divNat #[
            length, tailSize, nativeIndex, tailBound,
            hTailBound, hNativeIndex]
        let currentIndexType ← inferType semanticIndex
        let divideCurrent ←
          withLocalDeclD `flatCoordinate currentIndexType
              fun currentIndex => do
            let quotient ←
              mkAppOptM ``Fin.divNat #[
                some length, some tailSize, some currentIndex]
            mkLambdaFVars #[currentIndex] quotient
        let hSemanticHead ←
          mkAppM ``congrArg #[divideCurrent, hNativeSemantic]
        let hDivideOne ←
          mkAppOptM ``USize.div_one #[some nativeIndex]
        let hDivideOne ←
          withTransparency .all <|
            mkExpectedTypeHint hDivideOne
              (← mkEq nativeQuotient nativeIndex)
        let hDirectValue ←
          mkAppM ``congrArg #[
            mkConst ``USize.toNat, ← mkAppM ``Eq.symm #[hDivideOne]]
        let hDirectQuotient ←
          mkAppOptM ``Fin.ext #[
            some length, some nativeHead,
            some quotientHead, some hDirectValue]
        let hHead ←
          mkAppM ``Eq.trans #[
            hDirectQuotient,
            ← mkAppM ``Eq.trans #[hNativeHeadDiv, hSemanticHead]]
        let hHead ←
          withTransparency .all <|
            mkExpectedTypeHint hHead
              (← mkEq nativeHead semanticHead)
        return ([nativeHead], [semanticHead], [hHead])
    | length :: tailLengths => do
        let tailSize ← mkAppM ``Shape.size #[← shapeExpr tailLengths]
        let some (tailBound, hTailBound) ← nativeLoopBound? tailSize
          | throwError
              "internal error: a portable flattened contraction has a \
                nonportable row-major tail"
        let nativeQuotient ←
          mkAppM ``HDiv.hDiv #[nativeIndex, tailBound]
        let nativeHeadValue ←
          mkAppM ``USize.toNat #[nativeQuotient]
        let hNativeHead ←
          mkAppM ``Native.toNat_div_lt_of_lt_mul #[
            length, tailSize, nativeIndex, tailBound,
            hTailBound, hNativeIndex]
        let nativeHead ←
          mkAppOptM ``Fin.mk #[
            some length, some nativeHeadValue, some hNativeHead]
        let semanticHead ←
          mkAppOptM ``Fin.divNat #[
            some length, some tailSize, some semanticIndex]
        let hNativeHeadDiv ←
          mkAppM ``Native.fin_div_eq_divNat #[
            length, tailSize, nativeIndex, tailBound,
            hTailBound, hNativeIndex]
        let currentIndexType ← inferType semanticIndex
        let divideCurrent ←
          withLocalDeclD `flatCoordinate currentIndexType
              fun currentIndex => do
            let quotient ←
              mkAppOptM ``Fin.divNat #[
                some length, some tailSize, some currentIndex]
            mkLambdaFVars #[currentIndex] quotient
        let hSemanticHead ←
          mkAppM ``congrArg #[divideCurrent, hNativeSemantic]
        let hHead ←
          mkAppM ``Eq.trans #[hNativeHeadDiv, hSemanticHead]
        let hHead ←
          withTransparency .all <|
            mkExpectedTypeHint hHead
              (← mkEq nativeHead semanticHead)
        let hTailPositive ←
          mkDecideProof (← mkLT (mkNatLit 0) tailSize)
        let nativeRemainder ←
          mkAppM ``HMod.hMod #[nativeIndex, tailBound]
        let nativeTailValue ←
          mkAppM ``USize.toNat #[nativeRemainder]
        let hNativeTail ←
          mkAppM ``Native.toNat_mod_lt_of_pos #[
            tailSize, nativeIndex, tailBound,
            hTailBound, hTailPositive]
        let nativeTailIndex ←
          mkAppOptM ``Fin.mk #[
            some tailSize, some nativeTailValue, some hNativeTail]
        let semanticTailIndex ←
          mkAppOptM ``Fin.modNat #[
            some length, some tailSize, some semanticIndex]
        let hNativeTailMod ←
          mkAppM ``Native.fin_mod_eq_modNat #[
            length, tailSize, nativeIndex, tailBound,
            hTailBound, hNativeIndex, hTailPositive]
        let reduceCurrent ←
          withLocalDeclD `flatCoordinate currentIndexType
              fun currentIndex => do
            let remainder ←
              mkAppOptM ``Fin.modNat #[
                some length, some tailSize, some currentIndex]
            mkLambdaFVars #[currentIndex] remainder
        let hSemanticTail ←
          mkAppM ``congrArg #[reduceCurrent, hNativeSemantic]
        let hTail ←
          mkAppM ``Eq.trans #[hNativeTailMod, hSemanticTail]
        let hTail ←
          withTransparency .all <|
            mkExpectedTypeHint hTail
              (← mkEq nativeTailIndex semanticTailIndex)
        let (nativeTail, semanticTail, hTailComponents) ←
          visit tailLengths nativeRemainder hNativeTail
            semanticTailIndex hTail
        return (nativeHead :: nativeTail,
          semanticHead :: semanticTail, hHead :: hTailComponents)
  visit lengths flatIndex hFlatIndex nativeFlatIndex
    (← mkEqRefl nativeFlatIndex)

/--
Compile a large concrete multi-axis contraction as one native flat loop.

The executable callback decodes its row-major counter with native quotient
and remainder operations. Its certificate identifies those coordinates with
`Coord.unlinearize`, then transports the native fold to `Coord.foldl`.
The optimization applies only when the callback is the update of a tile in
`outputTiles` and the contraction has at least its `minimumFlatTerms` terms,
so every scalar accumulator stays live across the complete contraction.
-/
def compileFlatCoordinateFold?
    (lengths : List Expr) (step initial reference : Expr) :
    TermElabM (Option (Expr × Expr)) := do
  let some concreteLengths ← concreteNatExpressions? lengths
    | return none
  if concreteLengths.length < 2 ||
      concreteLengths.any (· == 0) then
    return none
  let stateType ←
    withTransparency .reducible <| whnf (← inferType initial)
  unless stateType.isAppOfArity ``Vector 2 do
    return none
  let some tileWidth ← getNatValue? stateType.getAppArgs[1]!
    | return none
  let some tile := Native.outputTile? tileWidth
    | return none
  if Shape.size concreteLengths < tile.minimumFlatTerms then
    return none
  let shape ← shapeExpr lengths
  let flatSize ← mkAppM ``Shape.size #[shape]
  let some (nativeBound, hBound) ← nativeLoopBound? flatSize
    | return none
  let stateType ← inferType initial
  let flatIndexType ← mkAppM ``Fin #[flatSize]
  let coordinateType ← mkAppM ``Coord #[shape]
  let flatStep ←
    withLocalDeclD `state stateType fun state =>
      withLocalDeclD `flatCoordinate flatIndexType
          fun flatCoordinate => do
        let coordinate ←
          mkAppOptM ``Coord.unlinearize #[
            some shape, some flatCoordinate]
        mkLambdaFVars #[state, flatCoordinate] <|
          step.beta #[state, coordinate]
  let flatFinLoop ←
    mkAppM ``Fin.foldl #[flatSize, flatStep, initial]
  let scalarized? ←
    withLocalDeclD `state stateType fun nativeState =>
      withLocalDeclD `nativeFlatCoordinate (mkConst ``USize)
          fun nativeFlatCoordinate => do
        let nativeFlatCoordinateNat ←
          mkAppM ``USize.toNat #[nativeFlatCoordinate]
        let nativeFlatCoordinateBound ←
          mkLT nativeFlatCoordinateNat flatSize
        withLocalDeclD `hCoordinate nativeFlatCoordinateBound
            fun hFlatCoordinate => do
          let flatCoordinate ←
            mkAppOptM ``Fin.mk #[
              some flatSize, some nativeFlatCoordinateNat,
              some hFlatCoordinate]
          let (nativeComponents, semanticComponents,
              hComponents) ←
            nativeCoordinateComponents lengths
              nativeFlatCoordinate hFlatCoordinate
          let nativeCoordinate ←
            coordinateFromComponents nativeComponents
          let semanticCoordinate ←
            coordinateFromComponents semanticComponents
          let hNativeSemantic ←
            coordinateFromComponentsEq
              nativeComponents semanticComponents hComponents
          let unlinearizedCoordinate ←
            mkAppOptM ``Coord.unlinearize #[
              some shape, some flatCoordinate]
          let hSemanticUnlinearized ←
            withTransparency .all <|
              mkExpectedTypeHint
                (← mkEqRefl semanticCoordinate)
                (← mkEq semanticCoordinate unlinearizedCoordinate)
          let hNativeUnlinearized ←
            mkAppM ``Eq.trans #[
              hNativeSemantic, hSemanticUnlinearized]
          let nativeBody :=
            step.beta #[nativeState, nativeCoordinate]
          let semanticBody :=
            step.beta #[nativeState, unlinearizedCoordinate]
          let applyStep ←
            withLocalDeclD `coordinate coordinateType fun coordinate => do
              mkLambdaFVars #[coordinate] <|
                step.beta #[nativeState, coordinate]
          let hBody ←
            mkAppM ``congrArg #[applyStep, hNativeUnlinearized]
          let hBody ←
            withTransparency .all <|
              mkExpectedTypeHint hBody
                (← mkEq nativeBody semanticBody)
          let callbackLocals :=
            #[nativeState, nativeFlatCoordinate, hFlatCoordinate]
          let nativeCallback ←
            mkLambdaFVars callbackLocals nativeBody
          let semanticCallback ←
            mkLambdaFVars callbackLocals semanticBody
          let mut hNativeCallback := hBody
          for localValue in callbackLocals.reverse do
            let hFunction ←
              mkLambdaFVars #[localValue] hNativeCallback
            hNativeCallback ← mkAppM ``funext #[hFunction]
          let nativeLoop ←
            mkAppM ``Native.finFoldl #[
              flatSize, nativeBound, hBound,
              nativeCallback, initial]
          let semanticLoop ←
            mkAppM ``Native.finFoldl #[
              flatSize, nativeBound, hBound,
              semanticCallback, initial]
          let hNativeSemanticLoop ←
            withLocalDeclD `callback (← inferType nativeCallback)
                fun callback => do
              let loop ←
                mkAppM ``Native.finFoldl #[
                  flatSize, nativeBound, hBound,
                  callback, initial]
              let preserveCallback ←
                mkLambdaFVars #[callback] loop
              mkAppM ``congrArg #[
                preserveCallback, hNativeCallback]
          let hSemanticFin ←
            mkAppM ``Native.finFoldl_eq_fin_foldl #[
              flatSize, nativeBound, hBound, flatStep, initial]
          let hSemanticFin ←
            withTransparency .all <|
              mkExpectedTypeHint hSemanticFin
                (← mkEq semanticLoop flatFinLoop)
          let hNativeFin ←
            mkAppM ``Eq.trans #[
              hNativeSemanticLoop, hSemanticFin]
          scalarizeTileNativeFold?
            flatSize nativeBound hBound nativeState
            nativeFlatCoordinate hFlatCoordinate nativeBody initial
            nativeLoop hNativeFin
  let some (scalarLoop, hScalarFin) := scalarized?
    | return none
  let hReferenceFin ←
    mkAppM ``Coord.foldl_eq_fin_foldl #[
      shape, step, initial]
  let hReferenceFin ←
    withTransparency .all <|
      mkExpectedTypeHint hReferenceFin
        (← mkEq reference flatFinLoop)
  let hReferenceScalar ←
    mkAppM ``Eq.trans #[
      hReferenceFin, ← mkAppM ``Eq.symm #[hScalarFin]]
  return some (scalarLoop, hReferenceScalar)

end TorchLean.Einops.Elab
