/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum.Contraction.Flat
public meta import NN.Tensor.Internal.Elab.Einsum.Contraction.Hoist
public import NN.Tensor.Internal.Elab.Einsum.Contraction.Flat
public import NN.Tensor.Internal.Elab.Einsum.Contraction.Hoist

/-!
# Verified contraction-loop generation

The contraction compiler is organized as certified native-index
normalization, loop-invariant motion, scalar-register lowering, flat
coordinate traversal, and arbitrary-rank coordinate folds; this module is the
last of those stages and is what importers of the contraction compiler name.

This module compiles arbitrary-rank coordinate folds and sums to certified
native loops. Concrete, symbolic, flattened, and compile-time-unrolled axes all
share the same finite-fold correctness boundary.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Compile a statically shaped coordinate-state fold into verified native loops.

The loop nest is constructed directly from the reflected dimension list. Tiny
literal axes are expanded within a fixed code-size budget. Other concrete
lengths that fit every Lean target use unboxed `USize` counters; symbolic or
oversized lengths retain `Fin.foldl`. Every optimized loop carries a proof
that it equals the corresponding standard fold. Simplification then cancels
coordinate wrappers and a final pass floats staged index bases out of inner
loops.
-/
def compileCoordinateFold
    (lengths : List Expr) (step initial reference : Expr) :
    TermElabM (Expr × Expr) := do
  if ← hasConcreteZeroDimension lengths then
    let shape ← shapeExpr lengths
    let hSize ← certifyWithTactic
      "that a contraction containing a zero axis has no coordinates"
      (← mkEq (← mkAppM ``Shape.size #[shape]) (mkNatLit 0))
      (← `(tactic| simp [Shape.size]))
    let hEmpty ← mkAppM ``coordinateFoldl_eq_initial_of_size_eq_zero #[
      shape, step, initial, hSize]
    let hReferenceInitial ← withTransparency .all <|
      mkExpectedTypeHint hEmpty (← mkEq reference initial)
    return (initial, hReferenceInitial)
  if let some (flattened, hReferenceFlattened) ←
      compileFlatCoordinateFold? lengths step initial reference then
    return ←
      optimizeCoordinateFoldExpression flattened hReferenceFlattened
  let stateType ← inferType initial
  let maxUnrolledAxisLength := 8
  let totalUnrollBudget := 16
  let rec
    /-- Build nested executable folds with their generic reference and equality proof. -/
    buildLoops
      (remainingLengths : List Expr)
      (coordinates : List Expr)
      (accumulator : Expr)
      (unrollBudget : Nat) : TermElabM (Expr × Expr × Expr) := do
    match remainingLengths with
    | [] => do
        let coordinate ← coordinateFromComponents coordinates
        let body := step.beta #[accumulator, coordinate]
        return (body, body, ← mkEqRefl body)
    | length :: remainingLengths => do
        let unrolledLength? ←
          match ← getNatValue? length with
          | some lengthValue =>
              if !length.hasFVar &&
                  lengthValue ≤ maxUnrolledAxisLength &&
                  (lengthValue = 0 || lengthValue ≤ unrollBudget) then
                if ← withTransparency .all <|
                    isDefEq length (mkNatLit lengthValue) then
                  pure (some lengthValue)
                else
                  pure none
              else
                pure none
          | _ => pure none
        if unrolledLength? == some 0 then
          return (accumulator, accumulator, ← mkEqRefl accumulator)
        let childUnrollBudget :=
          match unrolledLength? with
          | some lengthValue => unrollBudget / lengthValue
          | none => unrollBudget
        let coordinateType ← mkAppM ``Fin #[length]
        withLocalDeclD `state stateType fun state =>
          withLocalDeclD
              (Name.mkSimple s!"contractedAxis{coordinates.length}")
              coordinateType fun coordinate => do
            let (nativeBody, referenceBody, hBody) ←
              buildLoops remainingLengths
                (coordinates.concat coordinate) state childUnrollBudget
            let nativeStep ←
              mkLambdaFVars #[state, coordinate] nativeBody
            let referenceStep ←
              mkLambdaFVars #[state, coordinate] referenceBody
            let hCoordinateFunction ←
              mkLambdaFVars #[coordinate] hBody
            let hForState ← mkAppM ``funext #[hCoordinateFunction]
            let hStateFunction ←
              mkLambdaFVars #[state] hForState
            let hStep ← mkAppM ``funext #[hStateFunction]
            let nativeFinLoop ←
              mkAppM ``Fin.foldl #[length, nativeStep, accumulator]
            let referenceLoop ←
              mkAppM ``Fin.foldl #[length, referenceStep, accumulator]
            let hFinLoops ←
              withLocalDeclD `step (← inferType nativeStep) fun step => do
                let fold ←
                  mkAppM ``Fin.foldl #[length, step, accumulator]
                let preserveStep ← mkLambdaFVars #[step] fold
                mkAppM ``congrArg #[preserveStep, hStep]
            if let some unrolledLength := unrolledLength? then
              let mut unrolled := accumulator
              for index in [:unrolledLength] do
                let coordinateLiteral := mkNatLit index
                let hCoordinate ←
                  mkDecideProof (← mkLT coordinateLiteral length)
                let coordinate ←
                  mkAppOptM ``Fin.mk #[
                    some length, some coordinateLiteral, some hCoordinate]
                unrolled := nativeStep.beta #[unrolled, coordinate]
              let hUnrolledFin ←
                certifyWithTactic
                  "that a compile-time-expanded contraction fold equals \
                    its finite-fold semantics"
                  (← mkEq unrolled nativeFinLoop)
                  (← `(tactic|
                    simp only [Fin.foldl_succ, Fin.foldl_zero] <;>
                    congr))
              let hUnrolledReference ←
                mkAppM ``Eq.trans #[hUnrolledFin, hFinLoops]
              return (unrolled, referenceLoop, hUnrolledReference)
            let some (nativeBound, hBound) ← nativeLoopBound? length
              | return (nativeFinLoop, referenceLoop, hFinLoops)
            let (nativeLoop, hNativeFin) ←
              withLocalDeclD `state stateType fun nativeState =>
                withLocalDeclD
                    (Name.mkSimple
                      s!"nativeContractedAxis{coordinates.length}")
                    (mkConst ``USize) fun nativeCoordinate => do
                  let nativeCoordinateNat ←
                    mkAppM ``USize.toNat #[nativeCoordinate]
                  let nativeCoordinateBound ←
                    mkLT nativeCoordinateNat length
                  withLocalDeclD `hCoordinate nativeCoordinateBound
                      fun hCoordinate => do
                    let semanticCoordinate ←
                      mkAppOptM ``Fin.mk #[
                        some length, some nativeCoordinateNat,
                        some hCoordinate]
                    let body :=
                      nativeStep.beta #[
                        nativeState, semanticCoordinate]
                    let callbackLocals :=
                      #[nativeState, nativeCoordinate, hCoordinate]
                    let nativeCallback ←
                      mkLambdaFVars callbackLocals body
                    let ordinaryLoop ←
                      mkAppM ``nativeFinFoldl #[
                        length, nativeBound, hBound,
                        nativeCallback, accumulator]
                    let hOrdinaryFin ←
                      mkAppM ``nativeFinFoldl_eq_fin_foldl #[
                        length, nativeBound, hBound,
                        nativeStep, accumulator]
                    let hOrdinaryFin ←
                      withTransparency .all <|
                        mkExpectedTypeHint hOrdinaryFin
                          (← mkEq ordinaryLoop nativeFinLoop)
                    let some scalarized ←
                        scalarizeTileNativeFold?
                          length nativeBound hBound nativeState
                          nativeCoordinate hCoordinate body accumulator
                          ordinaryLoop hOrdinaryFin
                      | return (ordinaryLoop, hOrdinaryFin)
                    return scalarized
            let hNativeReference ←
              mkAppM ``Eq.trans #[hNativeFin, hFinLoops]
            return (nativeLoop, referenceLoop, hNativeReference)
  let (nativeLoops, referenceLoops, hNativeReference) ←
    buildLoops lengths [] initial totalUnrollBudget
  let hReferenceLoops ←
    withTransparency .all <|
      mkExpectedTypeHint
        (← mkEqRefl reference)
        (← mkEq reference referenceLoops)
  let hReferenceNative ←
    mkAppM ``Eq.trans #[
      hReferenceLoops, ← mkAppM ``Eq.symm #[hNativeReference]]
  optimizeCoordinateFoldExpression nativeLoops hReferenceNative

/--
Compile a statically shaped coordinate sum into verified native loops.

This wrapper instantiates the general state-fold compiler with scalar
addition, then transports its certificate to `Semantics.coordinateSum`.
-/
def compileCoordinateSum
    (lengths : List Expr) (values initial reference : Expr) :
    TermElabM (Expr × Expr) := do
  let scalarType ← inferType initial
  let shape ← shapeExpr lengths
  let coordinateType ← mkAppM ``Coord #[shape]
  withLocalDeclD `total scalarType fun total =>
    withLocalDeclD `coordinate coordinateType fun coordinate => do
      let body ← mkAdd total (values.beta #[coordinate])
      let step ← mkLambdaFVars #[total, coordinate] body
      let foldReference ←
        mkAppM ``coordinateFoldl #[shape, step, initial]
      let (optimized, hFoldOptimized) ←
        compileCoordinateFold lengths step initial foldReference
      let rawSum ←
        mkAppM ``Semantics.coordinateSum #[shape, values, initial]
      let hFoldRaw ←
        mkAppM ``coordinateFoldl_add_eq_coordinateSum #[
          shape, values, initial]
      let hRawFold ← mkAppM ``Eq.symm #[hFoldRaw]
      let hReferenceRaw ←
        withTransparency .all <|
          mkExpectedTypeHint (← mkEqRefl reference)
            (← mkEq reference rawSum)
      let hReferenceFold ←
        mkAppM ``Eq.trans #[hReferenceRaw, hRawFold]
      let hReferenceOptimized ←
        mkAppM ``Eq.trans #[hReferenceFold, hFoldOptimized]
      return (optimized, hReferenceOptimized)

end TorchLean.Tensor.Internal.Elab.Impl
