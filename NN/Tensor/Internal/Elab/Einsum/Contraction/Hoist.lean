/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Einsum.Contraction.Index
public import NN.Tensor.Internal.Elab.Einsum.Index
public import NN.Tensor.Internal.Elab.Einsum.Contraction.Index

/-!
# Contraction-loop invariant motion

This module floats compiler-generated index bases to the earliest contraction
loop where all referenced coordinates are available, preserving a certificate
for the optimized expression.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Float generated index-base lets to the earliest contraction loop where all of
their coordinates are available.

The pass is deliberately limited to the standard, native, and tiled fold
expressions emitted by the einsum compiler. It does not rewrite user terms.
-/
partial def hoistCoordinateLoopLets (value : Expr) : MetaM Expr := do
  let value := value.consumeMData
  let isFinFold := value.isAppOfArity ``Fin.foldl 4
  let isNativeFold := value.isAppOfArity ``nativeFinFoldl 6
  let isNativeSum4 := value.isAppOfArity ``nativeFinSum4 10
  let isNativeSum4Push := value.isAppOfArity ``nativeFinSum4Push 12
  let isNativeSum8 := value.isAppOfArity ``nativeFinSum8 14
  let isNativeSum8Push := value.isAppOfArity ``nativeFinSum8Push 16
  if isFinFold || isNativeFold ||
      isNativeSum4 || isNativeSum4Push ||
      isNativeSum8 || isNativeSum8Push then
    let arguments := value.getAppArgs
    let callbackPositions :=
      if isFinFold then
        [2]
      else if isNativeFold then
        [4]
      else if isNativeSum4 then
        (List.range 4).map (5 + ·)
      else if isNativeSum4Push then
        (List.range 4).map (6 + ·)
      else if isNativeSum8 then
        (List.range 8).map (5 + ·)
      else
        (List.range 8).map (6 + ·)
    let expectedLocals := if isNativeFold then 3 else 2
    let rec
      /-- Hoist invariants from every callback before rebuilding the loop. -/
      rewriteCallbacks
          (positions : List Nat) (arguments : Array Expr) : MetaM Expr := do
        match positions with
        | [] => pure (mkAppN value.getAppFn arguments)
        | callbackPosition :: positions =>
            lambdaTelescope arguments[callbackPosition]!
                fun foldLocals body => do
              unless foldLocals.size = expectedLocals do
                throwError
                  "internal error: a generated contraction fold has an \
                    unexpected callback shape"
              let body ← hoistCoordinateLoopLets body
              let body ←
                match body with
                | .app function argument =>
                    exposeFinalArgumentLets function argument
                | _ => pure body
              withFoldInvariantLets body foldLocals fun remainingBody => do
                withFoldInvariantNativeIndexExpressions
                    remainingBody foldLocals fun nativeBody => do
                  let foldFunction ← mkLambdaFVars foldLocals nativeBody
                  rewriteCallbacks positions <|
                    arguments.set! callbackPosition foldFunction
    rewriteCallbacks callbackPositions arguments
  else
    match value with
    | .app function argument =>
        pure (mkApp function (← hoistCoordinateLoopLets argument))
    | .letE name type assignment body _ =>
        withLetDecl name type assignment fun localValue => do
          let body ←
            hoistCoordinateLoopLets (body.instantiate1 localValue)
          mkLetFVars
            (generalizeNondepLet := false) #[localValue] body
    | _ => pure value

/--
Normalize native index conversions and float generated index bases out of
contraction loops.

The input certificate must identify the semantic reference with the generated
expression. The returned certificate has the same orientation after both
normalization passes and loop-invariant let motion.
-/
def optimizeCoordinateFoldExpression
    (generated hReferenceGenerated : Expr) :
    TermElabM (Expr × Expr) := do
  let mut nativeConversionTheorems : SimpTheorems := {}
  nativeConversionTheorems :=
    nativeConversionTheorems.addDeclToUnfoldCore ``Nat.toUSize
  let nativeConversionContext ←
    Simp.mkContext
      (config := {
        iota := false
        zeta := false
        failIfUnchanged := false
      })
      (simpTheorems := #[nativeConversionTheorems])
      (congrTheorems := ← getSimpCongrTheorems)
  let simplifyNativeConversions (expression : Expr) :
      TermElabM (Expr × Expr) := do
    unless hasNativeIndexRedex expression do
      return (expression, ← mkEqRefl expression)
    simplifyNativeIndexExpressions expression nativeConversionContext
  let (reduced, hGeneratedReduced) ←
    simplifyNativeConversions generated
  let hoisted ← hoistCoordinateLoopLets reduced
  let hGeneratedHoisted ←
    withTransparency .all <|
      mkExpectedTypeHint hGeneratedReduced
        (← mkEq generated hoisted)
  -- Hoisting can join a native conversion round trip that was previously
  -- split across nested lets.
  let (optimized, hHoistedOptimized) ←
    simplifyNativeConversions hoisted
  let hGeneratedOptimized ←
    mkAppM ``Eq.trans #[hGeneratedHoisted, hHoistedOptimized]
  let hReferenceOptimized ←
    mkAppM ``Eq.trans #[hReferenceGenerated, hGeneratedOptimized]
  return (optimized, hReferenceOptimized)

end TorchLean.Tensor.Internal.Elab.Impl
