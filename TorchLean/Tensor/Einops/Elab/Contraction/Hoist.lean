/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Einops.Elab.Contraction.Index
public meta import TorchLean.Tensor.Native.Tile.Registers
public import TorchLean.Tensor.Einops.Elab.Einsum.Index
public import TorchLean.Tensor.Einops.Elab.Contraction.Index

/-!
# Contraction-loop invariant motion

Loop-invariant motion floats compiler-generated index bases to the earliest
contraction loop where all of their coordinates are available, and carries the
certificate of the generated expression over to the optimized one.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Float generated index-base lets to the earliest contraction loop where all of
their coordinates are available.

The pass is limited on purpose to the standard, native and tiled folds that the
einsum compiler emits, so it never rewrites user terms.
-/
partial def hoistLoopLets (value : Expr) : MetaM Expr := do
  let value := value.consumeMData
  let isFinFold := value.isAppOfArity ``Fin.foldl 4
  let isNativeFold := value.isAppOfArity ``Native.finFoldl 6
  -- A scalar-state sum has five arguments before its lane callbacks and one after them.
  -- The appending sum adds a storage instance before the callbacks and a buffer after them.
  let tileCallbacks? : Option (List Nat) := Id.run do
    for tile in Native.outputTiles do
      if value.isAppOfArity tile.sum (tile.width + 6) then
        return some ((List.range tile.width).map (5 + ·))
      if value.isAppOfArity tile.sumPush (tile.width + 8) then
        return some ((List.range tile.width).map (6 + ·))
    return none
  if isFinFold || isNativeFold || tileCallbacks?.isSome then
    let arguments := value.getAppArgs
    let callbackPositions :=
      if isFinFold then
        [2]
      else if isNativeFold then
        [4]
      else
        tileCallbacks?.getD []
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
              let body ← hoistLoopLets body
              let body ←
                match body with
                | .app function argument =>
                    exposeFinalArgumentLets function argument
                | _ => pure body
              withFoldInvariantLets body foldLocals fun remainingBody => do
                withFoldInvariantIndices
                    remainingBody foldLocals fun nativeBody => do
                  let foldFunction ← mkLambdaFVars foldLocals nativeBody
                  rewriteCallbacks positions <|
                    arguments.set! callbackPosition foldFunction
    rewriteCallbacks callbackPositions arguments
  else
    match value with
    | .app function argument =>
        pure (mkApp function (← hoistLoopLets argument))
    | .letE name type assignment body _ =>
        withLetDecl name type assignment fun localValue => do
          let body ←
            hoistLoopLets (body.instantiate1 localValue)
          mkLetFVars
            (generalizeNondepLet := false) #[localValue] body
    | _ => pure value

/--
Normalize native index conversions and float generated index bases out of
contraction loops.

The input certificate must identify the semantic reference with the generated
expression, and the returned certificate keeps that orientation across both
normalization passes and the loop-invariant let motion between them.
-/
def optimizeCoordinateFold
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
  let hoisted ← hoistLoopLets reduced
  let hGeneratedHoisted ←
    withTransparency .all <|
      mkExpectedTypeHint hGeneratedReduced
        (← mkEq generated hoisted)
  -- A native conversion round trip split across nested lets can become one
  -- redex after hoisting, so the conversions are normalized a second time.
  let (optimized, hHoistedOptimized) ←
    simplifyNativeConversions hoisted
  let hGeneratedOptimized ←
    mkAppM ``Eq.trans #[hGeneratedHoisted, hHoistedOptimized]
  let hReferenceOptimized ←
    mkAppM ``Eq.trans #[hReferenceGenerated, hGeneratedOptimized]
  return (optimized, hReferenceOptimized)

end TorchLean.Einops.Elab
