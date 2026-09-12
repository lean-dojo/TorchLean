/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import Aesop.BuiltinRules
meta import Mathlib.Tactic.ToAdditive
public import NN.Tensor.Internal.Elab.Einsum.Tiling.Width4
public import NN.Tensor.Internal.Elab.Einsum.Tiling.Width8
public meta import NN.Tensor.Internal.Elab.Einsum.Tiling -- shake: keep

/-!
# Proof-producing einsum output fusion

This module recognizes a scalar-register contraction immediately consumed by
an output append. It emits the fused loop together with an equality proof to
the ordinary tiled append.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Meta

/--
Fuse a scalarized tiled contraction with the output-buffer append that
immediately consumes it.

When `finalizers` is nonempty, each lane function is applied only after its
scalar accumulator is complete. Leading generated lets remain outside the
fused call. The returned certificate identifies the direct append with the
original tiled push expression.
-/
partial def fuseNativeFinSumPush
    (tileWidth : Nat) (output rawTile emittedTile : Expr)
    (finalizers : List Expr) :
    MetaM (Option (Expr × Expr)) := do
  unless finalizers.isEmpty || finalizers.length = tileWidth do
    return none
  let usesFinalizers := !finalizers.isEmpty
  let some (pushTileName, scalarLoopName, scalarLoopArity,
      fusedLoopName, fusedLoopCorrectnessName) :=
      match tileWidth, usesFinalizers with
      | 4, false =>
          some
            (``pushTile4, ``nativeFinSum4, 10,
              ``nativeFinSum4Push, ``nativeFinSum4Push_eq_pushTile4)
      | 4, true =>
          some
            (``pushTile4, ``nativeFinSum4, 10,
              ``nativeFinSum4FinalizePush,
              ``nativeFinSum4FinalizePush_eq_pushTile4)
      | 8, false =>
          some
            (``pushTile8, ``nativeFinSum8, 14,
              ``nativeFinSum8Push, ``nativeFinSum8Push_eq_pushTile8)
      | 8, true =>
          some
            (``pushTile8, ``nativeFinSum8, 14,
              ``nativeFinSum8FinalizePush,
              ``nativeFinSum8FinalizePush_eq_pushTile8)
      | _, _ => none
    | return none
  let rec
    /-- Open generated lets until the scalar tiled loop is visible. -/
    visit (remaining : Expr) : MetaM (Option (Expr × Expr)) := do
      match remaining.consumeMData with
      | .letE name type assignment body _ =>
          withLetDecl name type assignment fun localValue => do
            let openedBody := body.instantiate1 localValue
            let some (fusedBody, hFusedBody) ← visit openedBody
              | return none
            let fused ←
              mkLetFVars
                (generalizeNondepLet := false) #[localValue] fusedBody
            let hFused ←
              mkLetFVars
                (generalizeNondepLet := false) #[localValue] hFusedBody
            return some (fused, hFused)
      | scalarLoop =>
          unless scalarLoop.isAppOfArity
              scalarLoopName scalarLoopArity do
            return none
          let arguments := scalarLoop.getAppArgs
          let initial := arguments[scalarLoopArity - 1]!
          unless initial.isAppOfArity ``Vector.replicate 3 do
            return none
          let initialArguments := initial.getAppArgs
          unless (← getNatValue? initialArguments[1]!) == some tileWidth do
            return none
          let mut fusedArguments :=
            arguments.extract 2 (scalarLoopArity - 1)
          for finalizer in finalizers do
            fusedArguments := fusedArguments.push finalizer
          fusedArguments := fusedArguments.push output
          fusedArguments := fusedArguments.push initialArguments[2]!
          let fused ← mkAppM fusedLoopName fusedArguments
          let hFused ←
            mkAppM fusedLoopCorrectnessName fusedArguments
          return some (fused, hFused)
  let some (fused, hFused) ← visit rawTile
    | return none
  let ordinary ← mkAppM pushTileName #[output, emittedTile]
  let hFused ←
    withTransparency .all <|
      mkExpectedTypeHint hFused (← mkEq fused ordinary)
  return some (fused, hFused)

end TorchLean.Tensor.Internal.Elab.Impl
