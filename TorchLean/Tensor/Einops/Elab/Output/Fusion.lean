/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Tile.Registers -- shake: keep
public meta import TorchLean.Tensor.Native.Tile.Registers

/-!
# Proof-producing einsum output fusion

Output fusion recognizes a scalar-register contraction that an output append
consumes immediately, and emits the fused loop together with an equality proof
against the ordinary tiled append.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Meta

/--
Fuse a scalarized tiled contraction with the output-buffer append that
immediately consumes it.

The scalar loop must start from a `Vector.replicate` of the tile width, and
`finalizers` is either empty or holds one function per lane, applied only after
that lane's scalar accumulator is complete. Leading generated lets stay outside
the fused call, and the returned certificate identifies the fused append with
the original tiled push expression.
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
      -- A scalar-state sum takes its six fixed arguments and one callback per lane.
      match Native.outputTile? tileWidth, usesFinalizers with
      | some tile, false =>
          some
            (tile.push, tile.sum, tile.width + 6,
              tile.sumPush, tile.sumPushEqPush)
      | some tile, true =>
          some
            (tile.push, tile.sum, tile.width + 6,
              tile.sumFinalizePush,
              tile.sumFinalizePushEqPush)
      | none, _ => none
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

end TorchLean.Einops.Elab
