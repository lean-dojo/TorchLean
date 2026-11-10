/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Native.Tile.Registers -- shake: keep
public meta import TorchLean.Tensor.Native.Tile.Registers
public meta import TorchLean.Tensor.Einops.Elab.Einsum.Index -- shake: keep
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Data.Rat.Cast.Order

/-!
# Scalar-register contraction lowering

A native fold whose step is the generated vector update of a tile in
`outputTiles` is lowered to a loop that carries each lane total as its own
scalar. The certificate shows that both loops produce the same vector, so the
lane count affects only the machine representation.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Replace a native fold whose body is the generated update of a tile in
`Native.outputTiles` with the scalar-state sum of that tile, which carries the lane
totals as separate scalar arguments.

The replacement applies only when the update acts on the incoming vector state
and no lane value depends on that state. The returned theorem composes the
scalar-loop equivalence with the caller's certificate for the ordinary native
fold.
-/
def scalarizeTileNativeFold?
    (length nativeBound hBound nativeState nativeCoordinate hCoordinate
      body initial ordinaryLoop hOrdinaryReference : Expr) :
    MetaM (Option (Expr × Expr)) := do
  let body := body.consumeMData
  let some (bodyArity, scalarLoopName, scalarLoopCorrectnessName) :=
      -- A tile update takes its three fixed arguments and one value per lane.
      Native.outputTiles.findSome? fun tile =>
        if body.isAppOfArity tile.update (tile.width + 3) then
          some
            (tile.width + 3, tile.sum, tile.sumEqFinFoldl)
        else
          none
    | return none
  let arguments := body.getAppArgs
  let values := arguments.extract 3 bodyArity
  unless arguments[2]!.consumeMData == nativeState &&
      !values.any fun value =>
        value.containsFVar nativeState.fvarId! do
    return none
  let mut sumArguments := #[length, nativeBound, hBound]
  for value in values do
    let valueCallback ←
      mkLambdaFVars #[nativeCoordinate, hCoordinate] value
    sumArguments := sumArguments.push valueCallback
  sumArguments := sumArguments.push initial
  let scalarLoop ← mkAppM scalarLoopName sumArguments
  let hScalarOrdinary ←
    mkAppM scalarLoopCorrectnessName sumArguments
  let hScalarOrdinary ←
    withTransparency .all <|
      mkExpectedTypeHint hScalarOrdinary
        (← mkEq scalarLoop ordinaryLoop)
  let hScalarReference ←
    mkAppM ``Eq.trans #[hScalarOrdinary, hOrdinaryReference]
  return some (scalarLoop, hScalarReference)

end TorchLean.Einops.Elab
