/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Tiling.Width4
public import NN.Tensor.Internal.Elab.Einsum.Tiling.Width8
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public meta import NN.Tensor.Internal.Elab.Einsum.Index -- shake: keep

/-!
# Scalar-register contraction lowering

This module recognizes a generic tiled vector fold and certifies its lowering
to concrete scalar-register loops. Lane count affects machine representation,
not the contraction semantics.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Replace an exact generated tiled vector update with the corresponding native
loop whose lane totals remain separate scalar arguments.

The replacement is valid only when no lane value depends on the incoming
vector state. The returned theorem composes the scalar-loop equivalence with
the caller's certificate for the ordinary native fold.
-/
def scalarizeTileNativeFold?
    (length nativeBound hBound nativeState nativeCoordinate hCoordinate
      body initial ordinaryLoop hOrdinaryReference : Expr) :
    MetaM (Option (Expr × Expr)) := do
  let body := body.consumeMData
  let some (bodyArity, scalarLoopName, scalarLoopCorrectnessName) :=
      if body.isAppOfArity ``updateTile4 7 then
        some
          (7, ``nativeFinSum4, ``nativeFinSum4_eq_nativeFinFoldl)
      else if body.isAppOfArity ``updateTile8 11 then
        some
          (11, ``nativeFinSum8, ``nativeFinSum8_eq_nativeFinFoldl)
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

end TorchLean.Tensor.Internal.Elab.Impl
