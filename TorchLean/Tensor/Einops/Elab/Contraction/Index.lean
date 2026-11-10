/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Einops.Elab.Einsum.Index -- shake: keep
public import TorchLean.Tensor.Native.Tile.Registers -- shake: keep
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Data.Rat.Cast.Order

/-!
# Certified native-index normalization

Native-index normalization recognizes the native coordinate arithmetic emitted
by the einsum compiler and replaces conversion and quotient-remainder round
trips with expressions proved equal to them.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Report whether a generated kernel contains a conversion or quotient-remainder
expression handled by the focused native-index simplifier.

The scan keeps the simplifier away from unrelated symbolic and scalar terms,
and it ignores natural-number division in symbolic indices, which needs no
native-word normalization.
-/
def hasNativeIndexRedex (value : Expr) : Bool :=
  (value.find? fun subexpression =>
    subexpression.isConstOf ``Nat.toUSize ||
      subexpression.isConstOf ``USize.ofNat ||
      ((subexpression.isAppOfArity ``HDiv.hDiv 6 ||
          subexpression.isAppOfArity ``HMod.hMod 6) &&
        subexpression.getAppArgs[0]!.isConstOf ``USize &&
        subexpression.getAppArgs[1]!.isConstOf ``USize &&
        subexpression.getAppArgs[2]!.isConstOf ``USize)).isSome

/--
Remove a generated native-to-natural-to-native conversion round trip.

`Nat.toUSize` is an abbreviation for `USize.ofNat`, and unfolding it can expose
the left-hand side of `USize.ofNat_toNat` only after the ordinary simplifier has
visited that root, so a focused post-simplifier performs the rewrite.
-/
private def simpRoundTrip
    (value : Expr) : SimpM Simp.Step := do
  let value := value.consumeMData
  unless value.isAppOfArity ``Nat.toUSize 1 ||
      value.isAppOfArity ``USize.ofNat 1 do
    return .continue
  let naturalValue := value.appArg!.consumeMData
  unless naturalValue.isAppOfArity ``USize.toNat 1 do
    return .continue
  let nativeValue := naturalValue.appArg!
  let correctness ←
    mkAppOptM ``USize.ofNat_toNat #[some nativeValue]
  let correctness ←
    withTransparency .all <|
      mkExpectedTypeHint correctness (← mkEq value nativeValue)
  return .done {
    expr := nativeValue
    proof? := some correctness
    cache := false
  }

/--
Return the operands of a binary native-word operation.

The generated index compiler uses the ordinary `HAdd`, `HMul`, `HDiv` and
`HMod` instances at `USize`, and checking all three type arguments keeps this
matcher away from scalar operations at another type.
-/
private def nativeBinaryOperands?
    (operation : Name) (value : Expr) : Option (Expr × Expr) := do
  let value := value.consumeMData
  guard <| value.isAppOfArity operation 6
  let arguments := value.getAppArgs
  guard <| arguments[0]!.isConstOf ``USize
  guard <| arguments[1]!.isConstOf ``USize
  guard <| arguments[2]!.isConstOf ``USize
  some (arguments[4]!, arguments[5]!)

/--
Remove division by one from a generated native index.

Flattened tile coordinates can contain `(index % tileSize) / 1`, and removing
that final unit stride exposes the quotient-remainder reconstruction handled by
`simpRecombine`.
-/
private def simpDivByOne
    (value : Expr) : SimpM Simp.Step := do
  let some (dividend, divisor) :=
      nativeBinaryOperands? ``HDiv.hDiv value
    | return .continue
  let some (divisorValue, _) ← getOfNatValue? divisor ``USize
    | return .continue
  unless divisorValue = 1 do
    return .continue
  let correctness ←
    mkAppOptM ``USize.div_one #[some dividend]
  let correctness ←
    withTransparency .all <|
      mkExpectedTypeHint correctness (← mkEq value dividend)
  return .done {
    expr := dividend
    proof? := some correctness
    cache := false
  }

/--
Cancel quotient-remainder decoding that a flattened contraction or output
traversal immediately re-encodes as a tensor-buffer index.

Both the plain and the scaled case need matching quotient and remainder
operands. The scaled case also needs concrete native numerals whose natural
values satisfy `stride = scale * divisor`. Its proof is assembled from
`USize.ofNat_mul`, so the rewrite holds on every Lean target whatever its
machine-word width.
-/
private def simpRecombine
    (value : Expr) : SimpM Simp.Step := do
  let some (left, right) :=
      nativeBinaryOperands? ``HAdd.hAdd value
    | return .continue
  let some (quotientTerm, base) :=
      nativeBinaryOperands? ``HAdd.hAdd right
    | return .continue
  if let some (scale, remainder) :=
      nativeBinaryOperands? ``HMul.hMul left then
    let some (index, divisor) :=
        nativeBinaryOperands? ``HMod.hMod remainder
      | return .continue
    let some (stride, quotient) :=
        nativeBinaryOperands? ``HMul.hMul quotientTerm
      | return .continue
    let some (quotientIndex, quotientDivisor) :=
        nativeBinaryOperands? ``HDiv.hDiv quotient
      | return .continue
    unless index.consumeMData == quotientIndex.consumeMData &&
        divisor.consumeMData == quotientDivisor.consumeMData do
      return .continue
    let some (scaleValue, _) ← getOfNatValue? scale ``USize
      | return .continue
    let some (strideValue, _) ← getOfNatValue? stride ``USize
      | return .continue
    let some (divisorValue, _) ← getOfNatValue? divisor ``USize
      | return .continue
    unless strideValue = scaleValue * divisorValue do
      return .continue
    let hStride ←
      mkAppM ``USize.ofNat_mul #[
        mkNatLit scaleValue, mkNatLit divisorValue]
    let expectedStride ← mkEq stride (← mkMul scale divisor)
    let hStride ←
      withTransparency .all <|
        mkExpectedTypeHint hStride expectedStride
    let replacement ← mkAdd (← mkMul scale index) base
    let correctness ←
      mkAppM ``Native.recombine_scaled_div_mod #[
        scale, stride, index, divisor, base, hStride]
    let correctness ←
      withTransparency .all <|
        mkExpectedTypeHint correctness (← mkEq value replacement)
    return .done {
      expr := replacement
      proof? := some correctness
      cache := false
    }
  let some (index, divisor) :=
      nativeBinaryOperands? ``HMod.hMod left
    | return .continue
  let some (productLeft, productRight) :=
      nativeBinaryOperands? ``HMul.hMul quotientTerm
    | return .continue
  let quotientFirst :=
    match nativeBinaryOperands? ``HDiv.hDiv productLeft with
    | some (quotientIndex, quotientDivisor) =>
        index.consumeMData == quotientIndex.consumeMData &&
          divisor.consumeMData == quotientDivisor.consumeMData &&
          divisor.consumeMData == productRight.consumeMData
    | none => false
  let divisorFirst :=
    match nativeBinaryOperands? ``HDiv.hDiv productRight with
    | some (quotientIndex, quotientDivisor) =>
        divisor.consumeMData == productLeft.consumeMData &&
          index.consumeMData == quotientIndex.consumeMData &&
          divisor.consumeMData == quotientDivisor.consumeMData
    | none => false
  unless quotientFirst || divisorFirst do
    return .continue
  let hProduct ←
    if quotientFirst then
      mkEqRefl quotientTerm
    else
      mkAppM ``USize.mul_comm #[productLeft, productRight]
  let replacement ← mkAdd index base
  let correctness ←
    mkAppM ``Native.recombine_div_mod #[
      index, divisor, quotientTerm, base, hProduct]
  let correctness ←
    withTransparency .all <|
      mkExpectedTypeHint correctness (← mkEq value replacement)
  return .done {
    expr := replacement
    proof? := some correctness
    cache := false
  }

/--
Normalize generated native-index expressions in two certified phases.

The first pass removes native-to-natural-to-native conversions. The second then
drops division by one and replaces each complete quotient-remainder
reconstruction with the decoded native index, and the returned equality
composes both simplifier certificates.
-/
def simplifyNativeIndexExpressions
    (expression : Expr) (context : Simp.Context)
    (pre : Expr → SimpM Simp.Step := fun _ => pure .continue) :
    MetaM (Expr × Expr) := do
  let (converted, _) ←
    Simp.main expression context
      (methods := {
        pre := pre >> Simp.rewritePre
        post :=
          simpRoundTrip >>
            Simp.rewritePost
      })
  let hExpressionConverted ← converted.getProof' expression
  let (recombined, _) ←
    Simp.main converted.expr context
      (methods := {
        pre := pre >> Simp.rewritePre
        post :=
          simpDivByOne >>
            simpRecombine >>
            Simp.rewritePost
      })
  let hConvertedRecombined ← recombined.getProof' converted.expr
  let hExpressionRecombined ←
    mkAppM ``Eq.trans #[
      hExpressionConverted, hConvertedRecombined]
  return (recombined.expr, hExpressionRecombined)

end TorchLean.Einops.Elab
