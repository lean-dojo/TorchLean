/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import Aesop.BuiltinRules
public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
meta import Mathlib.Tactic.ToAdditive
public import NN.Tensor.Internal.Elab.Einsum.Loop
public meta import NN.Tensor.Internal.Elab.Einsum.Index -- shake: keep
public import NN.Tensor.Internal.Elab.Einsum.Tiling -- shake: keep

/-!
# Certified native-index normalization

This module recognizes native coordinate arithmetic emitted by the einsum
compiler and replaces conversion and quotient-remainder round trips with
proved equal expressions.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Report whether a generated kernel contains a conversion or quotient-remainder
expression handled by the focused native-index simplifier.

The scan keeps the simplifier away from unrelated symbolic and scalar terms.
-/
def hasNativeIndexRedex (value : Expr) : Bool :=
  (value.find? fun subexpression =>
    subexpression.isConstOf ``Nat.toUSize ||
      subexpression.isConstOf ``USize.ofNat ||
      subexpression.isConstOf ``HDiv.hDiv ||
      subexpression.isConstOf ``HMod.hMod).isSome

/--
Remove a generated native-to-natural-to-native conversion round trip.

`Nat.toUSize` is an abbreviation for `USize.ofNat`. A focused post-simplifier
is needed because unfolding the abbreviation can expose the theorem's
left-hand side only after the ordinary simplifier has already visited that
root.
-/
private def simplifyNativeConversionRoundTrip
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

The generated index compiler uses the ordinary `HAdd`, `HMul`, `HDiv`, and
`HMod` instances at `USize`; checking all three type arguments prevents this
matcher from inspecting scalar operations at another type.
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

Flattened tile coordinates can contain `(index % tileSize) / 1`. Eliminating
that final unit stride exposes the quotient-remainder reconstruction handled
by `simplifyNativeCoordinateRecombination`.
-/
private def simplifyNativeDivisionByOne
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
Cancel quotient-remainder decoding that the flattened contraction compiler
immediately re-encodes as a tensor-buffer index.

Both cases require matching quotient and remainder operands. The scaled case
also accepts only concrete native numerals whose natural values satisfy
`stride = scale * divisor`. Its proof is assembled from `USize.ofNat_mul`, so
the rewrite is valid on every Lean target and does not assume a particular
machine-word width.
-/
private def simplifyNativeCoordinateRecombination
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
      mkAppM ``native_recombine_scaled_div_mod #[
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
    mkAppM ``native_recombine_div_mod #[
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

The first pass removes native-to-natural-to-native conversions. The second
then sees complete quotient-remainder reconstructions and replaces them with
direct native indexing. The returned equality composes both simplifier
certificates.
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
          simplifyNativeConversionRoundTrip >>
            Simp.rewritePost
      })
  let hExpressionConverted ← converted.getProof' expression
  let (recombined, _) ←
    Simp.main converted.expr context
      (methods := {
        pre := pre >> Simp.rewritePre
        post :=
          simplifyNativeDivisionByOne >>
            simplifyNativeCoordinateRecombination >>
            Simp.rewritePost
      })
  let hConvertedRecombined ← recombined.getProof' converted.expr
  let hExpressionRecombined ←
    mkAppM ``Eq.trans #[
      hExpressionConverted, hConvertedRecombined]
  return (recombined.expr, hExpressionRecombined)

end TorchLean.Tensor.Internal.Elab.Impl
