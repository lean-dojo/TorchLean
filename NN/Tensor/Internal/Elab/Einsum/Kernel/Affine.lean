/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Laws.MixedRadix
public meta import NN.Tensor.Internal.Elab.Einsum.Index

/-!
# Certified affine-index normalization

This module removes mixed-radix decode/encode pairs from generated source
indices. It recognizes the canonical row-major form
`remainder + radix * digit`, obtains the range proof from the `Fin` value used
as the remainder, and emits an equality certificate for every replacement.

The pass is independent of tensor rank and concrete axis names. Symbolic
radices are simplified whenever the remainder carries the matching `Fin`
bound; unsupported arithmetic remains unchanged.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Meta

/--
Recognize `remainder + radix * digit`, accepting either summand order.

Generated row-major expressions use the canonical product order. Accepting
the outer addition in either order also handles the compiler's associative
index regrouping without adding a separate rewrite theorem.
-/
private def mixedRadixEncoding?
    (value radix : Expr) : Option (Expr × Expr) := do
  let (left, right) ←
    natOperationOperands? ``Nat.add ``HAdd.hAdd value
  let fromProduct (remainder product : Expr) : Option (Expr × Expr) := do
    let (productRadix, digit) ←
      natOperationOperands? ``Nat.mul ``HMul.hMul product
    guard <| productRadix.consumeMData == radix.consumeMData
    return (remainder, digit)
  fromProduct left right <|> fromProduct right left

/-- Recover the bounded value whose projection is a generated remainder. -/
private def finSource? (value : Expr) : Option Expr := do
  let value := value.consumeMData
  match value with
  | .proj ``Fin 0 source => return source
  | _ =>
      guard <| value.isAppOfArity ``Fin.val 2
      return value.getAppArgs[1]!

/-- Prove that a concrete reflected radix is positive. -/
private def positiveConcreteRadix?
    (radix : Expr) : MetaM (Option Expr) := do
  let some value ← getNatValue? radix
    | return none
  unless 0 < value do
    return none
  let proof ←
    mkDecideProof (← mkLT (mkNatLit 0) (mkNatLit value))
  return some <|
    ← withTransparency .all <|
      mkExpectedTypeHint proof (← mkLT (mkNatLit 0) radix)

/--
Build the range proof carried either by a `Fin` digit or by a concrete
remainder operation.

After one row-major stage has been partially evaluated, later stages often
see the same bounded digit as `value % radix` rather than as a `Fin`
projection. Recognizing both representations lets composition cancel
decode/encode pairs without expanding a complete transform chain.
-/
private def remainderBound?
    (remainder radix : Expr) : MetaM (Option Expr) := do
  let expected ← mkLT remainder radix
  if let some source := finSource? remainder then
    let sourceType ← inferType source
    let sourceType := sourceType.consumeMData
    if sourceType.isAppOfArity ``Fin 1 then
      let sourceBound := sourceType.appArg!
      if ← withTransparency .reducible <| isDefEq sourceBound radix then
        let proof ← mkAppM ``Fin.isLt #[source]
        return some <|
          ← withTransparency .all <| mkExpectedTypeHint proof expected
  let some (value, modulus) :=
      natOperationOperands? ``Nat.mod ``HMod.hMod remainder
    | return none
  unless ← withTransparency .reducible <| isDefEq modulus radix do
    return none
  let some hPositive ← positiveConcreteRadix? radix
    | return none
  let proof ← mkAppM ``Nat.mod_lt #[value, hPositive]
  return some <|
    ← withTransparency .all <| mkExpectedTypeHint proof expected

/--
Cancel one mixed-radix quotient or remainder using the bound stored in its
lower digit.
-/
private def simplifyMixedRadix
    (value : Expr) : SimpM Simp.Step := do
  let operation? :=
    match natOperationOperands? ``Nat.div ``HDiv.hDiv value with
    | some operands => some (true, operands)
    | none =>
        (false, ·) <$> natOperationOperands? ``Nat.mod ``HMod.hMod value
  let some (isDivision, (numerator, radix)) := operation?
    | return .continue
  let some (remainder, digit) :=
      mixedRadixEncoding? numerator radix
    | return .continue
  let some hRemainder ← remainderBound? remainder radix
    | return .continue
  let replacement := if isDivision then digit else remainder
  let theoremName :=
    if isDivision then ``TorchLean.Tensor.Internal.MixedRadix.div_encode
    else ``TorchLean.Tensor.Internal.MixedRadix.mod_encode
  let correctness ←
    mkAppM theoremName #[remainder, digit, radix, hRemainder]
  let correctness ←
    withTransparency .all <|
      mkExpectedTypeHint correctness (← mkEq value replacement)
  return .done {
    expr := replacement
    proof? := some correctness
    cache := false
  }

/--
Normalize mixed-radix arithmetic in one generated flat index.

The returned equality is oriented from the original expression to the
optimized expression, ready to compose with the operand-view certificate.
-/
def simplifyAffineIndex (value : Expr) : MetaM (Expr × Expr) := do
  let mut theorems : SimpTheorems := {}
  for theoremName in #[
      ``Function.comp_apply,
      ``id_eq] do
    theorems ← theorems.addConst theoremName
  let simpContext ←
    Simp.mkContext
      (config := { zeta := true, failIfUnchanged := false })
      (simpTheorems := #[theorems])
      (congrTheorems := ← getSimpCongrTheorems)
  let (result, _) ←
    Simp.main value simpContext
      (methods := {
        post := simplifyMixedRadix >> Simp.rewritePost
      })
  return (result.expr, ← result.getProof' value)

end TorchLean.Tensor.Internal.Elab.Impl
