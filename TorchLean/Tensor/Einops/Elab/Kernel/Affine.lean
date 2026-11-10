/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.MixedRadix
public meta import TorchLean.Tensor.Einops.Elab.Einsum.Index
import Mathlib.Algebra.Order.Field.Basic
import TorchLean.Tensor.Einops.Elab.Einsum.Index
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Data.Finset.Attr
public import Mathlib.Data.Rat.Cast.Order

/-!
# Certified affine-index normalization

Affine-index normalization removes mixed-radix decode/encode pairs from
generated source indices. It recognizes the canonical row-major form
`remainder + radix * digit` and emits an equality certificate for every
replacement. The range proof comes from the `Fin` value used as the remainder
or from a remainder `value % radix` with a concrete positive radix.

The pass works for any tensor rank and any axis names. Symbolic radices are
simplified whenever the remainder carries the matching `Fin` bound, and other
arithmetic is left unchanged.
-/

public meta section

namespace TorchLean.Einops.Elab

open Lean
open Lean.Meta

/--
Recognize `remainder + radix * digit`, accepting either summand order.

Generated row-major expressions use the canonical product order, and accepting
the outer addition in either order also handles the compiler's associative
index regrouping without a separate rewrite theorem.
-/
private def mixedRadixEncoding?
    (value radix : Expr) : Option (Expr × Expr × Bool) := do
  let (left, right) ←
    natOperationOperands? ``Nat.add ``HAdd.hAdd value
  let fromProduct (remainder product : Expr) (swapped : Bool) :
      Option (Expr × Expr × Bool) := do
    let (productRadix, digit) ←
      natOperationOperands? ``Nat.mul ``HMul.hMul product
    guard <| productRadix.consumeMData == radix.consumeMData
    return (remainder, digit, swapped)
  fromProduct left right false <|> fromProduct right left true

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

After one row-major stage has been partially evaluated, later stages often see
the same bounded digit as `value % radix`. Recognizing both representations
lets composition cancel decode/encode pairs without expanding a complete
transform chain, and the `%` form needs a concrete positive radix.
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
private def simpMixedRadix
    (value : Expr) : SimpM Simp.Step := do
  let operation? :=
    match natOperationOperands? ``Nat.div ``HDiv.hDiv value with
    | some operands => some (true, operands)
    | none =>
        (false, ·) <$> natOperationOperands? ``Nat.mod ``HMod.hMod value
  let some (isDivision, (numerator, radix)) := operation?
    | return .continue
  let some (remainder, digit, swapped) :=
      mixedRadixEncoding? numerator radix
    | return .continue
  let some hRemainder ← remainderBound? remainder radix
    | return .continue
  let replacement := if isDivision then digit else remainder
  let theoremName :=
    if isDivision then ``Einops.MixedRadix.div_encode
    else ``Einops.MixedRadix.mod_encode
  let correctness ←
    mkAppM theoremName #[remainder, digit, radix, hRemainder]
  let correctness ←
    if swapped then
      -- The mixed-radix theorem requires the remainder before the product.
      let product ← mkAppM ``Nat.mul #[radix, digit]
      let hNumerator ← mkAppM ``Nat.add_comm #[product, remainder]
      let decode ←
        withLocalDeclD `numerator (mkConst ``Nat) fun numerator => do
          let result ←
            mkAppM (if isDivision then ``Nat.div else ``Nat.mod) #[numerator, radix]
          mkLambdaFVars #[numerator] result
      let hDecode ← mkAppM ``congrArg #[decode, hNumerator]
      mkAppM ``Eq.trans #[hDecode, correctness]
    else
      pure correctness
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
        post := simpMixedRadix >> Simp.rewritePost
      })
  return (result.expr, ← result.getProof' value)

end TorchLean.Einops.Elab
