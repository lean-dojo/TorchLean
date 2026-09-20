/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Contraction.Hoist
public meta import NN.Tensor.Internal.Elab.Einsum.Contraction.Loop -- shake: keep

/-!
# Native index normalization for generated einsum outputs

This module removes native-to-natural conversion round trips after output
coordinates have been substituted into a generated kernel. The simplifier is
restricted to compiler-generated coordinates, index bases, tensor reads, and
their dependent bound proofs; user scalar expressions are left untouched.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/--
Build the focused simplifier contexts used before and after output-coordinate
substitution.

The first context preserves coordinate wrappers so selected generated lets can
be unfolded explicitly. The second also performs iota reduction after a native
coordinate has been substituted.
-/
def nativeOutputIndexSimpContexts :
    MetaM (Simp.Context × Simp.Context) := do
  let mut theorems : SimpTheorems := {}
  theorems ← theorems.addConst ``id_eq
  theorems ← theorems.addConst ``Fin.val_castLE
  theorems ← theorems.addConst ``toUSize_mkDivMod
  let congrTheorems ← getSimpCongrTheorems
  let indexContext ←
    Simp.mkContext
      (config := {
        iota := false
        zeta := false
        zetaDelta := false
        failIfUnchanged := false
      })
      (simpTheorems := #[theorems])
      (congrTheorems := congrTheorems)
  let coordinateContext ←
    Simp.mkContext
      (config := {
        iota := true
        zeta := false
        zetaDelta := false
        failIfUnchanged := false
      })
      (simpTheorems := #[theorems])
      (congrTheorems := congrTheorems)
  pure (indexContext, coordinateContext)

/--
Inline only generated output-axis lets.

Other lets may contain tensor reads or staged index arithmetic, so preserving
them avoids duplicating executable work during native-index normalization.
-/
private partial def inlineOutputAxisLets (value : Expr) : Expr :=
  match value with
  | .forallE name domain body binderInfo =>
      .forallE name (inlineOutputAxisLets domain)
        (inlineOutputAxisLets body) binderInfo
  | .lam name domain body binderInfo =>
      .lam name (inlineOutputAxisLets domain)
        (inlineOutputAxisLets body) binderInfo
  | .letE name type assignment body nondep =>
      let type := inlineOutputAxisLets type
      let assignment := inlineOutputAxisLets assignment
      let body := inlineOutputAxisLets body
      if name.toString.startsWith "outputAxis" then
        body.instantiate1 assignment
      else
        .letE name type assignment body nondep
  | .app function argument =>
      .app (inlineOutputAxisLets function)
        (inlineOutputAxisLets argument)
  | .mdata data body =>
      .mdata data (inlineOutputAxisLets body)
  | .proj typeName fieldIndex value =>
      .proj typeName fieldIndex (inlineOutputAxisLets value)
  | value => value

/--
Inline a generated native index base and simplify the resulting expression in
the active focused simplifier context.
-/
private def inlineNativeIndexBase (value : Expr) : SimpM Simp.Step := do
  let .letE name _ assignment body _ := value
    | return .continue
  unless name.toString.startsWith "nativeIndexBase" do
    return .continue
  let inlinedBody := body.instantiate1 assignment
  let simplifiedBody ← Simp.simp inlinedBody
  let hSimplified ← simplifiedBody.getProof' inlinedBody
  let hSimplified ←
    withTransparency .all <|
      mkExpectedTypeHint hSimplified
        (← mkEq value simplifiedBody.expr)
  return .done {
    expr := simplifiedBody.expr
    proof? := some hSimplified
    cache := false
  }

/--
Normalize the native index of a generated tensor read and transport its
dependent bounds along the same equality.
-/
private def simplifyNativeTensorRead (context : Simp.Context)
    (value : Expr) : SimpM Simp.Step := do
  unless value.isAppOfArity ``Rep.getFlatUSize 6 do
    return .continue
  let arguments := value.getAppArgs
  let index := arguments[4]!
  let (simplifiedIndex, hIndex) ←
    simplifyNativeIndexExpressions index context
  if simplifiedIndex == index then
    return .done { expr := value }
  let boundPredicate ←
    withLocalDeclD `inputIndex (mkConst ``USize) fun inputIndex => do
      let inputIndexNat ←
        mkAppM ``USize.toNat #[inputIndex]
      let inputSize ←
        mkAppM ``Shape.size #[arguments[2]!]
      let bound ← mkLT inputIndexNat inputSize
      mkLambdaFVars #[inputIndex] bound
  let hBound ←
    mkAppM ``congrArg #[boundPredicate, hIndex]
  let simplifiedBound ←
    mkAppM ``Eq.mp #[hBound, arguments[5]!]
  let simplifiedArguments :=
    (arguments.set! 4 simplifiedIndex).set! 5 simplifiedBound
  let simplifiedRead :=
    mkAppN value.getAppFn simplifiedArguments
  let hRead ←
    mkAppM ``Rep.getFlatUSize_congr #[
      arguments[3]!, hIndex, arguments[5]!, simplifiedBound]
  let hRead ←
    withTransparency .all <|
      mkExpectedTypeHint hRead
        (← mkEq value simplifiedRead)
  return .done {
    expr := simplifiedRead
    proof? := some hRead
    cache := false
  }

/--
Keep the focused index simplifier out of erased proof arguments.

Dependent tensor reads are handled as complete applications before this guard,
so their bounds are still transported when the executable index changes.
-/
private def stopAtProof (value : Expr) : SimpM Simp.Step := do
  if ← isProof value then
    return .done { expr := value }
  return .continue

/--
Normalize native indices throughout a completed generated output expression
and return an equality from the normalized expression to the original one.
-/
def simplifyNativeOutputIndices
    (context : Simp.Context) (body : Expr) :
    TermElabM (Expr × Expr) := do
  unless hasNativeIndexRedex body do
    return (body, ← mkEqRefl body)
  let inlinedBody := inlineOutputAxisLets body
  let hInlinedBody ←
    withTransparency .all <|
      mkExpectedTypeHint (← mkEqRefl inlinedBody)
        (← mkEq inlinedBody body)
  let (simplifiedBody, hInlinedSimplified) ←
    simplifyNativeIndexExpressions inlinedBody context
      (inlineNativeIndexBase >>
        simplifyNativeTensorRead context >>
        stopAtProof)
  -- Simplification expands dependent index bases. Float them back to the
  -- earliest loop that supplies all of their coordinates.
  let rehoistedBody ← hoistCoordinateLoopLets simplifiedBody
  let hRehoistedSimplified ←
    withTransparency .all <|
      mkExpectedTypeHint (← mkEqRefl rehoistedBody)
        (← mkEq rehoistedBody simplifiedBody)
  let hSimplifiedInlined ←
    mkAppM ``Eq.symm #[hInlinedSimplified]
  let hSimplifiedBody ←
    mkAppM ``Eq.trans #[hSimplifiedInlined, hInlinedBody]
  let hRehoistedBody ←
    mkAppM ``Eq.trans #[hRehoistedSimplified, hSimplifiedBody]
  pure (rehoistedBody, hRehoistedBody)

/--
Normalize generated native indices in one scalar product after its output-axis
lets have been opened.
-/
def simplifyNativeProductIndices
    (baseContext : Simp.Context) (leadingLocals : Array Expr)
    (product : Expr) : TermElabM (Expr × Option Expr) := do
  unless hasNativeIndexRedex product do
    return (product, none)
  let mut outputAxisIds : FVarIdSet := {}
  for localValue in leadingLocals do
    let declaration ← getFVarLocalDecl localValue
    if declaration.userName.toString.startsWith "outputAxis" then
      outputAxisIds := outputAxisIds.insert localValue.fvarId!
  let context :=
    baseContext.setZetaDeltaSet outputAxisIds {}
  let (result, hProductResult) ←
    simplifyNativeIndexExpressions product context
      (inlineNativeIndexBase >>
        simplifyNativeTensorRead context >>
        stopAtProof)
  pure (result, some hProductResult)

end TorchLean.Tensor.Internal.Elab.Impl
