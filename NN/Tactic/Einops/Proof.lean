/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Transform.View
public import NN.Tensor.Internal.Lowering.Pack
public import NN.Tensor.Internal.Laws.MixedRadix -- shake: keep
public import NN.Tensor.Internal.Lowering.Reduce -- shake: keep
import Mathlib.Algebra.Order.Field.Basic

/-!
# Proof automation for verified tensor transformations

`einops` closes tensor-layout goals with compiler-correctness theorems and
general tensor laws. Equality between concrete rearrangements is reduced to
equality of compact row-major index maps. Other goals use focused e-graphs,
a deterministic simplification pass, and finally Mathlib's registered
`grind` rules.

Every successful run produces an ordinary kernel-checked proof term. The
tactic does not evaluate propositions through a separate native trust path.
The e-graph normalizes proof terms; executable tensors continue to use the
native `Array` lowerings.
-/

/--
Bound a quotient when the available product bound lists the quotient bound
before the divisor.
-/
private theorem Nat.div_lt_of_lt_mul_comm {value divisor bound : Nat}
    (h : value < bound * divisor) :
    value / divisor < bound := by
  apply Nat.div_lt_of_lt_mul
  simpa only [Nat.mul_comm] using h

/-- Bound one decoded digit after identifying the finite index size. -/
private theorem Fin.div_lt_of_size_eq_mul_bound
    {size divisor bound : Nat} (index : Fin size)
    (hSize : size = bound * divisor) :
    index.val / divisor < bound := by
  subst size
  exact Nat.div_lt_of_lt_mul_comm index.isLt

public meta section

namespace TorchLean.Tensor.Internal

open Lean Elab Tactic Meta

/--
Inline only the outer `let` chain of a generated term, exposing its lowering
head without unfolding the lowering itself.

The lowering proof helpers and the `einops?` report decoders use this traversal.
-/
partial def instantiateOuterLets : Expr → Expr
  | .letE _ _ value body _ =>
      instantiateOuterLets (body.instantiate1 value)
  | expression => expression

/-- Recognize a reflected rearrangement and return its plan, kind witness, and input. -/
private def rearrangeArguments? (expression : Expr) :
    Option (Expr × Expr × Expr) :=
  let expression := instantiateOuterLets expression
  if expression.isAppOfArity ``Lowering.rearrangeTensor 5 then
    let arguments := expression.getAppArgs
    some (arguments[2]!, arguments[3]!, arguments[4]!)
  else
    none

/--
Normalize an ordinary or fused rearrangement to one coordinate pullback.

The returned equality certifies the normalization, so the tactic may compare
the compact coordinate maps without trusting reflection.
-/
private partial def rearrangePullProgram? (expression : Expr) :
    MetaM (Option (Expr × Expr × Expr × Expr)) := do
  let expression := instantiateOuterLets expression
  if expression.isAppOfArity
      ``Elab.Impl.nativeTensorKernel 8 then
    let arguments := expression.getAppArgs
    let some (coordinateMap, inputTensor, normalized, hReference) ←
        rearrangePullProgram? arguments[3]!
      | return none
    let hNative ←
      mkAppM ``Elab.Impl.nativeTensorKernel_correct #[
        arguments[3]!, arguments[4]!, arguments[5]!, arguments[6]!,
        arguments[7]!]
    let hNormalized ← mkAppM ``Eq.trans #[hNative, hReference]
    return some
      (coordinateMap, inputTensor, normalized, hNormalized)
  else if expression.isAppOfArity ``Lowering.rearrangeTensor 5 then
    let arguments := expression.getAppArgs
    let checked := arguments[2]!
    let hKind := arguments[3]!
    let inputTensor := arguments[4]!
    let coordinateEquiv ←
      mkAppM ``Check.CheckedTransform.rearrangeCoordinateEquiv #[
        checked, hKind]
    let coordinateMap ← mkAppM ``Equiv.toFun #[coordinateEquiv]
    let normalized ←
      mkAppM ``Rep.pull #[coordinateMap, inputTensor]
    let correctness ←
      mkAppM ``Lowering.rearrangeTensor_correct #[
        checked, hKind, inputTensor]
    let hNormalized ←
      withTransparency .all <|
        mkExpectedTypeHint correctness (← mkEq expression normalized)
    if let some
        (inputMap, sourceTensor, _, hInputNormalized) ←
        rearrangePullProgram? inputTensor then
      let combinedMap ←
        mkAppM ``Function.comp #[inputMap, coordinateMap]
      let combinedNormalized ←
        mkAppM ``Rep.pull #[combinedMap, sourceTensor]
      let inputTensorType ← inferType inputTensor
      let pullOuter ←
        withLocalDeclD `intermediateTensor inputTensorType fun
            intermediateTensor => do
          let body ←
            mkAppM ``Rep.pull #[coordinateMap, intermediateTensor]
          mkLambdaFVars #[intermediateTensor] body
      let hInputPulled ←
        mkAppM ``congrArg #[pullOuter, hInputNormalized]
      let hComposed ←
        mkAppM ``Rep.pull_comp #[
          inputMap, coordinateMap, sourceTensor]
      let hNormalized ←
        mkAppM ``Eq.trans #[
          hNormalized,
          ← mkAppM ``Eq.trans #[hInputPulled, hComposed]]
      return some
        (combinedMap, sourceTensor, combinedNormalized, hNormalized)
    return some (coordinateMap, inputTensor, normalized, hNormalized)
  else if expression.isAppOfArity
      ``Lowering.transformTensorFused 8 then
    let arguments := expression.getAppArgs
    let checked := arguments[2]!
    let hAxes := arguments[3]!
    let inputMap := arguments[4]!
    let inputTensor := arguments[7]!
    let checkedMap ←
      mkAppM ``Check.CheckedTransform.inputCoordinateOfOutput #[
        checked, hAxes]
    let coordinateMap ←
      mkAppM ``Function.comp #[inputMap, checkedMap]
    let normalized ←
      mkAppM ``Rep.pull #[coordinateMap, inputTensor]
    let hNormalized ←
      mkAppM ``Lowering.transformTensorFused_correct #[
        checked, hAxes, inputMap, arguments[5]!, arguments[6]!, inputTensor]
    return some (coordinateMap, inputTensor, normalized, hNormalized)
  else
    return none

-- Reuse the elaborator's checked input and output shape view.
open Elab.Impl (checkedTransformShapes)

/-- Recover the scalar type of a native tensor expression. -/
private def tensorScalarType (inputTensor : Expr) : MetaM Expr := do
  let inputType ← inferType inputTensor
  unless inputType.isAppOfArity ``Rep 3 do
    throwError "expected a tensor input"
  let arguments := inputType.getAppArgs
  let scalarType := arguments[0]!
  if scalarType.hasLooseBVars then
    throwError "expected a tensor with one scalar type"
  return scalarType

/-- Recover the static shape of a native tensor expression. -/
private def tensorShape (inputTensor : Expr) : MetaM Expr := do
  let inputType ← inferType inputTensor
  unless inputType.isAppOfArity ``Rep 3 do
    throwError "expected a tensor input"
  return inputType.getAppArgs[1]!

/-- Recover the physical storage selected for a native tensor expression. -/
private def tensorStorage (inputTensor : Expr) : MetaM Expr := do
  let inputType ← inferType inputTensor
  unless inputType.isAppOfArity ``Rep 3 do
    throwError "expected a tensor input"
  return inputType.getAppArgs[2]!

/--
Normalize a tensor to an identity coordinate pullback.

This lets the fused rearrangement prover compare an optimized inverse chain
directly with its original tensor through the same row-major map criterion.
-/
private def identityPullProgram (inputTensor : Expr) :
    MetaM (Expr × Expr × Expr × Expr) := do
  let inputShape ← tensorShape inputTensor
  let coordinateType ← mkAppM ``Coord #[inputShape]
  let coordinateMap ←
    withLocalDeclD `coordinate coordinateType fun coordinate =>
      mkLambdaFVars #[coordinate] coordinate
  let normalized ← mkAppM ``Rep.pull #[coordinateMap, inputTensor]
  let hNormalized ←
    mkAppM ``Eq.symm #[← mkAppM ``Rep.pull_id #[inputTensor]]
  return (coordinateMap, inputTensor, normalized, hNormalized)

/--
Recover the checked transform and rearrange witness from one direct generated
equivalence application.

The matcher deliberately does not search arbitrary descendants. A composed
map may contain several checked transforms, and pairing an inner transform
with the outer coordinate would produce an ill-typed certificate. The focused
simplifier exposes compositions first, then this matcher certifies each direct
`Equiv.toFun` application separately.
-/
private def checkedRearrangementFunction? (function : Expr) :
    Option (Expr × Expr) := do
  let function := instantiateOuterLets function.consumeMData
  guard <| function.isAppOfArity ``Equiv.toFun 3
  let coordinateEquiv :=
    instantiateOuterLets function.getAppArgs[2]!.consumeMData
  guard <| coordinateEquiv.isAppOfArity
    ``Check.CheckedTransform.rearrangeCoordinateEquiv 2
  let arguments := coordinateEquiv.getAppArgs
  some (arguments[0]!, arguments[1]!)

/--
Recognize the general checked output-to-input projection used by fused
rearrange and repeat kernels.
-/
private def checkedProjectionFunction? (function : Expr) :
    Option (Expr × Expr) := do
  let function := instantiateOuterLets function.consumeMData
  guard <| function.isAppOfArity
    ``Check.CheckedTransform.inputCoordinateOfOutput 2
  let arguments := function.getAppArgs
  some (arguments[0]!, arguments[1]!)

/--
Replace one checked rearrangement coordinate application with its certified
row-major index calculation.

This custom simplifier step fires before Lean visits the large checked-plan
value. It therefore keeps proof cost proportional to the compact index
formula instead of the parser and checker certificate stored in the plan.
-/
private def simplifyCheckedRearrangementLinearize
    (value : Expr) : SimpM Simp.Step := do
  let value := value.consumeMData
  if value.hasLooseBVars then
    return .continue
  let linearized? :=
    match value with
    | .proj ``Fin 0 linearized => some linearized
    | _ =>
        if value.isAppOfArity ``Fin.val 2 then
          some value.getAppArgs[1]!
        else
          none
  let some linearized := linearized?
    | return .continue
  let linearized := linearized.consumeMData
  unless linearized.isAppOfArity ``Coord.linearize 2 do
    return .continue
  let coordinate := linearized.getAppArgs[1]!.consumeMData
  let .app function outputCoordinate := coordinate
    | return .continue
  if function.isAppOfArity ``Function.comp 5 then
    let arguments := function.getAppArgs
    let correctness ←
      mkAppM ``linearize_comp_apply_val #[
        arguments[3]!, arguments[4]!, outputCoordinate]
    let correctnessType ← inferType correctness
    let some (_, _, replacement) := correctnessType.eq?
      | throwError
          "internal error: coordinate-composition theorem is not an equality"
    let correctness ←
      withTransparency .all <|
        mkExpectedTypeHint correctness (← mkEq value replacement)
    return .visit {
      expr := replacement
      proof? := some correctness
      cache := false
    }
  if let some (checked, hAxes) :=
      checkedProjectionFunction? function then
    let correctness ←
      mkAppM
        ``Check.CheckedTransform.inputCoordinateOfOutput_linearize_coord
        #[checked, hAxes, outputCoordinate]
    let correctnessType ← inferType correctness
    let some (_, _, replacement) := correctnessType.eq?
      | throwError
          "internal error: compact transform theorem is not an equality"
    let correctness ←
      withTransparency .all <|
        mkExpectedTypeHint correctness (← mkEq value replacement)
    return .visit {
      expr := replacement
      proof? := some correctness
      cache := false
    }
  let some (checked, hKind) :=
      checkedRearrangementFunction? function
    | return .continue
  let correctness ←
    mkAppM
      ``Check.CheckedTransform.rearrangeCoordinateEquiv_toFun_linearize_coord
      #[checked, hKind, outputCoordinate]
  let correctnessType ← inferType correctness
  let some (_, _, replacement) := correctnessType.eq?
    | throwError
        "internal error: compact rearrangement theorem is not an equality"
  let correctness ←
    withTransparency .all <|
      mkExpectedTypeHint correctness (← mkEq value replacement)
  return .visit {
    expr := replacement
    proof? := some correctness
    cache := false
  }

/--
Build the deliberately small simplifier context for coordinate-map
normalization.

General `[simp]` lemmas are excluded because unfolding a checked transform
before its compact theorem fires expands parser and validation certificates.
-/
private def linearIndexSimpContext : MetaM Simp.Context := do
  let mut theorems : SimpTheorems := {}
  for theoremName in [
      ``Equiv.toFun_as_coe,
      ``Function.comp_apply,
      ``Coord.linearize_unlinearize,
      ``linearize_cast_val,
      ``linearize_equivCast_val,
      ``id_eq,
      ``finCongr_apply_coe] do
    theorems ← theorems.addConst theoremName
  Simp.mkContext
    (config := {
      zeta := true
      zetaDelta := false
      failIfUnchanged := false
    })
    (simpTheorems := #[theorems])
    (congrTheorems := ← getSimpCongrTheorems)

/--
Normalize all checked coordinate maps in a compact-index goal and install the
result as the new target together with the simplifier's equality certificate.
-/
private def normalizeLinearIndexGoal (goal : MVarId) : TacticM MVarId :=
  goal.withContext do
    let target ← instantiateMVars (← goal.getType)
    let context ← linearIndexSimpContext
    let (result, _) ←
      Simp.main target context
        (methods := {
          pre :=
            simplifyCheckedRearrangementLinearize >>
              Simp.rewritePre
          post :=
            simplifyCheckedRearrangementLinearize >>
              Simp.rewritePost
        })
    applySimpResultToTarget goal target result

/-- Recover the operands of a direct or heterogeneously elaborated operation. -/
private def natBinaryOperands? (direct heterogeneous : Name)
    (expression : Expr) : Option (Expr × Expr) :=
  if expression.isAppOfArity direct 2 then
    let arguments := expression.getAppArgs
    some (arguments[0]!, arguments[1]!)
  else if expression.isAppOfArity heterogeneous 6 then
    let arguments := expression.getAppArgs
    if arguments[0]!.isConstOf ``Nat &&
        arguments[1]!.isConstOf ``Nat &&
        arguments[2]!.isConstOf ``Nat then
      some (arguments[4]!, arguments[5]!)
    else
      none
  else
    none

/--
Recover the operands of any supported natural-number binary operation.
-/
private def namedNatBinaryOperands? (names : List Name)
    (expression : Expr) : Option (Expr × Expr) :=
  let rec first
      (operations : List (Name × Name)) : Option (Expr × Expr) :=
    match operations with
    | [] => none
    | (direct, heterogeneous) :: rest =>
        if names.contains direct then
          match natBinaryOperands? direct heterogeneous expression with
          | some operands => some operands
          | none => first rest
        else
          first rest
  first [
    (``Nat.add, ``HAdd.hAdd),
    (``Nat.mul, ``HMul.hMul),
    (``Nat.div, ``HDiv.hDiv),
    (``Nat.mod, ``HMod.hMod)]

/--
Collect distinct natural-number binary applications together with their
operands, including heterogeneously elaborated arithmetic notation.
-/
private partial def collectNatBinaryApplications
    (names : List Name) (expression : Expr)
    (applications : Array (Expr × Expr × Expr) := #[]) :
    Array (Expr × Expr × Expr) :=
  let applications :=
    match expression with
    | .forallE _ domain body _ =>
        collectNatBinaryApplications names body <|
          collectNatBinaryApplications names domain applications
    | .lam _ domain body _ =>
        collectNatBinaryApplications names body <|
          collectNatBinaryApplications names domain applications
    | .letE _ type value body _ =>
        collectNatBinaryApplications names body <|
          collectNatBinaryApplications names value <|
            collectNatBinaryApplications names type applications
    | .app function argument =>
        collectNatBinaryApplications names argument <|
          collectNatBinaryApplications names function applications
    | .mdata _ body =>
        collectNatBinaryApplications names body applications
    | .proj _ _ body =>
        collectNatBinaryApplications names body applications
    | _ => applications
  let operands? := namedNatBinaryOperands? names expression
  if let some (left, right) := operands? then
    if applications.any fun application => application.1 == expression then
      applications
    else
      applications.push (expression, left, right)
  else
    applications

/--
Remove irrelevant data-valued locals from an arithmetic side goal while
retaining every proposition that can constrain it.
-/
private def clearNonPropositionalLocals (goal : MVarId) : MetaM MVarId :=
  goal.withContext do
    let mut candidates : Array FVarId := #[]
    for localDecl in (← getLCtx) do
      unless ← isProp localDecl.type do
        candidates := candidates.push localDecl.fvarId
    return (← goal.tryClearMany' candidates).1

/-- Ask Omega for a closed certificate without disturbing the caller's goal list. -/
private def omegaCertificate? (proposition : Expr) :
    TacticM (Option Expr) := do
  let certificateGoal ← mkFreshExprSyntheticOpaqueMVar proposition
  let omegaGoal ← clearNonPropositionalLocals certificateGoal.mvarId!
  let savedGoals ← getGoals
  let result ←
    try
      setGoals [omegaGoal]
      try
        evalTactic <| ← `(tactic| omega)
      catch
        | .error _ _ => return none
        | error => throw error
      unless (← getUnsolvedGoals).isEmpty do
        return none
      let certificate ← instantiateMVars certificateGoal
      ensureHasNoMVars certificate
      pure (some certificate)
    finally
      setGoals savedGoals
  return result

/--
Ask the ordinary simplifier for a small closed certificate.

This is used for static shape-size equalities, where definitional reduction
may stop at identities such as `0 + dimension`.
-/
private def simpCertificate? (proposition : Expr) :
    TacticM (Option Expr) := do
  let certificateGoal ← mkFreshExprSyntheticOpaqueMVar proposition
  let savedGoals ← getGoals
  let result ←
    try
      setGoals [certificateGoal.mvarId!]
      try
        evalTactic <| ← `(tactic| simp)
      catch
        | .error _ _ => return none
        | error => throw error
      unless (← getUnsolvedGoals).isEmpty do
        return none
      let certificate ← instantiateMVars certificateGoal
      ensureHasNoMVars certificate
      pure (some certificate)
    finally
      setGoals savedGoals
  return result

/-- Add an arithmetic fact to a goal only when Omega can certify it. -/
private def noteOmegaFact? (goal : MVarId) (proposition : Expr) :
    TacticM (Option MVarId) :=
  goal.withContext do
    let some certificate ← omegaCertificate? proposition
      | return none
    let (_, nextGoal) ←
      goal.note (← mkFreshUserName `hEinopsIndexBound) certificate
        (some proposition)
    return some nextGoal

/--
If a division numerator encodes `remainder + radix * digit`, recover the
remainder whose range condition permits exact mixed-radix decoding.
-/
private def mixedRadixRemainder? (numerator radix : Expr) : Option Expr := do
  let (left, right) ←
    namedNatBinaryOperands? [``Nat.add] numerator.consumeMData
  let remainderFromProduct? (remainder product : Expr) : Option Expr := do
    let (firstFactor, secondFactor) ←
      namedNatBinaryOperands? [``Nat.mul] product.consumeMData
    if firstFactor.consumeMData == radix.consumeMData ||
        secondFactor.consumeMData == radix.consumeMData then
      some remainder
    else
      none
  remainderFromProduct? left right <|>
    remainderFromProduct? right left

/--
Add the quotient and remainder bounds needed to normalize a concrete
mixed-radix index formula.

Each fact is proved by Omega before it enters the main goal. Iterating over
`%` and `/` subexpressions lets an outer bound use facts already established
for its inner digits without enumerating the finite index.
-/
private def addMixedRadixBounds (goal : MVarId) (outputIndex target : Expr) :
    TacticM MVarId := do
  let outputBound ← goal.withContext <| mkAppM ``Fin.isLt #[outputIndex]
  let outputBoundType ← goal.withContext <| inferType outputBound
  let (_, goal) ←
    goal.note (← mkFreshUserName `hEinopsOutputIndex) outputBound
      (some outputBoundType)
  let remainderApplications :=
    (collectNatBinaryApplications [``Nat.mod] target).insertionSort fun
      (_, leftNumerator, _) (_, rightNumerator, _) =>
        leftNumerator.approxDepth < rightNumerator.approxDepth
  let divisionApplications :=
    (collectNatBinaryApplications [``Nat.div] target).insertionSort fun
      (_, leftNumerator, _) (_, rightNumerator, _) =>
        leftNumerator.approxDepth < rightNumerator.approxDepth
  let mut goal := goal
  for (_, numerator, modulus) in remainderApplications do
    let hPositive? ← goal.withContext do
      let proposition ← mkLT (mkNatLit 0) modulus
      try
        pure (some (← mkDecideProof proposition))
      catch
        | .error _ _ => pure none
        | error => throw error
    if let some hPositive := hPositive? then
      let hRemainder ←
        goal.withContext <| mkAppM ``Nat.mod_lt #[numerator, hPositive]
      let proposition ← goal.withContext <| inferType hRemainder
      let (_, nextGoal) ←
        goal.note (← mkFreshUserName `hEinopsRemainder) hRemainder
          (some proposition)
      goal := nextGoal

  let outputValue ← goal.withContext <| mkAppM ``Fin.val #[outputIndex]
  let mut candidateBounds : Array (Expr × Expr × Expr) := #[]
  for (_, numerator, denominator) in
      remainderApplications ++ divisionApplications do
    unless ← goal.withContext <|
        withTransparency .reducible <| isDefEq numerator outputValue do
      unless candidateBounds.any fun bound =>
          bound.1 == numerator && bound.2.1 == denominator do
        let proposition ← goal.withContext <| mkLT numerator denominator
        candidateBounds :=
          candidateBounds.push (numerator, denominator, proposition)
  for (_, numerator, denominator) in divisionApplications do
    if let some remainder :=
        mixedRadixRemainder? numerator denominator then
      unless candidateBounds.any fun bound =>
          bound.1 == remainder && bound.2.1 == denominator do
        let proposition ← goal.withContext <| mkLT remainder denominator
        candidateBounds :=
          candidateBounds.push (remainder, denominator, proposition)

  let mut pendingBounds : Array (Expr × Expr × Expr) := #[]
  for (numerator, denominator, proposition) in candidateBounds do
    let directCertificate? ← goal.withContext do
      let numerator := numerator.consumeMData
      let divisionArguments? :=
        namedNatBinaryOperands? [``Nat.div] numerator
      let some (dividend, divisor) := divisionArguments?
        | return none
      unless ← withTransparency .reducible <|
          isDefEq dividend outputValue do
        return none
      try
        let outputIndexType ← inferType outputIndex
        unless outputIndexType.isAppOfArity ``Fin 1 do
          return none
        let size := outputIndexType.appArg!
        let product ← mkMul denominator divisor
        let hSizeType ← mkEq size product
        let some hSize ← simpCertificate? hSizeType
          | return none
        unless ← withTransparency .reducible <|
            isDefEq (← inferType hSize) hSizeType do
          return none
        let certificate ←
          withTransparency .all <|
            mkAppOptM ``Fin.div_lt_of_size_eq_mul_bound #[
              some size, some divisor, some denominator,
              some outputIndex, some hSize]
        let certificateType ← inferType certificate
        if ← withTransparency .reducible <|
            isDefEq certificateType proposition then
          pure (some certificate)
        else
          pure none
      catch
        | .error _ _ => pure none
        | error => throw error
    if let some certificate := directCertificate? then
      let (_, nextGoal) ←
        goal.note (← mkFreshUserName `hEinopsIndexBound) certificate
          (some proposition)
      goal := nextGoal
    else
      pendingBounds := pendingBounds.push
        (numerator, denominator, proposition)

  -- Most concrete layouts expose all digit bounds at once. Proving their
  -- conjunction gives Omega one small arithmetic problem instead of one
  -- invocation per quotient layer.
  if !pendingBounds.isEmpty then
    let allBounds :=
      pendingBounds.foldr
        (fun (_, _, proposition) rest => mkAnd proposition rest)
        (mkConst ``True)
    if let some allCertificates ←
        goal.withContext <| omegaCertificate? allBounds then
      let mut certificates := allCertificates
      for (_, _, proposition) in pendingBounds do
        let certificate ←
          goal.withContext <| mkAppM ``And.left #[certificates]
        let (_, nextGoal) ←
          goal.note (← mkFreshUserName `hEinopsIndexBound) certificate
            (some proposition)
        goal := nextGoal
        certificates ←
          goal.withContext <| mkAppM ``And.right #[certificates]
      return goal

  let mut provedBounds : Array (Expr × Expr) := #[]
  for _ in [:pendingBounds.size + 1] do
    let mut madeProgress := false
    for (numerator, denominator, proposition) in pendingBounds do
      if provedBounds.any fun bound =>
          bound.1 == numerator && bound.2 == denominator then
        continue
      if let some nextGoal ← noteOmegaFact? goal proposition then
        goal := nextGoal
        provedBounds := provedBounds.push (numerator, denominator)
        madeProgress := true
    unless madeProgress do
      break
  return goal

/--
Instantiate a rearrangement equivalence criterion with a certified equality of
its compact row-major index maps and assign the original goal.
-/
private def closeWithLinearIndex (goal : MVarId) (target : Expr)
    (criterion inputTensor : Expr) : TacticM Unit := do
  let criterionType ← inferType criterion
  let .forallE _ linearIndexType _ _ := criterionType
    | throwError "internal error: expected a linear-index premise"
  let .forallE indexName indexType indexBody _ := linearIndexType
    | throwError
        "internal error: expected a quantified linear-index premise"
  let indexType ←
    if indexType.isAppOfArity ``Fin 1 then
      let indexBound := indexType.appArg!
      let indexBound :=
        match ← getNatValue? indexBound with
        | some value => mkNatLit value
        | none => indexBound
      pure (mkApp (mkConst ``Fin) indexBound)
    else
      pure indexType
  let hLinearIndex ←
    withLocalDeclD indexName indexType fun outputIndex => do
      let equality := indexBody.instantiate1 outputIndex
      let some (_, lhs, rhs) := equality.eq?
        | throwError
            "internal error: linear-index premise is not an equality"
      let hEquality ←
        if lhs == rhs then
          mkExpectedTypeHint (← mkEqRefl lhs) equality
        else
          let equalityGoal ← mkFreshExprSyntheticOpaqueMVar equality
          let compactGoal ←
            clearNonPropositionalLocals equalityGoal.mvarId!
          let savedGoals ← getGoals
          try
            setGoals [compactGoal]
            let compactGoal ← normalizeLinearIndexGoal compactGoal
            setGoals [compactGoal]
            try
              evalTactic <| ←
                `(tactic|
                  simp [
                    rearrangeLinearIndex,
                    Check.TransformPlan.axisLength,
                    Check.TransformPlan.normalized,
                    Check.NormalizedTransform.inputAxes,
                    Check.NormalizedTransform.outputAxes,
                    Check.PartialAxisLengths.set,
                    Check.PartialAxisLengths.seed])
            catch
              | .internal id _ =>
                throwError
                  "compact index simplifier aborted ({id.toString}) on:{indentExpr
                    equality}"
              | error => throw error
            let remainingGoals ← getUnsolvedGoals
            if remainingGoals.length > 1 then
              throwError
                "compact index simplification produced \
                  {remainingGoals.length} goals"
            if let some arithmeticGoal := remainingGoals[0]? then
              let arithmeticTarget ← arithmeticGoal.getType
              let arithmeticGoal ←
                addMixedRadixBounds arithmeticGoal outputIndex
                  arithmeticTarget
              let arithmeticGoal ←
                clearNonPropositionalLocals arithmeticGoal
              setGoals [arithmeticGoal]
              try
                evalTactic <| ←
                  `(tactic|
                  simp_all [
                      MixedRadix.div_encode,
                      MixedRadix.mod_encode,
                      Nat.mod_add_div,
                      Nat.mod_eq_of_lt,
                      Nat.div_eq_of_lt,
                      Nat.add_mul_div_left,
                      finCongr_apply_coe])
              catch
                | .internal id _ =>
                  throwError
                    "mixed-radix simplifier aborted ({id.toString}) on:{indentExpr
                      arithmeticTarget}"
                | error => throw error
              unless (← getUnsolvedGoals).isEmpty do
                try
                  evalTactic <| ← `(tactic| omega)
                catch
                  | .internal id _ =>
                    throwError
                      "mixed-radix arithmetic aborted ({id.toString}) on:{indentExpr
                        arithmeticTarget}"
                  | .error _ _ => pure ()
              let remainingGoals ← getUnsolvedGoals
              unless remainingGoals.isEmpty do
                throwError
                  "could not prove compact index goal:{indentExpr
                    (← remainingGoals[0]!.getType)}"
          finally
            setGoals savedGoals
          let hEquality ← instantiateMVars equalityGoal
          ensureHasNoMVars hEquality
          pure hEquality
      mkLambdaFVars #[outputIndex] hEquality
  ensureHasNoMVars hLinearIndex
  let certificate := mkApp2 criterion hLinearIndex inputTensor
  ensureHasNoMVars certificate
  goal.assign (← mkExpectedTypeHint certificate target)
  unless ← goal.isAssigned do
    throwError
      "internal error: the rearrangement certificate did not close its goal"
  replaceMainGoal []

/--
Close equality involving a fused rearrangement by comparing its complete
output-to-source map with the map on the other side.

The arithmetic certificate is passed to `Rep.pull_eq_of_linearIndex_eq`;
the kernel then transports it back through each lowering-correctness theorem.
-/
private def closeFusedPullbacks : TacticM Bool :=
  withMainContext do
    let goal ← getMainGoal
    let target :=
      instantiateOuterLets (← instantiateMVars (← goal.getType))
    let some (_, left, right) := target.eq?
      | return false
    let leftProgram? ← rearrangePullProgram? left
    let rightProgram? ← rearrangePullProgram? right
    let (leftMap, leftInput, leftNormalized, hLeft,
        rightMap, rightInput, rightNormalized, hRight) ←
      match leftProgram?, rightProgram? with
      | some leftProgram, some rightProgram =>
          pure (leftProgram.1, leftProgram.2.1, leftProgram.2.2.1,
            leftProgram.2.2.2, rightProgram.1, rightProgram.2.1,
            rightProgram.2.2.1, rightProgram.2.2.2)
      | some leftProgram, none =>
          unless instantiateOuterLets right == leftProgram.2.1 do
            return false
          let rightProgram ← identityPullProgram right
          pure (leftProgram.1, leftProgram.2.1, leftProgram.2.2.1,
            leftProgram.2.2.2, rightProgram.1, rightProgram.2.1,
            rightProgram.2.2.1, rightProgram.2.2.2)
      | none, some rightProgram =>
          unless instantiateOuterLets left == rightProgram.2.1 do
            return false
          let leftProgram ← identityPullProgram left
          pure (leftProgram.1, leftProgram.2.1, leftProgram.2.2.1,
            leftProgram.2.2.2, rightProgram.1, rightProgram.2.1,
            rightProgram.2.2.1, rightProgram.2.2.2)
      | none, none => return false
    unless ← withTransparency .reducible <| isDefEq leftInput rightInput do
      return false
    let normalizedTarget ← mkEq leftNormalized rightNormalized
    let scalarType ← tensorScalarType leftInput
    let storage ← tensorStorage leftInput
    let sourceShape ← tensorShape leftInput
    let outputShape ← tensorShape leftNormalized
    let criterion ←
      mkAppOptM ``Rep.pull_eq_of_linearIndex_eq #[
        some scalarType, some storage, some sourceShape, some outputShape,
        some leftMap, some rightMap]
    let certificateGoal ←
      mkFreshExprSyntheticOpaqueMVar normalizedTarget
    let savedGoals ← getGoals
    try
      setGoals [certificateGoal.mvarId!]
      closeWithLinearIndex certificateGoal.mvarId! normalizedTarget
        criterion leftInput
    finally
      setGoals savedGoals
    let hMaps ← instantiateMVars certificateGoal
    ensureHasNoMVars hMaps
    let hRightSymm ← mkAppM ``Eq.symm #[hRight]
    let certificate ←
      mkAppM ``Eq.trans #[
        hLeft, ← mkAppM ``Eq.trans #[hMaps, hRightSymm]]
    goal.assign (← mkExpectedTypeHint certificate target)
    unless ← goal.isAssigned do
      throwError
        "internal error: the fused rearrangement certificate did not close its goal"
    replaceMainGoal []
    return true

/--
Close equality of two literal rearrangements by comparing their compact
row-major index maps. The reflected plans are passed directly to the general
criterion theorem, avoiding unification against dependent proof fields.
-/
private def closeEquivalentRearrangements : TacticM Bool :=
  withMainContext do
    let goal ← getMainGoal
    let target :=
      instantiateOuterLets (← instantiateMVars (← goal.getType))
    let some (_, left, right) := target.eq?
      | return false
    let some (leftPlan, hLeftKind, leftInput) :=
        rearrangeArguments? left
      | return false
    let some (rightPlan, hRightKind, rightInput) :=
        rearrangeArguments? right
      | return false

    if leftInput == rightInput then
      let (leftInputShape, leftOutputShape) ←
        checkedTransformShapes leftPlan
      let (rightInputShape, rightOutputShape) ←
        checkedTransformShapes rightPlan
      let hInputShape ←
        mkExpectedTypeHint (← mkEqRefl leftInputShape)
          (← mkEq leftInputShape rightInputShape)
      let hOutputShape ←
        mkExpectedTypeHint (← mkEqRefl leftOutputShape)
          (← mkEq leftOutputShape rightOutputShape)
      let scalarType ← tensorScalarType leftInput
      let criterion ←
        mkAppOptM ``Lowering.rearrangeTensor_eq_of_linearIndex_eq #[
          some scalarType, some leftPlan, some rightPlan, some hLeftKind,
          some hRightKind, some hInputShape, some hOutputShape]
      closeWithLinearIndex goal target criterion leftInput
      return true
    else
      let some (innerPlan, hInnerKind, inputTensor) :=
          rearrangeArguments? leftInput
        | return false
      unless inputTensor == rightInput do
        return false
      let (innerInputShape, innerOutputShape) ←
        checkedTransformShapes innerPlan
      let (outerInputShape, outerOutputShape) ←
        checkedTransformShapes leftPlan
      let (directInputShape, directOutputShape) ←
        checkedTransformShapes rightPlan
      let hMiddleShape ←
        mkExpectedTypeHint (← mkEqRefl innerOutputShape)
          (← mkEq innerOutputShape outerInputShape)
      let hInputShape ←
        mkExpectedTypeHint (← mkEqRefl innerInputShape)
          (← mkEq innerInputShape directInputShape)
      let hOutputShape ←
        mkExpectedTypeHint (← mkEqRefl outerOutputShape)
          (← mkEq outerOutputShape directOutputShape)
      let scalarType ← tensorScalarType inputTensor
      let criterion ←
        mkAppOptM
          ``Lowering.rearrangeTensor_comp_eq_of_linearIndex_eq #[
            some scalarType, some innerPlan, some leftPlan, some rightPlan,
            some hInnerKind, some hLeftKind, some hRightKind,
            some hMiddleShape, some hInputShape, some hOutputShape]
      closeWithLinearIndex goal target criterion inputTensor
      return true

/--
Whether a target contains a tensor term handled by the `einops` theorem set.

This domain guard prevents the final e-graph pass from acting as a general
hypothesis-closing tactic on unrelated propositions.
-/
private def containsEinopsTerm (target : Expr) : Bool :=
  (target.find? fun expression =>
    match expression.getAppFn with
    | .const name _ =>
        (`TorchLean.Tensor.Internal.Rep).isPrefixOf name ||
          (`TorchLean.Tensor.Internal.Semantics).isPrefixOf name ||
          (`TorchLean.Tensor.Internal.Lowering).isPrefixOf name
    | _ => false).isSome

/-- Whether a target contains an unpack operation that may expose a pack inverse. -/
private def containsUnpackTerm (target : Expr) : Bool :=
  (target.find? fun expression =>
    let headName? := expression.getAppFn.constName?
    headName? == some ``Lowering.unpackTensor ||
      headName? == some ``Semantics.denoteUnpack).isSome

/--
Close a tensor-layout identity using TorchLean.Tensor.Internal compiler-correctness theorems,
the library's tensor algebra, and Mathlib's registered `grind` rules.
-/
syntax (name := einopsTactic) "einops" : tactic

elab_rules : tactic
  | `(tactic| einops) => do
      unless ← closeFusedPullbacks do
        unless ← closeEquivalentRearrangements do
          let normalizeUnpack ← withMainContext do
            containsUnpackTerm <$> instantiateMVars (← getMainTarget)
          if normalizeUnpack then
            evalTactic <| ←
              `(tactic|
                try simp only [
                  Elab.Impl.nativeTensorKernel_correct,
                  Lowering.unpackTensor_packTensor,
                  Lowering.packTensor_unpackTensor,
                  Fin.cases_zero,
                  Fin.cases_succ,
                  true_and,
                  and_true])
          else
            evalTactic <| ←
              `(tactic|
                try rw [Elab.Impl.nativeTensorKernel_correct])
          if (← getGoals).isEmpty then
            return
          withMainContext do
            let target ← instantiateMVars (← getMainTarget)
            unless containsEinopsTerm target do
              throwError
                "`einops` found no tensor, semantic operation, or lowering in the target"
          evalTactic <| ←
            `(tactic|
            solve
            | (with_unfolding_all
                grw (transparency := all) [
                  Lowering.rearrangeTensor_correct]
               with_unfolding_all
                exact Semantics.sum_denoteRearrange _ _ _)
            | (with_unfolding_all
                grw (transparency := all) [
                  Lowering.repeatTensor_correct]
               with_unfolding_all
                exact Semantics.sum_denoteRepeat _ _ _)
            | grind only [
                = Rep.pull_id,
                = Rep.reindex_refl,
                = Rep.reindex_symm_reindex,
                = Rep.map_reindex,
                = Rep.zipWith_reindex,
                = Rep.flatten_unflatten,
                = Rep.unflatten_flatten,
                = Rep.reshape_eq_reindex,
                = Rep.reshape_rfl,
                = Rep.reshape_symm_reshape]
            | grind only [
                = Rep.map_stack,
                = Rep.zipWith_stack,
                = Rep.map_unstack,
                = Rep.zipWith_unstack,
                = Rep.map_reshape,
                = Rep.zipWith_reshape,
                = Rep.dot_stack,
                = Rep.dot_reindex_reindex]
            | grind only [
                = Rep.reduce_reindex_input,
                = Rep.reindex_reduce,
                = Rep.reduceNonempty_reindex_input,
                = Rep.reindex_reduceNonempty]
            | grind only [
                = Lowering.transformTensorFused_correct,
                = Lowering.transformTensorFused_rearrange_correct,
                = Lowering.rearrangeTensor_correct,
                = Semantics.inverse_denoteRearrange,
                = Semantics.map_denoteRearrange,
                = Semantics.zipWith_denoteRearrange,
                = Semantics.sum_denoteRearrange,
                = Semantics.dot_denoteRearrange,
                = Semantics.dot_denoteRearrange_denoteRearrange]
            | grind only [
                = Lowering.transformTensorFused_correct,
                = Lowering.transformTensorFused_repeat_correct,
                = Lowering.repeatTensor_correct,
                = Semantics.push_denoteRepeat,
                = Semantics.sum_denoteRepeat,
                = Semantics.dot_denoteRepeat]
            | grind only [
                = Lowering.reduceFoldTensor_sum,
                = Lowering.reduceFoldTensor_prod,
                = Lowering.reduceFoldTensor_any,
                = Lowering.reduceFoldTensor_all,
                = Lowering.reduceFoldTensor_mean,
                = Lowering.reduceFoldTensor_ordered_correct,
                = Lowering.reduceNonemptyFoldTensor_ordered_correct,
                = Lowering.reduceTensor_correct,
                = Lowering.reduceNonemptyTensor_correct,
                = Semantics.sum_denoteReduce_sum,
                = Semantics.dot_denoteReduce_sum]
            | grind only [
                = Lowering.einsumTensor_correct,
                = Lowering.einsumTensorKernel_correct,
                = Semantics.einsumProductTensor_update_add,
                = Semantics.denoteEinsum_update_add,
                = Semantics.dot_denoteEinsum_update_eq_dot_einsumOperandVjp,
                = Semantics.sum_denoteEinsum]
            | grind only [
                = Lowering.packTensor_correct,
                = Lowering.unpackTensor_correct,
                = Lowering.unpackTensor_packTensor,
                = Lowering.packTensor_unpackTensor,
                = Semantics.denoteUnpack_denotePack,
                = Semantics.denotePack_denoteUnpack,
                = Semantics.dot_denotePack_denotePack,
                = Lowering.dot_packTensor,
                usr Lowering.dot_unpackTensor]
            | simp (config := { dsimp := false })
                (discharger := assumption) only [
                Lowering.transformTensorFused_correct,
                Lowering.transformTensorFused_rearrange_correct,
                Lowering.rearrangeTensor_correct,
                Lowering.transformTensorFused_repeat_correct,
                Lowering.repeatTensor_correct,
                Lowering.reduceFoldTensor_sum,
                Lowering.reduceFoldTensor_prod,
                Lowering.reduceFoldTensor_any,
                Lowering.reduceFoldTensor_all,
                Lowering.reduceFoldTensor_mean,
                Lowering.reduceFoldTensor_ordered_correct,
                Lowering.reduceNonemptyFoldTensor_ordered_correct,
                Lowering.reduceTensor_correct,
                Lowering.reduceNonemptyTensor_correct,
                Lowering.einsumTensor_correct,
                Lowering.einsumTensorKernel_correct,
                Lowering.packTensor_correct,
                Lowering.unpackTensor_correct,
                Semantics.inverse_denoteRearrange,
                Semantics.map_denoteRearrange,
                Semantics.zipWith_denoteRearrange,
                Semantics.sum_denoteRearrange,
                Semantics.dot_denoteRearrange,
                Semantics.dot_denoteRearrange_denoteRearrange,
                Semantics.push_denoteRepeat,
                Semantics.sum_denoteRepeat,
                Semantics.dot_denoteRepeat,
                Semantics.sum_denoteReduce_sum,
                Semantics.dot_denoteReduce_sum,
                Semantics.dot_map_snd_denoteReduce_prod_dualNumber,
                Semantics.sum_minEqualShareTieWeight,
                Semantics.sum_maxEqualShareTieWeight,
                Semantics.dot_equalShareTieReduce,
                Semantics.dot_denoteMeanReduce,
                Semantics.denoteEinsum_update_add,
                Semantics.dot_denoteEinsum_update_eq_dot_einsumOperandVjp,
                Semantics.sum_denoteEinsum,
                Semantics.denoteUnpack_denotePack,
                Semantics.denotePack_denoteUnpack,
                Semantics.dot_denotePack_denotePack,
                Lowering.dot_packTensor,
                Lowering.dot_unpackTensor,
                Rep.map_id,
                Rep.map_map,
                Rep.map_zipWith,
                Rep.zipWith_map_map,
                Rep.stack_apply,
                Rep.unstack_apply,
                Rep.map_stack,
                Rep.zipWith_stack,
                Rep.map_unstack,
                Rep.zipWith_unstack,
                Rep.pull_id,
                Rep.pull_comp,
                Rep.reindex_refl,
                Rep.reindex_trans,
                Rep.reindex_symm_reindex,
                Rep.map_reindex,
                Rep.zipWith_reindex,
                Rep.map_reshape,
                Rep.zipWith_reshape,
                Rep.reshape_rfl,
                Rep.reshape_symm_reshape,
                Rep.push_equiv,
                Rep.push_id,
                Rep.push_comp,
                Rep.sum_push,
                Rep.sum_reindex,
                Rep.dot_reindex_eq_dot_reindex_symm,
                Rep.dot_reindex_reindex,
                Rep.dot_push_eq_dot_pull,
                Rep.dot_productReduceDifferential_eq_dot_productReduceVjp,
                Rep.dot_equalShareTieDifferential_eq_dot_equalShareTieVjp,
                Fin.cases_zero,
                Fin.cases_succ,
                Fin.cases_succ',
                true_and,
              and_true] <;>
              simp only [Semantics.denoteEinsum_update_add] <;>
              rfl
            | grind)

end TorchLean.Tensor.Internal
