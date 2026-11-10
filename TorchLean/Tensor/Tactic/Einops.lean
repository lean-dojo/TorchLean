/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Elab.Transform.View
public import TorchLean.Tensor.Einops.Lowering.Pack
public import TorchLean.Tensor.Einops.MixedRadix -- shake: keep
public import TorchLean.Tensor.Einops.Lowering.Reduce -- shake: keep
import Mathlib.Algebra.Order.Field.Basic
public meta import Mathlib.Util.SynthesizeUsing
public meta import TorchLean.Tensor.Tactic.Einops.Table
public meta import TorchLean.Tensor.Tactic.Init
public import Mathlib.Util.SynthesizeUsing
public import TorchLean.Tensor.Tactic.Einops.Table
public import TorchLean.Tensor.Tactic.Init

/-!
# Proof automation for verified tensor transformations

`einops` closes tensor-layout goals with compiler-correctness theorems and general tensor laws.
An equality between concrete rearrangements reduces to an equality of compact row-major index
maps. Other goals use one e-graph over the `einops` lemma set, a deterministic simplification pass
over `einops_simps`, and finally Mathlib's registered `grind` rules.

Every successful run produces an ordinary proof term checked by the kernel, with no native
evaluation in the trust path. The e-graph rewrites proof terms only, and executable tensors run
on the native `Array` lowerings.
-/

/-- If `size = bound * divisor`, then the digit `index / divisor` of an index below `size` is below
`bound`. -/
private theorem Fin.div_lt_of_size_eq_mul_bound
    {size divisor bound : Nat} (index : Fin size)
    (hSize : size = bound * divisor) :
    index.val / divisor < bound := by
  subst size
  exact Nat.div_lt_of_lt_mul (by simpa only [Nat.mul_comm] using index.isLt)

public meta section

namespace TorchLean.Einops

open Lean Elab Tactic Meta
open Tensor (Rep)
open TorchLean.Shape (Coord)

/-- Recognize a reflected rearrangement and return its plan, kind witness, and input. -/
private def rearrangeArguments? (expression : Expr) :
    Option (Expr × Expr × Expr) :=
  let expression := instantiateOuterLets expression
  if Lowering.rearrangeTensor.isApp expression then
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
  if Lowering.nativeTensorKernel.isApp expression then
    let arguments := expression.getAppArgs
    let some (coordinateMap, inputTensor, normalized, hReference) ←
        rearrangePullProgram? arguments[3]!
      | return none
    let hNative ←
      mkAppM Lowering.nativeTensorKernel.correctness #[
        arguments[3]!, arguments[4]!, arguments[5]!, arguments[6]!,
        arguments[7]!]
    let hNormalized ← mkAppM ``Eq.trans #[hNative, hReference]
    return some
      (coordinateMap, inputTensor, normalized, hNormalized)
  else if Lowering.rearrangeTensor.isApp expression then
    let arguments := expression.getAppArgs
    let checked := arguments[2]!
    let hKind := arguments[3]!
    let inputTensor := arguments[4]!
    let coordinateEquiv ←
      mkAppM ``Transform.rearrangeEquiv #[
        checked, hKind]
    let coordinateMap ← mkAppM ``Equiv.toFun #[coordinateEquiv]
    let normalized ←
      mkAppM ``Rep.pull #[coordinateMap, inputTensor]
    let correctness ←
      mkAppM Lowering.rearrangeTensor.correctness #[
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
  else if Lowering.transformTensorFused.isApp expression then
    let arguments := expression.getAppArgs
    let checked := arguments[3]!
    let hAxes := arguments[4]!
    let inputMap := arguments[5]!
    let inputTensor := arguments[8]!
    let checkedMap ←
      mkAppM ``Transform.inputOfOutput #[
        checked, hAxes]
    let coordinateMap ←
      mkAppM ``Function.comp #[inputMap, checkedMap]
    let normalized ←
      mkAppM ``Rep.pull #[coordinateMap, inputTensor]
    let hNormalized ←
      mkAppM Lowering.transformTensorFused.correctness #[
        checked, hAxes, inputMap, arguments[6]!, arguments[7]!, inputTensor]
    return some (coordinateMap, inputTensor, normalized, hNormalized)
  else
    return none

-- Reuse the elaborator's checked input and output shape view.
open Einops.Elab (checkedTransformShapes)

/--
Recover the scalar type, the static shape and the physical storage of a native tensor
expression: the three arguments of its type `Rep α s st`.
-/
private def repArguments (inputTensor : Expr) : MetaM (Expr × Expr × Expr) := do
  let inputType ← withTransparency .reducible <| whnf (← inferType inputTensor)
  unless inputType.isAppOfArity ``Rep 3 do
    throwError "expected a tensor input"
  let arguments := inputType.getAppArgs
  let scalarType := arguments[0]!
  if scalarType.hasLooseBVars then
    throwError "expected a tensor with one scalar type"
  return (scalarType, arguments[1]!, arguments[2]!)

/--
Normalize a tensor to the identity coordinate pullback.

With it the fused rearrangement prover can compare an optimized inverse chain with its original
tensor by the same row-major map criterion.
-/
private def identityPullProgram (inputTensor : Expr) :
    MetaM (Expr × Expr × Expr × Expr) := do
  let (_, inputShape, _) ← repArguments inputTensor
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

The matcher inspects only the head application. A composed map may contain several checked
transforms, and pairing an inner transform with the outer coordinate would produce an ill-typed
certificate. The focused simplifier exposes compositions first, then this matcher certifies each
`Equiv.toFun` application separately.
-/
private def checkedRearrangementFunction? (function : Expr) :
    Option (Expr × Expr) := do
  let function := instantiateOuterLets function.consumeMData
  guard <| function.isAppOfArity ``Equiv.toFun 3
  let coordinateEquiv :=
    instantiateOuterLets function.getAppArgs[2]!.consumeMData
  guard <| coordinateEquiv.isAppOfArity
    ``Transform.rearrangeEquiv 2
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
    ``Transform.inputOfOutput 2
  let arguments := function.getAppArgs
  some (arguments[0]!, arguments[1]!)

/--
Replace one checked rearrangement coordinate application with its certified
row-major index calculation.

The step fires as a `simp` pre-step, before Lean visits the large checked-plan value, so proof cost
scales with the compact index formula whatever the size of the parser and checker certificates
stored in the plan.
-/
private def simpCheckedLinearize
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
      mkAppM ``Coord.linearize_comp_apply_val #[
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
        ``Transform.inputOfOutput_linearize_coord
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
      ``Transform.rearrangeEquiv_linearize_coord
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
      ``Coord.linearize_cast_val,
      ``Equiv.cast_apply,
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
Normalize all checked coordinate maps in a compact-index goal and replace the target with the
result, justified by the simplifier's equality proof.
-/
private def normalizeLinearIndexGoal (goal : MVarId) : TacticM MVarId :=
  goal.withContext do
    let target ← instantiateMVars (← goal.getType)
    let context ← linearIndexSimpContext
    let (result, _) ←
      Simp.main target context
        (methods := {
          pre :=
            simpCheckedLinearize >>
              Simp.rewritePre
          post :=
            simpCheckedLinearize >>
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
Recover the operands of a natural-number `+`, `*`, `/` or `%` whose operation is listed in `names`.
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
private def collectNatBinaryApplications
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
termination_by structural expression

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

/--
Run `tactic` on a fresh goal for `proposition` without disturbing the caller's goal list, and
return the closed proof term, or `none` when the tactic fails or leaves goals open.
-/
private def certificate? (proposition : Expr) (tactic : TacticM Unit) :
    TacticM (Option Expr) := do
  let result? ←
    try
      let result ← synthesizeUsing (u := .zero) proposition tactic
      pure (some result)
    catch
      | .error _ _ => pure none
      | error => throw error
  let some (goals, certificate) := result?
    | return none
  unless goals.isEmpty do return none
  ensureHasNoMVars certificate
  return some certificate

/-- Ask Omega for a closed certificate without disturbing the caller's goal list. -/
private def omegaCertificate? (proposition : Expr) : TacticM (Option Expr) :=
  certificate? proposition do
    setGoals [← clearNonPropositionalLocals (← getMainGoal)]
    evalTactic <| ← `(tactic| omega)

/--
Ask the default simplifier for a closed proof of `proposition`.

Static shape-size equalities need it, because definitional reduction can stop at identities such
as `0 + dimension`.
-/
private def simpCertificate? (proposition : Expr) : TacticM (Option Expr) :=
  certificate? proposition do
    evalTactic <| ← `(tactic| simp)

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

Every fact is proved before it enters the main goal: a remainder bound by `Nat.mod_lt`, a digit
of the output index by `Fin.div_lt_of_size_eq_mul_bound`, and the rest by Omega, first as one
conjunction and then in repeated passes. Each pass can use the bounds proved for inner digits, so
the finite index is never enumerated.
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

  -- Most concrete layouts expose all digit bounds at once, so one Omega call on their
  -- conjunction replaces a call per quotient layer.
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
                    Rearrangement.linearIndex,
                    TransformPlan.axisLength,
                    TransformPlan.normalized,
                    NormalForm.inputAxes,
                    NormalForm.outputAxes,
                    PartialAxisLengths.set,
                    PartialAxisLengths.seed])
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
Close an equality involving a fused rearrangement by comparing its complete
output-to-source map with the map on the other side.

The arithmetic certificate goes to `Rep.pull_eq_of_linearIndex_eq`, and the lowering-correctness
equalities of both sides carry the result back to the original goal.
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
    let (scalarType, sourceShape, storage) ← repArguments leftInput
    let (_, outputShape, _) ← repArguments leftNormalized
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
Close an equality between two literal rearrangements of the same input, or between a composition
of two rearrangements and a single one, by comparing their compact row-major index maps. The
criterion theorem receives the reflected plans as arguments, which avoids unification against
dependent proof fields.
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
      let (scalarType, _, _) ← repArguments leftInput
      let criterion ←
        mkAppOptM ``Lowering.rearrange_eq_of_linearIndex #[
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
      let (scalarType, _, _) ← repArguments inputTensor
      let criterion ←
        mkAppOptM
          ``Lowering.rearrange_comp_of_linearIndex #[
            some scalarType, some innerPlan, some leftPlan, some rightPlan,
            some hInnerKind, some hLeftKind, some hRightKind,
            some hMiddleShape, some hInputShape, some hOutputShape]
      closeWithLinearIndex goal target criterion inputTensor
      return true

/--
Whether a target mentions a constant from `Tensor.Rep`, `Semantics` or `Lowering`, the terms that
the `einops` theorem set handles.

The guard keeps the final `grind` passes from acting as a general hypothesis-closing tactic on
unrelated propositions.
-/
private def containsTensorTerm (target : Expr) : Bool :=
  (target.find? fun expression =>
    match expression.getAppFn with
    | .const name _ =>
        (`TorchLean.Tensor.Rep).isPrefixOf name ||
          (`TorchLean.Einops.Semantics).isPrefixOf name ||
          (`TorchLean.Einops.Lowering).isPrefixOf name ||
          (`TorchLean.Einops.Pack).isPrefixOf name ||
          name == `TorchLean.Shape.Coord.reshapeEquiv ||
          name == `TorchLean.Shape.Coord.concatEquiv ||
          -- the operations that the `Tensor.Rep` modules define under their public names
          [``TorchLean.Tensor.full, ``TorchLean.Tensor.map, ``TorchLean.Tensor.zipWith,
            ``TorchLean.Tensor.zipWith3, ``TorchLean.Tensor.dim, ``TorchLean.Tensor.unstack,
            ``TorchLean.Tensor.set, ``TorchLean.Tensor.modify, ``TorchLean.Tensor.foldl,
            ``TorchLean.Tensor.cast].contains name
    | _ => false).isSome

/-- Whether a target contains an unpack operation that may expose a pack inverse. -/
private def containsUnpackTerm (target : Expr) : Bool :=
  (target.find? fun expression =>
    let headName? := expression.getAppFn.constName?
    headName? == some Lowering.unpackTensor.head ||
      headName? == some Lowering.unpackTensor.denotation).isSome

/--
Close a tensor-layout identity with the einops compiler-correctness theorems, the library's
tensor algebra and the registered `grind` rules.
-/
syntax (name := einopsTactic) "einops" : tactic

elab_rules : tactic
  | `(tactic| einops) => focus do
      evalTactic <| ←
        `(tactic| dsimp (config := { failIfUnchanged := false })
          only [id, TorchLean.Tensor.castShape])
      if (← getGoals).isEmpty then
        return
      unless ← closeFusedPullbacks do
        unless ← closeEquivalentRearrangements do
          let normalizeUnpack ← withMainContext do
            containsUnpackTerm <$> instantiateMVars (← getMainTarget)
          if normalizeUnpack then
            evalTactic <| ←
              `(tactic|
                try simp only [
                  $(mkIdent Lowering.nativeTensorKernel.correctness):term,
                  Lowering.unpack_pack,
                  Lowering.pack_unpack,
                  Fin.cases_zero,
                  Fin.cases_succ,
                  true_and,
                  and_true])
          else
            evalTactic <| ←
              `(tactic|
                try rw [$(mkIdent Lowering.nativeTensorKernel.correctness):term])
          if (← getGoals).isEmpty then
            return
          withMainContext do
            let target ← instantiateMVars (← getMainTarget)
            unless containsTensorTerm target do
              throwError
                "`einops` found no tensor, semantic operation, or lowering in the target"
          evalTactic <| ←
            `(tactic|
            solve
            | (with_unfolding_all
                grw (transparency := all) [
                  $(mkIdent Lowering.rearrangeTensor.correctness):term]
               with_unfolding_all
                exact Semantics.sum_denoteRearrange _ _ _)
            | (with_unfolding_all
                grw (transparency := all) [
                  $(mkIdent Lowering.repeatTensor.correctness):term]
               with_unfolding_all
                exact Semantics.sum_denoteExpand _ _ _)
            | grind only [$(mkIdent `einops):term]
            | simp (config := { dsimp := false })
                (discharger := assumption) only [
                einops_simps,
                Semantics.sum_equalShareTieWeight Reduction.min Reduction.min_mem,
                Semantics.sum_equalShareTieWeight Reduction.max Reduction.max_mem,
                Fin.cases_zero,
                Fin.cases_succ,
                Fin.cases_succ',
                true_and,
              and_true] <;>
              simp only [Semantics.denoteEinsum_update_add] <;>
              rfl
            | grind)

end TorchLean.Einops
