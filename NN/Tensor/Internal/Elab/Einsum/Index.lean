/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Elab.Einsum.Loop
public import NN.Tensor.Internal.Lowering.Einsum.Planning
public import Lean.Meta.Tactic.SplitIf
public meta import NN.Tensor.Internal.Elab.Einsum.Symbolic -- shake: keep
import NN.Tensor.Internal.Elab.Common

/-!
# Reflected einsum index construction

This module builds and certifies the coordinate and flat-index expressions used by
the generated einsum kernels.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- Build a natural-number product without emitting statically neutral arithmetic. -/
def natMulExpr (left right : Expr) : MetaM Expr := do
  let leftValue? ← getNatValue? left
  let rightValue? ← getNatValue? right
  match leftValue?, rightValue? with
  | some leftValue, some rightValue =>
      pure (mkNatLit (leftValue * rightValue))
  | some 0, _ | _, some 0 => pure (mkNatLit 0)
  | some 1, _ => pure right
  | _, some 1 => pure left
  | _, _ => mkAppM ``Nat.mul #[left, right]

/-- Build a natural-number sum without emitting statically neutral arithmetic. -/
def natAddExpr (left right : Expr) : MetaM Expr := do
  let leftValue? ← getNatValue? left
  let rightValue? ← getNatValue? right
  match leftValue?, rightValue? with
  | some leftValue, some rightValue =>
      pure (mkNatLit (leftValue + rightValue))
  | some 0, _ => pure right
  | _, some 0 => pure left
  | _, _ => mkAppM ``Nat.add #[left, right]

/-- Build the right-associated product used by `Shape.size`. -/
def shapeSizeExpr : List Expr → MetaM Expr
  | [] => pure (mkNatLit 1)
  | dimension :: dimensions => do
      mkAppM ``Nat.mul #[dimension, ← shapeSizeExpr dimensions]

/-- Compute a runtime stride while folding literal and neutral factors. -/
def runtimeShapeSizeExpr : List Expr → MetaM Expr
  | [] => pure (mkNatLit 1)
  | dimension :: dimensions => do
      natMulExpr dimension (← runtimeShapeSizeExpr dimensions)

/--
Use native index arithmetic only when an operand's complete concrete buffer
fits the platform word size. Symbolic shapes retain the general `Nat` kernel.
-/
def supportsNativeInputIndex (dimensions : List Expr) : MetaM Bool := do
  let size ← runtimeShapeSizeExpr dimensions
  match ← getNatValue? size with
  | some size => pure (size < USize.size)
  | none => pure false

/--
Report whether a reflected shape contains a dimension definitionally equal to
zero. The compiler uses this only for static dead-loop elimination; symbolic
dimensions continue through the fully general lowering.
-/
def hasConcreteZeroDimension (lengths : List Expr) : MetaM Bool := do
  for length in lengths do
    if let some 0 ← getNatValue? length then
      if ← withTransparency .all <| isDefEq length (mkNatLit 0) then
        return true
  return false

/--
Produce a portable native bound for a definitionally concrete loop length.

Every Lean target represents at least `2^32` native indices. Restricting this
optimization to that common range keeps generated code portable across
32-bit and 64-bit targets. The returned equality is erased, while the
literal `USize` survives as the executable loop bound. Symbolic and larger
lengths retain `Fin.foldl`.
-/
def nativeLoopBound? (length : Expr) :
    MetaM (Option (Expr × Expr)) := do
  let reducedLength ← withTransparency .all <| whnf length
  let some lengthValue ← getNatValue? reducedLength
    | return none
  if length.hasFVar then
    return none
  let lengthLiteral := mkNatLit lengthValue
  unless ← withTransparency .all <| isDefEq length lengthLiteral do
    return none
  let portableLimit : Nat := 2 ^ 32
  if portableLimit ≤ lengthValue then
    return none
  let hPortable ←
    mkDecideProof (← mkLT lengthLiteral (mkNatLit portableLimit))
  let bound ← mkNumeral (mkConst ``USize) lengthValue
  let hBound ←
    mkAppM ``USize.toNat_ofNat_of_lt_32 #[hPortable]
  let hBound ←
    withTransparency .all <|
      mkExpectedTypeHint hBound
        (← mkEq (← mkAppM ``USize.toNat #[bound]) length)
  return some (bound, hBound)

/--
Return the operands of a direct or notation-elaborated binary `Nat`
operation.

Metaprogram-generated arithmetic can retain either kernel representation
after proof-producing simplification. Recognizing both keeps later native
lowering independent of that representation choice.
-/
def natOperationOperands?
    (direct heterogeneous : Name) (value : Expr) :
    Option (Expr × Expr) := do
  let value := value.consumeMData
  if value.isAppOfArity direct 2 then
    let arguments := value.getAppArgs
    return (arguments[0]!, arguments[1]!)
  guard <| value.isAppOfArity heterogeneous 6
  let arguments := value.getAppArgs
  guard <| arguments[0]!.consumeMData.isConstOf ``Nat
  guard <| arguments[1]!.consumeMData.isConstOf ``Nat
  guard <| arguments[2]!.consumeMData.isConstOf ``Nat
  return (arguments[4]!, arguments[5]!)

/--
Translate a concrete row-major `Nat` expression to platform-native arithmetic
and report every addition or multiplication that could wrap.

The reported natural-number expressions let callers certify intermediate
smallness from their loop bounds before proving that the complete native
expression has the intended value.
-/
partial def nativeIndexValueWithIntermediates
    (value : Expr) : MetaM (Expr × List Expr) := do
  let value := value.consumeMData
  if let some literal ← getNatValue? value then
    return (← mkNumeral (mkConst ``USize) literal, [])
  if value.isAppOfArity ``USize.toNat 1 then
    return (value.getAppArgs[0]!, [])
  if let some (left, right) :=
      natOperationOperands? ``Nat.add ``HAdd.hAdd value then
    let (nativeLeft, leftIntermediates) ←
      nativeIndexValueWithIntermediates left
    let (nativeRight, rightIntermediates) ←
      nativeIndexValueWithIntermediates right
    let nativeValue ← mkAdd nativeLeft nativeRight
    let intermediates :=
      leftIntermediates ++ rightIntermediates ++ [value]
    return (nativeValue, intermediates)
  if let some (left, right) :=
      natOperationOperands? ``Nat.mul ``HMul.hMul value then
    let (nativeLeft, leftIntermediates) ←
      nativeIndexValueWithIntermediates left
    let (nativeRight, rightIntermediates) ←
      nativeIndexValueWithIntermediates right
    let nativeValue ← mkMul nativeLeft nativeRight
    let intermediates :=
      leftIntermediates ++ rightIntermediates ++ [value]
    return (nativeValue, intermediates)
  if let some (left, right) :=
      natOperationOperands? ``Nat.div ``HDiv.hDiv value then
    let (nativeLeft, leftIntermediates) ←
      nativeIndexValueWithIntermediates left
    let (nativeRight, rightIntermediates) ←
      nativeIndexValueWithIntermediates right
    let nativeValue ←
      mkAppM ``HDiv.hDiv #[nativeLeft, nativeRight]
    return (nativeValue, leftIntermediates ++ rightIntermediates)
  if let some (left, right) :=
      natOperationOperands? ``Nat.mod ``HMod.hMod value then
    let (nativeLeft, leftIntermediates) ←
      nativeIndexValueWithIntermediates left
    let (nativeRight, rightIntermediates) ←
      nativeIndexValueWithIntermediates right
    let nativeValue ←
      mkAppM ``HMod.hMod #[nativeLeft, nativeRight]
    return (nativeValue, leftIntermediates ++ rightIntermediates)
  if let some (left, right) :=
      natOperationOperands? ``Nat.sub ``HSub.hSub value then
    let (nativeLeft, leftIntermediates) ←
      nativeIndexValueWithIntermediates left
    let (nativeRight, rightIntermediates) ←
      nativeIndexValueWithIntermediates right
    let nativeValue ←
      mkAppM ``HSub.hSub #[nativeLeft, nativeRight]
    return (nativeValue, leftIntermediates ++ rightIntermediates)
  return (← mkAppM ``Nat.toUSize #[value], [])

/--
Translate a concrete row-major `Nat` expression to platform-native arithmetic.

Literal factors remain native constants. Dynamic leaves are converted when
they reach the generated expression, while additions and multiplications run
as `USize` operations. Division, remainder, subtraction, and unsupported
forms remain one compact `Nat.toUSize` leaf. A separate erased certificate
proves that this modular expression has exactly the original natural-number
value.

This compact translation is used by ordinary einsum operands. Native
operation lowerings that provide explicit loop invariants use
`certifiedNativeIndexValue?` instead, which can certify the larger arithmetic
vocabulary compositionally.
-/
partial def nativeIndexValue (value : Expr) : MetaM Expr := do
  let value := value.consumeMData
  if let some literal ← getNatValue? value then
    return ← mkNumeral (mkConst ``USize) literal
  if value.isAppOfArity ``USize.toNat 1 then
    return value.getAppArgs[0]!
  if let some (left, right) :=
      natOperationOperands? ``Nat.add ``HAdd.hAdd value then
    return ← mkAdd
      (← nativeIndexValue left)
      (← nativeIndexValue right)
  if let some (left, right) :=
      natOperationOperands? ``Nat.mul ``HMul.hMul value then
    return ← mkMul
      (← nativeIndexValue left)
      (← nativeIndexValue right)
  mkAppM ``Nat.toUSize #[value]

/--
Project one axis value from the nested product representation of `Coord`.

Literal einsums use these direct projections inside the generated nested
loops, avoiding division and remainder operations in the scalar kernel.
-/
def coordinateAxisExpr : Nat → Expr → MetaM Expr
  | 0, coordinate => withTransparency .all do
      mkAppM ``Fin.val #[← mkAppM ``Prod.fst #[coordinate]]
  | axisPosition + 1, coordinate => withTransparency .all do
      coordinateAxisExpr axisPosition <|
        ← mkAppM ``Prod.snd #[coordinate]

/--
Expand row-major unlinearization into bounded quotient/remainder components.

Binding these components before constructing the temporary proof-level
coordinate lets native code decode each output axis once without allocating
the nested `Prod` representation of `Coord`.
-/
def coordinateComponentsFromFlatIndex :
    List Expr → Expr → MetaM (List Expr)
  | [], _ => pure []
  | dimension :: dimensions, flatIndex => withTransparency .all do
      let tailSize ← shapeSizeExpr dimensions
      let headCoordinate ←
        mkAppOptM ``Fin.divNat #[
          some dimension, some tailSize, some flatIndex]
      let tailFlatIndex ←
        mkAppOptM ``Fin.modNat #[
          some dimension, some tailSize, some flatIndex]
      let tailCoordinates ←
        coordinateComponentsFromFlatIndex dimensions tailFlatIndex
      pure (headCoordinate :: tailCoordinates)

/--
Construct the operand index produced by repeatedly unfolding `Fin.foldl`.

The first operand is `0`; each later operand is one more `Fin.succ` around
zero in the remaining finite type. Matching this canonical form lets
generated scalar products be certified directly by the standard fold laws.
-/
def foldOperandIndexExpr
    (operandCount operandIndex : Nat) : MetaM Expr := do
  unless operandIndex < operandCount do
    throwError
      "internal error: einsum operand index {operandIndex} is outside \
        operand count {operandCount}"
  let remainingCount := operandCount - operandIndex
  let remainingType ← mkAppM ``Fin #[mkNatLit remainingCount]
  let mut operand ← mkNumeral remainingType 0
  for _ in [:operandIndex] do
    operand ← mkAppM ``Fin.succ #[operand]
  return operand

/-- Rebuild the nested coordinate representation from bounded components. -/
def coordinateFromComponents : List Expr → MetaM Expr
  | [] =>
      pure (mkConst ``PUnit.unit [Level.succ Level.zero])
  | coordinate :: coordinates => do
      mkAppM ``Prod.mk #[
        coordinate, ← coordinateFromComponents coordinates]

/--
Combine componentwise coordinate equalities into equality of nested `Coord`
values.
-/
def coordinateFromComponentsEquality :
    List Expr → List Expr → List Expr → MetaM Expr
  | [], [], [] =>
      mkEqRefl (mkConst ``PUnit.unit [Level.succ Level.zero])
  | nativeHead :: nativeTail, semanticHead :: semanticTail,
      hHead :: hTail => do
      let nativeTailCoordinate ← coordinateFromComponents nativeTail
      let hTailCoordinate ←
        coordinateFromComponentsEquality nativeTail semanticTail hTail
      let headType ← inferType nativeHead
      let replaceHead ←
        withLocalDeclD `headCoordinate headType fun headCoordinate => do
          let coordinate ←
            mkAppM ``Prod.mk #[headCoordinate, nativeTailCoordinate]
          mkLambdaFVars #[headCoordinate] coordinate
      let hReplaceHead ← mkAppM ``congrArg #[replaceHead, hHead]
      let tailType ← inferType nativeTailCoordinate
      let replaceTail ←
        withLocalDeclD `tailCoordinate tailType fun tailCoordinate => do
          let coordinate ←
            mkAppM ``Prod.mk #[semanticHead, tailCoordinate]
          mkLambdaFVars #[tailCoordinate] coordinate
      let hReplaceTail ←
        mkAppM ``congrArg #[replaceTail, hTailCoordinate]
      mkAppM ``Eq.trans #[hReplaceHead, hReplaceTail]
  | _, _, _ =>
      throwError
        "internal error: generated coordinate components and equalities have \
          different lengths"

/--
Introduce generated values as ordinary `let` bindings and pass their local
variables to the body in source order. Both returned expressions receive the
same bindings, allowing an optimized value and its correctness proof to be
constructed together.
-/
def withGeneratedLetPair (bindings : List (Name × Expr))
    (body : List Expr → TermElabM (Expr × Expr)) :
    TermElabM (Expr × Expr) := do
  let rec
    /-- Traverse pending bindings while retaining their local variables in source order. -/
    visit (remaining : List (Name × Expr))
      (values : List Expr) : TermElabM (Expr × Expr) := do
    match remaining with
    | [] => body values
    | (name, value) :: remaining =>
        withLetDecl name (← inferType value) value fun localValue => do
          let (result, correctness) ←
            visit remaining (values.concat localValue)
          let result ←
            mkLetFVars
              (generalizeNondepLet := false) #[localValue] result
          let correctness ←
            mkLetFVars
              (generalizeNondepLet := false) #[localValue] correctness
          return (result, correctness)
  visit bindings []

/-- Abstract one ordinary local while retaining the body's generated lets. -/
def mkLambdaPreservingLets (localExpr body : Expr) : MetaM Expr := do
  let declaration ← getFVarLocalDecl localExpr
  let body ← body.abstractM #[localExpr]
  return .lam declaration.userName declaration.type body declaration.binderInfo

/--
Expose the leading lets of a generated expression as local declarations.

The optimized expression and its certificate are wrapped in the same lets
after compilation, preserving loop-invariant values outside nested folds.
The continuation also receives the opened locals so a caller can selectively
unfold generated aliases without enabling unrestricted zeta reduction.
-/
partial def withLeadingLetPair (value : Expr)
    (body : Array Expr → Expr → TermElabM (Expr × Expr)) :
    TermElabM (Expr × Expr) := do
  let rec
    /-- Open the let chain while retaining its locals in source order. -/
    visit (remaining : Expr) (locals : Array Expr) :
        TermElabM (Expr × Expr) := do
      match remaining with
      | .letE name type assignment letBody _ =>
          withLetDecl name type assignment fun localValue => do
            let (result, correctness) ←
              visit (letBody.instantiate1 localValue)
                (locals.push localValue)
            let result ←
              mkLetFVars
                (generalizeNondepLet := false) #[localValue] result
            let correctness ←
              mkLetFVars
                (generalizeNondepLet := false) #[localValue] correctness
            return (result, correctness)
      | _ => body locals remaining
  visit value #[]

/--
A symbolic `Fin` coordinate is zero when its dimension is known locally to be
one. This form lets the generated-kernel certifier simplify dependent
coordinates without first substituting through an entire tensor coordinate.
-/
theorem fin_val_eq_zero_of_eq_one
    {dimension : Nat} (coordinate : Fin dimension)
    (hDimension : dimension = 1) :
    coordinate.val = 0 := by
  subst dimension
  exact Fin.val_eq_zero coordinate

/--
Evaluating a conditionally selected operand plan is the same as selecting the
corresponding evaluated index. This exposes only plan-selection conditions to
the certifier, without distributing arbitrary applications over conditionals.
-/
theorem evaluateInputFlatIndexPlan_ite
    {condition : Prop} [Decidable condition]
    {outputShape contractedShape : Shape}
    (outputCoordinate : Coord outputShape)
    (contractionCoordinate : Coord contractedShape)
    (positivePlan negativePlan : List (Bool × Nat × Nat)) :
    Lowering.evaluateInputFlatIndexPlan outputShape contractedShape
        outputCoordinate contractionCoordinate
        (if condition then positivePlan else negativePlan) =
      if condition then
        Lowering.evaluateInputFlatIndexPlan outputShape contractedShape
          outputCoordinate contractionCoordinate positivePlan
      else
        Lowering.evaluateInputFlatIndexPlan outputShape contractedShape
          outputCoordinate contractionCoordinate negativePlan := by
  split <;> rfl

/--
Prove that generated row-major arithmetic agrees with the verified generic
kernel after reducing the statically known pattern and operand family.
-/
def certifyCompiledKernel
    (description : String) (proposition : Expr) :
    TermElabM Expr := do
  unless proposition.isEq do
    throwError "internal error: expected an equality while proving {description}"
  let kernelGoal ← mkFreshExprSyntheticOpaqueMVar proposition
  let mut simpTheorems ← getSimpTheorems
  for declaration in #[
      ``Lowering.inputFlatIndexPlan,
      ``Lowering.evaluateInputFlatIndexPlan,
      ``Lowering.coordinateAt,
      ``Lowering.coordinatePermutationEquiv,
      ``AxisTuple.selectEquiv,
      ``AxisTuple.select,
      ``AxisTuple.coordEquiv] do
    if let some equations ← getEqnsFor? declaration then
      for equation in equations do
        simpTheorems ← simpTheorems.addConst equation
    else
      simpTheorems := simpTheorems.addDeclToUnfoldCore declaration
  simpTheorems ←
    simpTheorems.addConst ``evaluateInputFlatIndexPlan_ite (post := false)
  for theoremName in #[
      ``Fin.val_eq_zero,
      ``fin_val_eq_zero_of_eq_one,
      ``Equiv.trans_apply,
      ``Equiv.prodCongr_apply,
      ``Fin.consEquiv_apply,
      ``Fin.consEquiv_symm_apply,
      ``Nat.add_assoc, ``Nat.add_comm, ``Nat.add_left_comm] do
    simpTheorems ← simpTheorems.addConst theoremName
  let simpContext ←
    Simp.mkContext
      (config := { zeta := true, failIfUnchanged := false })
      (simpTheorems := #[simpTheorems])
      (congrTheorems := ← getSimpCongrTheorems)
  let simprocs := #[← Simp.getSimprocs]
  let closeLeaf (goal : MVarId) : TermElabM Unit := do
    let (contextGoal?, _) ←
      simpAll goal simpContext (simprocs := simprocs)
    let some contextGoal := contextGoal?
      | return
    let omegaTactic ← `(tactic| omega)
    let remainingGoals ←
      Lean.Elab.Tactic.run contextGoal do
        Lean.Elab.Tactic.evalTactic omegaTactic
    unless remainingGoals.isEmpty do
      let target ← instantiateMVars (← remainingGoals[0]!.getType)
      throwError
        "internal error: could not prove {description}:\
          {indentExpr target}"
  let rec
    /-- Split generated conditionals before simplifying and closing each resulting leaf. -/
    closeGoal (fuel : Nat) (goal : MVarId) : TermElabM Unit := do
    let (remainingGoal?, _) ←
      simpTarget goal simpContext (simprocs := simprocs)
    let some remainingGoal := remainingGoal?
      | return
    match fuel with
    | 0 => closeLeaf remainingGoal
    | fuel + 1 =>
        if let some (positive, negative) ←
            splitIfTarget? remainingGoal (useNewSemantics := true) then
          closeGoal fuel positive.mvarId
          closeGoal fuel negative.mvarId
        else
          closeLeaf remainingGoal
  closeGoal proposition.sizeWithoutSharing kernelGoal.mvarId!
  let certificate ← instantiateMVars kernelGoal
  if certificate.hasMVar then
    throwError
      "internal error: the proof of {description} still contains metavariables"
  return certificate

/--
Certify a self-contained generated arithmetic fact after removing ambient
locals that cannot occur in its statement.

This is intentionally narrower than `certifyWithTactic`: callers must provide
a proposition whose proof needs no separate hypotheses from the surrounding
elaboration context. Pruning those hypotheses keeps erased certificates from
capturing unrelated contraction coordinates and blocking loop-invariant code
motion.
-/
private def certifySelfContained
    (description : String) (proposition : Expr) (tactic : Syntax) :
    TermElabM Expr := do
  let certificateGoal ← mkFreshExprSyntheticOpaqueMVar proposition
  let focusedGoal ← certificateGoal.mvarId!.withContext do
    let mut candidates : Array FVarId := #[]
    for localDecl in (← getLCtx) do
      candidates := candidates.push localDecl.fvarId
    return (← certificateGoal.mvarId!.tryClearMany' candidates).1
  let savedState ← saveState
  let remainingGoals ←
    try
      Lean.Elab.Tactic.run focusedGoal do
        let result ←
          Mathlib.Tactic.withResetServerInfo <|
            Lean.Elab.Tactic.evalTactic tactic
        if result.result?.isNone || result.msgs.hasErrors then
          throwError "generated certificate tactic failed"
    catch _ =>
      restoreState savedState
      pure [focusedGoal]
  unless remainingGoals.isEmpty do
    let target ← instantiateMVars (← remainingGoals[0]!.getType)
    throwError
      "internal error: could not prove {description}:{indentExpr target}"
  let certificate ← instantiateMVars certificateGoal
  if certificate.hasMVar then
    throwError
      "internal error: the proof of {description} still contains metavariables"
  if certificate.hasSorry then
    throwError
      "internal error: the proof of {description} contains a synthetic sorry"
  let proposition ← instantiateMVars proposition
  sealCertificate proposition certificate

/--
Certify that native word arithmetic has neither wrapped nor changed a
generated concrete input index.

Lean supports 32-bit and 64-bit `USize`; splitting that platform theorem lets
the simplifier expose the corresponding modulus before `omega` discharges the
coordinate bounds.
-/
def certifyNativeIndex (proposition : Expr) : TermElabM Expr := do
  let tactic ←
    `(tactic|
      rcases System.Platform.numBits_eq with hBits | hBits <;>
      simp_all (config := { zeta := true }) [hBits] <;>
      omega)
  certifySelfContained
    "that native einsum index arithmetic equals its natural-number reference"
    proposition tactic

/--
Certify that one generated addition or multiplication remains below the
native word modulus under explicit loop-index hypotheses.
-/
def certifyNativeIntermediateBound
    (value : Expr) (assumptions : Array Expr) : TermElabM Expr := do
  let mut proposition ← mkLT value (mkConst ``USize.size)
  for assumption in assumptions.reverse do
    proposition ← mkArrow (← inferType assumption) proposition
  let tactic ←
    `(tactic|
      (intros
       apply Nat.lt_of_lt_of_le ?_ USize.le_size
       simp only [Shape.size] at *
       omega))
  let certificate ←
    certifySelfContained
      "that an intermediate native index operation cannot wrap"
      proposition tactic
  let mut proof := certificate
  for assumption in assumptions do
    proof := mkApp proof assumption
  return proof

/--
Certify one generated arithmetic side condition under explicit loop-index
hypotheses.

The returned proof is closed over exactly the supplied assumptions. Native
subtraction uses this to establish that its segment prefix does not exceed the
selected packed-axis position.
-/
def certifyNativeArithmeticFact
    (description : String) (proposition : Expr)
    (assumptions : Array Expr) : TermElabM Expr := do
  let mut closedProposition := proposition
  for assumption in assumptions.reverse do
    closedProposition ← mkArrow (← inferType assumption) closedProposition
  let tactic ←
    `(tactic|
      first
      | (intros; assumption)
      | (intros; omega)
      | (intros; simp_all (config := { zeta := true }) <;> omega))
  let certificate ←
    certifySelfContained description closedProposition tactic
  let mut proof := certificate
  for assumption in assumptions do
    proof := mkApp proof assumption
  return proof

/--
Translate generated `Nat` arithmetic to native words while constructing its
value theorem one operation at a time.

This compositional route is used when a caller supplies explicit loop
invariants. It avoids asking one arithmetic tactic to rediscover every nested
machine-word nonwrapping fact after the complete expression has been built.
-/
partial def certifiedNativeIndexValue?
    (value : Expr) (assumptions : Array Expr) :
    TermElabM (Option (Expr × Expr)) := do
  let value := value.consumeMData
  if let some literal ← getNatValue? value then
    let portableLimit : Nat := 2 ^ 32
    unless literal < portableLimit do
      return none
    let nativeValue ← mkNumeral (mkConst ``USize) literal
    let hPortable ←
      mkDecideProof (← mkLT (mkNatLit literal) (mkNatLit portableLimit))
    let hValue ←
      mkAppM ``USize.toNat_ofNat_of_lt_32 #[hPortable]
    let hValue ←
      withTransparency .all <|
        mkExpectedTypeHint hValue
          (← mkEq (← mkAppM ``USize.toNat #[nativeValue]) value)
    return some (nativeValue, hValue)
  if value.isAppOfArity ``USize.toNat 1 then
    return some (value.getAppArgs[0]!, ← mkEqRefl value)
  if let some (left, right) :=
      natOperationOperands? ``Nat.add ``HAdd.hAdd value then
    let some (nativeLeft, hLeft) ←
        certifiedNativeIndexValue? left assumptions
      | return none
    let some (nativeRight, hRight) ←
        certifiedNativeIndexValue? right assumptions
      | return none
    let nativeValue ← mkAdd nativeLeft nativeRight
    let hBound ← certifyNativeIntermediateBound value assumptions
    let hValue ←
      mkAppM ``native_add_toNat_of_eq #[
        nativeLeft, nativeRight, left, right, hLeft, hRight, hBound]
    return some (nativeValue, hValue)
  if let some (left, right) :=
      natOperationOperands? ``Nat.mul ``HMul.hMul value then
    let some (nativeLeft, hLeft) ←
        certifiedNativeIndexValue? left assumptions
      | return none
    let some (nativeRight, hRight) ←
        certifiedNativeIndexValue? right assumptions
      | return none
    let nativeValue ← mkMul nativeLeft nativeRight
    let hBound ← certifyNativeIntermediateBound value assumptions
    let hValue ←
      mkAppM ``native_mul_toNat_of_eq #[
        nativeLeft, nativeRight, left, right, hLeft, hRight, hBound]
    return some (nativeValue, hValue)
  if let some (left, right) :=
      natOperationOperands? ``Nat.div ``HDiv.hDiv value then
    let some (nativeLeft, hLeft) ←
        certifiedNativeIndexValue? left assumptions
      | return none
    let some (nativeRight, hRight) ←
        certifiedNativeIndexValue? right assumptions
      | return none
    let nativeValue ←
      mkAppM ``HDiv.hDiv #[nativeLeft, nativeRight]
    let hValue ←
      mkAppM ``native_div_toNat_of_eq #[
        nativeLeft, nativeRight, left, right, hLeft, hRight]
    return some (nativeValue, hValue)
  if let some (left, right) :=
      natOperationOperands? ``Nat.mod ``HMod.hMod value then
    let some (nativeLeft, hLeft) ←
        certifiedNativeIndexValue? left assumptions
      | return none
    let some (nativeRight, hRight) ←
        certifiedNativeIndexValue? right assumptions
      | return none
    let nativeValue ←
      mkAppM ``HMod.hMod #[nativeLeft, nativeRight]
    let hValue ←
      mkAppM ``native_mod_toNat_of_eq #[
        nativeLeft, nativeRight, left, right, hLeft, hRight]
    return some (nativeValue, hValue)
  if let some (left, right) :=
      natOperationOperands? ``Nat.sub ``HSub.hSub value then
    let some (nativeLeft, hLeft) ←
        certifiedNativeIndexValue? left assumptions
      | return none
    let some (nativeRight, hRight) ←
        certifiedNativeIndexValue? right assumptions
      | return none
    let nativeValue ←
      mkAppM ``HSub.hSub #[nativeLeft, nativeRight]
    let hLe ←
      certifyNativeArithmeticFact
        "that a native segment offset does not exceed its packed position"
        (← mkLE right left) assumptions
    let hValue ←
      mkAppM ``native_sub_toNat_of_eq #[
        nativeLeft, nativeRight, left, right, hLeft, hRight, hLe]
    return some (nativeValue, hValue)
  return none

/--
Prove that a concrete generated row-major index lies inside its input buffer.

The index expression itself supplies every coordinate needed by the proof.
Discarding unrelated ambient locals ensures the erased certificate has the
same loop dependencies as the executable index.
-/
def certifyInputIndexBound (index inputSize : Expr) : TermElabM Expr := do
  let proposition ← mkLT index inputSize
  let tactic ←
    `(tactic|
      simp_all (config := { zeta := true }) [Shape.size] <;>
      omega)
  certifySelfContained
    "that a generated einsum input index is within its tensor buffer"
    proposition tactic

/--
Move leading lets from the final argument of an application around the
application itself.

The generated contraction loops have the form `total + body`. Exposing the
leading lets of `body` lets the loop pass below identify which index bases are
independent of the current loop coordinate.
-/
partial def exposeFinalArgumentLets
    (function argument : Expr) : MetaM Expr := do
  match argument with
  | .letE name type assignment body _ =>
      withLetDecl name type assignment fun localValue => do
        let result ←
          exposeFinalArgumentLets function (body.instantiate1 localValue)
        mkLetFVars
          (generalizeNondepLet := false) #[localValue] result
  | _ => pure (mkApp function argument)

/--
Partition a generated leading-let chain around one contraction fold.

Lets independent of the fold move outside it even when they follow a
coordinate-dependent let. A let depending on an earlier dependent local stays
inside as well, so the partition accounts for transitive dependencies while
preserving the original order within both groups.

Every rewrite is definitionally equal to its source. A value depending on an
outer contraction coordinate therefore moves across inner folds, but remains
inside the fold that introduces that coordinate.
-/
partial def withFoldInvariantLets
    (body : Expr) (foldLocals : Array Expr)
    (continuation : Expr → MetaM Expr) : MetaM Expr := do
  let dependsOn
      (type assignment : Expr) (locals : Array Expr) : Bool :=
    locals.any fun localValue =>
      let localId := localValue.fvarId!
      type.containsFVar localId || assignment.containsFVar localId
  let rec
    /--
    Open the complete chain before rebuilding it with invariant locals around
    the fold and dependent locals inside the callback.
    -/
    visit (remainingBody : Expr)
        (invariantLocals dependentLocals : Array Expr) : MetaM Expr := do
      match remainingBody with
      | .letE name type assignment letBody _ =>
          let isDependent :=
            dependsOn type assignment foldLocals ||
              dependsOn type assignment dependentLocals
          withLetDecl name type assignment fun localValue => do
            if isDependent then
              visit (letBody.instantiate1 localValue)
                invariantLocals (dependentLocals.push localValue)
            else
              visit (letBody.instantiate1 localValue)
                (invariantLocals.push localValue) dependentLocals
      | _ => do
          let innerBody ←
            mkLetFVars (generalizeNondepLet := false)
              dependentLocals remainingBody
          let result ← continuation innerBody
          mkLetFVars
            (generalizeNondepLet := false) invariantLocals result
  visit body #[] #[]

/--
Hoist maximal native-index fragments that are invariant under the current
contraction fold.

`nativeIndexValue` emits only native constants and arithmetic.
Lean's C compiler does not always move compound fragments of that tree out of
nested loops. Binding the largest independent fragment once removes repeated
stride arithmetic while preserving definitional equality. Recursing through
dependent let assignments lets an outer fold subsequently hoist the nested
base that is independent of its own coordinate. A bare stride product is not
bound: passing a singleton base through a generated callback is slower than
recomputing it in simple kernels.
-/
partial def withFoldInvariantNativeIndexExpressions
    (body : Expr) (foldLocals : Array Expr)
    (continuation : Expr → MetaM Expr) : MetaM Expr := do
  let isUSizeType (type : Expr) : Bool :=
    match type.consumeMData with
    | .const ``USize _ => true
    | _ => false
  let isShareableNativeIndexExpression (value : Expr) : Bool :=
    if value.isAppOfArity ``USize.ofNat 1 then
      true
    else if value.isAppOfArity ``HAdd.hAdd 6 then
      let arguments := value.getAppArgs
      isUSizeType arguments[0]! &&
        isUSizeType arguments[1]! &&
        isUSizeType arguments[2]!
    else
      false
  let isInvariantNativeIndexExpression (value : Expr) : Bool :=
    isShareableNativeIndexExpression value &&
      value.hasFVar &&
      !value.hasLooseBVars &&
      !foldLocals.any fun localValue =>
        value.containsFVar localValue.fvarId!
  let rec
    /-- Collect distinct maximal fragments that are independent of the current fold. -/
    collect (value : Expr)
      (expressions : Array Expr) : Array Expr :=
    let value := value.consumeMData
    if isInvariantNativeIndexExpression value then
      if expressions.any fun expression => expression == value then
        expressions
      else
        expressions.push value
    else
      match value with
      | .forallE _ domain body _ | .lam _ domain body _ =>
          collect body (collect domain expressions)
      | .letE _ type assignment body _ =>
          collect body (collect assignment (collect type expressions))
      | .app function argument =>
          collect argument (collect function expressions)
      | .proj _ _ value =>
          collect value expressions
      | _ => expressions
  let rec
    /-- Bind each shared fragment once and replace all of its occurrences in the body. -/
    bind (nextIndex : Nat) (expressions : List Expr)
      (body : Expr) : MetaM Expr := do
    match expressions with
    | [] => continuation body
    | expression :: expressions =>
        let name ←
          if expression.isAppOfArity ``USize.ofNat 1 then
            let argument := expression.getAppArgs[0]!
            match argument.consumeMData with
            | .fvar _ => do
                let declaration ← getFVarLocalDecl argument
                pure (declaration.userName.appendAfter "Native")
            | _ =>
                pure (Name.mkSimple s!"nativeIndexBase{nextIndex}")
          else
            pure (Name.mkSimple s!"nativeIndexBase{nextIndex}")
        withLetDecl name (mkConst ``USize) expression fun nativeValue => do
          let body :=
            body.replace fun candidate =>
              if candidate.consumeMData == expression then
                some nativeValue
              else
                none
          let result ← bind (nextIndex + 1) expressions body
          mkLetFVars
            (generalizeNondepLet := false) #[nativeValue] result
  bind 0 (collect body #[]).toList body

end TorchLean.Tensor.Internal.Elab.Impl
