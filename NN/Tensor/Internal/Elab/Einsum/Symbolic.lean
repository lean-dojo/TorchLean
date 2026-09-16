/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tensor.Internal.Elab.Common
public import NN.Tensor.Internal.Elab.Common

/-!
# Symbolic einsum certificates

This module constructs `Check.CheckedEinsum` values when tensor dimensions are
symbolic `Nat` expressions rather than reducible literals. The pattern and
operand ranks are still checked by computation. Repeated labels,
singleton broadcasting, logical-axis sources, and the expected output shape
are then proved against the original symbolic dimensions.

The resulting certificate feeds the same semantic lowering and native kernel
compiler used for concrete shapes. Symbolic dimensions change how the checked
plan is built, not what einsum means.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.Impl

open Lean
open Lean.Elab
open Lean.Elab.Term
open Lean.Meta

/-- Find the symbolic length assigned to a reflected einsum axis. -/
def einsumAxisExpression?
    (assignments : List (Check.EinsumAxis × Expr))
    (axis : Check.EinsumAxis) : Option Expr :=
  (assignments.find? fun assignment => assignment.1 == axis).map Prod.snd

/--
Discharge one symbolic einsum dimension obligation without unfolding the
parser, ellipsis expansion, or complete checked-plan API.
-/
private def certifySymbolicEinsumInvariant
    (description : String) (proposition : Expr) : TermElabM Expr := do
  let tactic ←
    `(tactic|
      simp (config := { zeta := true }) [
        Check.repeatedEinsumDimensionsAgree] <;>
      first
      | assumption
      | rfl
      | omega)
  certifyWithTactic description proposition tactic

/--
Record the first symbolic length chosen for an axis, preserving global-axis
order and preventing later occurrences from silently changing it.
-/
private def appendEinsumAxisExpression
    (assignments : List (Check.EinsumAxis × Expr))
    (axis : Check.EinsumAxis) (length : Expr) :
    List (Check.EinsumAxis × Expr) :=
  if (einsumAxisExpression? assignments axis).isSome then
    assignments
  else
    assignments.concat (axis, length)

/--
Resolve a logical einsum length from all of its physical occurrences.

Definitionally singleton dimensions are ignored. Definitionally equal
symbolic dimensions retain their original expression. Only genuinely
ambiguous symbolic broadcasting introduces a conditional that selects the
first non-singleton occurrence.
-/
private def symbolicBroadcastAxisLength (dimensions : List Expr) :
    MetaM Expr := do
  let one := mkNatLit 1
  let mut nonSingleton : List Expr := []
  for dimension in dimensions do
    unless ← withTransparency .reducible <| isDefEq dimension one do
      nonSingleton := nonSingleton.concat dimension
  let some first := nonSingleton.head?
    | return one
  let mut resolved := first
  for dimension in nonSingleton.drop 1 do
    unless ← withTransparency .reducible <| isDefEq resolved dimension do
      let resolvedIsSingleton ← mkEq resolved one
      resolved ← mkAppM ``ite #[resolvedIsSingleton, dimension, resolved]
  return resolved

/--
Build the dependent axis-length function stored in a symbolic einsum
certificate.
-/
private def symbolicEinsumAxisLengthExpr
    (assignments : List (Check.EinsumAxis × Expr)) :
    MetaM Expr := do
  withLocalDeclD `axis (mkConst ``Check.EinsumAxis) fun axis => do
    let mut length := mkNatLit 1
    for assignment in assignments.reverse do
      let axisMatches ← mkEq axis (Lean.toExpr assignment.1)
      length ← mkAppM ``ite #[axisMatches, assignment.2, length]
    mkLambdaFVars #[axis] length

/--
Assemble a `List.Forall₂` certificate from certificates for corresponding
elements.
-/
private def mkForall₂ (leftType rightType relation : Expr)
    (headCertificates : List Expr) : MetaM Expr := do
  let mut certificate ←
    mkAppOptM ``List.Forall₂.nil
      #[some leftType, some rightType, some relation]
  for headCertificate in headCertificates.reverse do
    certificate ←
      mkAppOptM ``List.Forall₂.cons #[
        some leftType, some rightType, some relation,
        none, none, none, none, some headCertificate, some certificate]
  return certificate

/--
Construct the operand-level relation stating that repeated labels select equal
physical dimensions.
-/
private def repeatedEinsumDimensionsRelation : MetaM Expr := do
  let shapeType ← mkAppM ``List #[mkConst ``Nat]
  let axesType ←
    mkAppM ``List #[mkConst ``Check.EinsumAxis]
  withLocalDeclD `inputShape shapeType fun inputShape =>
    withLocalDeclD `axes axesType fun axes => do
      let agreement ←
        mkAppM ``Check.repeatedEinsumDimensionsAgree
          #[axes, inputShape]
      let proposition ← mkEq agreement (Lean.toExpr true)
      mkLambdaFVars #[inputShape, axes] proposition

/--
Certify repeated-label agreement independently for each symbolic operand, then
package the results into the checker invariant.
-/
private def certifyRepeatedDimensions
    (pattern inputShapes : Expr)
    (inputAxes : List (List Check.EinsumAxis))
    (inputDimensions : List (List Expr)) : TermElabM Expr := do
  let mut operandCertificates : List Expr := []
  for (axes, dimensions) in inputAxes.zip inputDimensions do
    unless axes.length = dimensions.length do
      throwError
        "internal error: symbolic einsum axes and dimensions have different ranks"
    let axesExpr := Lean.toExpr axes
    let inputShape ← shapeExpr dimensions
    let agreement ←
      mkAppM ``Check.repeatedEinsumDimensionsAgree
        #[axesExpr, inputShape]
    let proposition ← mkEq agreement (Lean.toExpr true)
    let certificate ←
      certifySymbolicEinsumInvariant
        "that repeated labels in one einsum operand have equal dimensions"
        proposition
    operandCertificates := operandCertificates.concat certificate
  let shapeType ← mkAppM ``List #[mkConst ``Nat]
  let axesType ←
    mkAppM ``List #[mkConst ``Check.EinsumAxis]
  let relation ← repeatedEinsumDimensionsRelation
  let certificate ←
    mkForall₂ shapeType axesType relation operandCertificates
  let expected ←
    mkAppM ``Check.EinsumRepeatedDimensions #[pattern, inputShapes]
  let certificateType ← inferType certificate
  unless ← withTransparency .all <| isDefEq certificateType expected do
    throwError
      "internal error: symbolic repeated-dimension proof has type\
        {indentExpr certificateType}\nexpected{indentExpr expected}"
  sealCertificate expected certificate

/--
Construct the per-dimension broadcasting relation: a physical dimension is
either the logical axis length or the singleton length.
-/
private def einsumInputDimensionRelation (axisLength : Expr) :
    MetaM Expr := do
  withLocalDeclD `dimension (mkConst ``Nat) fun dimension =>
    withLocalDeclD `axis (mkConst ``Check.EinsumAxis) fun axis => do
      let resolvedLength := mkApp axisLength axis
      let dimensionMatches ← mkEq dimension resolvedLength
      let singleton ← mkEq dimension (mkNatLit 1)
      let proposition ← mkAppM ``Or #[dimensionMatches, singleton]
      mkLambdaFVars #[dimension, axis] proposition

/--
Certify that every symbolic operand dimension broadcasts to its resolved
logical axis.
-/
private def certifyInputBroadcasting
    (pattern inputShapes axisLength : Expr)
    (inputAxes : List (List Check.EinsumAxis))
    (inputDimensions : List (List Expr)) : TermElabM Expr := do
  let dimensionRelation ← einsumInputDimensionRelation axisLength
  let mut operandCertificates : List Expr := []
  for (axes, dimensions) in inputAxes.zip inputDimensions do
    unless axes.length = dimensions.length do
      throwError
        "internal error: symbolic einsum axes and dimensions have different ranks"
    let mut dimensionCertificates : List Expr := []
    for (axis, dimension) in axes.zip dimensions do
      let resolvedLength := mkApp axisLength (Lean.toExpr axis)
      let dimensionMatches ← mkEq dimension resolvedLength
      let singleton ← mkEq dimension (mkNatLit 1)
      let proposition ← mkAppM ``Or #[dimensionMatches, singleton]
      let certificate ←
        certifySymbolicEinsumInvariant
          "that an einsum input dimension broadcasts to its logical axis"
          proposition
      dimensionCertificates := dimensionCertificates.concat certificate
    let operandCertificate ←
      mkForall₂ (mkConst ``Nat) (mkConst ``Check.EinsumAxis)
        dimensionRelation dimensionCertificates
    operandCertificates := operandCertificates.concat operandCertificate
  let shapeType ← mkAppM ``List #[mkConst ``Nat]
  let axesType ←
    mkAppM ``List #[mkConst ``Check.EinsumAxis]
  let outerRelation ←
    withLocalDeclD `inputShape shapeType fun inputShape =>
      withLocalDeclD `axes axesType fun axes => do
        let proposition ←
          mkAppM ``List.Forall₂
            #[dimensionRelation, inputShape, axes]
        mkLambdaFVars #[inputShape, axes] proposition
  let certificate ←
    mkForall₂ shapeType axesType outerRelation operandCertificates
  let expected ←
    mkAppM ``Check.EinsumInputDimensions
      #[pattern, inputShapes, axisLength]
  let certificateType ← inferType certificate
  unless ← withTransparency .all <| isDefEq certificateType expected do
    throwError
      "internal error: symbolic broadcasting proof has type\
        {indentExpr certificateType}\nexpected{indentExpr expected}"
  sealCertificate expected certificate

/--
An explicit occurrence certifies that a logical axis length has an input
source. Keeping the witness outside `List.any` lets symbolic elaboration prove
only the dimension equality instead of simplifying every earlier occurrence.
-/
theorem einsum_axis_source_of_occurrence
    (axisLength : Check.EinsumAxis → Nat)
    (axis : Check.EinsumAxis) (length : Nat)
    (before after : List (Check.EinsumAxis × Nat))
    (hLength : length = axisLength axis) :
    (axisLength axis == 1 ||
        (before ++ (axis, length) :: after).any fun occurrence =>
          occurrence.1 == axis &&
            occurrence.2 == axisLength axis) =
      true := by
  subst length
  simp

/--
Certify that each resolved logical length is either singleton or supplied by
an actual input occurrence.
-/
private def certifyAxisSources
    (pattern inputShapes axisLength : Expr)
    (inputAxes : List (List Check.EinsumAxis))
    (inputDimensions : List (List Expr))
    (globalAxes : List Check.EinsumAxis) : TermElabM Expr := do
  let axisType := Lean.mkConst ``Check.EinsumAxis
  let occurrenceType ←
    mkAppM ``Prod #[axisType, mkConst ``Nat]
  let occurrences :=
    (inputAxes.zip inputDimensions).flatMap fun operand =>
      operand.1.zip operand.2
  let occurrenceExpressions ←
    occurrences.mapM fun occurrence =>
      mkAppM ``Prod.mk #[Lean.toExpr occurrence.1, occurrence.2]
  let occurrencesExpr ←
    mkListLit occurrenceType occurrenceExpressions
  let axisPredicate ←
    withLocalDeclD `axis axisType fun axis => do
      let resolvedLength := mkApp axisLength axis
      let isSingleton ←
        mkAppM ``BEq.beq #[resolvedLength, mkNatLit 1]
      let occurrencePredicate ←
        withLocalDeclD `occurrence occurrenceType fun occurrence => do
          let occurrenceAxis ← mkAppM ``Prod.fst #[occurrence]
          let occurrenceLength ← mkAppM ``Prod.snd #[occurrence]
          let sameAxis ← mkAppM ``BEq.beq #[occurrenceAxis, axis]
          let sameLength ←
            mkAppM ``BEq.beq #[occurrenceLength, resolvedLength]
          let suppliesLength ←
            mkAppM ``Bool.and #[sameAxis, sameLength]
          mkLambdaFVars #[occurrence] suppliesLength
      let hasSource ←
        mkAppM ``List.any #[occurrencesExpr, occurrencePredicate]
      let justified ← mkAppM ``Bool.or #[isSingleton, hasSource]
      mkLambdaFVars #[axis] justified
  let sourceLengthTactic ←
    `(tactic|
      simp only [reduceCtorEq, ↓reduceIte] <;>
      first
      | assumption
      | rfl
      | (repeat' split <;> simp_all)
      | omega)
  let rec certifyAxes (axes : List Check.EinsumAxis) : TermElabM Expr := do
    let axesExpression := Lean.toExpr axes
    let directCheck ←
      mkAppM ``List.all #[axesExpression, axisPredicate]
    let expected ← mkEq directCheck (Lean.toExpr true)
    match axes with
    | [] =>
        withTransparency .all <|
          mkExpectedTypeHint (← mkEqRefl (Lean.toExpr true)) expected
    | axis :: remainingAxes =>
        let axisExpression := Lean.toExpr axis
        let axisCheck := mkApp axisPredicate axisExpression
        let axisProposition ← mkEq axisCheck (Lean.toExpr true)
        let resolvedLength := mkApp axisLength axisExpression
        let mut exactCandidates : List
            ((Check.EinsumAxis × Expr) × Nat) := []
        let mut remainingCandidates : List
            ((Check.EinsumAxis × Expr) × Nat) := []
        for candidate in occurrences.zipIdx do
          if candidate.1.1 == axis then
            if ← withTransparency .reducible <|
                isDefEq candidate.1.2 resolvedLength then
              exactCandidates := exactCandidates.concat candidate
            else
              remainingCandidates := remainingCandidates.concat candidate
        let candidates := exactCandidates ++ remainingCandidates
        let mut axisCertificate? : Option Expr := none
        for candidate in candidates do
          if axisCertificate?.isNone then
            let lengthProposition ←
              mkEq candidate.1.2 resolvedLength
            let lengthCertificate? ←
              observing? <|
                certifyWithTactic
                  "that an input occurrence supplies its logical einsum length"
                  lengthProposition sourceLengthTactic
            let lengthCertificate? ←
              match lengthCertificate? with
              | some certificate => pure (some certificate)
              | none =>
                  observing? <|
                    certifySymbolicEinsumInvariant
                      "that an input occurrence supplies its logical einsum length"
                      lengthProposition
            if let some lengthCertificate := lengthCertificate? then
              let before ←
                mkListLit occurrenceType <|
                  occurrenceExpressions.take candidate.2
              let after ←
                mkListLit occurrenceType <|
                  occurrenceExpressions.drop (candidate.2 + 1)
              let certificate ←
                mkAppM ``einsum_axis_source_of_occurrence #[
                  axisLength, axisExpression, candidate.1.2,
                  before, after, lengthCertificate]
              let certificateType ← inferType certificate
              unless ← withTransparency .all <|
                  isDefEq certificateType axisProposition do
                throwError
                  "internal error: symbolic axis-source proof has type\
                    {indentExpr certificateType}\nexpected\
                    {indentExpr axisProposition}"
              axisCertificate? := some certificate
        let some axisCertificate := axisCertificate?
          | throwError
              "could not prove that one logical einsum length is supplied \
                by an input occurrence:{indentExpr axisProposition}\n\
                Add the shape equality or singleton-broadcasting hypothesis \
                needed by this operation."
        let remainingCertificate ← certifyAxes remainingAxes
        let conjunctionCertificate ←
          mkAppM ``And.intro #[axisCertificate, remainingCertificate]
        let remainingCheck ←
          mkAppM ``List.all #[Lean.toExpr remainingAxes, axisPredicate]
        let conjunctionEquality ←
          mkAppM ``Bool.and_eq_true #[axisCheck, remainingCheck]
        let certificate ←
          mkAppM ``Eq.mpr #[conjunctionEquality, conjunctionCertificate]
        withTransparency .all <|
          mkExpectedTypeHint certificate expected
  let certificate ← certifyAxes globalAxes
  let expectedCheck ←
    mkAppM ``Check.einsumAxisLengths
      #[pattern, inputShapes, axisLength]
  let expected ← mkEq expectedCheck (Lean.toExpr true)
  let certificateType ← inferType certificate
  unless ← withTransparency .all <| isDefEq certificateType expected do
    throwError
      "internal error: symbolic axis-length proof has type\
        {indentExpr certificateType}\nexpected{indentExpr expected}"
  sealCertificate expected certificate

/--
Construct a checked einsum directly from symbolic natural-number dimensions.

The executable checker validates the complete rank and pattern structure on
same-rank singleton witnesses. The actual dimensions are then used in every
proof-valued field of `CheckedEinsum`. An expected output tensor shape, when
available, names the corresponding logical output lengths and is checked by
the same broadcasting and source-occurrence invariants.
-/
def symbolicCheckedEinsumExpr (source : String)
    (pattern : Syntax.EinsumPattern)
    (inputDimensions : List (List Expr))
    (expectedType? : Option Expr) :
    TermElabM
      (Expr × List (Check.EinsumAxis × Expr) × Expr × Expr) := do
  let dummyShapes :=
    inputDimensions.map fun dimensions =>
      List.replicate dimensions.length 1
  match Check.checkEinsum pattern dummyShapes with
  | .error diagnostic =>
      throwPatternDiagnostic "einsum" "shape" source
        diagnostic.message diagnostic.span
  | .ok _ => pure ()
  let inputAxes := Check.einsumInputAxes pattern dummyShapes
  let outputAxes := Check.einsumOutputAxes pattern dummyShapes
  let globalAxes := Check.einsumGlobalAxes pattern dummyShapes
  let occurrences :=
    (inputAxes.zip inputDimensions).flatMap fun operand =>
      operand.1.zip operand.2
  let mut expectedAssignments : List (Check.EinsumAxis × Expr) := []
  if let some expectedShape ← expectedTensorShape? expectedType? then
    if let some expectedDimensions ← staticListElements? expectedShape then
      if expectedDimensions.length = outputAxes.length then
        for (axis, dimension) in outputAxes.zip expectedDimensions do
          expectedAssignments :=
            appendEinsumAxisExpression expectedAssignments axis dimension
  -- Preserve first-input-occurrence order in the generated dependent function.
  -- Expected dimensions replace only genuinely ambiguous broadcast expressions.
  let mut assignments : List (Check.EinsumAxis × Expr) := []
  for axis in globalAxes do
    let dimensions :=
      occurrences.filterMap fun occurrence =>
        if occurrence.1 == axis then some occurrence.2 else none
    let inferredLength ← symbolicBroadcastAxisLength dimensions
    let length ←
      match einsumAxisExpression? expectedAssignments axis with
      | none => pure inferredLength
      | some expectedLength =>
          if ← withTransparency .reducible <|
              isDefEq inferredLength expectedLength then
            pure inferredLength
          else
            pure expectedLength
    assignments :=
      appendEinsumAxisExpression assignments axis length
  let inputShapeExpressions ←
    inputDimensions.mapM fun dimensions => shapeExpr dimensions
  let inputShapes ← shapesExpr inputShapeExpressions
  let compactOutputDimensions ←
    outputAxes.mapM fun axis =>
      match einsumAxisExpression? assignments axis with
      | some length => pure length
      | none =>
          throwError
            "internal error: a symbolic einsum output axis has no resolved length"
  let compactOutputShape ← shapeExpr compactOutputDimensions
  let patternExpr := Lean.toExpr pattern
  let axisLength ← symbolicEinsumAxisLengthExpr assignments
  let repeatedDimensions ←
    certifyRepeatedDimensions
      patternExpr inputShapes inputAxes inputDimensions
  let hInputDimensions ←
    certifyInputBroadcasting
      patternExpr inputShapes axisLength inputAxes inputDimensions
  let axisLengths ←
    certifyAxisSources
      patternExpr inputShapes axisLength inputAxes inputDimensions globalAxes
  let patternInputs ←
    mkAppM ``Syntax.EinsumPattern.inputs #[patternExpr]
  let inputCountProposition ←
    mkEq
      (← mkAppM ``List.length #[patternInputs])
      (← mkAppM ``List.length #[inputShapes])
  let inputCount ←
    mkDecideProof inputCountProposition
  let inputCount ←
    sealCertificate inputCountProposition inputCount
  let surfaceSupportedCheck ←
    mkAppM ``Check.einsumPatternSupported #[patternExpr]
  let surfaceSupportedProposition ←
    mkEq surfaceSupportedCheck (Lean.toExpr true)
  let surfaceSupported ←
    mkDecideProof surfaceSupportedProposition
  let surfaceSupported ←
    sealCertificate surfaceSupportedProposition surfaceSupported
  let outputAxesExpression ←
    mkAppM ``Check.einsumOutputAxes #[patternExpr, inputShapes]
  let checkedOutputDimensions ←
    mkAppM ``List.map #[axisLength, outputAxesExpression]
  let concreteOutputAxesExpression := Lean.toExpr outputAxes
  let concreteOutputDimensions ←
    mkAppM ``List.map #[axisLength, concreteOutputAxesExpression]
  let concreteOutputAxesNodup ←
    mkAppM ``List.Nodup #[Lean.toExpr outputAxes]
  let outputAxesNodup ← mkDecideProof concreteOutputAxesNodup
  let expectedOutputAxesNodup ←
    mkAppM ``List.Nodup #[outputAxesExpression]
  let outputAxesNodupType ← inferType outputAxesNodup
  unless ← withTransparency .all <|
      isDefEq outputAxesNodupType expectedOutputAxesNodup do
    throwError
      "internal error: expanded einsum output axes disagree with the \
        symbolic checked expression"
  let outputAxesNodup ←
    sealCertificate expectedOutputAxesNodup outputAxesNodup
  let globalAxesExpression ←
    mkAppM ``Check.einsumGlobalAxes #[patternExpr, inputShapes]
  let concreteOutputAxesKnownCheck ←
    mkAppM ``Check.einsumAxesSubset #[
      Lean.toExpr outputAxes, Lean.toExpr globalAxes]
  let outputAxesKnown ←
    mkDecideProof
      (← mkEq concreteOutputAxesKnownCheck (Lean.toExpr true))
  let expectedOutputAxesKnownCheck ←
    mkAppM ``Check.einsumAxesSubset #[
      outputAxesExpression, globalAxesExpression]
  let expectedOutputAxesKnown ←
    mkEq expectedOutputAxesKnownCheck (Lean.toExpr true)
  let outputAxesKnownType ← inferType outputAxesKnown
  unless ← withTransparency .all <|
      isDefEq outputAxesKnownType expectedOutputAxesKnown do
    throwError
      "internal error: expanded einsum output axes are not supplied by the \
        symbolic input axes"
  let outputAxesKnown ←
    sealCertificate expectedOutputAxesKnown outputAxesKnown
  -- Each certificate above is checked against its exact field type. Applying
  -- all fields at once avoids unfolding ellipsis expansion once per dependent
  -- constructor argument.
  let constructor := Lean.mkConst ``Check.CheckedEinsum.mk
  let checked :=
    mkAppN constructor #[
      patternExpr,
      inputShapes,
      axisLength,
      inputCount,
      surfaceSupported,
      repeatedDimensions,
      hInputDimensions,
      axisLengths,
      outputAxesNodup,
      outputAxesKnown]
  let checkedOutputShape :=
    mkApp (mkConst ``Check.CheckedEinsum.output) checked
  let checkedOutputShapeDefinition ←
    withTransparency .all <|
      mkExpectedTypeHint
        (← mkEqRefl checkedOutputShape)
        (← mkEq checkedOutputShape checkedOutputDimensions)
  let checkedOutputDimensionsAgreement ←
    withTransparency .all <|
      mkExpectedTypeHint
        (← mkEqRefl checkedOutputDimensions)
        (← mkEq checkedOutputDimensions concreteOutputDimensions)
  let concreteOutputShapeAgreement ←
    certifySymbolicEinsumInvariant
      "that the symbolic einsum output dimensions match the checked output shape"
      (← mkEq concreteOutputDimensions compactOutputShape)
  let compactOutputShapeAgreement ←
    mkAppM ``Eq.trans #[
      checkedOutputDimensionsAgreement, concreteOutputShapeAgreement]
  let outputShapeAgreement ←
    mkAppM ``Eq.trans #[
      checkedOutputShapeDefinition, compactOutputShapeAgreement]
  return (checked, assignments, compactOutputShape, outputShapeAgreement)

end TorchLean.Tensor.Internal.Elab.Impl
