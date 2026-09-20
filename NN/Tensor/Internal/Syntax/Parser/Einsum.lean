/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Expression -- shake: keep

/-!
# Einsum-pattern parsing

This module parses comma-separated operands, the output expression, and their
canonical rendering round trip.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

open Parser.Impl

namespace Parser.Impl

/-- Split an einsum input token stream at top-level commas. -/
def splitCommas : List Token → List Token → List (List Token) → List (List Token)
  | [], currentRev, groupsRev => (currentRev.reverse :: groupsRev).reverse
  | token :: rest, currentRev, groupsRev =>
      match token.value with
      | .comma => splitCommas rest [] (currentRev.reverse :: groupsRev)
      | _ => splitCommas rest (token :: currentRev) groupsRev

end Parser.Impl

/-- Intercalating a nonempty group list extended by one group exposes the final separator. -/
private theorem intercalate_append_singleton
    {α : Type} (separator : List α) (groups : List (List α))
    (last : List α) (hGroups : groups ≠ []) :
    List.intercalate separator (groups ++ [last]) =
      List.intercalate separator groups ++ separator ++ last := by
  induction groups with
  | nil => contradiction
  | cons group groups induction =>
      cases groups with
      | nil => simp [List.intercalate]
      | cons next rest =>
          simp only [List.cons_append, List.intercalate, List.intersperse]
          simp [List.intercalate, List.append_assoc] at induction ⊢
          exact induction

/-- Intercalating reversed groups separates the final group exactly when one precedes it. -/
private theorem intercalate_reverse_cons
    {α : Type} (separator group : List α)
    (groups : List (List α)) :
    List.intercalate separator (group :: groups).reverse =
      List.intercalate separator groups.reverse ++
        (if groups.isEmpty then [] else separator) ++ group := by
  cases groups with
  | nil => simp [List.intercalate]
  | cons first rest =>
      rw [List.reverse_cons]
      rw [intercalate_append_singleton]
      · simp
      · simp

/-- Splitting on commas preserves the complete canonical token stream. -/
private theorem splitCommas_reconstruct_canonical
    (tokens currentRev : List Token) (groupsRev : List (List Token)) :
    List.intercalate [.comma]
        ((splitCommas tokens currentRev groupsRev).map fun group =>
          group.map fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value) =
      List.intercalate [.comma]
          (groupsRev.reverse.map fun group =>
            group.map fun token =>
              canonicalTokenKind IdentifierPolicy.pythonUnicode
                token.value) ++
        (if groupsRev.isEmpty then [] else [.comma]) ++
          currentRev.reverse.map
              (fun token =>
                canonicalTokenKind IdentifierPolicy.pythonUnicode
                  token.value) ++
            tokens.map fun token =>
              canonicalTokenKind IdentifierPolicy.pythonUnicode
                token.value := by
  induction tokens generalizing currentRev groupsRev with
  | nil =>
      simpa [splitCommas, List.map_reverse, List.map_map,
          Function.comp_def] using
        intercalate_reverse_cons [.comma]
          (currentRev.reverse.map fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value)
          (groupsRev.map fun group =>
            group.map fun token =>
              canonicalTokenKind IdentifierPolicy.pythonUnicode token.value)
  | cons token tokens induction =>
      by_cases hComma : token.value = .comma
      · have hStep :
            splitCommas (token :: tokens) currentRev groupsRev =
              splitCommas tokens []
                (currentRev.reverse :: groupsRev) := by
          simp [splitCommas, hComma]
        rw [hStep]
        rw [induction (currentRev := [])
          (groupsRev := currentRev.reverse :: groupsRev)]
        simp only [List.map_reverse, List.map_cons, List.map_nil,
          List.reverse_nil]
        rw [intercalate_reverse_cons [.comma]
          ((currentRev.map fun item =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode item.value).reverse)
          (groupsRev.map fun group =>
            group.map fun item =>
              canonicalTokenKind IdentifierPolicy.pythonUnicode item.value)]
        simp [canonicalTokenKind, hComma, List.append_assoc]
      · have hStep :
            splitCommas (token :: tokens) currentRev groupsRev =
              splitCommas tokens (token :: currentRev) groupsRev := by
          cases hValue : token.value <;>
            simp [splitCommas, hValue] at hComma ⊢
        rw [hStep]
        simpa [List.reverse_cons, List.map_append,
            List.append_assoc] using
          induction (currentRev := token :: currentRev)
            (groupsRev := groupsRev)

/-- Comma splitting commutes with canonical token replacement. -/
private theorem splitCommas_canonical
    (sourceTokens targetTokens sourceCurrentRev targetCurrentRev : List Token)
    (sourceGroupsRev targetGroupsRev : List (List Token))
    (hTokens :
      targetTokens.map Located.value =
        sourceTokens.map fun token =>
          canonicalTokenKind IdentifierPolicy.pythonUnicode token.value)
    (hCurrent :
      targetCurrentRev.map Located.value =
        sourceCurrentRev.map fun token =>
          canonicalTokenKind IdentifierPolicy.pythonUnicode token.value)
    (hGroups :
      targetGroupsRev.map (List.map Located.value) =
        sourceGroupsRev.map fun group =>
          group.map fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value) :
    (splitCommas targetTokens targetCurrentRev targetGroupsRev).map
        (List.map Located.value) =
      (splitCommas sourceTokens sourceCurrentRev sourceGroupsRev).map
        fun group =>
          group.map fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value := by
  induction sourceTokens generalizing
      targetTokens sourceCurrentRev targetCurrentRev sourceGroupsRev
      targetGroupsRev with
  | nil =>
      cases targetTokens with
      | nil =>
          simp [splitCommas, List.map_reverse, hCurrent, hGroups]
      | cons targetToken targetTokens =>
          simp at hTokens
  | cons sourceToken sourceTokens induction =>
      cases targetTokens with
      | nil =>
          simp at hTokens
      | cons targetToken targetTokens =>
          simp only [List.map_cons, List.cons.injEq] at hTokens
          rcases hTokens with ⟨hValue, hTokens⟩
          by_cases hSourceComma : sourceToken.value = .comma
          · have hTargetComma : targetToken.value = .comma := by
              rw [hValue]
              exact
                (canonicalTokenKind_eq_comma
                  IdentifierPolicy.pythonUnicode sourceToken.value).2
                  hSourceComma
            have hSourceStep :
                splitCommas (sourceToken :: sourceTokens) sourceCurrentRev
                    sourceGroupsRev =
                  splitCommas sourceTokens []
                    (sourceCurrentRev.reverse :: sourceGroupsRev) := by
              cases hSourceValue : sourceToken.value <;>
                simp [splitCommas, hSourceValue] at hSourceComma ⊢
            have hTargetStep :
                splitCommas (targetToken :: targetTokens) targetCurrentRev
                    targetGroupsRev =
                  splitCommas targetTokens []
                    (targetCurrentRev.reverse :: targetGroupsRev) := by
              simp [splitCommas, hTargetComma]
            rw [hSourceStep, hTargetStep]
            apply induction
              (targetTokens := targetTokens)
              (sourceCurrentRev := [])
              (targetCurrentRev := [])
              (sourceGroupsRev :=
                sourceCurrentRev.reverse :: sourceGroupsRev)
              (targetGroupsRev :=
                targetCurrentRev.reverse :: targetGroupsRev)
              hTokens (by simp)
            simp only [List.map_cons, List.cons.injEq]
            refine ⟨?_, hGroups⟩
            simpa [List.map_reverse] using congrArg List.reverse hCurrent
          · have hTargetNotComma : targetToken.value ≠ .comma := by
              intro hTargetComma
              apply hSourceComma
              apply
                (canonicalTokenKind_eq_comma
                  IdentifierPolicy.pythonUnicode sourceToken.value).1
              exact hValue.symm.trans hTargetComma
            have hSourceStep :
                splitCommas (sourceToken :: sourceTokens) sourceCurrentRev
                    sourceGroupsRev =
                  splitCommas sourceTokens
                    (sourceToken :: sourceCurrentRev) sourceGroupsRev := by
              cases hSourceValue : sourceToken.value <;>
                simp [splitCommas, hSourceValue] at hSourceComma ⊢
            have hTargetStep :
                splitCommas (targetToken :: targetTokens) targetCurrentRev
                    targetGroupsRev =
                  splitCommas targetTokens
                    (targetToken :: targetCurrentRev) targetGroupsRev := by
              cases hTargetValue : targetToken.value <;>
                simp [splitCommas, hTargetValue] at hTargetNotComma ⊢
            rw [hSourceStep, hTargetStep]
            apply induction
              (targetTokens := targetTokens)
              (sourceCurrentRev := sourceToken :: sourceCurrentRev)
              (targetCurrentRev := targetToken :: targetCurrentRev)
              (sourceGroupsRev := sourceGroupsRev)
              (targetGroupsRev := targetGroupsRev)
              hTokens
            · simp [hValue, hCurrent]
            · exact hGroups

namespace Parser.Impl

/-- Compute the smallest span covering a token list, or use the fallback when empty. -/
def tokenListSpan (fallback : Span) : List Token → Span
  | [] => fallback
  | first :: rest =>
      match rest.getLast? with
      | none => first.span
      | some last => Span.join first.span last.span

/-- Parse comma-separated einsum operands in source order. -/
def parseEinsumInputs (policy : IdentifierPolicy) :
    List (List Token) → List Expression → Result (List Expression)
  | [], parsedRev => .ok parsedRev.reverse
  | tokens :: rest, parsedRev => do
      let span := tokenListSpan (Span.point 0) tokens
      let parsed ← parseExpressionTokens policy .einsumInput span tokens {}
      parseEinsumInputs policy rest (parsed :: parsedRev)

end Parser.Impl

/-- Successful operand parsing records each input expression's canonical token kinds. -/
private theorem tokenKinds_eq_of_parseEinsumInputs_eq_ok
    (policy : IdentifierPolicy) (groups : List (List Token))
    (parsedRev expressions : List Expression)
    (hParse :
      parseEinsumInputs policy groups parsedRev = .ok expressions) :
    expressions.map Expression.tokenKinds =
      parsedRev.reverse.map Expression.tokenKinds ++
        groups.map fun group =>
          group.map fun token => canonicalTokenKind policy token.value := by
  induction groups generalizing parsedRev expressions with
  | nil =>
      simp [parseEinsumInputs] at hParse
      subst expressions
      simp
  | cons group groups induction =>
      cases hExpressionParse :
          parseExpressionTokens policy .einsumInput
            (tokenListSpan (Span.point 0) group) group {} with
      | error diagnostic =>
          simp [parseEinsumInputs, hExpressionParse] at hParse
          change
            (Except.error diagnostic : Result (List Expression)) =
              .ok expressions at hParse
          contradiction
      | ok expression =>
          simp [parseEinsumInputs, hExpressionParse] at hParse
          change
            parseEinsumInputs policy groups (expression :: parsedRev) =
              .ok expressions at hParse
          have hKinds :=
            tokenKinds_eq_of_parseExpressionTokens_eq_ok policy .einsumInput
              (tokenListSpan (Span.point 0) group) group {} expression
              hExpressionParse
          simp [ExpressionState.tokenKinds] at hKinds
          rw [induction (parsedRev := expression :: parsedRev)
            (expressions := expressions) hParse]
          simp [List.reverse_cons, List.map_append, hKinds,
            List.append_assoc]

/-- Canonically equivalent operand groups parse to equivalent expression lists. -/
private theorem parseEinsumInputs_canonical
    (sourceGroups targetGroups : List (List Token))
    (sourceParsedRev targetParsedRev sourceExpressions : List Expression)
    (hGroups :
      targetGroups.map (List.map Located.value) =
        sourceGroups.map fun group =>
          group.map fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value)
    (hParsed :
      targetParsedRev.map Expression.tokenKinds =
        sourceParsedRev.map Expression.tokenKinds)
    (hParse :
      parseEinsumInputs IdentifierPolicy.pythonUnicode sourceGroups
          sourceParsedRev =
        .ok sourceExpressions) :
    ∃ targetExpressions,
      parseEinsumInputs IdentifierPolicy.pythonUnicode targetGroups
          targetParsedRev =
          .ok targetExpressions ∧
        targetExpressions.map Expression.tokenKinds =
          sourceExpressions.map Expression.tokenKinds := by
  induction sourceGroups generalizing
      targetGroups sourceParsedRev targetParsedRev sourceExpressions with
  | nil =>
      cases targetGroups with
      | cons targetGroup targetGroups =>
          simp at hGroups
      | nil =>
          simp [parseEinsumInputs] at hParse
          subst sourceExpressions
          refine ⟨targetParsedRev.reverse, by simp [parseEinsumInputs], ?_⟩
          simpa [List.map_reverse] using congrArg List.reverse hParsed
  | cons sourceGroup sourceGroups induction =>
      cases targetGroups with
      | nil =>
          simp at hGroups
      | cons targetGroup targetGroups =>
          simp only [List.map_cons, List.cons.injEq] at hGroups
          rcases hGroups with ⟨hGroup, hGroups⟩
          cases hSourceExpressionParse :
              parseExpressionTokens IdentifierPolicy.pythonUnicode .einsumInput
                (tokenListSpan (Span.point 0) sourceGroup) sourceGroup {} with
          | error diagnostic =>
              simp [parseEinsumInputs, hSourceExpressionParse] at hParse
              change
                (Except.error diagnostic : Result (List Expression)) =
                  .ok sourceExpressions at hParse
              contradiction
          | ok sourceExpression =>
              simp [parseEinsumInputs, hSourceExpressionParse] at hParse
              change
                parseEinsumInputs IdentifierPolicy.pythonUnicode sourceGroups
                    (sourceExpression :: sourceParsedRev) =
                  .ok sourceExpressions at hParse
              have hInitialState :
                  ExpressionState.CanonicalEq
                    ({} : ExpressionState) ({} : ExpressionState) :=
                ⟨rfl, rfl, rfl⟩
              obtain
                  ⟨targetExpression, hTargetExpressionParse,
                    hExpressionKinds⟩ :=
                parseExpressionTokens_canonical .einsumInput
                  (tokenListSpan (Span.point 0) sourceGroup)
                  (tokenListSpan (Span.point 0) targetGroup)
                  sourceGroup targetGroup {} {} sourceExpression hGroup
                  hInitialState hSourceExpressionParse
              have hNextParsed :
                  (targetExpression :: targetParsedRev).map
                      Expression.tokenKinds =
                    (sourceExpression :: sourceParsedRev).map
                      Expression.tokenKinds := by
                simp [hExpressionKinds, hParsed]
              obtain ⟨targetExpressions, hTargetParse, hKinds⟩ :=
                induction
                  (targetGroups := targetGroups)
                  (sourceParsedRev := sourceExpression :: sourceParsedRev)
                  (targetParsedRev := targetExpression :: targetParsedRev)
                  (sourceExpressions := sourceExpressions)
                  hGroups hNextParsed hParse
              refine ⟨targetExpressions, ?_, hKinds⟩
              simp only [parseEinsumInputs]
              rw [hTargetExpressionParse]
              change
                parseEinsumInputs IdentifierPolicy.pythonUnicode targetGroups
                    (targetExpression :: targetParsedRev) =
                  .ok targetExpressions
              exact hTargetParse

/-- Parse an einsum pattern with one or more comma-separated input expressions. -/
def parseEinsumPattern (source : String)
    (policy := IdentifierPolicy.pythonUnicode) : Result EinsumPattern := do
  let whole := sourceSpan source
  let tokens ← lex source policy .transformation
  let (inputTokens, arrow, outputTokens) ← splitArrow (Span.point whole.stop) tokens []
  let inputs ← parseEinsumInputs policy (splitCommas inputTokens [] []) []
  let outputSpan := Span.between arrow.span.stop whole.stop
  let output ← parseExpressionTokens policy .einsumOutput outputSpan outputTokens {}
  .ok { inputs, arrow := arrow.span, output, span := whole }

/-- Successful component parses assemble into a successful einsum parse. -/
private theorem parseEinsumPattern_eq_ok_of_components
    (source : String)
    (tokens inputTokens outputTokens : List Token) (arrow : Token)
    (inputs : List Expression) (output : Expression)
    (hLex :
      lex source IdentifierPolicy.pythonUnicode .transformation = .ok tokens)
    (hSplit :
      splitArrow (Span.point (sourceSpan source).stop) tokens [] =
        .ok (inputTokens, arrow, outputTokens))
    (hInputs :
      parseEinsumInputs IdentifierPolicy.pythonUnicode
          (splitCommas inputTokens [] []) [] =
        .ok inputs)
    (hOutput :
      parseExpressionTokens IdentifierPolicy.pythonUnicode .einsumOutput
          (Span.between arrow.span.stop (sourceSpan source).stop)
          outputTokens {} =
        .ok output) :
    parseEinsumPattern source =
      .ok
        { inputs
          arrow := arrow.span
          output
          span := sourceSpan source } := by
  unfold parseEinsumPattern
  rw [hLex]
  change
    (do
      let (parsedInputTokens, parsedArrow, parsedOutputTokens) ←
        splitArrow (Span.point (sourceSpan source).stop) tokens []
      let parsedInputs ←
        parseEinsumInputs IdentifierPolicy.pythonUnicode
          (splitCommas parsedInputTokens [] []) []
      let parsedOutput ←
        parseExpressionTokens IdentifierPolicy.pythonUnicode .einsumOutput
          (Span.between parsedArrow.span.stop (sourceSpan source).stop)
          parsedOutputTokens {}
      .ok
        ({ inputs := parsedInputs
           arrow := parsedArrow.span
           output := parsedOutput
           span := sourceSpan source } :
          EinsumPattern)) =
      .ok
        { inputs
          arrow := arrow.span
          output
          span := sourceSpan source }
  rw [hSplit]
  change
    (do
      let parsedInputs ←
        parseEinsumInputs IdentifierPolicy.pythonUnicode
          (splitCommas inputTokens [] []) []
      let parsedOutput ←
        parseExpressionTokens IdentifierPolicy.pythonUnicode .einsumOutput
          (Span.between arrow.span.stop (sourceSpan source).stop)
          outputTokens {}
      .ok
        ({ inputs := parsedInputs
           arrow := arrow.span
           output := parsedOutput
           span := sourceSpan source } :
          EinsumPattern)) =
      .ok
        { inputs
          arrow := arrow.span
          output
          span := sourceSpan source }
  rw [hInputs]
  change
    (do
      let parsedOutput ←
        parseExpressionTokens IdentifierPolicy.pythonUnicode .einsumOutput
          (Span.between arrow.span.stop (sourceSpan source).stop)
          outputTokens {}
      .ok
        ({ inputs
           arrow := arrow.span
           output := parsedOutput
           span := sourceSpan source } :
          EinsumPattern)) =
      .ok
        { inputs
          arrow := arrow.span
          output
          span := sourceSpan source }
  rw [hOutput]
  rfl

/-- A successful einsum parse decomposes into lexing, splitting, inputs, and output. -/
private theorem parseEinsumPattern_eq_ok_decomposition
    (source : String) (pattern : EinsumPattern)
    (hParse : parseEinsumPattern source = .ok pattern) :
    ∃ (sourceTokens sourceInputTokens sourceOutputTokens : List Token)
        (sourceArrow : Token) (sourceInputs : List Expression)
        (sourceOutput : Expression),
      lex source IdentifierPolicy.pythonUnicode .transformation =
          .ok sourceTokens ∧
        splitArrow (Span.point (sourceSpan source).stop) sourceTokens [] =
          .ok (sourceInputTokens, sourceArrow, sourceOutputTokens) ∧
        parseEinsumInputs IdentifierPolicy.pythonUnicode
            (splitCommas sourceInputTokens [] []) [] =
          .ok sourceInputs ∧
        parseExpressionTokens IdentifierPolicy.pythonUnicode .einsumOutput
            (Span.between sourceArrow.span.stop (sourceSpan source).stop)
            sourceOutputTokens {} =
          .ok sourceOutput ∧
        pattern =
          { inputs := sourceInputs
            arrow := sourceArrow.span
            output := sourceOutput
            span := sourceSpan source } := by
  unfold parseEinsumPattern at hParse
  cases hSourceLex :
      lex source IdentifierPolicy.pythonUnicode .transformation with
  | error diagnostic =>
      rw [hSourceLex] at hParse
      contradiction
  | ok sourceTokens =>
      rw [hSourceLex] at hParse
      change
        (do
          let (parsedInputTokens, parsedArrow, parsedOutputTokens) ←
            splitArrow (Span.point (sourceSpan source).stop) sourceTokens []
          let parsedInputs ←
            parseEinsumInputs IdentifierPolicy.pythonUnicode
              (splitCommas parsedInputTokens [] []) []
          let parsedOutput ←
            parseExpressionTokens IdentifierPolicy.pythonUnicode .einsumOutput
              (Span.between parsedArrow.span.stop (sourceSpan source).stop)
              parsedOutputTokens {}
          .ok
            ({ inputs := parsedInputs
               arrow := parsedArrow.span
               output := parsedOutput
               span := sourceSpan source } :
              EinsumPattern)) =
          .ok pattern at hParse
      cases hSourceSplit :
          splitArrow (Span.point (sourceSpan source).stop) sourceTokens [] with
      | error diagnostic =>
          rw [hSourceSplit] at hParse
          change
            (Except.error diagnostic : Result EinsumPattern) =
              .ok pattern at hParse
          contradiction
      | ok split =>
          obtain
              ⟨sourceInputTokens, sourceArrow, sourceOutputTokens⟩ := split
          rw [hSourceSplit] at hParse
          change
            (do
              let parsedInputs ←
                parseEinsumInputs IdentifierPolicy.pythonUnicode
                  (splitCommas sourceInputTokens [] []) []
              let parsedOutput ←
                parseExpressionTokens IdentifierPolicy.pythonUnicode
                  .einsumOutput
                  (Span.between sourceArrow.span.stop
                    (sourceSpan source).stop)
                  sourceOutputTokens {}
              .ok
                ({ inputs := parsedInputs
                   arrow := sourceArrow.span
                   output := parsedOutput
                   span := sourceSpan source } :
                  EinsumPattern)) =
              .ok pattern at hParse
          cases hSourceInputs :
              parseEinsumInputs IdentifierPolicy.pythonUnicode
                (splitCommas sourceInputTokens [] []) [] with
          | error diagnostic =>
              rw [hSourceInputs] at hParse
              change
                (Except.error diagnostic : Result EinsumPattern) =
                  .ok pattern at hParse
              contradiction
          | ok sourceInputs =>
              rw [hSourceInputs] at hParse
              change
                (do
                  let parsedOutput ←
                    parseExpressionTokens IdentifierPolicy.pythonUnicode
                      .einsumOutput
                      (Span.between sourceArrow.span.stop
                        (sourceSpan source).stop)
                      sourceOutputTokens {}
                  .ok
                    ({ inputs := sourceInputs
                       arrow := sourceArrow.span
                       output := parsedOutput
                       span := sourceSpan source } :
                      EinsumPattern)) =
                  .ok pattern at hParse
              cases hSourceOutput :
                  parseExpressionTokens IdentifierPolicy.pythonUnicode
                    .einsumOutput
                    (Span.between sourceArrow.span.stop
                      (sourceSpan source).stop)
                    sourceOutputTokens {} with
              | error diagnostic =>
                  rw [hSourceOutput] at hParse
                  change
                    (Except.error diagnostic : Result EinsumPattern) =
                      .ok pattern at hParse
                  contradiction
              | ok sourceOutput =>
                  rw [hSourceOutput] at hParse
                  injection hParse with hPattern
                  exact
                    ⟨sourceTokens, sourceInputTokens, sourceOutputTokens,
                      sourceArrow, sourceInputs, sourceOutput, rfl,
                      hSourceSplit, hSourceInputs, hSourceOutput,
                      hPattern.symm⟩

/-- Lexing a rendered einsum recovers its canonical comma-and-arrow token sequence. -/
private theorem lex_einsumPattern_render_eq_ok
    (pattern : EinsumPattern)
    (hWords :
      ∀ text,
        .word text ∈
            ((pattern.inputs.map Expression.tokenKinds).intersperse
                [.comma]).flatten ++
              [.arrow] ++ pattern.output.tokenKinds →
          text.toList ≠ [] ∧
            ∀ character ∈ text.toList,
              IdentifierPolicy.pythonUnicode.isWordChar character = true) :
    (lex pattern.render IdentifierPolicy.pythonUnicode
        LexicalWhitespace.transformation).map (List.map Located.value) =
      .ok
        (((pattern.inputs.map Expression.tokenKinds).intersperse
            [.comma]).flatten ++
          [.arrow] ++ pattern.output.tokenKinds) := by
  rw [EinsumPattern.render]
  apply TokenKind.lex_renderSequence_transformation
  exact hWords

/--
Parsing a successful einsum pattern's canonical rendering succeeds with the
same canonical syntax.

The theorem applies to any number of input tensors. It preserves their order
and comma boundaries, including empty expressions denoting scalar operands.
Repeated labels and ellipses inside inputs are retained; only source locations
and anonymous-axis occurrence offsets are intentionally refreshed.
-/
theorem parseEinsumPattern_render_eq_ok (source : String)
    (pattern : EinsumPattern)
    (hParse : parseEinsumPattern source = .ok pattern) :
    (parseEinsumPattern pattern.render).map EinsumPattern.render =
      .ok pattern.render := by
  obtain
      ⟨sourceTokens, sourceInputTokens, sourceOutputTokens, sourceArrow,
        sourceInputs, sourceOutput, hSourceLex, hSourceSplit,
        hSourceInputs, hSourceOutput, hPattern⟩ :=
    parseEinsumPattern_eq_ok_decomposition source pattern hParse
  subst pattern
  have hInputKinds :=
    tokenKinds_eq_of_parseEinsumInputs_eq_ok
      IdentifierPolicy.pythonUnicode
      (splitCommas sourceInputTokens [] []) [] sourceInputs hSourceInputs
  simp at hInputKinds
  have hOutputKinds :=
    tokenKinds_eq_of_parseExpressionTokens_eq_ok
      IdentifierPolicy.pythonUnicode .einsumOutput
      (Span.between sourceArrow.span.stop (sourceSpan source).stop)
      sourceOutputTokens {} sourceOutput hSourceOutput
  simp [ExpressionState.tokenKinds] at hOutputKinds
  obtain ⟨hSourceTokens, hSourceArrow⟩ :=
    splitArrow_eq_ok_decomposition
      (Span.point (sourceSpan source).stop) sourceTokens []
      sourceInputTokens sourceOutputTokens sourceArrow hSourceSplit
  simp only [List.reverse_nil, List.nil_append] at hSourceTokens
  have hInputReconstruction :=
    splitCommas_reconstruct_canonical sourceInputTokens [] []
  simp at hInputReconstruction
  have hRenderedKinds :
      ((sourceInputs.map Expression.tokenKinds).intersperse
            [.comma]).flatten ++
          [.arrow] ++ sourceOutput.tokenKinds =
        sourceTokens.map fun token =>
          canonicalTokenKind IdentifierPolicy.pythonUnicode token.value := by
    calc
      ((sourceInputs.map Expression.tokenKinds).intersperse
              [.comma]).flatten ++
            [.arrow] ++ sourceOutput.tokenKinds =
          List.intercalate [.comma]
                ((splitCommas sourceInputTokens [] []).map fun group =>
                  group.map fun token =>
                    canonicalTokenKind IdentifierPolicy.pythonUnicode
                      token.value) ++
            [.arrow] ++ sourceOutput.tokenKinds := by
        rw [hInputKinds]
        rfl
      _ =
          sourceInputTokens.map
                (fun token =>
                  canonicalTokenKind IdentifierPolicy.pythonUnicode
                    token.value) ++
            [.arrow] ++
              sourceOutputTokens.map fun token =>
                canonicalTokenKind IdentifierPolicy.pythonUnicode
                  token.value := by
        rw [hInputReconstruction, hOutputKinds]
      _ =
          (sourceInputTokens ++ sourceArrow :: sourceOutputTokens).map
            fun token =>
              canonicalTokenKind IdentifierPolicy.pythonUnicode
                token.value := by
        simp [hSourceArrow, canonicalTokenKind, List.append_assoc]
      _ =
          sourceTokens.map fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value := by
        rw [hSourceTokens]
  have hWords :
      ∀ text,
        .word text ∈
            ((sourceInputs.map Expression.tokenKinds).intersperse
                [.comma]).flatten ++
              [.arrow] ++ sourceOutput.tokenKinds →
          text.toList ≠ [] ∧
            ∀ character ∈ text.toList,
              IdentifierPolicy.pythonUnicode.isWordChar character = true := by
    intro text hText
    apply
      canonical_words_valid_of_lex_eq_ok source .transformation sourceTokens
        hSourceLex text
    rw [← hRenderedKinds]
    exact hText
  let sourcePattern : EinsumPattern :=
    { inputs := sourceInputs
      arrow := sourceArrow.span
      output := sourceOutput
      span := sourceSpan source }
  have hTargetLexValues :
      (lex sourcePattern.render IdentifierPolicy.pythonUnicode
          LexicalWhitespace.transformation).map (List.map Located.value) =
        .ok
          (((sourceInputs.map Expression.tokenKinds).intersperse
              [.comma]).flatten ++
            [.arrow] ++ sourceOutput.tokenKinds) := by
    exact lex_einsumPattern_render_eq_ok sourcePattern hWords
  cases hTargetLex :
      lex sourcePattern.render IdentifierPolicy.pythonUnicode
        LexicalWhitespace.transformation with
  | error diagnostic =>
      rw [hTargetLex] at hTargetLexValues
      contradiction
  | ok targetTokens =>
      rw [hTargetLex] at hTargetLexValues
      have hTargetKinds :
          targetTokens.map Located.value =
            ((sourceInputs.map Expression.tokenKinds).intersperse
                [.comma]).flatten ++
              [.arrow] ++ sourceOutput.tokenKinds := by
        simpa [Except.map] using hTargetLexValues
      have hTokens :
          targetTokens.map Located.value =
            sourceTokens.map fun token =>
              canonicalTokenKind IdentifierPolicy.pythonUnicode
                token.value :=
        hTargetKinds.trans hRenderedKinds
      let targetWhole := sourceSpan sourcePattern.render
      obtain
          ⟨targetInputTokens, targetArrow, targetOutputTokens,
            hTargetSplit, hTargetInputKinds, hTargetOutputKinds⟩ :=
        splitArrow_canonical
          (Span.point (sourceSpan source).stop)
          (Span.point targetWhole.stop)
          sourceTokens targetTokens [] [] sourceInputTokens
          sourceOutputTokens sourceArrow hTokens (by simp) hSourceSplit
      have hTargetGroups :
          (splitCommas targetInputTokens [] []).map
              (List.map Located.value) =
            (splitCommas sourceInputTokens [] []).map fun group =>
              group.map fun token =>
                canonicalTokenKind IdentifierPolicy.pythonUnicode
                  token.value := by
        exact
          splitCommas_canonical sourceInputTokens targetInputTokens
            [] [] [] [] hTargetInputKinds (by simp) (by simp)
      obtain
          ⟨targetInputs, hTargetInputsParse, hTargetInputsKinds⟩ :=
        parseEinsumInputs_canonical
          (splitCommas sourceInputTokens [] [])
          (splitCommas targetInputTokens [] [])
          [] [] sourceInputs hTargetGroups (by simp) hSourceInputs
      have hInitialState :
          ExpressionState.CanonicalEq
            ({} : ExpressionState) ({} : ExpressionState) :=
        ⟨rfl, rfl, rfl⟩
      obtain
          ⟨targetOutput, hTargetOutputParse, hTargetOutputExpressionKinds⟩ :=
        parseExpressionTokens_canonical .einsumOutput
          (Span.between sourceArrow.span.stop (sourceSpan source).stop)
          (Span.between targetArrow.span.stop targetWhole.stop)
          sourceOutputTokens targetOutputTokens {} {} sourceOutput
          hTargetOutputKinds hInitialState hSourceOutput
      have hTargetParse :
          parseEinsumPattern sourcePattern.render =
            .ok
              { inputs := targetInputs
                arrow := targetArrow.span
                output := targetOutput
                span := targetWhole } := by
        exact
          parseEinsumPattern_eq_ok_of_components sourcePattern.render
            targetTokens targetInputTokens targetOutputTokens targetArrow
            targetInputs targetOutput hTargetLex hTargetSplit
            hTargetInputsParse hTargetOutputParse
      change
        (parseEinsumPattern sourcePattern.render).map
            EinsumPattern.render =
          .ok sourcePattern.render
      rw [hTargetParse]
      simp only [Except.map]
      have hRender :
          ({ inputs := targetInputs
             arrow := targetArrow.span
             output := targetOutput
             span := targetWhole } :
            EinsumPattern).render =
            sourcePattern.render := by
        simp only [EinsumPattern.render, sourcePattern]
        rw [hTargetInputsKinds, hTargetOutputExpressionKinds]
      exact congrArg Except.ok hRender

end TorchLean.Tensor.Internal.Syntax
