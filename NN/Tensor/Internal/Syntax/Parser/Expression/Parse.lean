/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Expression.State
public import NN.Tensor.Internal.Syntax.Parser.Expression.Config -- shake: keep
public import NN.Tensor.Internal.Syntax.Lexer -- shake: keep
public import NN.Tensor.Internal.Syntax.Render -- shake: keep

/-!
# Expression token parsing

The recursive parser and its canonical-token theorem turn a checked token
stream into one physical-axis expression.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

namespace Parser.Impl

/--
Parse one indexing expression while maintaining grouping and duplicate-axis
state.
-/
def parseExpressionTokens (policy : IdentifierPolicy)
    (config : ExpressionConfig) (expressionSpan : Span) :
    List Token → ExpressionState → Result Expression
  | [], state =>
      match state.group with
      | some (openSpan, _) =>
          .error
            { code := .unbalancedParenthesis
              message := "unclosed parenthesis in indexing expression"
              span := Span.join openSpan (Span.point expressionSpan.stop) }
      | none =>
          .ok { axes := state.axesRev.reverse, span := expressionSpan }
  | token :: rest, state => do
      match token.value with
      | .word _ =>
          let (axis, state) ← decodeWord policy config token state
          parseExpressionTokens policy config expressionSpan rest (pushAxis axis state)
      | .ellipsis =>
          let (axis, state) ← decodeEllipsis config token state
          parseExpressionTokens policy config expressionSpan rest (pushAxis axis state)
      | .leftParen =>
          match state.group with
          | some _ =>
              .error
                { code := .nestedParenthesis
                  message := "axis composition is one-level"
                  span := token.span }
          | none =>
              parseExpressionTokens policy config expressionSpan rest
                { state with group := some (token.span, []) }
      | .rightParen =>
          match state.group with
          | none =>
              .error
                { code := .unbalancedParenthesis
                  message := "closing parenthesis has no matching opening parenthesis"
                  span := token.span }
          | some (openSpan, axesRev) =>
              let composite :=
                { axes := axesRev.reverse
                  parenthesized := true
                  span := Span.join openSpan token.span }
              parseExpressionTokens policy config expressionSpan rest
                { state with group := none, axesRev := composite :: state.axesRev }
      | _ =>
          .error
            { code := .unexpectedToken
              message := "unexpected punctuation in indexing expression"
              span := token.span }

/-- A successful expression parse records the canonical token sequence it consumed. -/
theorem tokenKinds_eq_of_parseExpressionTokens_eq_ok
    (policy : IdentifierPolicy) (config : ExpressionConfig)
    (expressionSpan : Span) (tokens : List Token)
    (state : ExpressionState) (expression : Expression)
    (hParse :
      parseExpressionTokens policy config expressionSpan tokens state =
        .ok expression) :
    expression.tokenKinds =
      state.tokenKinds ++
        tokens.map fun token => canonicalTokenKind policy token.value := by
  induction tokens generalizing state with
  | nil =>
      cases hGroup : state.group with
      | none =>
          simp [parseExpressionTokens, hGroup] at hParse
          subst expression
          simp [Expression.tokenKinds, ExpressionState.tokenKinds, hGroup]
      | some group =>
          simp [parseExpressionTokens, hGroup] at hParse
  | cons token tokens induction =>
      cases hValue : token.value with
      | word text =>
          cases hDecode :
              decodeWord policy config token state with
          | error diagnostic =>
              simp [parseExpressionTokens, hValue, hDecode] at hParse
              change
                (Except.error diagnostic : Result Expression) =
                  .ok expression at hParse
              contradiction
          | ok decoded =>
              obtain ⟨axis, nextState⟩ := decoded
              simp [parseExpressionTokens, hValue, hDecode] at hParse
              change
                parseExpressionTokens policy config expressionSpan tokens
                    (pushAxis axis nextState) =
                  .ok expression at hParse
              rw [induction (state := pushAxis axis nextState) hParse]
              rw [tokenKinds_pushAxis_decodeWord policy config token state
                axis nextState hDecode]
              simp [List.append_assoc]
      | ellipsis =>
          cases hDecode :
              decodeEllipsis config token state with
          | error diagnostic =>
              simp [parseExpressionTokens, hValue, hDecode] at hParse
              change
                (Except.error diagnostic : Result Expression) =
                  .ok expression at hParse
              contradiction
          | ok decoded =>
              obtain ⟨axis, nextState⟩ := decoded
              simp [parseExpressionTokens, hValue, hDecode] at hParse
              change
                parseExpressionTokens policy config expressionSpan tokens
                    (pushAxis axis nextState) =
                  .ok expression at hParse
              rw [induction (state := pushAxis axis nextState) hParse]
              rw [tokenKinds_pushAxis_decodeEllipsis config token state
                axis nextState hDecode]
              simp [canonicalTokenKind, hValue, List.append_assoc]
      | leftParen =>
          cases hGroup : state.group with
          | none =>
              simp [parseExpressionTokens, hValue, hGroup] at hParse
              rw [induction
                (state := { state with group := some (token.span, []) })
                hParse]
              simp [ExpressionState.tokenKinds, hGroup, canonicalTokenKind,
                hValue, List.append_assoc]
          | some group =>
              simp [parseExpressionTokens, hValue, hGroup] at hParse
      | rightParen =>
          cases hGroup : state.group with
          | none =>
              simp [parseExpressionTokens, hValue, hGroup] at hParse
          | some group =>
              obtain ⟨openSpan, axesRev⟩ := group
              simp [parseExpressionTokens, hValue, hGroup] at hParse
              rw [induction
                (state :=
                  { state with
                    group := none
                    axesRev :=
                      { axes := axesRev.reverse
                        parenthesized := true
                        span := Span.join openSpan token.span } ::
                        state.axesRev })
                hParse]
              simp [ExpressionState.tokenKinds, hGroup,
                CompositeAxis.tokenKinds, canonicalTokenKind, hValue,
                List.append_assoc]
      | arrow =>
          simp [parseExpressionTokens, hValue] at hParse
      | comma =>
          simp [parseExpressionTokens, hValue] at hParse
      | star =>
          simp [parseExpressionTokens, hValue] at hParse

/-- Canonically equivalent token streams parse to expressions with equal token structure. -/
theorem parseExpressionTokens_canonical
    (config : ExpressionConfig)
    (sourceExpressionSpan targetExpressionSpan : Span)
    (sourceTokens targetTokens : List Token)
    (sourceState targetState : ExpressionState)
    (sourceExpression : Expression)
    (hTokens :
      targetTokens.map Located.value =
        sourceTokens.map fun token =>
          canonicalTokenKind IdentifierPolicy.pythonUnicode token.value)
    (hState : ExpressionState.CanonicalEq sourceState targetState)
    (hParse :
      parseExpressionTokens IdentifierPolicy.pythonUnicode config
          sourceExpressionSpan sourceTokens sourceState =
        .ok sourceExpression) :
    ∃ targetExpression,
      parseExpressionTokens IdentifierPolicy.pythonUnicode config
          targetExpressionSpan targetTokens targetState =
          .ok targetExpression ∧
        targetExpression.tokenKinds = sourceExpression.tokenKinds := by
  induction sourceTokens generalizing
      targetTokens sourceState targetState sourceExpression with
  | nil =>
      cases targetTokens with
      | nil =>
          cases hSourceGroup : sourceState.group with
          | none =>
              simp [parseExpressionTokens, hSourceGroup] at hParse
              subst sourceExpression
              cases hTargetGroup : targetState.group with
              | none =>
                  refine
                    ⟨{ axes := targetState.axesRev.reverse
                       span := targetExpressionSpan },
                      ?_, ?_⟩
                  · simp [parseExpressionTokens, hTargetGroup]
                  · simpa [Expression.tokenKinds, ExpressionState.tokenKinds,
                      hSourceGroup, hTargetGroup] using hState.2.1.symm
              | some group =>
                  have hGroups := hState.2.2
                  simp [hSourceGroup, hTargetGroup] at hGroups
          | some group =>
              simp [parseExpressionTokens, hSourceGroup] at hParse
      | cons targetToken targetTokens =>
          simp at hTokens
  | cons sourceToken sourceTokens induction =>
      cases targetTokens with
      | nil =>
          simp at hTokens
      | cons targetToken targetTokens =>
          simp only [List.map_cons, List.cons.injEq] at hTokens
          rcases hTokens with ⟨hValue, hTokens⟩
          cases hSourceValue : sourceToken.value with
          | word sourceText =>
              cases hDecode :
                  decodeWord IdentifierPolicy.pythonUnicode config sourceToken
                    sourceState with
              | error diagnostic =>
                  simp [parseExpressionTokens, hSourceValue, hDecode] at hParse
                  change
                    (Except.error diagnostic : Result Expression) =
                      .ok sourceExpression at hParse
                  contradiction
              | ok decoded =>
                  obtain ⟨sourceAxis, sourceNextState⟩ := decoded
                  simp [parseExpressionTokens, hSourceValue, hDecode] at hParse
                  change
                    parseExpressionTokens IdentifierPolicy.pythonUnicode config
                        sourceExpressionSpan sourceTokens
                        (pushAxis sourceAxis sourceNextState) =
                      .ok sourceExpression at hParse
                  obtain
                      ⟨targetAxis, targetNextState, hTargetDecode, hNextState⟩ :=
                    decodeWord_canonical config sourceToken targetToken
                      sourceState targetState sourceAxis sourceNextState
                      hValue hState hDecode
                  obtain ⟨targetText, hTargetValue⟩ :
                      ∃ targetText, targetToken.value = .word targetText := by
                    cases hTargetValue : targetToken.value with
                    | word targetText => exact ⟨targetText, rfl⟩
                    | leftParen =>
                        simp [decodeWord, hTargetValue] at hTargetDecode
                    | rightParen =>
                        simp [decodeWord, hTargetValue] at hTargetDecode
                    | ellipsis =>
                        simp [decodeWord, hTargetValue] at hTargetDecode
                    | arrow =>
                        simp [decodeWord, hTargetValue] at hTargetDecode
                    | comma =>
                        simp [decodeWord, hTargetValue] at hTargetDecode
                    | star =>
                        simp [decodeWord, hTargetValue] at hTargetDecode
                  obtain ⟨targetExpression, hTargetParse, hTargetKinds⟩ :=
                    induction
                      (targetTokens := targetTokens)
                      (sourceState := pushAxis sourceAxis sourceNextState)
                      (targetState := pushAxis targetAxis targetNextState)
                      (sourceExpression := sourceExpression)
                      hTokens hNextState hParse
                  refine ⟨targetExpression, ?_, hTargetKinds⟩
                  simp only [parseExpressionTokens, hTargetValue]
                  rw [hTargetDecode]
                  change
                    parseExpressionTokens IdentifierPolicy.pythonUnicode config
                        targetExpressionSpan targetTokens
                        (pushAxis targetAxis targetNextState) =
                      .ok targetExpression
                  exact hTargetParse
          | ellipsis =>
              have hTargetValue : targetToken.value = .ellipsis := by
                simpa [canonicalTokenKind, hSourceValue] using hValue
              cases hDecode :
                  decodeEllipsis config sourceToken sourceState with
              | error diagnostic =>
                  simp [parseExpressionTokens, hSourceValue, hDecode] at hParse
                  change
                    (Except.error diagnostic : Result Expression) =
                      .ok sourceExpression at hParse
                  contradiction
              | ok decoded =>
                  obtain ⟨sourceAxis, sourceNextState⟩ := decoded
                  simp [parseExpressionTokens, hSourceValue, hDecode] at hParse
                  change
                    parseExpressionTokens IdentifierPolicy.pythonUnicode config
                        sourceExpressionSpan sourceTokens
                        (pushAxis sourceAxis sourceNextState) =
                      .ok sourceExpression at hParse
                  obtain
                      ⟨targetAxis, targetNextState, hTargetDecode, hNextState⟩ :=
                    decodeEllipsis_canonical config sourceToken targetToken
                      sourceState targetState sourceAxis sourceNextState
                      hState hDecode
                  obtain ⟨targetExpression, hTargetParse, hTargetKinds⟩ :=
                    induction
                      (targetTokens := targetTokens)
                      (sourceState := pushAxis sourceAxis sourceNextState)
                      (targetState := pushAxis targetAxis targetNextState)
                      (sourceExpression := sourceExpression)
                      hTokens hNextState hParse
                  refine ⟨targetExpression, ?_, hTargetKinds⟩
                  simp only [parseExpressionTokens, hTargetValue]
                  rw [hTargetDecode]
                  change
                    parseExpressionTokens IdentifierPolicy.pythonUnicode config
                        targetExpressionSpan targetTokens
                        (pushAxis targetAxis targetNextState) =
                      .ok targetExpression
                  exact hTargetParse
          | leftParen =>
              have hTargetValue : targetToken.value = .leftParen := by
                simpa [canonicalTokenKind, hSourceValue] using hValue
              cases hSourceGroup : sourceState.group with
              | none =>
                  simp [parseExpressionTokens, hSourceValue, hSourceGroup]
                    at hParse
                  cases hTargetGroup : targetState.group with
                  | none =>
                      have hNextState :=
                        ExpressionState.canonicalEq_openGroup
                          sourceState targetState sourceToken.span
                          targetToken.span hState hSourceGroup hTargetGroup
                      obtain ⟨targetExpression, hTargetParse, hTargetKinds⟩ :=
                        induction
                          (targetTokens := targetTokens)
                          (sourceState :=
                            { sourceState with
                              group := some (sourceToken.span, []) })
                          (targetState :=
                            { targetState with
                              group := some (targetToken.span, []) })
                          (sourceExpression := sourceExpression)
                          hTokens hNextState hParse
                      refine ⟨targetExpression, ?_, hTargetKinds⟩
                      simpa [parseExpressionTokens, hTargetValue, hTargetGroup]
                        using hTargetParse
                  | some group =>
                      have hGroups := hState.2.2
                      simp [hSourceGroup, hTargetGroup] at hGroups
              | some group =>
                  simp [parseExpressionTokens, hSourceValue, hSourceGroup]
                    at hParse
          | rightParen =>
              have hTargetValue : targetToken.value = .rightParen := by
                simpa [canonicalTokenKind, hSourceValue] using hValue
              cases hSourceGroup : sourceState.group with
              | none =>
                  simp [parseExpressionTokens, hSourceValue, hSourceGroup]
                    at hParse
              | some sourceGroup =>
                  obtain ⟨sourceOpenSpan, sourceAxesRev⟩ := sourceGroup
                  simp [parseExpressionTokens, hSourceValue, hSourceGroup]
                    at hParse
                  cases hTargetGroup : targetState.group with
                  | none =>
                      have hGroups := hState.2.2
                      simp [hSourceGroup, hTargetGroup] at hGroups
                  | some targetGroup =>
                      obtain ⟨targetOpenSpan, targetAxesRev⟩ := targetGroup
                      have hNextState :=
                        ExpressionState.canonicalEq_closeGroup
                          sourceState targetState sourceOpenSpan
                          targetOpenSpan sourceToken.span targetToken.span
                          sourceAxesRev targetAxesRev hState
                          hSourceGroup hTargetGroup
                      obtain ⟨targetExpression, hTargetParse, hTargetKinds⟩ :=
                        induction
                          (targetTokens := targetTokens)
                          (sourceState :=
                            { sourceState with
                              group := none
                              axesRev :=
                                { axes := sourceAxesRev.reverse
                                  parenthesized := true
                                  span :=
                                    Span.join sourceOpenSpan sourceToken.span } ::
                                  sourceState.axesRev })
                          (targetState :=
                            { targetState with
                              group := none
                              axesRev :=
                                { axes := targetAxesRev.reverse
                                  parenthesized := true
                                  span :=
                                    Span.join targetOpenSpan targetToken.span } ::
                                  targetState.axesRev })
                          (sourceExpression := sourceExpression)
                          hTokens hNextState hParse
                      refine ⟨targetExpression, ?_, hTargetKinds⟩
                      simpa [parseExpressionTokens, hTargetValue, hTargetGroup]
                        using hTargetParse
          | arrow =>
              simp [parseExpressionTokens, hSourceValue] at hParse
          | comma =>
              simp [parseExpressionTokens, hSourceValue] at hParse
          | star =>
              simp [parseExpressionTokens, hSourceValue] at hParse

/-- Canonicalized words from a successful Python-policy lex remain valid lexer words. -/
theorem canonical_words_valid_of_lex_eq_ok (source : String)
    (whitespace : LexicalWhitespace) (tokens : List Token)
    (hLex :
      lex source IdentifierPolicy.pythonUnicode whitespace = .ok tokens) :
    ∀ text,
      TokenKind.word text ∈
          tokens.map (fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value) →
        text.toList ≠ [] ∧
          ∀ character ∈ text.toList,
            IdentifierPolicy.pythonUnicode.isWordChar character = true := by
  intro text hText
  obtain ⟨token, hToken, hCanonical⟩ := List.mem_map.mp hText
  cases hValue : token.value with
  | word sourceText =>
      by_cases hDecimal :
          IdentifierPolicy.pythonUnicode.isDecimal sourceText = true
      · cases hNat :
          IdentifierPolicy.pythonUnicode.toNat? sourceText with
        | none =>
            simp [canonicalTokenKind, hValue, hDecimal, hNat] at hCanonical
            subst text
            exact
              word_valid_of_lex_eq_ok source IdentifierPolicy.pythonUnicode
                whitespace hLex hToken hValue
        | some value =>
            simp [canonicalTokenKind, hValue, hDecimal, hNat] at hCanonical
            subst text
            exact IdentifierPolicy.pythonUnicode_word_toString value
      · simp [canonicalTokenKind, hValue, hDecimal] at hCanonical
        subst text
        exact
          word_valid_of_lex_eq_ok source IdentifierPolicy.pythonUnicode
            whitespace hLex hToken hValue
  | leftParen => simp [canonicalTokenKind, hValue] at hCanonical
  | rightParen => simp [canonicalTokenKind, hValue] at hCanonical
  | ellipsis => simp [canonicalTokenKind, hValue] at hCanonical
  | arrow => simp [canonicalTokenKind, hValue] at hCanonical
  | comma => simp [canonicalTokenKind, hValue] at hCanonical
  | star => simp [canonicalTokenKind, hValue] at hCanonical

/-- The span covering an entire source string. -/
def sourceSpan (source : String) : Span :=
  ⟨0, source.toList.length⟩


end Parser.Impl

end TorchLean.Tensor.Internal.Syntax
