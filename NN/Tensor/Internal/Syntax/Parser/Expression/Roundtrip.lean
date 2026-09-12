/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Expression.Parse
public import NN.Tensor.Internal.Syntax.Parser.Expression.Config -- shake: keep
public import NN.Tensor.Internal.Syntax.Render -- shake: keep

/-!
# Expression render round-trips

Successfully parsed expressions render to canonical source that parses back
to the same rendered token sequence.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

open Parser.Impl

/-- Parse one side of a transformation pattern. -/
def parseExpression (source : String)
    (config := ExpressionConfig.transformation)
    (policy := IdentifierPolicy.pythonUnicode) : Result Expression := do
  let tokens ← lex source policy .transformation
  parseExpressionTokens policy config (sourceSpan source) tokens {}

/-- Every word in a successfully parsed expression satisfies the Python word policy. -/
private theorem expression_words_valid_of_parseExpression_eq_ok
    (source : String) (config : ExpressionConfig) (expression : Expression)
    (hParse :
      parseExpression source config IdentifierPolicy.pythonUnicode =
        .ok expression) :
    ∀ text, TokenKind.word text ∈ expression.tokenKinds →
      text.toList ≠ [] ∧
        ∀ character ∈ text.toList,
          IdentifierPolicy.pythonUnicode.isWordChar character = true := by
  cases hLex :
      lex source IdentifierPolicy.pythonUnicode
        LexicalWhitespace.transformation with
  | error diagnostic =>
      unfold parseExpression at hParse
      rw [hLex] at hParse
      contradiction
  | ok tokens =>
      have hExpressionParse :
          parseExpressionTokens IdentifierPolicy.pythonUnicode config
              (sourceSpan source) tokens {} =
            .ok expression := by
        unfold parseExpression at hParse
        rw [hLex] at hParse
        exact hParse
      have hKinds :=
        tokenKinds_eq_of_parseExpressionTokens_eq_ok
          IdentifierPolicy.pythonUnicode config (sourceSpan source)
          tokens {} expression hExpressionParse
      simp [ExpressionState.tokenKinds] at hKinds
      intro text hText
      rw [hKinds] at hText
      exact
        canonical_words_valid_of_lex_eq_ok source .transformation tokens
          hLex text hText

/--
Parsing a successful expression's canonical rendering succeeds with the same
canonical syntax.

The reparsed tree has fresh source spans and fresh anonymous-axis occurrence
offsets. Comparing renderings captures every parser distinction used by later
checking while intentionally ignoring that source-location metadata.
-/
theorem parseExpression_render_eq_ok (source : String)
    (config : ExpressionConfig) (expression : Expression)
    (hParse : parseExpression source config = .ok expression) :
    (parseExpression expression.render config).map Expression.render =
      .ok expression.render := by
  have hWords :=
    expression_words_valid_of_parseExpression_eq_ok source config expression
      hParse
  have hTargetLexValues :
      (lex expression.render IdentifierPolicy.pythonUnicode
          LexicalWhitespace.transformation).map (List.map Located.value) =
        .ok expression.tokenKinds := by
    simpa [Expression.render] using
      TokenKind.lex_renderSequence_transformation expression.tokenKinds hWords
  cases hSourceLex :
      lex source IdentifierPolicy.pythonUnicode
        LexicalWhitespace.transformation with
  | error diagnostic =>
      unfold parseExpression at hParse
      rw [hSourceLex] at hParse
      contradiction
  | ok sourceTokens =>
      have hSourceParse :
          parseExpressionTokens IdentifierPolicy.pythonUnicode config
              (sourceSpan source) sourceTokens {} =
            .ok expression := by
        unfold parseExpression at hParse
        rw [hSourceLex] at hParse
        exact hParse
      have hSourceKinds :=
        tokenKinds_eq_of_parseExpressionTokens_eq_ok
          IdentifierPolicy.pythonUnicode config (sourceSpan source)
          sourceTokens {} expression hSourceParse
      simp [ExpressionState.tokenKinds] at hSourceKinds
      cases hTargetLex :
          lex expression.render IdentifierPolicy.pythonUnicode
            LexicalWhitespace.transformation with
      | error diagnostic =>
          rw [hTargetLex] at hTargetLexValues
          contradiction
      | ok targetTokens =>
          rw [hTargetLex] at hTargetLexValues
          have hTargetTokenKinds :
              targetTokens.map Located.value = expression.tokenKinds := by
            simpa [Except.map] using hTargetLexValues
          have hTokens :
              targetTokens.map Located.value =
                sourceTokens.map fun token =>
                  canonicalTokenKind IdentifierPolicy.pythonUnicode
                    token.value := by
            exact hTargetTokenKinds.trans hSourceKinds
          have hInitialState :
              ExpressionState.CanonicalEq
                ({} : ExpressionState) ({} : ExpressionState) :=
            ⟨rfl, rfl, rfl⟩
          obtain
              ⟨targetExpression, hTargetParse, hTargetKinds⟩ :=
            parseExpressionTokens_canonical config
              (sourceSpan source) (sourceSpan expression.render)
              sourceTokens targetTokens {} {} expression hTokens
              hInitialState hSourceParse
          unfold parseExpression
          rw [hTargetLex]
          change
            (parseExpressionTokens IdentifierPolicy.pythonUnicode config
                (sourceSpan expression.render) targetTokens {}).map
                Expression.render =
              .ok expression.render
          rw [hTargetParse]
          simp only [Except.map]
          have hRender :
              targetExpression.render = expression.render :=
            congrArg TokenKind.renderSequence hTargetKinds
          exact congrArg Except.ok hRender


end TorchLean.Tensor.Internal.Syntax
