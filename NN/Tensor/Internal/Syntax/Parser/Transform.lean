/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Expression.Parse
public import NN.Tensor.Internal.Syntax.Parser.Expression.Split
public import NN.Tensor.Internal.Syntax.Parser.Expression -- shake: keep

/-!
# Transformation-pattern parsing

This module parses and proves canonical round trips for `rearrange`, `repeat`,
and `reduce` patterns.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

open Parser.Impl

/-- Parse the shared pattern language of `rearrange`, `repeat`, and `reduce`. -/
def parseTransformPattern (source : String)
    (policy := IdentifierPolicy.pythonUnicode) : Result TransformPattern := do
  let whole := sourceSpan source
  let tokens ← lex source policy .transformation
  let (leftTokens, arrow, rightTokens) ← splitArrow (Span.point whole.stop) tokens []
  let leftSpan := Span.between whole.offset arrow.span.offset
  let rightSpan := Span.between arrow.span.stop whole.stop
  let left ← parseExpressionTokens policy .transformation leftSpan leftTokens {}
  let right ← parseExpressionTokens policy .transformation rightSpan rightTokens {}
  .ok { left, arrow := arrow.span, right, span := whole }

/-- Successful component parses assemble into a successful transformation parse. -/
private theorem parseTransformPattern_eq_ok_of_components
    (source : String)
    (tokens leftTokens rightTokens : List Token) (arrow : Token)
    (left right : Expression)
    (hLex :
      lex source IdentifierPolicy.pythonUnicode .transformation = .ok tokens)
    (hSplit :
      splitArrow (Span.point (sourceSpan source).stop) tokens [] =
        .ok (leftTokens, arrow, rightTokens))
    (hLeft :
      parseExpressionTokens IdentifierPolicy.pythonUnicode .transformation
          (Span.between (sourceSpan source).offset arrow.span.offset)
          leftTokens {} =
        .ok left)
    (hRight :
      parseExpressionTokens IdentifierPolicy.pythonUnicode .transformation
          (Span.between arrow.span.stop (sourceSpan source).stop)
          rightTokens {} =
        .ok right) :
    parseTransformPattern source =
      .ok
        { left
          arrow := arrow.span
          right
          span := sourceSpan source } := by
  have hAfterRight :
      (Except.ok
          ({ left
             arrow := arrow.span
             right
             span := sourceSpan source } :
            TransformPattern) :
        Result TransformPattern) =
        .ok
          { left
            arrow := arrow.span
            right
            span := sourceSpan source } :=
    rfl
  have hAfterLeft :
      (do
        let parsedRight ←
          parseExpressionTokens IdentifierPolicy.pythonUnicode
            .transformation
            (Span.between arrow.span.stop (sourceSpan source).stop)
            rightTokens {}
        .ok
          ({ left
             arrow := arrow.span
             right := parsedRight
             span := sourceSpan source } :
            TransformPattern)) =
        .ok
          { left
            arrow := arrow.span
            right
            span := sourceSpan source } := by
    rw [hRight]
    exact hAfterRight
  have hAfterSplit :
      (do
        let parsedLeft ←
          parseExpressionTokens IdentifierPolicy.pythonUnicode
            .transformation
            (Span.between (sourceSpan source).offset arrow.span.offset)
            leftTokens {}
        let parsedRight ←
          parseExpressionTokens IdentifierPolicy.pythonUnicode
            .transformation
            (Span.between arrow.span.stop (sourceSpan source).stop)
            rightTokens {}
        .ok
          ({ left := parsedLeft
             arrow := arrow.span
             right := parsedRight
             span := sourceSpan source } :
            TransformPattern)) =
        .ok
          { left
            arrow := arrow.span
            right
            span := sourceSpan source } := by
    rw [hLeft]
    exact hAfterLeft
  have hAfterLex :
      (do
        let split ←
          splitArrow (Span.point (sourceSpan source).stop) tokens []
        let parsedLeft ←
          parseExpressionTokens IdentifierPolicy.pythonUnicode
            .transformation
            (Span.between (sourceSpan source).offset
              split.2.1.span.offset)
            split.1 {}
        let parsedRight ←
          parseExpressionTokens IdentifierPolicy.pythonUnicode
            .transformation
            (Span.between split.2.1.span.stop (sourceSpan source).stop)
            split.2.2 {}
        .ok
          ({ left := parsedLeft
             arrow := split.2.1.span
             right := parsedRight
             span := sourceSpan source } :
            TransformPattern)) =
        .ok
          { left
            arrow := arrow.span
            right
            span := sourceSpan source } := by
    rw [hSplit]
    exact hAfterSplit
  unfold parseTransformPattern
  rw [hLex]
  exact hAfterLex

/-- A successful transformation parse decomposes into lexing, splitting, and two expressions. -/
private theorem parseTransformPattern_eq_ok_decomposition
    (source : String) (pattern : TransformPattern)
    (hParse : parseTransformPattern source = .ok pattern) :
    ∃ (sourceTokens sourceLeft sourceRight : List Token)
        (sourceArrow : Token)
        (sourceLeftExpression sourceRightExpression : Expression),
      lex source IdentifierPolicy.pythonUnicode .transformation =
          .ok sourceTokens ∧
        splitArrow (Span.point (sourceSpan source).stop) sourceTokens [] =
          .ok (sourceLeft, sourceArrow, sourceRight) ∧
        parseExpressionTokens IdentifierPolicy.pythonUnicode .transformation
            (Span.between (sourceSpan source).offset sourceArrow.span.offset)
            sourceLeft {} =
          .ok sourceLeftExpression ∧
        parseExpressionTokens IdentifierPolicy.pythonUnicode .transformation
            (Span.between sourceArrow.span.stop (sourceSpan source).stop)
            sourceRight {} =
          .ok sourceRightExpression ∧
        pattern =
          { left := sourceLeftExpression
            arrow := sourceArrow.span
            right := sourceRightExpression
            span := sourceSpan source } := by
  unfold parseTransformPattern at hParse
  cases hSourceLex :
      lex source IdentifierPolicy.pythonUnicode .transformation with
  | error diagnostic =>
      rw [hSourceLex] at hParse
      contradiction
  | ok sourceTokens =>
      have hAfterLex :
          (do
            let split ←
              splitArrow (Span.point (sourceSpan source).stop)
                sourceTokens []
            let left ←
              parseExpressionTokens IdentifierPolicy.pythonUnicode
                .transformation
                (Span.between (sourceSpan source).offset
                  split.2.1.span.offset)
                split.1 {}
            let right ←
              parseExpressionTokens IdentifierPolicy.pythonUnicode
                .transformation
                (Span.between split.2.1.span.stop
                  (sourceSpan source).stop)
                split.2.2 {}
            .ok
              ({ left
                 arrow := split.2.1.span
                 right
                 span := sourceSpan source } :
                TransformPattern)) =
            .ok pattern := by
        rw [hSourceLex] at hParse
        exact hParse
      cases hSourceSplit :
          splitArrow (Span.point (sourceSpan source).stop)
            sourceTokens [] with
      | error diagnostic =>
          rw [hSourceSplit] at hAfterLex
          contradiction
      | ok sourceSplit =>
          obtain ⟨sourceLeft, sourceArrow, sourceRight⟩ := sourceSplit
          have hAfterSplit :
              (do
                let left ←
                  parseExpressionTokens IdentifierPolicy.pythonUnicode
                    .transformation
                    (Span.between (sourceSpan source).offset
                      sourceArrow.span.offset)
                    sourceLeft {}
                let right ←
                  parseExpressionTokens IdentifierPolicy.pythonUnicode
                    .transformation
                    (Span.between sourceArrow.span.stop
                      (sourceSpan source).stop)
                    sourceRight {}
                .ok
                  ({ left
                     arrow := sourceArrow.span
                     right
                     span := sourceSpan source } :
                    TransformPattern)) =
                .ok pattern := by
            rw [hSourceSplit] at hAfterLex
            exact hAfterLex
          cases hLeftParse :
              parseExpressionTokens IdentifierPolicy.pythonUnicode
                .transformation
                (Span.between (sourceSpan source).offset
                  sourceArrow.span.offset)
                sourceLeft {} with
          | error diagnostic =>
              rw [hLeftParse] at hAfterSplit
              contradiction
          | ok sourceLeftExpression =>
              have hAfterLeft :
                  (do
                    let right ←
                      parseExpressionTokens IdentifierPolicy.pythonUnicode
                        .transformation
                        (Span.between sourceArrow.span.stop
                          (sourceSpan source).stop)
                        sourceRight {}
                    .ok
                      ({ left := sourceLeftExpression
                         arrow := sourceArrow.span
                         right
                         span := sourceSpan source } :
                        TransformPattern)) =
                    .ok pattern := by
                rw [hLeftParse] at hAfterSplit
                exact hAfterSplit
              cases hRightParse :
                  parseExpressionTokens IdentifierPolicy.pythonUnicode
                    .transformation
                    (Span.between sourceArrow.span.stop
                      (sourceSpan source).stop)
                    sourceRight {} with
              | error diagnostic =>
                  rw [hRightParse] at hAfterLeft
                  contradiction
              | ok sourceRightExpression =>
                  have hResult :
                      (Except.ok
                          ({ left := sourceLeftExpression
                             arrow := sourceArrow.span
                             right := sourceRightExpression
                             span := sourceSpan source } :
                            TransformPattern) :
                        Result TransformPattern) =
                        .ok pattern := by
                    rw [hRightParse] at hAfterLeft
                    exact hAfterLeft
                  injection hResult with hPattern
                  exact
                    ⟨sourceTokens, sourceLeft, sourceRight, sourceArrow,
                      sourceLeftExpression, sourceRightExpression,
                      rfl, hSourceSplit, hLeftParse, hRightParse,
                      hPattern.symm⟩

/-- Lexing a rendered transformation recovers its canonical token sequence. -/
private theorem lex_transformPattern_render_eq_ok
    (pattern : TransformPattern)
    (hWords :
      ∀ text,
        .word text ∈
            pattern.left.tokenKinds ++ [.arrow] ++
              pattern.right.tokenKinds →
          text.toList ≠ [] ∧
            ∀ character ∈ text.toList,
              IdentifierPolicy.pythonUnicode.isWordChar character = true) :
    (lex pattern.render IdentifierPolicy.pythonUnicode
        LexicalWhitespace.transformation).map (List.map Located.value) =
      .ok
        (pattern.left.tokenKinds ++ [.arrow] ++
          pattern.right.tokenKinds) := by
  rw [TransformPattern.render]
  apply TokenKind.lex_renderSequence_transformation
  exact hWords

/--
Parsing a successful transformation pattern's canonical rendering succeeds
with the same canonical syntax.

This applies uniformly to patterns used by `rearrange`, `repeat`, and
`reduce`; operation-specific shape conditions belong to the checker rather
than this syntax theorem.
-/
theorem parseTransformPattern_render_eq_ok (source : String)
    (pattern : TransformPattern)
    (hParse : parseTransformPattern source = .ok pattern) :
    (parseTransformPattern pattern.render).map TransformPattern.render =
      .ok pattern.render := by
  obtain
      ⟨sourceTokens, sourceLeft, sourceRight, sourceArrow,
        sourceLeftExpression, sourceRightExpression, hSourceLex,
        hSourceSplit, hLeftParse, hRightParse, hPattern⟩ :=
    parseTransformPattern_eq_ok_decomposition source pattern hParse
  subst pattern
  let sourceWhole := sourceSpan source
  let sourceLeftSpan :=
    Span.between sourceWhole.offset sourceArrow.span.offset
  let sourceRightSpan :=
    Span.between sourceArrow.span.stop sourceWhole.stop
  ·
                  have hLeftKinds :=
                    tokenKinds_eq_of_parseExpressionTokens_eq_ok
                      IdentifierPolicy.pythonUnicode .transformation
                      sourceLeftSpan sourceLeft {} sourceLeftExpression
                      hLeftParse
                  have hRightKinds :=
                    tokenKinds_eq_of_parseExpressionTokens_eq_ok
                      IdentifierPolicy.pythonUnicode .transformation
                      sourceRightSpan sourceRight {} sourceRightExpression
                      hRightParse
                  simp [ExpressionState.tokenKinds] at hLeftKinds hRightKinds
                  obtain ⟨hSourceTokens, hSourceArrow⟩ :=
                    splitArrow_eq_ok_decomposition
                      (Span.point sourceWhole.stop) sourceTokens []
                      sourceLeft sourceRight sourceArrow hSourceSplit
                  simp only [List.reverse_nil, List.nil_append] at hSourceTokens
                  have hRenderedKinds :
                      sourceLeftExpression.tokenKinds ++ [.arrow] ++
                          sourceRightExpression.tokenKinds =
                        sourceTokens.map fun token =>
                          canonicalTokenKind IdentifierPolicy.pythonUnicode
                            token.value := by
                    calc
                      sourceLeftExpression.tokenKinds ++ [.arrow] ++
                            sourceRightExpression.tokenKinds =
                          (sourceLeft.map fun token =>
                              canonicalTokenKind
                                IdentifierPolicy.pythonUnicode token.value) ++
                            [.arrow] ++
                            sourceRight.map fun token =>
                              canonicalTokenKind
                                IdentifierPolicy.pythonUnicode token.value := by
                        rw [hLeftKinds, hRightKinds]
                      _ =
                          (sourceLeft ++ sourceArrow :: sourceRight).map
                            fun token =>
                              canonicalTokenKind
                                IdentifierPolicy.pythonUnicode token.value := by
                        simp [hSourceArrow, canonicalTokenKind,
                          List.append_assoc]
                      _ =
                          sourceTokens.map fun token =>
                            canonicalTokenKind
                              IdentifierPolicy.pythonUnicode token.value := by
                        rw [← hSourceTokens]
                  have hWords :
                      ∀ text,
                        .word text ∈
                            sourceLeftExpression.tokenKinds ++ [.arrow] ++
                              sourceRightExpression.tokenKinds →
                          text.toList ≠ [] ∧
                            ∀ character ∈ text.toList,
                              IdentifierPolicy.pythonUnicode.isWordChar
                                character = true := by
                    intro text hText
                    apply
                      canonical_words_valid_of_lex_eq_ok source
                        .transformation sourceTokens hSourceLex text
                    rw [← hRenderedKinds]
                    exact hText
                  have hTargetLexValues :
                      (lex
                          ({ left := sourceLeftExpression
                             arrow := sourceArrow.span
                             right := sourceRightExpression
                             span := sourceWhole } :
                            TransformPattern).render
                          IdentifierPolicy.pythonUnicode
                          LexicalWhitespace.transformation).map
                            (List.map Located.value) =
                        .ok
                          (sourceLeftExpression.tokenKinds ++ [.arrow] ++
                            sourceRightExpression.tokenKinds) := by
                    exact
                      lex_transformPattern_render_eq_ok
                        { left := sourceLeftExpression
                          arrow := sourceArrow.span
                          right := sourceRightExpression
                          span := sourceWhole }
                        hWords
                  cases hTargetLex :
                      lex
                        ({ left := sourceLeftExpression
                           arrow := sourceArrow.span
                           right := sourceRightExpression
                           span := sourceWhole } :
                          TransformPattern).render
                        IdentifierPolicy.pythonUnicode
                        LexicalWhitespace.transformation with
                  | error diagnostic =>
                      rw [hTargetLex] at hTargetLexValues
                      contradiction
                  | ok targetTokens =>
                      rw [hTargetLex] at hTargetLexValues
                      have hTargetKinds :
                          targetTokens.map Located.value =
                            sourceLeftExpression.tokenKinds ++ [.arrow] ++
                              sourceRightExpression.tokenKinds := by
                        simpa [Except.map] using hTargetLexValues
                      have hTokens :
                          targetTokens.map Located.value =
                            sourceTokens.map fun token =>
                              canonicalTokenKind
                                IdentifierPolicy.pythonUnicode
                                token.value :=
                        hTargetKinds.trans hRenderedKinds
                      let targetWhole :=
                        sourceSpan
                          ({ left := sourceLeftExpression
                             arrow := sourceArrow.span
                             right := sourceRightExpression
                             span := sourceWhole } :
                            TransformPattern).render
                      obtain
                          ⟨targetLeft, targetArrow, targetRight,
                            hTargetSplit, hTargetLeft, hTargetRight⟩ :=
                        splitArrow_canonical
                          (Span.point sourceWhole.stop)
                          (Span.point targetWhole.stop)
                          sourceTokens targetTokens [] [] sourceLeft
                          sourceRight sourceArrow hTokens (by simp)
                          hSourceSplit
                      let targetLeftSpan :=
                        Span.between targetWhole.offset
                          targetArrow.span.offset
                      let targetRightSpan :=
                        Span.between targetArrow.span.stop targetWhole.stop
                      have hInitialState :
                          ExpressionState.CanonicalEq
                            ({} : ExpressionState)
                            ({} : ExpressionState) :=
                        ⟨rfl, rfl, rfl⟩
                      obtain
                          ⟨targetLeftExpression, hTargetLeftParse,
                            hTargetLeftKinds⟩ :=
                        parseExpressionTokens_canonical .transformation
                          sourceLeftSpan targetLeftSpan sourceLeft
                          targetLeft {} {} sourceLeftExpression hTargetLeft
                          hInitialState hLeftParse
                      obtain
                          ⟨targetRightExpression, hTargetRightParse,
                            hTargetRightKinds⟩ :=
                        parseExpressionTokens_canonical .transformation
                          sourceRightSpan targetRightSpan sourceRight
                          targetRight {} {} sourceRightExpression
                          hTargetRight hInitialState hRightParse
                      have hTargetParse :
                          parseTransformPattern
                              ({ left := sourceLeftExpression
                                 arrow := sourceArrow.span
                                 right := sourceRightExpression
                                 span := sourceWhole } :
                                TransformPattern).render =
                            .ok
                              { left := targetLeftExpression
                                arrow := targetArrow.span
                                right := targetRightExpression
                                span := targetWhole } := by
                        exact
                          parseTransformPattern_eq_ok_of_components
                            ({ left := sourceLeftExpression
                               arrow := sourceArrow.span
                               right := sourceRightExpression
                               span := sourceWhole } :
                              TransformPattern).render
                            targetTokens targetLeft targetRight targetArrow
                            targetLeftExpression targetRightExpression
                            hTargetLex hTargetSplit hTargetLeftParse
                            hTargetRightParse
                      rw [hTargetParse]
                      simp only [Except.map]
                      have hRender :
                          ({ left := targetLeftExpression
                             arrow := targetArrow.span
                             right := targetRightExpression
                             span := targetWhole } :
                            TransformPattern).render =
                            ({ left := sourceLeftExpression
                               arrow := sourceArrow.span
                               right := sourceRightExpression
                               span := sourceWhole } :
                              TransformPattern).render := by
                        simp only [TransformPattern.render]
                        rw [hTargetLeftKinds, hTargetRightKinds]
                      exact congrArg Except.ok hRender

end TorchLean.Tensor.Internal.Syntax
