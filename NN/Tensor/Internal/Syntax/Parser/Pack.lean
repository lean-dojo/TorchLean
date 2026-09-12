/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Expression.Parse
public import NN.Tensor.Internal.Syntax.Parser.Expression -- shake: keep

/-!
# Pack-pattern parsing

This module parses the fixed axes surrounding a unique packed `*` axis and
proves canonical rendering round trips.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

open Parser.Impl

namespace Parser.Impl

/--
Parse the fixed axes before and after the unique packed `*` axis.
-/
def parsePackTokens (policy : IdentifierPolicy) (whole : Span) :
    List Token → List (Located String) → Option Span →
      List (Located String) → List String → Result PackPattern
  | [], _, none, _, _ =>
      .error
        { code := .missingPackAxis
          message := "pack pattern must contain exactly one '*'"
          span := Span.point whole.stop }
  | [], beforeRev, some packed, afterRev, _ =>
      .ok
        { before := beforeRev.reverse
          packed
          after := afterRev.reverse
          span := whole }
  | token :: rest, beforeRev, packed, afterRev, seen => do
      match token.value with
      | .star =>
          match packed with
          | some _ =>
              .error
                { code := .duplicatePackAxis
                  message := "pack pattern contains more than one '*'"
                  span := token.span }
          | none =>
              parsePackTokens policy whole rest beforeRev (some token.span) afterRev seen
      | .word name =>
          if !policy.isIdentifier name ||
              name.startsWith "_" || name.endsWith "_" then
            .error
              { code := .invalidIdentifier
                message := "invalid axis identifier in pack pattern"
                span := token.span }
          else if name ∈ seen then
            .error (duplicateDiagnostic token.span)
          else
            let located := Located.mk name token.span
            match packed with
            | none =>
                parsePackTokens policy whole rest (located :: beforeRev) packed afterRev
                  (name :: seen)
            | some _ =>
                parsePackTokens policy whole rest beforeRev packed (located :: afterRev)
                  (name :: seen)
      | _ =>
          .error
            { code := .unexpectedToken
              message := "pack patterns contain only axis names and one '*'"
              span := token.span }

end Parser.Impl

/-- A successful pack-token parse records the fixed axes and packed marker in order. -/
private theorem tokenKinds_eq_of_parsePackTokens_eq_ok
    (policy : IdentifierPolicy) (whole : Span) (tokens : List Token)
    (beforeRev : List (Located String)) (packed : Option Span)
    (afterRev : List (Located String)) (seen : List String)
    (pattern : PackPattern)
    (hAfterUnpacked : packed = none → afterRev = [])
    (hParse :
      parsePackTokens policy whole tokens beforeRev packed afterRev seen =
        .ok pattern) :
    pattern.before.map (fun axis => TokenKind.word axis.value) ++
          [.star] ++
          pattern.after.map (fun axis => TokenKind.word axis.value) =
      beforeRev.reverse.map (fun axis => TokenKind.word axis.value) ++
        match packed with
        | none => tokens.map Located.value
        | some _ =>
            [.star] ++
              afterRev.reverse.map
                  (fun axis => TokenKind.word axis.value) ++
                tokens.map Located.value := by
  induction tokens generalizing
      beforeRev packed afterRev seen pattern with
  | nil =>
      cases packed with
      | none =>
          simp [parsePackTokens] at hParse
      | some packedSpan =>
          simp [parsePackTokens] at hParse
          subst pattern
          simp [List.map_reverse, List.append_assoc]
  | cons token tokens induction =>
      cases hValue : token.value with
      | word name =>
          by_cases hInvalid :
              (!policy.isIdentifier name ||
                name.startsWith "_" || name.endsWith "_") = true
          · simp [parsePackTokens, hValue, hInvalid] at hParse
          · by_cases hDuplicate : name ∈ seen
            · simp [parsePackTokens, hValue, hInvalid, hDuplicate] at hParse
            · cases packed with
              | none =>
                  simp [parsePackTokens, hValue, hInvalid, hDuplicate] at hParse
                  simpa [hValue, List.reverse_cons, List.map_append,
                      List.append_assoc] using
                    induction
                      (beforeRev := ⟨name, token.span⟩ :: beforeRev)
                      (packed := none) (afterRev := afterRev)
                      (seen := name :: seen) (pattern := pattern)
                      hAfterUnpacked hParse
              | some packedSpan =>
                  simp [parsePackTokens, hValue, hInvalid, hDuplicate] at hParse
                  simpa [hValue, List.reverse_cons, List.map_append,
                      List.append_assoc] using
                    induction (beforeRev := beforeRev)
                      (packed := some packedSpan)
                      (afterRev := ⟨name, token.span⟩ :: afterRev)
                      (seen := name :: seen) (pattern := pattern)
                      (by simp) hParse
      | star =>
          cases packed with
          | none =>
              simp [parsePackTokens, hValue] at hParse
              have hAfterRev : afterRev = [] := hAfterUnpacked rfl
              subst afterRev
              simpa [hValue, List.append_assoc] using
                induction (beforeRev := beforeRev)
                  (packed := some token.span) (afterRev := [])
                  (seen := seen) (pattern := pattern) (by simp) hParse
          | some packedSpan =>
              simp [parsePackTokens, hValue] at hParse
      | leftParen => simp [parsePackTokens, hValue] at hParse
      | rightParen => simp [parsePackTokens, hValue] at hParse
      | ellipsis => simp [parsePackTokens, hValue] at hParse
      | arrow => simp [parsePackTokens, hValue] at hParse
      | comma => simp [parsePackTokens, hValue] at hParse

/-- Equivalent pack token streams parse to patterns with the same canonical rendering. -/
private theorem parsePackTokens_canonical
    (policy : IdentifierPolicy) (sourceWhole targetWhole : Span)
    (sourceTokens targetTokens : List Token)
    (sourceBeforeRev targetBeforeRev : List (Located String))
    (sourcePacked targetPacked : Option Span)
    (sourceAfterRev targetAfterRev : List (Located String))
    (seen : List String) (sourcePattern : PackPattern)
    (hTokens :
      targetTokens.map Located.value = sourceTokens.map Located.value)
    (hBefore :
      targetBeforeRev.map (fun axis => TokenKind.word axis.value) =
        sourceBeforeRev.map (fun axis => TokenKind.word axis.value))
    (hPacked : targetPacked.isSome = sourcePacked.isSome)
    (hAfter :
      targetAfterRev.map (fun axis => TokenKind.word axis.value) =
        sourceAfterRev.map (fun axis => TokenKind.word axis.value))
    (hParse :
      parsePackTokens policy sourceWhole sourceTokens sourceBeforeRev
          sourcePacked sourceAfterRev seen =
        .ok sourcePattern) :
    ∃ targetPattern,
      parsePackTokens policy targetWhole targetTokens targetBeforeRev
          targetPacked targetAfterRev seen =
          .ok targetPattern ∧
        targetPattern.render = sourcePattern.render := by
  induction sourceTokens generalizing
      targetTokens sourceBeforeRev targetBeforeRev sourcePacked targetPacked
      sourceAfterRev targetAfterRev seen sourcePattern with
  | nil =>
      cases targetTokens with
      | cons targetToken targetTokens =>
          simp at hTokens
      | nil =>
          cases sourcePacked with
          | none =>
              simp [parsePackTokens] at hParse
          | some sourcePackedSpan =>
              cases targetPacked with
              | none =>
                  simp at hPacked
              | some targetPackedSpan =>
                  simp [parsePackTokens] at hParse
                  subst sourcePattern
                  have hBeforeReverse :
                      targetBeforeRev.reverse.map
                          (fun axis => TokenKind.word axis.value) =
                        sourceBeforeRev.reverse.map
                          (fun axis => TokenKind.word axis.value) := by
                    rw [List.map_reverse, List.map_reverse, hBefore]
                  have hAfterReverse :
                      targetAfterRev.reverse.map
                          (fun axis => TokenKind.word axis.value) =
                        sourceAfterRev.reverse.map
                          (fun axis => TokenKind.word axis.value) := by
                    rw [List.map_reverse, List.map_reverse, hAfter]
                  refine
                    ⟨{ before := targetBeforeRev.reverse
                       packed := targetPackedSpan
                       after := targetAfterRev.reverse
                       span := targetWhole },
                      by simp [parsePackTokens], ?_⟩
                  simp only [PackPattern.render]
                  apply congrArg TokenKind.renderSequence
                  rw [hBeforeReverse, hAfterReverse]
  | cons sourceToken sourceTokens induction =>
      cases targetTokens with
      | nil =>
          simp at hTokens
      | cons targetToken targetTokens =>
          simp only [List.map_cons, List.cons.injEq] at hTokens
          rcases hTokens with ⟨hValue, hTokens⟩
          cases hSourceValue : sourceToken.value with
          | word name =>
              by_cases hInvalid :
                  (!policy.isIdentifier name ||
                    name.startsWith "_" || name.endsWith "_") = true
              · simp [parsePackTokens, hSourceValue, hInvalid] at hParse
              · by_cases hDuplicate : name ∈ seen
                · simp [parsePackTokens, hSourceValue, hInvalid,
                    hDuplicate] at hParse
                · cases sourcePacked with
                  | none =>
                      cases targetPacked with
                      | none =>
                          simp [parsePackTokens, hSourceValue, hInvalid,
                            hDuplicate] at hParse
                          obtain
                              ⟨targetPattern, hTargetParse, hRender⟩ :=
                            induction (targetTokens := targetTokens)
                              (sourceBeforeRev :=
                                ⟨name, sourceToken.span⟩ :: sourceBeforeRev)
                              (targetBeforeRev :=
                                ⟨name, targetToken.span⟩ :: targetBeforeRev)
                              (sourcePacked := none) (targetPacked := none)
                              (sourceAfterRev := sourceAfterRev)
                              (targetAfterRev := targetAfterRev)
                              (seen := name :: seen)
                              (sourcePattern := sourcePattern)
                              hTokens (by simp [hBefore]) rfl hAfter hParse
                          refine ⟨targetPattern, ?_, hRender⟩
                          simpa [parsePackTokens, hValue, hSourceValue,
                            hInvalid, hDuplicate] using hTargetParse
                      | some targetPackedSpan =>
                          simp at hPacked
                  | some sourcePackedSpan =>
                      cases targetPacked with
                      | none =>
                          simp at hPacked
                      | some targetPackedSpan =>
                          simp [parsePackTokens, hSourceValue, hInvalid,
                            hDuplicate] at hParse
                          obtain
                              ⟨targetPattern, hTargetParse, hRender⟩ :=
                            induction (targetTokens := targetTokens)
                              (sourceBeforeRev := sourceBeforeRev)
                              (targetBeforeRev := targetBeforeRev)
                              (sourcePacked := some sourcePackedSpan)
                              (targetPacked := some targetPackedSpan)
                              (sourceAfterRev :=
                                ⟨name, sourceToken.span⟩ :: sourceAfterRev)
                              (targetAfterRev :=
                                ⟨name, targetToken.span⟩ :: targetAfterRev)
                              (seen := name :: seen)
                              (sourcePattern := sourcePattern)
                              hTokens hBefore rfl (by simp [hAfter]) hParse
                          refine ⟨targetPattern, ?_, hRender⟩
                          simpa [parsePackTokens, hValue, hSourceValue,
                            hInvalid, hDuplicate] using hTargetParse
          | star =>
              cases sourcePacked with
              | none =>
                  cases targetPacked with
                  | none =>
                      simp [parsePackTokens, hSourceValue] at hParse
                      obtain
                          ⟨targetPattern, hTargetParse, hRender⟩ :=
                        induction (targetTokens := targetTokens)
                          (sourceBeforeRev := sourceBeforeRev)
                          (targetBeforeRev := targetBeforeRev)
                          (sourcePacked := some sourceToken.span)
                          (targetPacked := some targetToken.span)
                          (sourceAfterRev := sourceAfterRev)
                          (targetAfterRev := targetAfterRev)
                          (seen := seen) (sourcePattern := sourcePattern)
                          hTokens hBefore rfl hAfter hParse
                      refine ⟨targetPattern, ?_, hRender⟩
                      simpa [parsePackTokens, hValue, hSourceValue] using
                        hTargetParse
                  | some targetPackedSpan =>
                      simp at hPacked
              | some sourcePackedSpan =>
                  simp [parsePackTokens, hSourceValue] at hParse
          | leftParen => simp [parsePackTokens, hSourceValue] at hParse
          | rightParen => simp [parsePackTokens, hSourceValue] at hParse
          | ellipsis => simp [parsePackTokens, hSourceValue] at hParse
          | arrow => simp [parsePackTokens, hSourceValue] at hParse
          | comma => simp [parsePackTokens, hSourceValue] at hParse

/-- Parse the separate `pack` and `unpack` pattern grammar. -/
def parsePackPattern (source : String)
    (policy := IdentifierPolicy.pythonUnicode) : Result PackPattern := do
  let whole := sourceSpan source
  let tokens ← lex source policy .general
  parsePackTokens policy whole tokens [] none [] []

/-- Every word produced by a successful lex satisfies the selected word policy. -/
private theorem words_valid_of_lex_eq_ok
    (source : String) (policy : IdentifierPolicy)
    (whitespace : LexicalWhitespace) (tokens : List Token)
    (hLex : lex source policy whitespace = .ok tokens) :
    ∀ text, .word text ∈ tokens.map Located.value →
      text.toList ≠ [] ∧
        ∀ character ∈ text.toList,
          policy.isWordChar character = true := by
  intro text hText
  obtain ⟨token, hToken, hValue⟩ := List.mem_map.mp hText
  cases hTokenValue : token.value with
  | word sourceText =>
      simp [hTokenValue] at hValue
      subst text
      exact
        word_valid_of_lex_eq_ok source policy whitespace hLex hToken
          hTokenValue
  | leftParen => simp [hTokenValue] at hValue
  | rightParen => simp [hTokenValue] at hValue
  | ellipsis => simp [hTokenValue] at hValue
  | arrow => simp [hTokenValue] at hValue
  | comma => simp [hTokenValue] at hValue
  | star => simp [hTokenValue] at hValue

/-- Lexing a rendered pack pattern recovers its canonical token sequence. -/
private theorem lex_packPattern_render_eq_ok
    (pattern : PackPattern)
    (hWords :
      ∀ text,
        .word text ∈
            pattern.before.map (fun axis => TokenKind.word axis.value) ++
              [.star] ++
                pattern.after.map
                  (fun axis => TokenKind.word axis.value) →
          text.toList ≠ [] ∧
            ∀ character ∈ text.toList,
              IdentifierPolicy.pythonUnicode.isWordChar character = true) :
    (lex pattern.render IdentifierPolicy.pythonUnicode
        LexicalWhitespace.general).map (List.map Located.value) =
      .ok
        (pattern.before.map (fun axis => TokenKind.word axis.value) ++
          [.star] ++
            pattern.after.map
              (fun axis => TokenKind.word axis.value)) := by
  rw [PackPattern.render]
  apply TokenKind.lex_renderSequence_general
  exact hWords

/--
Parsing a successful `pack` or `unpack` pattern's canonical rendering
succeeds with the same ordered fixed axes and packed-segment boundary.

The reparsed pattern has fresh source spans. Comparing renderings intentionally
ignores those diagnostic locations while preserving every syntactic choice
used by pack and unpack checking.
-/
theorem parsePackPattern_render_eq_ok (source : String)
    (pattern : PackPattern)
    (hParse : parsePackPattern source = .ok pattern) :
    (parsePackPattern pattern.render).map PackPattern.render =
      .ok pattern.render := by
  cases hSourceLex :
      lex source IdentifierPolicy.pythonUnicode LexicalWhitespace.general with
  | error diagnostic =>
      unfold parsePackPattern at hParse
      rw [hSourceLex] at hParse
      contradiction
  | ok sourceTokens =>
      have hSourceParse :
          parsePackTokens IdentifierPolicy.pythonUnicode (sourceSpan source)
              sourceTokens [] none [] [] =
            .ok pattern := by
        unfold parsePackPattern at hParse
        rw [hSourceLex] at hParse
        exact hParse
      have hRenderedKinds :
          pattern.before.map (fun axis => TokenKind.word axis.value) ++
                [.star] ++
                pattern.after.map
                  (fun axis => TokenKind.word axis.value) =
            sourceTokens.map Located.value := by
        simpa using
          tokenKinds_eq_of_parsePackTokens_eq_ok
            IdentifierPolicy.pythonUnicode (sourceSpan source) sourceTokens
            [] none [] [] pattern (by simp) hSourceParse
      have hWords :
          ∀ text,
            .word text ∈
                pattern.before.map
                    (fun axis => TokenKind.word axis.value) ++
                  [.star] ++
                    pattern.after.map
                      (fun axis => TokenKind.word axis.value) →
              text.toList ≠ [] ∧
                ∀ character ∈ text.toList,
                  IdentifierPolicy.pythonUnicode.isWordChar character =
                    true := by
        intro text hText
        apply
          words_valid_of_lex_eq_ok source
            IdentifierPolicy.pythonUnicode .general sourceTokens hSourceLex
            text
        rw [← hRenderedKinds]
        exact hText
      have hTargetLexValues :
          (lex pattern.render IdentifierPolicy.pythonUnicode
              LexicalWhitespace.general).map (List.map Located.value) =
            .ok
              (pattern.before.map
                    (fun axis => TokenKind.word axis.value) ++
                [.star] ++
                  pattern.after.map
                    (fun axis => TokenKind.word axis.value)) :=
        lex_packPattern_render_eq_ok pattern hWords
      cases hTargetLex :
          lex pattern.render IdentifierPolicy.pythonUnicode
            LexicalWhitespace.general with
      | error diagnostic =>
          rw [hTargetLex] at hTargetLexValues
          contradiction
      | ok targetTokens =>
          rw [hTargetLex] at hTargetLexValues
          have hTargetKinds :
              targetTokens.map Located.value =
                pattern.before.map
                    (fun axis => TokenKind.word axis.value) ++
                  [.star] ++
                    pattern.after.map
                      (fun axis => TokenKind.word axis.value) := by
            simpa [Except.map] using hTargetLexValues
          have hTokens :
              targetTokens.map Located.value =
                sourceTokens.map Located.value :=
            hTargetKinds.trans hRenderedKinds
          obtain ⟨targetPattern, hTargetParseTokens, hRender⟩ :=
            parsePackTokens_canonical IdentifierPolicy.pythonUnicode
              (sourceSpan source) (sourceSpan pattern.render)
              sourceTokens targetTokens [] [] none none [] [] [] pattern
              hTokens rfl rfl rfl hSourceParse
          have hTargetParse :
              parsePackPattern pattern.render = .ok targetPattern := by
            unfold parsePackPattern
            rw [hTargetLex]
            exact hTargetParseTokens
          rw [hTargetParse]
          simp only [Except.map]
          exact congrArg Except.ok hRender

end TorchLean.Tensor.Internal.Syntax
