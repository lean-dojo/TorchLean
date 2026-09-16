/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Parser.Expression.State

/-!
# Transformation arrow splitting

Arrow splitting rejects missing or duplicate arrows and preserves canonical
left and right token sequences.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal.Syntax

namespace Parser.Impl

/-- Split a transformation token stream at its unique arrow. -/
def splitArrow (eof : Span) :
    List Token → List Token → Result (List Token × Token × List Token)
  | [], _ =>
      .error
        { code := .missingArrow
          message := "pattern must contain exactly one '->'"
          span := eof }
  | token :: rest, leftRev =>
      match token.value with
      | .arrow =>
          if let some duplicate := rest.find? fun item => item.value == .arrow then
            .error
              { code := .duplicateArrow
                message := "pattern contains more than one '->'"
                span := duplicate.span }
          else
            .ok (leftRev.reverse, token, rest)
      | _ => splitArrow eof rest (token :: leftRev)

/-- Splitting canonical tokens preserves the left and right token sequences. -/
theorem splitArrow_canonical
    (sourceEof targetEof : Span)
    (sourceTokens targetTokens sourceLeftRev targetLeftRev : List Token)
    (sourceLeft sourceRight : List Token) (sourceArrow : Token)
    (hTokens :
      targetTokens.map Located.value =
        sourceTokens.map fun token =>
          canonicalTokenKind IdentifierPolicy.pythonUnicode token.value)
    (hLeftRev :
      targetLeftRev.map Located.value =
        sourceLeftRev.map fun token =>
          canonicalTokenKind IdentifierPolicy.pythonUnicode token.value)
    (hSplit :
      splitArrow sourceEof sourceTokens sourceLeftRev =
        .ok (sourceLeft, sourceArrow, sourceRight)) :
    ∃ targetLeft targetArrow targetRight,
      splitArrow targetEof targetTokens targetLeftRev =
          .ok (targetLeft, targetArrow, targetRight) ∧
        targetLeft.map Located.value =
          sourceLeft.map (fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value) ∧
        targetRight.map Located.value =
          sourceRight.map fun token =>
            canonicalTokenKind IdentifierPolicy.pythonUnicode token.value := by
  induction sourceTokens generalizing
      targetTokens sourceLeftRev targetLeftRev with
  | nil =>
      simp [splitArrow] at hSplit
  | cons sourceToken sourceTokens induction =>
      cases targetTokens with
      | nil =>
          simp at hTokens
      | cons targetToken targetTokens =>
          simp only [List.map_cons, List.cons.injEq] at hTokens
          rcases hTokens with ⟨hValue, hTokens⟩
          by_cases hSourceArrow : sourceToken.value = .arrow
          · cases hFind :
              sourceTokens.find? (fun token => token.value == .arrow) with
            | some duplicate =>
                simp [splitArrow, hSourceArrow, hFind] at hSplit
            | none =>
                simp [splitArrow, hSourceArrow, hFind] at hSplit
                rcases hSplit with ⟨rfl, rfl, rfl⟩
                have hTargetValue : targetToken.value = .arrow := by
                  simpa [hSourceArrow, canonicalTokenKind] using hValue
                have hTargetFind :=
                  findArrow_eq_none_of_canonical
                    IdentifierPolicy.pythonUnicode sourceTokens targetTokens
                    hTokens hFind
                refine
                  ⟨targetLeftRev.reverse, targetToken, targetTokens,
                    ?_, ?_, hTokens⟩
                · simp [splitArrow, hTargetValue, hTargetFind]
                · simpa [List.map_reverse] using
                    congrArg List.reverse hLeftRev
          · have hTargetNotArrow : targetToken.value ≠ .arrow := by
              intro hTargetArrow
              apply hSourceArrow
              have hArrow :
                  (sourceToken.value == .arrow) = true := by
                rw [← canonicalTokenKind_beq_arrow
                  IdentifierPolicy.pythonUnicode]
                rw [← hValue, hTargetArrow]
                simp only [beq_self_eq_true]
              simpa using hArrow
            have hSourceStep :
                splitArrow sourceEof (sourceToken :: sourceTokens)
                    sourceLeftRev =
                  splitArrow sourceEof sourceTokens
                    (sourceToken :: sourceLeftRev) := by
              cases hSourceValue : sourceToken.value <;>
                simp [splitArrow, hSourceValue] at hSourceArrow ⊢
            have hTargetStep :
                splitArrow targetEof (targetToken :: targetTokens)
                    targetLeftRev =
                  splitArrow targetEof targetTokens
                    (targetToken :: targetLeftRev) := by
              cases hTargetValue : targetToken.value <;>
                simp [splitArrow, hTargetValue] at hTargetNotArrow ⊢
            have hNextLeftRev :
                (targetToken :: targetLeftRev).map Located.value =
                  (sourceToken :: sourceLeftRev).map fun token =>
                    canonicalTokenKind IdentifierPolicy.pythonUnicode
                      token.value := by
              simp only [List.map_cons, List.cons.injEq]
              exact ⟨hValue, hLeftRev⟩
            rw [hSourceStep] at hSplit
            obtain
                ⟨targetLeft, targetArrow, targetRight, hTargetSplit,
                  hTargetLeft, hTargetRight⟩ :=
              induction
                (targetTokens := targetTokens)
                (sourceLeftRev := sourceToken :: sourceLeftRev)
                (targetLeftRev := targetToken :: targetLeftRev)
                hTokens hNextLeftRev hSplit
            refine
              ⟨targetLeft, targetArrow, targetRight, ?_,
                hTargetLeft, hTargetRight⟩
            rw [hTargetStep]
            exact hTargetSplit

/-- A successful arrow split reconstructs the original stream around an arrow token. -/
theorem splitArrow_eq_ok_decomposition
    (eof : Span) (tokens leftRev left right : List Token) (arrow : Token)
    (hSplit :
      splitArrow eof tokens leftRev = .ok (left, arrow, right)) :
    leftRev.reverse ++ tokens = left ++ arrow :: right ∧
      arrow.value = .arrow := by
  induction tokens generalizing leftRev with
  | nil =>
      simp [splitArrow] at hSplit
  | cons token tokens induction =>
      by_cases hArrow : token.value = .arrow
      · cases hFind :
          tokens.find? (fun item => item.value == .arrow) with
        | some duplicate =>
            simp [splitArrow, hArrow, hFind] at hSplit
        | none =>
            simp [splitArrow, hArrow, hFind] at hSplit
            rcases hSplit with ⟨rfl, rfl, rfl⟩
            simp [hArrow]
      · have hStep :
            splitArrow eof (token :: tokens) leftRev =
              splitArrow eof tokens (token :: leftRev) := by
          cases hValue : token.value <;>
            simp [splitArrow, hValue] at hArrow ⊢
        rw [hStep] at hSplit
        have hResult := induction (token :: leftRev) hSplit
        simpa [List.reverse_cons, List.append_assoc] using hResult

end Parser.Impl

end TorchLean.Tensor.Internal.Syntax
