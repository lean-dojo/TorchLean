/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Syntax.Expression.State

/-!
# Transformation arrow splitting

Arrow splitting rejects missing or duplicate arrows and returns the tokens on
each side of the arrow.
-/

@[expose] public section

namespace TorchLean.Einops.Syntax

/--
Split a token stream at its only arrow into the tokens before it, the arrow and the tokens after
it.

The second list accumulates the tokens already passed, in reverse, and starts as `[]`. A stream
with no arrow fails at `eof`, and a second arrow fails at its own span.
-/
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

end TorchLean.Einops.Syntax
