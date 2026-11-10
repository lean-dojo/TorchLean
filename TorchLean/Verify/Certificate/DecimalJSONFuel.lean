/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Certificate.DecimalJSON
import all Std.Internal.Parsec.Basic
import all Std.Internal.Parsec.String
import all Lean.Data.Json.Parser

/-!
# The fuel of the decimal JSON reader

`parse` in `TorchLean.Verify.Certificate.DecimalJSON` runs its nested calls on a fuel of
`2 * s.utf8ByteSize + 2`, and we prove here that this fuel suffices. No parser of the reader moves
back (`Forward`), and between two nested calls of `value` the reader reads at least one byte, so a
fuel of twice the bytes left plus one never runs out.

`value_fuel_congr` gives `value` the same result at `it` for any two fuels of at least
`2 * it.2.remainingBytes + 1` (`arrayItems_fuel_congr` and `objectFields_fuel_congr` do the same
with `+ 2`), and `run_value_eq_parse` gives the result of `parse s` for every fuel of at least
`2 * s.utf8ByteSize + 1`. No module imports this one, so the users of `parse` do not wait for these
proofs.
-/

public section

namespace TorchLean.Certificate.DecimalJson

open Std.Internal.Parsec Std.Internal.Parsec.String
open DecimalJson.Parser (value decimal)

/-- `peek!` never moves back. -/
theorem forward_peek! : Forward (peek! : Parser Char) := by
  intro it it' c h
  unfold peek! at h
  split at h
  · cases h
    exact Nat.le_refl _
  · cases h

/-- `peek?` never moves back. -/
theorem forward_peek? : Forward (peek? : Parser (Option Char)) := by
  intro it it' c h
  unfold peek? at h
  split at h <;> (cases h; exact Nat.le_refl _)

/-- When `skip` succeeds, fewer bytes of the input are left. -/
theorem skip_lt {it it' : Sigma String.Pos} {u : Unit}
    (h : (skip : Parser Unit) it = .success it' u) :
    it'.2.remainingBytes < it.2.remainingBytes := by
  unfold skip at h
  split at h
  next hn =>
    cases h
    exact remainingBytes_next_lt (not_isAtEnd_of_hasNext hn)
  next => cases h

/-- `skip` never moves back. -/
theorem forward_skip : Forward (skip : Parser Unit) :=
  fun _ _ _ h => Nat.le_of_lt (skip_lt h)

private theorem remainingBytes_skipWs_le {s : String} (p : s.Pos) :
    (skipWs p).remainingBytes ≤ p.remainingBytes := by
  fun_induction skipWs p
  all_goals first
    | exact Nat.le_refl _
    | (rename_i ih; exact Nat.le_trans ih (Nat.le_of_lt (remainingBytes_next_lt ‹_›)))

/-- `ws` never moves back. -/
theorem forward_ws : Forward ws := by
  intro it it' u h
  cases h
  exact remainingBytes_skipWs_le it.2

/-- `pstring t` never moves back. -/
theorem forward_pstring (t : String) : Forward (pstring t) := by
  intro it it' u h
  unfold pstring at h
  split at h
  · cases h
    exact (String.Pos.le_iff_remainingBytes_le _ _).1 String.Pos.le_nextn
  · cases h

/-- `skipString t` never moves back. -/
theorem forward_skipString (t : String) : Forward (skipString t) :=
  Forward.bind (forward_pstring t) fun _ => forward_pure ()

/-- `Parser.nextIs p` never moves back. -/
theorem forward_nextIs (p : Char → Bool) : Forward (Parser.nextIs p) :=
  Forward.bind forward_peek? fun _ => forward_pure _

/-- `Lean.Json.Parser.lookahead p desc` never moves back. -/
theorem forward_lookahead (p : Char → Prop) (desc : String) [DecidablePred p] :
    Forward (Lean.Json.Parser.lookahead p desc) := by
  unfold Lean.Json.Parser.lookahead
  exact Forward.bind forward_peek! fun c => by
    split
    · exact forward_pure _
    · exact forward_fail _

/-- `strCore acc` never moves back. -/
theorem forward_strCore (acc : String) : Forward (strCore acc) := by
  intro it
  fun_induction strCore acc it
  all_goals intro it' r h
  all_goals first
    | (cases h; done)
    | (cases h; exact Nat.le_of_lt (remainingBytes_next_lt ‹_›))
    | (rename_i he ih
       exact Nat.le_trans (ih h) (Nat.le_trans (forward_escapedChar he)
         (Nat.le_of_lt (remainingBytes_next_lt ‹_›))))
    | (rename_i ih; exact Nat.le_trans (ih h) (Nat.le_of_lt (remainingBytes_next_lt ‹_›)))

/-- `natCore acc` never moves back. -/
theorem forward_natCore (acc : Nat) : Forward (natCore acc) := by
  intro it it' r h
  fun_induction natCore acc it
  all_goals first
    | (cases h; exact Nat.le_refl _)
    | (rename_i ih; exact Nat.le_trans (ih h) (Nat.le_of_lt (remainingBytes_next_lt ‹_›)))

/-- `natCoreNumDigits acc digits` never moves back. -/
theorem forward_natCoreNumDigits (acc digits : Nat) :
    Forward (natCoreNumDigits acc digits) := by
  intro it it' r h
  fun_induction natCoreNumDigits acc digits it
  all_goals first
    | (cases h; exact Nat.le_refl _)
    | (rename_i ih; exact Nat.le_trans (ih h) (Nat.le_of_lt (remainingBytes_next_lt ‹_›)))

/-- `decimal` never moves back. -/
theorem forward_decimal : Forward decimal := by
  unfold decimal
  repeat' first
    | with_reducible exact forward_pure _
    | with_reducible exact forward_skip
    | with_reducible exact forward_peek!
    | with_reducible exact forward_nextIs _
    | with_reducible exact forward_lookahead _ _
    | with_reducible exact forward_natCore _
    | with_reducible exact forward_natCoreNumDigits _ _
    | with_reducible refine Forward.bind ?_ fun _ => ?_
    | with_reducible refine iteInduction (motive := Forward) (fun _ => ?_) fun _ => ?_

mutual

/-- `Parser.arrayItems n acc` never moves back, at every fuel `n`. -/
theorem forward_arrayItems : ∀ (n : Nat) (acc : Array Value), Forward (Parser.arrayItems n acc)
  | 0, _ => by
    rw [Parser.arrayItems]
    exact forward_fail _
  | n + 1, acc => by
    rw [Parser.arrayItems]
    exact Forward.bind (forward_value n) fun _ => Forward.bind forward_any fun _ =>
      iteInduction (motive := Forward) (fun _ => Forward.bind forward_ws fun _ => forward_pure _)
        fun _ => iteInduction (motive := Forward)
          (fun _ => Forward.bind forward_ws fun _ => forward_arrayItems n _)
          fun _ => forward_fail _

/-- `Parser.objectFields n acc` never moves back, at every fuel `n`. -/
theorem forward_objectFields :
    ∀ (n : Nat) (acc : Array (String × Value)), Forward (Parser.objectFields n acc)
  | 0, _ => by
    rw [Parser.objectFields]
    exact forward_fail _
  | n + 1, acc => by
    rw [Parser.objectFields]
    exact Forward.bind (forward_lookahead _ _) fun _ => Forward.bind forward_skip fun _ =>
      Forward.bind (forward_strCore _) fun _ => Forward.bind forward_ws fun _ =>
      Forward.bind (forward_lookahead _ _) fun _ => Forward.bind forward_skip fun _ =>
      Forward.bind forward_ws fun _ => iteInduction (motive := Forward)
        (fun _ _ _ _ h => nomatch h)
        fun _ => Forward.bind (forward_value n) fun _ => Forward.bind forward_any fun _ =>
          iteInduction (motive := Forward)
            (fun _ => Forward.bind forward_ws fun _ => forward_pure _)
            fun _ => iteInduction (motive := Forward)
              (fun _ => Forward.bind forward_ws fun _ => forward_objectFields n _)
              fun _ => forward_fail _

/-- `value n` never moves back, at every fuel `n`. -/
theorem forward_value : ∀ n : Nat, Forward (value n)
  | 0 => by
    rw [value]
    exact forward_fail _
  | n + 1 => by
    rw [value]
    exact Forward.bind forward_peek! fun _ => iteInduction (motive := Forward)
      (fun _ => Forward.bind forward_skip fun _ => Forward.bind forward_ws fun _ =>
        Forward.bind (forward_nextIs _) fun _ => iteInduction (motive := Forward)
          (fun _ => Forward.bind forward_skip fun _ => Forward.bind forward_ws fun _ =>
            forward_pure _)
          fun _ => Forward.bind (forward_arrayItems n _) fun _ => forward_pure _)
      fun _ => iteInduction (motive := Forward)
      (fun _ => Forward.bind forward_skip fun _ => Forward.bind forward_ws fun _ =>
        Forward.bind (forward_nextIs _) fun _ => iteInduction (motive := Forward)
          (fun _ => Forward.bind forward_skip fun _ => Forward.bind forward_ws fun _ =>
            forward_pure _)
          fun _ => Forward.bind (forward_objectFields n _) fun _ => forward_pure _)
      fun _ => iteInduction (motive := Forward)
      (fun _ => Forward.bind forward_skip fun _ => Forward.bind (forward_strCore _) fun _ =>
        Forward.bind forward_ws fun _ => forward_pure _)
      fun _ => iteInduction (motive := Forward)
      (fun _ => Forward.bind (forward_skipString _) fun _ => Forward.bind forward_ws fun _ =>
        forward_pure _)
      fun _ => iteInduction (motive := Forward)
      (fun _ => Forward.bind (forward_skipString _) fun _ => Forward.bind forward_ws fun _ =>
        forward_pure _)
      fun _ => iteInduction (motive := Forward)
      (fun _ => Forward.bind (forward_skipString _) fun _ => Forward.bind forward_ws fun _ =>
        forward_pure _)
      fun _ => iteInduction (motive := Forward)
      (fun _ => Forward.bind forward_decimal fun _ => Forward.bind forward_ws fun _ =>
        forward_pure _)
      fun _ => forward_fail _

end

/-- `p >>= f` and `q >>= g` agree at `it` when `p` and `q` agree at `it` and, after each success
of `p` at `it`, `f` and `g` agree. -/
theorem bind_apply_congr {α β : Type} {p q : Parser α} {f g : α → Parser β}
    {it : Sigma String.Pos} (h : p it = q it)
    (hfg : ∀ it' a, p it = .success it' a → f a it' = g a it') :
    (p >>= f) it = (q >>= g) it := by
  change Std.Internal.Parsec.bind p f it = Std.Internal.Parsec.bind q g it
  unfold Std.Internal.Parsec.bind
  rw [← h]
  split
  next it' a ha => exact hfg it' a ha
  next => rfl

mutual

/-- Two fuels of at least `2 * it.2.remainingBytes + 2` give `Parser.arrayItems` the same result at
`it`. -/
theorem arrayItems_fuel_congr : ∀ {n m : Nat} (acc : Array Value) {it : Sigma String.Pos},
    2 * it.2.remainingBytes + 2 ≤ n → 2 * it.2.remainingBytes + 2 ≤ m →
    Parser.arrayItems n acc it = Parser.arrayItems m acc it
  | 0, _, _, _, hn, _ => absurd hn (by omega)
  | _ + 1, 0, _, _, _, hm => absurd hm (by omega)
  | n + 1, m + 1, acc, it, hn, hm => by
    rw [Parser.arrayItems, Parser.arrayItems]
    refine bind_apply_congr (value_fuel_congr (by omega) (by omega)) fun it₁ v h₁ =>
      bind_apply_congr rfl fun it₂ c h₂ => ?_
    have := forward_value n h₁
    have := any_lt h₂
    by_cases h1 : (c == ']') = true
    · rw [ite_eq_left h1, ite_eq_left h1]
    rw [ite_eq_right h1, ite_eq_right h1]
    by_cases h2 : (c == ',') = true
    · rw [ite_eq_left h2, ite_eq_left h2]
      exact bind_apply_congr rfl fun it₃ _ h₃ => by
        have := forward_ws h₃
        exact arrayItems_fuel_congr _ (by omega) (by omega)
    rw [ite_eq_right h2, ite_eq_right h2]

/-- Two fuels of at least `2 * it.2.remainingBytes + 2` give `Parser.objectFields` the same result
at `it`. -/
theorem objectFields_fuel_congr :
    ∀ {n m : Nat} (acc : Array (String × Value)) {it : Sigma String.Pos},
    2 * it.2.remainingBytes + 2 ≤ n → 2 * it.2.remainingBytes + 2 ≤ m →
    Parser.objectFields n acc it = Parser.objectFields m acc it
  | 0, _, _, _, hn, _ => absurd hn (by omega)
  | _ + 1, 0, _, _, _, hm => absurd hm (by omega)
  | n + 1, m + 1, acc, it, hn, hm => by
    rw [Parser.objectFields, Parser.objectFields]
    refine bind_apply_congr rfl fun it₁ _ h₁ => bind_apply_congr rfl fun it₂ _ h₂ =>
      bind_apply_congr rfl fun it₃ k h₃ => bind_apply_congr rfl fun it₄ _ h₄ =>
      bind_apply_congr rfl fun it₅ _ h₅ => bind_apply_congr rfl fun it₆ _ h₆ =>
      bind_apply_congr rfl fun it₇ _ h₇ => ?_
    have := forward_lookahead _ _ h₁
    have := skip_lt h₂
    have := forward_strCore _ h₃
    have := forward_ws h₄
    have := forward_lookahead _ _ h₅
    have := skip_lt h₆
    have := forward_ws h₇
    dsimp only
    by_cases h0 : (acc.any fun x => x.1 == k) = true
    · rw [ite_eq_left h0, ite_eq_left h0]
      exact bind_apply_congr rfl fun _ _ h => nomatch h
    rw [ite_eq_right h0, ite_eq_right h0]
    refine bind_apply_congr (value_fuel_congr (by omega) (by omega)) fun it₈ v h₈ =>
      bind_apply_congr rfl fun it₉ c h₉ => ?_
    have := forward_value n h₈
    have := any_lt h₉
    by_cases h1 : (c == '}') = true
    · rw [ite_eq_left h1, ite_eq_left h1]
    rw [ite_eq_right h1, ite_eq_right h1]
    by_cases h2 : (c == ',') = true
    · rw [ite_eq_left h2, ite_eq_left h2]
      exact bind_apply_congr rfl fun it₁₀ _ h₁₀ => by
        have := forward_ws h₁₀
        exact objectFields_fuel_congr _ (by omega) (by omega)
    rw [ite_eq_right h2, ite_eq_right h2]

/-- Two fuels of at least `2 * it.2.remainingBytes + 1` give `value` the same result at `it`. -/
theorem value_fuel_congr : ∀ {n m : Nat} {it : Sigma String.Pos},
    2 * it.2.remainingBytes + 1 ≤ n → 2 * it.2.remainingBytes + 1 ≤ m → value n it = value m it
  | 0, _, _, hn, _ => absurd hn (by omega)
  | _ + 1, 0, _, _, hm => absurd hm (by omega)
  | n + 1, m + 1, it, hn, hm => by
    rw [value, value]
    refine bind_apply_congr rfl fun it₁ c h₁ => ?_
    have := forward_peek! h₁
    by_cases h1 : (c == '[') = true
    · rw [ite_eq_left h1, ite_eq_left h1]
      refine bind_apply_congr rfl fun it₂ _ h₂ => bind_apply_congr rfl fun it₃ _ h₃ =>
        bind_apply_congr rfl fun it₄ b h₄ => ?_
      have := skip_lt h₂
      have := forward_ws h₃
      have := forward_nextIs _ h₄
      cases b
      · exact bind_apply_congr (arrayItems_fuel_congr _ (by omega) (by omega)) fun _ _ _ => rfl
      · rfl
    rw [ite_eq_right h1, ite_eq_right h1]
    by_cases h2 : (c == '{') = true
    · rw [ite_eq_left h2, ite_eq_left h2]
      refine bind_apply_congr rfl fun it₂ _ h₂ => bind_apply_congr rfl fun it₃ _ h₃ =>
        bind_apply_congr rfl fun it₄ b h₄ => ?_
      have := skip_lt h₂
      have := forward_ws h₃
      have := forward_nextIs _ h₄
      cases b
      · exact bind_apply_congr (objectFields_fuel_congr _ (by omega) (by omega)) fun _ _ _ => rfl
      · rfl
    rw [ite_eq_right h2, ite_eq_right h2]

end

/-- The reader of `parse` with any fuel `n ≥ 2 * s.utf8ByteSize + 1` gives `parse s`. -/
theorem run_value_eq_parse (s : String) {n : Nat} (hn : 2 * s.utf8ByteSize + 1 ≤ n) :
    Std.Internal.Parsec.String.Parser.run
      (do Std.Internal.Parsec.String.ws; let v ← value n; Std.Internal.Parsec.eof; pure v) s =
    parse s := by
  have h0 : s.startPos.remainingBytes = s.utf8ByteSize := by
    simp [String.Pos.remainingBytes_eq]
  unfold parse Std.Internal.Parsec.String.Parser.run
  congr 1
  refine bind_apply_congr rfl fun it _ ha =>
    bind_apply_congr (value_fuel_congr ?_ ?_) fun _ _ _ => rfl
  all_goals
    have := forward_ws ha
    dsimp only at this
    omega

end TorchLean.Certificate.DecimalJson
