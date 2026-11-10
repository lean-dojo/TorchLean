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
import all Lean.Data.Json.Printer

/-!
# Reading rendered JSON back

We prove that `parse` reads `Value.render v` back as `v` (`parse_render`) for every value `v`
with `Value.Canonical v`: the keys of each object are pairwise distinct and a decimal with
significand `0` has the exponent of the text that `Decimal.render` writes for it. Every string and
key goes through `Lean.Json.renderString`, and we prove from the bodies of `Lean.Json.escape`, its
`needEscape` byte table and `Lean.Json.Parser.escapedChar` that `strCore` reads each one back.
-/

public section

namespace TorchLean.Certificate.DecimalJson

open Std.Internal.Parsec Std.Internal.Parsec.String
open FloatLib.Numerics.DecimalText (Decimal)
open DecimalJson.Parser (nextIs)

/-- The values covered by `parse_render`: the keys of each object are pairwise distinct and a
decimal with significand `0` has exponent `-1` when negative and `0` otherwise (the texts `-0.0`
and `0` that `Decimal.render` writes). -/
inductive Value.Canonical : Value → Prop
  /-- `null` is canonical. -/
  | null : Value.Canonical .null
  /-- Every Boolean is canonical. -/
  | bool (b : Bool) : Value.Canonical (.bool b)
  /-- A decimal is canonical when a zero significand comes with exponent `-1` if negative and
  `0` otherwise. -/
  | num (d : Decimal) (h : d.significand = 0 → d.exponent = if d.negative then -1 else 0) :
      Value.Canonical (.num d)
  /-- Every string is canonical. -/
  | str (s : String) : Value.Canonical (.str s)
  /-- An array of canonical values is canonical. -/
  | arr (xs : Array Value) (h : ∀ x ∈ xs, Value.Canonical x) : Value.Canonical (.arr xs)
  /-- An object with pairwise distinct keys and canonical values is canonical. -/
  | obj (fields : Array (String × Value)) (hk : (fields.toList.map Prod.fst).Nodup)
      (h : ∀ f ∈ fields, Value.Canonical f.2) : Value.Canonical (.obj fields)

section Steps

private theorem bind_of_success {α β : Type} {P : Parser α} {f : α → Parser β}
    {it it' : Sigma String.Pos} {a : α} (h : P it = .success it' a) :
    (P >>= f) it = f a it' := by
  change Std.Internal.Parsec.bind P f it = _
  unfold Std.Internal.Parsec.bind
  rw [h]

private theorem pure_bind_apply {α β : Type} (a : α) (f : α → Parser β)
    (it : Sigma String.Pos) : (pure a >>= f) it = f a it := rfl

private theorem pure_apply {α : Type} (a : α) (it : Sigma String.Pos) :
    (pure a : Parser α) it = .success it a := rfl

private theorem hasNext_of_splits {s : String} {p : s.Pos} {t₁ t₂ : String} {c : Char}
    (h : p.Splits t₁ (String.singleton c ++ t₂)) :
    Input.hasNext (⟨s, p⟩ : Sigma String.Pos) = true :=
  decide_eq_true h.ne_endPos_of_singleton

private theorem curr'_of_splits {s : String} {p : s.Pos} {t₁ t₂ : String} {c : Char}
    (h : p.Splits t₁ (String.singleton c ++ t₂)) :
    Input.curr' (⟨s, p⟩ : Sigma String.Pos) (hasNext_of_splits h) = c :=
  h.next.get_eq_of_singleton

private theorem peek!_eq {s : String} {p : s.Pos} {t₁ t₂ : String} {c : Char}
    (h : p.Splits t₁ (String.singleton c ++ t₂)) :
    (peek! : Parser Char) ⟨s, p⟩ = .success ⟨s, p⟩ c := by
  unfold peek!
  rw [dite_eq_left (hasNext_of_splits h), curr'_of_splits h]

private theorem skip_eq {s : String} {p : s.Pos} {t₁ t₂ : String} {c : Char}
    (h : p.Splits t₁ (String.singleton c ++ t₂)) :
    (skip : Parser Unit) ⟨s, p⟩ = .success ⟨s, p.next h.ne_endPos_of_singleton⟩ () := by
  unfold skip
  exact dite_eq_left (hasNext_of_splits h)

private theorem any_eq {s : String} {p : s.Pos} {t₁ t₂ : String} {c : Char}
    (h : p.Splits t₁ (String.singleton c ++ t₂)) :
    (any : Parser Char) ⟨s, p⟩ = .success ⟨s, p.next h.ne_endPos_of_singleton⟩ c := by
  unfold any
  rw [dite_eq_left (hasNext_of_splits h)]
  dsimp only
  rw [curr'_of_splits h]
  rfl

private theorem peek?_eq {s : String} {p : s.Pos} {t₁ t₂ : String} (h : p.Splits t₁ t₂) :
    (peek? : Parser (Option Char)) ⟨s, p⟩ = .success ⟨s, p⟩ t₂.toList.head? := by
  unfold peek?
  by_cases hp : p = s.endPos
  · have hn : ¬Input.hasNext (⟨s, p⟩ : Sigma String.Pos) = true :=
      fun hn => not_isAtEnd_of_hasNext hn hp
    rw [dite_eq_right hn, h.eq_endPos_iff.1 hp]
    simp
  · obtain ⟨t, rfl⟩ := h.exists_eq_singleton_append hp
    rw [dite_eq_left (hasNext_of_splits h), curr'_of_splits h]
    simp

private theorem nextIs_eq (f : Char → Bool) {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ t₂) :
    nextIs f ⟨s, p⟩ = .success ⟨s, p⟩ ((t₂.toList.head?.map f).getD false) := by
  unfold nextIs
  rw [bind_of_success (peek?_eq h)]
  rfl

private theorem nextIs_cons (f : Char → Bool) {s : String} {p : s.Pos} {t₁ t₂ : String}
    {c : Char} (h : p.Splits t₁ (String.singleton c ++ t₂)) :
    nextIs f ⟨s, p⟩ = .success ⟨s, p⟩ (f c) := by
  rw [nextIs_eq f h]
  simp

private theorem nextIs_false (f : Char → Bool) {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ t₂) (hf : ∀ c ∈ t₂.toList.head?, f c = false) :
    nextIs f ⟨s, p⟩ = .success ⟨s, p⟩ false := by
  rw [nextIs_eq f h]
  cases hc : t₂.toList.head? with
  | none => rfl
  | some c => simp [hf c (by simp [hc])]

private theorem ws_eq {s : String} {p : s.Pos} {t₁ t₂ : String} (h : p.Splits t₁ t₂)
    (hw : ∀ c ∈ t₂.toList.head?, ¬(c = '\t' ∨ c = '\n' ∨ c = '\r' ∨ c = ' ')) :
    ws ⟨s, p⟩ = .success ⟨s, p⟩ () := by
  show (.success ⟨s, skipWs p⟩ () : ParseResult Unit (Sigma String.Pos)) = _
  rw [skipWs.eq_def]
  by_cases hp : p = s.endPos
  · rw [dite_eq_right (not_not.2 hp)]
  · rw [dite_eq_left hp]
    obtain ⟨t, ht⟩ := h.exists_eq_singleton_append hp
    dsimp only
    rw [ite_eq_right (hw (p.get hp) (by simp [ht]))]

private theorem splits_nextn {s : String} {p : s.Pos} {t₁ u t₂ : String}
    (h : p.Splits t₁ (u ++ t₂)) : (p.nextn u.length).Splits (t₁ ++ u) t₂ :=
  (h.nextn u.length).of_eq
    (by rw [String.toList_append, List.take_left' String.length_toList, String.ofList_toList])
    (by rw [String.toList_append, List.drop_left' String.length_toList, String.ofList_toList])

private theorem pstring_eq {s : String} {p : s.Pos} {t₁ u t₂ : String}
    (h : p.Splits t₁ (u ++ t₂)) :
    pstring u ⟨s, p⟩ = .success ⟨s, p.nextn u.length⟩ u := by
  have hc : (s.sliceFrom p).startsWith u = true := by
    simp [h.copy_sliceFrom_eq]
  unfold pstring
  rw [ite_eq_left hc]

private theorem skipString_eq {s : String} {p : s.Pos} {t₁ u t₂ : String}
    (h : p.Splits t₁ (u ++ t₂)) :
    skipString u ⟨s, p⟩ = .success ⟨s, p.nextn u.length⟩ () := by
  show (pstring u >>= fun _ => pure ()) ⟨s, p⟩ = _
  rw [bind_of_success (pstring_eq h)]
  rfl

private theorem lookahead_eq (f : Char → Prop) [DecidablePred f] (desc : String) {s : String}
    {p : s.Pos} {t₁ t₂ : String} {c : Char} (h : p.Splits t₁ (String.singleton c ++ t₂))
    (hc : f c) : Lean.Json.Parser.lookahead f desc ⟨s, p⟩ = .success ⟨s, p⟩ () := by
  unfold Lean.Json.Parser.lookahead
  rw [bind_of_success (peek!_eq h), ite_eq_left hc]
  rfl

private theorem eof_eq {s : String} {p : s.Pos} {t₁ : String} (h : p.Splits t₁ "") :
    (eof : Parser Unit) ⟨s, p⟩ = .success ⟨s, p⟩ () := by
  have hp : p = s.endPos := h.eq_endPos_iff.2 rfl
  have hn : ¬Input.hasNext (⟨s, p⟩ : Sigma String.Pos) = true :=
    fun hn => not_isAtEnd_of_hasNext hn hp
  unfold eof
  rw [ite_eq_right hn]

end Steps

section Digits

private theorem digit_cond (c : Char) : (decide ('0' ≤ c) && decide (c ≤ '9')) = c.isDigit :=
  rfl

private theorem digit_sub {c : Char} (hc : c.isDigit = true) :
    (c.val - '0'.val).toNat = c.toNat - '0'.toNat := by
  have h0 : '0'.val ≤ c.val := UInt32.le_iff_toNat_le.2 (Char.isDigit_iff_toNat.1 hc).1
  rw [UInt32.toNat_sub_of_le _ _ h0]
  rfl

private theorem natCore_end {acc : Nat} {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ t₂) (ht : ∀ c ∈ t₂.toList.head?, c.isDigit = false) :
    natCore acc ⟨s, p⟩ = .success ⟨s, p⟩ acc := by
  rw [natCore.eq_def]
  by_cases hp : p = s.endPos
  · rw [dite_eq_left hp]
  · rw [dite_eq_right hp]
    obtain ⟨t, ht'⟩ := h.exists_eq_singleton_append hp
    have hc : (p.get hp).isDigit = false := ht _ (by simp [ht'])
    simp only [digit_cond, hc, Bool.false_eq_true, ↓reduceIte]

private theorem natCore_cons {acc : Nat} {s : String} {p : s.Pos} {t₁ t₂ : String} {c : Char}
    (h : p.Splits t₁ (String.singleton c ++ t₂)) (hc : c.isDigit = true) :
    natCore acc ⟨s, p⟩ =
      natCore (10 * acc + (c.val - '0'.val).toNat) ⟨s, p.next h.ne_endPos_of_singleton⟩ := by
  have hget : p.get h.ne_endPos_of_singleton = c := h.next.get_eq_of_singleton
  rw [natCore.eq_def, dite_eq_right h.ne_endPos_of_singleton]
  simp only [hget, digit_cond, hc, ↓reduceIte]

private theorem natCore_eq (ds : List Char) (hds : ∀ c ∈ ds, c.isDigit = true) :
    ∀ (acc : Nat) {s : String} {p : s.Pos} {t₁ t₂ : String},
      p.Splits t₁ (String.ofList ds ++ t₂) → (∀ c ∈ t₂.toList.head?, c.isDigit = false) →
      ∃ q : s.Pos, q.Splits (t₁ ++ String.ofList ds) t₂ ∧
        natCore acc ⟨s, p⟩ = .success ⟨s, q⟩ (Nat.ofDigitChars 10 ds acc) := by
  induction ds with
  | nil =>
    intro acc s p t₁ t₂ h ht
    have h' : p.Splits t₁ t₂ := by simpa using h
    refine ⟨p, by simpa using h', ?_⟩
    rw [Nat.ofDigitChars_nil]
    exact natCore_end h' ht
  | cons d ds ih =>
    intro acc s p t₁ t₂ h ht
    have hd : d.isDigit = true := hds d (by simp)
    have h' : p.Splits t₁ (String.singleton d ++ (String.ofList ds ++ t₂)) := by
      rwa [String.ofList_cons, String.append_assoc] at h
    obtain ⟨q, hq, hrun⟩ :=
      ih (fun c hc => hds c (by simp [hc])) (10 * acc + (d.val - '0'.val).toNat) h'.next ht
    refine ⟨q, by rw [String.ofList_cons, ← String.append_assoc]; exact hq, ?_⟩
    rw [natCore_cons h' hd, hrun, Nat.ofDigitChars_cons, digit_sub hd]

private theorem natCoreNumDigits_end {acc k : Nat} {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ t₂) (ht : ∀ c ∈ t₂.toList.head?, c.isDigit = false) :
    natCoreNumDigits acc k ⟨s, p⟩ = .success ⟨s, p⟩ (acc, k) := by
  rw [natCoreNumDigits.eq_def]
  by_cases hp : p = s.endPos
  · rw [dite_eq_left hp]
  · rw [dite_eq_right hp]
    obtain ⟨t, ht'⟩ := h.exists_eq_singleton_append hp
    have hc : (p.get hp).isDigit = false := ht _ (by simp [ht'])
    simp only [digit_cond, hc, Bool.false_eq_true, ↓reduceIte]

private theorem natCoreNumDigits_cons {acc k : Nat} {s : String} {p : s.Pos} {t₁ t₂ : String}
    {c : Char} (h : p.Splits t₁ (String.singleton c ++ t₂)) (hc : c.isDigit = true) :
    natCoreNumDigits acc k ⟨s, p⟩ =
      natCoreNumDigits (10 * acc + (c.val - '0'.val).toNat) (k + 1)
        ⟨s, p.next h.ne_endPos_of_singleton⟩ := by
  have hget : p.get h.ne_endPos_of_singleton = c := h.next.get_eq_of_singleton
  rw [natCoreNumDigits.eq_def, dite_eq_right h.ne_endPos_of_singleton]
  simp only [hget, digit_cond, hc, ↓reduceIte]

private theorem natCoreNumDigits_eq (ds : List Char) (hds : ∀ c ∈ ds, c.isDigit = true) :
    ∀ (acc k : Nat) {s : String} {p : s.Pos} {t₁ t₂ : String},
      p.Splits t₁ (String.ofList ds ++ t₂) → (∀ c ∈ t₂.toList.head?, c.isDigit = false) →
      ∃ q : s.Pos, q.Splits (t₁ ++ String.ofList ds) t₂ ∧
        natCoreNumDigits acc k ⟨s, p⟩ =
          .success ⟨s, q⟩ (Nat.ofDigitChars 10 ds acc, k + ds.length) := by
  induction ds with
  | nil =>
    intro acc k s p t₁ t₂ h ht
    have h' : p.Splits t₁ t₂ := by simpa using h
    refine ⟨p, by simpa using h', ?_⟩
    rw [Nat.ofDigitChars_nil, List.length_nil, Nat.add_zero]
    exact natCoreNumDigits_end h' ht
  | cons d ds ih =>
    intro acc k s p t₁ t₂ h ht
    have hd : d.isDigit = true := hds d (by simp)
    have h' : p.Splits t₁ (String.singleton d ++ (String.ofList ds ++ t₂)) := by
      rwa [String.ofList_cons, String.append_assoc] at h
    obtain ⟨q, hq, hrun⟩ :=
      ih (fun c hc => hds c (by simp [hc])) (10 * acc + (d.val - '0'.val).toNat) (k + 1)
        h'.next ht
    refine ⟨q, by rw [String.ofList_cons, ← String.append_assoc]; exact hq, ?_⟩
    have hlen : k + 1 + ds.length = k + (d :: ds).length := by
      simp only [List.length_cons]
      omega
    rw [natCoreNumDigits_cons h' hd, hrun, Nat.ofDigitChars_cons, digit_sub hd, hlen]

end Digits

section Number

open DecimalJson.Parser (decimal)

/-- The text after a value: a comma, a closing bracket or brace, or nothing. -/
private def Follows (t : String) : Prop := ∀ c ∈ t.toList.head?, c = ',' ∨ c = ']' ∨ c = '}'

private theorem Follows.head {t : String} (ht : Follows t) {P : Char → Prop} (h1 : P ',')
    (h2 : P ']') (h3 : P '}') : ∀ c ∈ t.toList.head?, P c := by
  intro c hc
  rcases ht c hc with rfl | rfl | rfl
  exacts [h1, h2, h3]

private theorem head_singleton_append {c : Char} {t : String} {P : Char → Prop} (h : P c) :
    ∀ x ∈ (String.singleton c ++ t).toList.head?, P x := by
  intro x hx
  simp only [String.toList_append, String.toList_singleton, List.singleton_append,
    List.head?_cons, Option.mem_def, Option.some.injEq] at hx
  exact hx ▸ h

private theorem splits_of_toList {s : String} {p : s.Pos} {t₁ t₂ u₁ u₂ : String}
    (h : p.Splits t₁ t₂) (h₁ : t₁.toList = u₁.toList) (h₂ : t₂.toList = u₂.toList) :
    p.Splits u₁ u₂ :=
  h.of_eq (String.toList_inj.1 h₁) (String.toList_inj.1 h₂)

private theorem char_le_iff {a b : Char} : a ≤ b ↔ a.toNat ≤ b.toNat :=
  Char.le_def.trans UInt32.le_iff_toNat_le

private theorem leading_digit {c : Char} (h : c.isDigit = true) (h0 : c ≠ '0') :
    (decide ('1' ≤ c) && decide (c ≤ '9')) = true := by
  have h1 := Char.isDigit_iff_toNat.1 h
  have h2 : c.toNat ≠ '0'.toNat := fun e => h0 (Char.toNat_inj.1 e)
  have h3 : '1'.toNat = '0'.toNat + 1 := by decide
  simp only [Bool.and_eq_true, decide_eq_true_eq, char_le_iff]
  omega

private theorem beq_of_isDigit {c d : Char} (h : c.isDigit = true) (hd : d.isDigit = false) :
    (c == d) = false :=
  beq_eq_false_iff_ne.2 fun e => by
    subst e
    rw [hd] at h
    exact Bool.false_ne_true h

private theorem toDigits_head (m : Nat) (hm : m ≠ 0) :
    ∃ c cs, Nat.toDigits 10 m = c :: cs ∧ c ≠ '0' := by
  by_cases h : m < 10
  · exact ⟨m.digitChar, [], Nat.toDigits_of_lt_base h, by simpa using hm⟩
  · obtain ⟨c, cs, hcs, hc⟩ := toDigits_head (m / 10) (by omega)
    refine ⟨c, cs ++ [Nat.digitChar (m % 10)], ?_, hc⟩
    rw [Nat.toDigits_of_base_le (b := 10) (n := m) (by decide) (by omega), hcs, List.cons_append]
termination_by m
decreasing_by omega

private theorem toString_int (e : Int) :
    toString e = (if decide (e < 0) = true then "-" else "") ++ toString e.natAbs := by
  rw [Int.toString_eq_repr, Int.repr_eq_ite]
  by_cases h : 0 ≤ e
  · rw [ite_eq_left h, decide_eq_false (show ¬e < 0 by omega),
      show e.toNat = e.natAbs by omega]
    simp
  · rw [ite_eq_right h, decide_eq_true (show e < 0 by omega),
      show (-e).toNat = e.natAbs by omega]
    simp

private theorem exp_natAbs (e : Int) :
    (if decide (e < 0) = true then -(e.natAbs : Int) else (e.natAbs : Int)) = e := by
  by_cases h : e < 0
  · rw [ite_eq_left (decide_eq_true h)]
    omega
  · rw [ite_eq_right (fun h' => h (of_decide_eq_true h'))]
    omega

private theorem toDigits_eq_digits (m : Nat) (hm : m ≠ 0) :
    Nat.toDigits 10 m = (Nat.digits 10 m).reverse.map Nat.digitChar := by
  by_cases h : m < 10
  · rw [Nat.toDigits_of_lt_base h, Nat.digits_of_lt 10 m hm h]
    rfl
  · rw [Nat.toDigits_of_base_le (b := 10) (n := m) (by decide) (by omega),
      toDigits_eq_digits (m / 10) (by omega), Nat.digits_def' (n := m) (by decide) (by omega)]
    simp
termination_by m
decreasing_by omega

private theorem naturalDigits_ten {m : Nat} (hm : m ≠ 0) :
    FloatLib.Numerics.RadixText.naturalDigits m 10 = Nat.toDigits 10 m := by
  rw [FloatLib.Numerics.RadixText.naturalDigits, ite_eq_right hm, toDigits_eq_digits m hm]

private theorem integerDigits_ne_zero {e : Int} (he : e ≠ 0) :
    FloatLib.Numerics.DecimalText.integerDigits e =
      (if decide (e < 0) = true then ['-'] else []) ++ Nat.toDigits 10 e.natAbs := by
  cases e with
  | ofNat v =>
      have hv : v ≠ 0 := fun h => he (by subst h; rfl)
      have hn : decide (Int.ofNat v < 0) = false :=
        decide_eq_false (Int.not_lt.mpr (Int.natCast_nonneg v))
      rw [FloatLib.Numerics.DecimalText.integerDigits, naturalDigits_ten hv, hn]
      rfl
  | negSucc v =>
      have hn : decide (Int.negSucc v < 0) = true := decide_eq_true (Int.negSucc_lt_zero v)
      rw [FloatLib.Numerics.DecimalText.integerDigits, naturalDigits_ten (by omega : v + 1 ≠ 0), hn]
      rfl

private theorem render_exp_zero {neg : Bool} {m : Nat} (hm : m ≠ 0) :
    Decimal.render ⟨neg, m, 0⟩ =
      (if neg = true then "-" else "") ++ String.ofList (Nat.toDigits 10 m) := by
  rw [← String.toList_inj]
  cases neg <;>
    simp [Decimal.render, hm, Decimal.format, Decimal.characters, naturalDigits_ten hm]

private theorem render_exp {neg : Bool} {m : Nat} {e : Int} (hm : m ≠ 0) (he : e ≠ 0) :
    Decimal.render ⟨neg, m, e⟩ = (if neg = true then "-" else "") ++
      (String.ofList (Nat.toDigits 10 m) ++ (String.singleton 'e' ++
        ((if decide (e < 0) = true then "-" else "") ++
          String.ofList (Nat.toDigits 10 e.natAbs)))) := by
  rw [← String.toList_inj]
  by_cases hlt : e < 0 <;> cases neg <;>
    simp [Decimal.render, hm, he, hlt, Decimal.format, Decimal.characters, naturalDigits_ten hm,
      integerDigits_ne_zero he]

private theorem nextIs_sign (f : Char → Bool) (b : Bool) {c : Char} (hc : f c = false)
    {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ ((if b = true then "-" else "") ++ (String.singleton c ++ t₂))) :
    nextIs f ⟨s, p⟩ = .success ⟨s, p⟩ (b && f '-') := by
  cases b
  · have h' : p.Splits t₁ (String.singleton c ++ t₂) := splits_of_toList h rfl (by simp)
    simp only [nextIs_cons f h', hc, Bool.false_and]
  · have h' : p.Splits t₁ (String.singleton '-' ++ (String.singleton c ++ t₂)) :=
      splits_of_toList h rfl (by simp)
    simp only [nextIs_cons f h', Bool.true_and]

private theorem sign_run {α : Type} (b : Bool) {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ ((if b = true then "-" else "") ++ t₂)) :
    ∃ q : s.Pos, q.Splits (t₁ ++ (if b = true then "-" else "")) t₂ ∧
      ∀ J : Unit → Parser α, (if b = true then skip >>= J else J ()) ⟨s, p⟩ = J () ⟨s, q⟩ := by
  cases b
  · exact ⟨p, splits_of_toList h (by simp) (by simp), fun J => rfl⟩
  · have h' : p.Splits t₁ (String.singleton '-' ++ t₂) := splits_of_toList h rfl (by simp)
    exact ⟨_, splits_of_toList h'.next (by simp) rfl, fun J => bind_of_success (skip_eq h')⟩

private theorem decimal_zero {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ ("0" ++ t₂)) (ht : Follows t₂) :
    ∃ q : s.Pos, q.Splits (t₁ ++ "0") t₂ ∧
      decimal ⟨s, p⟩ = .success ⟨s, q⟩ ⟨false, 0, 0⟩ := by
  have h1 : p.Splits t₁ (String.singleton '0' ++ t₂) := splits_of_toList h rfl (by simp)
  have h2 := h1.next
  refine ⟨p.next h1.ne_endPos_of_singleton, splits_of_toList h2 (by simp) rfl, ?_⟩
  simp only [decimal, bind_of_success (nextIs_cons _ h1), bind_of_success (peek!_eq h1),
    bind_of_success (skip_eq h1), pure_bind_apply,
    bind_of_success (nextIs_false (· == '.') h2 (ht.head (by decide) (by decide) (by decide))),
    bind_of_success (nextIs_false (fun c => c == 'e' || c == 'E') h2
      (ht.head (by decide) (by decide) (by decide))),
    pure_apply, ↓reduceIte, Bool.false_eq_true, Char.reduceBEq]
  rfl

private theorem decimal_neg_zero {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ ("-0.0" ++ t₂)) (ht : Follows t₂) :
    ∃ q : s.Pos, q.Splits (t₁ ++ "-0.0") t₂ ∧
      decimal ⟨s, p⟩ = .success ⟨s, q⟩ ⟨true, 0, -1⟩ := by
  have h1 : p.Splits t₁ (String.singleton '-' ++ (String.singleton '0' ++
      (String.singleton '.' ++ (String.singleton '0' ++ t₂)))) := splits_of_toList h rfl (by simp)
  have h2 := h1.next
  have h3 := h2.next
  have h4 := h3.next
  have h4' := splits_of_toList h4 rfl
    (by simp : (String.singleton '0' ++ t₂).toList = (String.ofList ['0'] ++ t₂).toList)
  obtain ⟨q, hq, hrun⟩ := natCoreNumDigits_eq ['0'] (by decide) 0 0 h4'
    (ht.head (by decide) (by decide) (by decide))
  refine ⟨q, splits_of_toList hq (by simp) rfl, ?_⟩
  simp only [Nat.ofDigitChars_cons, Nat.ofDigitChars_nil, List.length_cons, List.length_nil]
    at hrun
  simp only [decimal, bind_of_success (nextIs_cons _ h1), bind_of_success (skip_eq h1),
    bind_of_success (peek!_eq h2), bind_of_success (skip_eq h2), pure_bind_apply,
    bind_of_success (nextIs_cons _ h3), bind_of_success (skip_eq h3),
    bind_of_success (lookahead_eq (fun c => (decide ('0' ≤ c) && decide (c ≤ '9')) = true)
      "digit" h4 (by decide)),
    bind_of_success hrun,
    bind_of_success (nextIs_false (fun c => c == 'e' || c == 'E') hq
      (ht.head (by decide) (by decide) (by decide))),
    pure_apply, ↓reduceIte, Bool.false_eq_true, Char.reduceBEq]
  rfl

private theorem decimal_exp_zero {neg : Bool} {m : Nat} (hm : m ≠ 0) {s : String} {p : s.Pos}
    {t₁ t₂ : String} (h : p.Splits t₁ (Decimal.render ⟨neg, m, 0⟩ ++ t₂)) (ht : Follows t₂) :
    ∃ q : s.Pos, q.Splits (t₁ ++ Decimal.render ⟨neg, m, 0⟩) t₂ ∧
      decimal ⟨s, p⟩ = .success ⟨s, q⟩ ⟨neg, m, 0⟩ := by
  obtain ⟨c, cs, hds, hc⟩ := toDigits_head m hm
  have hdig : ∀ x ∈ Nat.toDigits 10 m, x.isDigit = true :=
    fun x hx => Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  have hcd : c.isDigit = true := hdig c (by simp [hds])
  rw [render_exp_zero hm] at h ⊢
  have h0 : p.Splits t₁ ((if neg = true then "-" else "") ++
      (String.singleton c ++ (String.ofList cs ++ t₂))) := splits_of_toList h rfl (by simp [hds])
  obtain ⟨q1, hq1, hJ⟩ := sign_run (α := Decimal) neg h0
  have hq1' : q1.Splits (t₁ ++ (if neg = true then "-" else ""))
      (String.ofList (Nat.toDigits 10 m) ++ t₂) := splits_of_toList hq1 rfl (by simp [hds])
  obtain ⟨q2, hq2, hnat⟩ :=
    natCore_eq _ hdig 0 hq1' (ht.head (by decide) (by decide) (by decide))
  rw [Nat.ofDigitChars_ten_toDigits] at hnat
  refine ⟨q2, splits_of_toList hq2 (by simp) rfl, ?_⟩
  simp only [decimal,
    bind_of_success (nextIs_sign (· == '-') neg (beq_of_isDigit hcd (by decide)) h0), hJ,
    bind_of_success (peek!_eq hq1), beq_eq_false_iff_ne.2 hc,
    bind_of_success (lookahead_eq (fun c => (decide ('1' ≤ c) && decide (c ≤ '9')) = true)
      "1-9" hq1 (leading_digit hcd hc)),
    bind_of_success hnat,
    bind_of_success (nextIs_false (· == '.') hq2 (ht.head (by decide) (by decide) (by decide))),
    bind_of_success (nextIs_false (fun c => c == 'e' || c == 'E') hq2
      (ht.head (by decide) (by decide) (by decide))),
    pure_apply, ↓reduceIte, Bool.false_eq_true, Char.reduceBEq, Bool.and_true,
    Nat.pow_zero, Nat.mul_one, Nat.add_zero, Int.ofNat_zero, Int.sub_zero]

private theorem decimal_exp {neg : Bool} {m : Nat} {e : Int} (hm : m ≠ 0) (he : e ≠ 0)
    {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ (Decimal.render ⟨neg, m, e⟩ ++ t₂)) (ht : Follows t₂) :
    ∃ q : s.Pos, q.Splits (t₁ ++ Decimal.render ⟨neg, m, e⟩) t₂ ∧
      decimal ⟨s, p⟩ = .success ⟨s, q⟩ ⟨neg, m, e⟩ := by
  obtain ⟨c, cs, hds, hc⟩ := toDigits_head m hm
  obtain ⟨c', cs', hds'⟩ :=
    List.exists_cons_of_ne_nil (Nat.toDigits_ne_nil (n := e.natAbs) (b := 10))
  have hdig : ∀ x ∈ Nat.toDigits 10 m, x.isDigit = true :=
    fun x hx => Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  have hdig' : ∀ x ∈ Nat.toDigits 10 e.natAbs, x.isDigit = true :=
    fun x hx => Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  have hcd : c.isDigit = true := hdig c (by simp [hds])
  have hcd' : c'.isDigit = true := hdig' c' (by simp [hds'])
  rw [render_exp hm he] at h ⊢
  have h0 : p.Splits t₁ ((if neg = true then "-" else "") ++ (String.singleton c ++
      (String.ofList cs ++ (String.singleton 'e' ++ ((if decide (e < 0) = true then "-" else "") ++
        (String.ofList (Nat.toDigits 10 e.natAbs) ++ t₂)))))) :=
    splits_of_toList h rfl (by simp [hds])
  obtain ⟨q1, hq1, hJ⟩ := sign_run (α := Decimal) neg h0
  have hq1' : q1.Splits (t₁ ++ (if neg = true then "-" else ""))
      (String.ofList (Nat.toDigits 10 m) ++ (String.singleton 'e' ++
        ((if decide (e < 0) = true then "-" else "") ++
          (String.ofList (Nat.toDigits 10 e.natAbs) ++ t₂)))) :=
    splits_of_toList hq1 rfl (by simp [hds])
  obtain ⟨q2, hq2, hnat⟩ := natCore_eq _ hdig 0 hq1' (head_singleton_append (by decide))
  rw [Nat.ofDigitChars_ten_toDigits] at hnat
  have hq3 := hq2.next
  have hq3' := splits_of_toList hq3 rfl
    (by simp [hds'] : (((if decide (e < 0) = true then "-" else "") ++
        (String.ofList (Nat.toDigits 10 e.natAbs) ++ t₂))).toList =
      ((if decide (e < 0) = true then "-" else "") ++
        (String.singleton c' ++ (String.ofList cs' ++ t₂))).toList)
  obtain ⟨q4, hq4, hJ'⟩ := sign_run (α := Decimal) (decide (e < 0)) hq3'
  have hq4' := splits_of_toList hq4 rfl
    (by simp [hds'] : (String.singleton c' ++ (String.ofList cs' ++ t₂)).toList =
      (String.ofList (Nat.toDigits 10 e.natAbs) ++ t₂).toList)
  obtain ⟨q5, hq5, hnat'⟩ :=
    natCore_eq _ hdig' 0 hq4' (ht.head (by decide) (by decide) (by decide))
  rw [Nat.ofDigitChars_ten_toDigits] at hnat'
  refine ⟨q5, splits_of_toList hq5 (by simp) rfl, ?_⟩
  simp only [decimal,
    bind_of_success (nextIs_sign (· == '-') neg (beq_of_isDigit hcd (by decide)) h0), hJ,
    bind_of_success (peek!_eq hq1), beq_eq_false_iff_ne.2 hc,
    bind_of_success (lookahead_eq (fun c => (decide ('1' ≤ c) && decide (c ≤ '9')) = true)
      "1-9" hq1 (leading_digit hcd hc)),
    bind_of_success hnat, bind_of_success (nextIs_cons _ hq2), bind_of_success (skip_eq hq2),
    bind_of_success (nextIs_sign (· == '-') (decide (e < 0)) (beq_of_isDigit hcd' (by decide))
      hq3'),
    bind_of_success (nextIs_sign (· == '+') (decide (e < 0)) (beq_of_isDigit hcd' (by decide))
      hq3'),
    hJ',
    bind_of_success (lookahead_eq (fun c => (decide ('0' ≤ c) && decide (c ≤ '9')) = true)
      "0-9" hq4 ((digit_cond c').trans hcd')),
    bind_of_success hnat',
    pure_apply, ↓reduceIte, Bool.false_eq_true, Char.reduceBEq, Bool.and_true, Bool.and_false,
    Bool.or_false, exp_natAbs, Nat.pow_zero, Nat.mul_one, Nat.add_zero,
    Int.ofNat_zero, Int.sub_zero]

private theorem decimal_render {d : Decimal}
    (hd : d.significand = 0 → d.exponent = if d.negative then -1 else 0)
    {s : String} {p : s.Pos} {t₁ t₂ : String} (h : p.Splits t₁ (Decimal.render d ++ t₂))
    (ht : Follows t₂) :
    ∃ q : s.Pos, q.Splits (t₁ ++ Decimal.render d) t₂ ∧ decimal ⟨s, p⟩ = .success ⟨s, q⟩ d := by
  obtain ⟨neg, m, e⟩ := d
  by_cases hm : m = 0
  · subst hm
    have he : e = if neg = true then -1 else 0 := hd rfl
    subst he
    cases neg
    · exact decimal_zero h ht
    · exact decimal_neg_zero h ht
  · by_cases he : e = 0
    · subst he
      exact decimal_exp_zero hm h ht
    · exact decimal_exp hm he h ht

end Number

section Strings

private theorem val_le_max (c : Char) : c.val ≤ 0x10ffff := by
  have h := c.valid
  unfold UInt32.isValidChar Nat.isValidChar at h
  rw [UInt32.le_iff_toNat_le]
  have h' : (0x10ffff : UInt32).toNat = 0x10ffff := rfl
  omega

private theorem escapeAux_plain {acc : String} {c : Char} (h : ' ' ≤ c ∧ c ≠ '"' ∧ c ≠ '\\') :
    Lean.Json.escapeAux acc c = acc.push c := by
  obtain ⟨h1, h2, h3⟩ := h
  have hn : c ≠ '\n' := by
    rintro rfl
    exact absurd h1 (by decide)
  have hr : c ≠ '\x0d' := by
    rintro rfl
    exact absurd h1 (by decide)
  have hv : 0x0020 ≤ c.val ∧ c.val ≤ 0x10ffff := ⟨h1, val_le_max c⟩
  unfold Lean.Json.escapeAux
  rw [ite_eq_right h2, ite_eq_right h3, ite_eq_right hn, ite_eq_right hr, ite_eq_left hv]

private theorem foldl_escapeAux_plain : ∀ (l : List Char) (acc : String),
    (∀ c ∈ l, ' ' ≤ c ∧ c ≠ '"' ∧ c ≠ '\\') →
      l.foldl Lean.Json.escapeAux acc = acc ++ String.ofList l
  | [], acc, _ => by simp
  | c :: l, acc, h => by
    rw [List.foldl_cons, escapeAux_plain (h c (by simp)),
      foldl_escapeAux_plain l _ fun d hd => h d (List.mem_cons_of_mem c hd),
      String.ofList_cons, String.push_eq_append, String.append_assoc]

private theorem strCore_quote {acc : String} {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ (String.singleton '"' ++ t₂)) :
    strCore acc ⟨s, p⟩ = .success ⟨s, p.next h.ne_endPos_of_singleton⟩ acc := by
  have hget : p.get h.ne_endPos_of_singleton = '"' := h.next.get_eq_of_singleton
  rw [strCore.eq_def, dite_eq_right h.ne_endPos_of_singleton]
  simp only [hget, beq_self_eq_true, ↓reduceIte]

private theorem strCore_plain {acc : String} {s : String} {p : s.Pos} {t₁ t₂ : String} {c : Char}
    (h : p.Splits t₁ (String.singleton c ++ t₂)) (hc : ' ' ≤ c ∧ c ≠ '"' ∧ c ≠ '\\') :
    strCore acc ⟨s, p⟩ = strCore (acc.push c) ⟨s, p.next h.ne_endPos_of_singleton⟩ := by
  obtain ⟨h1, h2, h3⟩ := hc
  have hget : p.get h.ne_endPos_of_singleton = c := h.next.get_eq_of_singleton
  have h1' : (0x0020 : UInt32) ≤ c.val := h1
  rw [strCore.eq_def, dite_eq_right h.ne_endPos_of_singleton]
  simp only [hget, beq_eq_false_iff_ne.2 h2, beq_eq_false_iff_ne.2 h3, decide_eq_true h1',
    decide_eq_true (val_le_max c), Bool.and_self, Bool.false_eq_true, ↓reduceIte]

private theorem escapeTable_ne_zero : ∀ k (hk : k < 256), (k < 32 ∨ k = 34 ∨ k = 92) →
    Lean.Json.escapeTable.val.get k (by rw [Lean.Json.escapeTable.property]; exact hk) ≠ 0 := by
  decide +kernel

private theorem needEscape_go_true (s : String) :
    ∀ (d i j : Nat) (hj : j < s.utf8ByteSize), j = i + d →
    Lean.Json.escapeTable.val.get (s.getUTF8Byte ⟨j⟩ hj).toNat
      (by rw [Lean.Json.escapeTable.property]; exact UInt8.toNat_lt_size _) ≠ 0 →
    Lean.Json.needEscape.go s i = true := by
  intro d
  induction d with
  | zero =>
    intro i j hj hji hb
    subst hji
    rw [Lean.Json.needEscape.go, dite_eq_left hj]
    dsimp only
    split
    · next h => exact absurd (beq_iff_eq.1 h) hb
    · rfl
  | succ d ih =>
    intro i j hj hji hb
    have hi : i < s.utf8ByteSize := by omega
    rw [Lean.Json.needEscape.go, dite_eq_left hi]
    have := ih (i + 1) j hj (by omega) hb
    dsimp only
    split
    · exact this
    · rfl

private theorem getUTF8Byte_of_toList {s : String} {l₁ l₂ : List Char} {c : Char}
    (hs : s.toList = l₁ ++ c :: l₂) (hc : c.utf8Size = 1) :
    ∃ (j : Nat) (hj : j < s.utf8ByteSize), s.getUTF8Byte ⟨j⟩ hj = c.val.toUInt8 := by
  have hs' : s.toByteArray = (String.ofList l₁).toByteArray ++
      ([c.val.toUInt8].toByteArray ++ (String.ofList l₂).toByteArray) := by
    rw [← String.utf8Encode_toList, hs, List.utf8Encode_append, List.utf8Encode_cons,
      List.utf8Encode_singleton, String.utf8EncodeChar_eq_singleton hc, String.toByteArray_ofList,
      String.toByteArray_ofList]
  have hsz : s.utf8ByteSize =
      (String.ofList l₁).utf8ByteSize + (1 + (String.ofList l₂).utf8ByteSize) := by
    rw [← String.size_toByteArray, hs', ByteArray.size_append, ByteArray.size_append,
      String.size_toByteArray, String.size_toByteArray, List.size_toByteArray]
    rfl
  refine ⟨(String.ofList l₁).utf8ByteSize, by omega, ?_⟩
  rw [String.getUTF8Byte_eq_getElem]
  simp only [hs']
  rw [ByteArray.getElem_append_right (Nat.le_of_eq String.size_toByteArray)]
  simp only [String.size_toByteArray, Nat.sub_self]
  simp

private theorem plain_of_needEscape {u : String} (h : Lean.Json.needEscape u = false) :
    ∀ c ∈ u.toList, ' ' ≤ c ∧ c ≠ '"' ∧ c ≠ '\\' := by
  intro c hc
  refine Classical.byContradiction fun hn => ?_
  have hlt : c.val.toNat < 32 ∨ c.val.toNat = 34 ∨ c.val.toNat = 92 := by
    by_cases h1 : ' ' ≤ c
    · by_cases h2 : c = '"'
      · subst h2
        decide
      · have h3 : c = '\\' := Classical.byContradiction fun h3 => hn ⟨h1, h2, h3⟩
        subst h3
        decide
    · left
      rw [Char.le_def, UInt32.le_iff_toNat_le] at h1
      have : (' ' : Char).val.toNat = 32 := rfl
      omega
  have h127 : (127 : UInt32).toNat = 127 := rfl
  have h1 : c.utf8Size = 1 := Char.utf8Size_eq_one_iff.2 (by rw [UInt32.le_iff_toNat_le]; omega)
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem hc
  obtain ⟨j, hj, hb⟩ := getUTF8Byte_of_toList hl h1
  have hb' : (u.getUTF8Byte ⟨j⟩ hj).toNat = c.val.toNat := by
    rw [hb, UInt32.toNat_toUInt8]
    exact Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le (show c.val.toNat < 128 by omega) (by decide))
  have ht := needEscape_go_true u j 0 j hj (by omega)
    (escapeTable_ne_zero _ (UInt8.toNat_lt_size _) (by rw [hb']; exact hlt))
  have h' : Lean.Json.needEscape.go u 0 = false := h
  rw [h'] at ht
  exact Bool.false_ne_true ht

private theorem escapeAux_append (acc : String) (c : Char) :
    Lean.Json.escapeAux acc c = acc ++ Lean.Json.escapeAux "" c := by
  unfold Lean.Json.escapeAux
  by_cases h1 : c = '"'
  · rw [ite_eq_left h1, ite_eq_left h1, String.empty_append]
  rw [ite_eq_right h1, ite_eq_right h1]
  by_cases h2 : c = '\\'
  · rw [ite_eq_left h2, ite_eq_left h2, String.empty_append]
  rw [ite_eq_right h2, ite_eq_right h2]
  by_cases h3 : c = '\n'
  · rw [ite_eq_left h3, ite_eq_left h3, String.empty_append]
  rw [ite_eq_right h3, ite_eq_right h3]
  by_cases h4 : c = '\x0d'
  · rw [ite_eq_left h4, ite_eq_left h4, String.empty_append]
  rw [ite_eq_right h4, ite_eq_right h4]
  by_cases h5 : 0x0020 ≤ c.val ∧ c.val ≤ 0x10ffff
  · rw [ite_eq_left h5, ite_eq_left h5, String.push_eq_append, String.push_eq_append,
      String.empty_append]
  rw [ite_eq_right h5, ite_eq_right h5]
  simp only [String.push_eq_append, String.empty_append, String.append_assoc]

private theorem foldl_escapeAux : ∀ (l : List Char) (acc : String),
    l.foldl Lean.Json.escapeAux acc = acc ++ l.foldl Lean.Json.escapeAux ""
  | [], acc => by simp
  | c :: l, acc => by
    rw [List.foldl_cons, List.foldl_cons, foldl_escapeAux l (Lean.Json.escapeAux acc c),
      foldl_escapeAux l (Lean.Json.escapeAux "" c), escapeAux_append acc c, String.append_assoc]

private theorem hexChar_digitChar {n : Nat} (hn : n < 16) {c : Char} {s : String} {p : s.Pos}
    {t₁ t₂ : String} (h : p.Splits t₁ (String.singleton c ++ t₂)) (hc : Nat.digitChar n = c) :
    Lean.Json.Parser.hexChar ⟨s, p⟩ =
      .success ⟨s, p.next h.ne_endPos_of_singleton⟩ n.toUInt16 := by
  unfold Lean.Json.Parser.hexChar
  rw [bind_of_success (any_eq h)]
  have : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨
    n = 10 ∨ n = 11 ∨ n = 12 ∨ n = 13 ∨ n = 14 ∨ n = 15 := by omega
  rcases this with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl
  · rw [Nat.digitChar_eq_zero.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_one.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_two.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_three.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_four.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_five.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_six.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_seven.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_eight.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_nine.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_a.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_b.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_c.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_d.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_e.2 rfl] at hc; subst hc; rfl
  · rw [Nat.digitChar_eq_f.2 rfl] at hc; subst hc; rfl

private theorem escapedChar_simple {s : String} {p : s.Pos} {t₁ t₂ : String} {c e : Char}
    (h : p.Splits t₁ (String.singleton c ++ t₂))
    (hce : c = '"' ∧ e = '"' ∨ c = '\\' ∧ e = '\\' ∨ c = 'n' ∧ e = '\n' ∨ c = 'r' ∧ e = '\x0d') :
    Lean.Json.Parser.escapedChar ⟨s, p⟩ = .success ⟨s, p.next h.ne_endPos_of_singleton⟩ e := by
  unfold Lean.Json.Parser.escapedChar
  rw [bind_of_success (any_eq h)]
  rcases hce with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> rfl

private theorem hexDigits_val : ∀ k, k < 32 →
    ((k / 4096).toUInt16 <<< (12 : UInt16) ||| ((k % 4096) / 256).toUInt16 <<< (8 : UInt16) |||
      ((k % 256) / 16).toUInt16 <<< (4 : UInt16) ||| (k % 16).toUInt16 : UInt16) < 0xD800 ∧
    ((k / 4096).toUInt16 <<< (12 : UInt16) ||| ((k % 4096) / 256).toUInt16 <<< (8 : UInt16) |||
      ((k % 256) / 16).toUInt16 <<< (4 : UInt16) ||| (k % 16).toUInt16 : UInt16).toUInt32.toNat =
        k := by
  decide

private theorem strCore_backslash {acc : String} {s : String} {p : s.Pos} {t₁ t₂ : String}
    {it : Sigma String.Pos} {e : Char} (h : p.Splits t₁ (String.singleton '\\' ++ t₂))
    (he : Lean.Json.Parser.escapedChar ⟨s, p.next h.ne_endPos_of_singleton⟩ = .success it e) :
    strCore acc ⟨s, p⟩ = strCore (acc.push e) it := by
  have hget : p.get h.ne_endPos_of_singleton = '\\' := h.next.get_eq_of_singleton
  rw [strCore.eq_def, dite_eq_right h.ne_endPos_of_singleton]
  simp only [hget, Char.reduceBEq, Bool.false_eq_true, ↓reduceIte]
  split
  · next it'' e' he' =>
    rw [he] at he'
    cases he'
    rfl
  · next it'' err he' =>
    rw [he] at he'
    cases he'

private theorem strCore_escapeAux (c : Char) (acc : String) {s : String} {p : s.Pos}
    {t₁ t₂ : String}
    (h : p.Splits t₁ (Lean.Json.escapeAux "" c ++ t₂)) :
    ∃ q : s.Pos, q.Splits (t₁ ++ Lean.Json.escapeAux "" c) t₂ ∧
      strCore acc ⟨s, p⟩ = strCore (acc.push c) ⟨s, q⟩ := by
  by_cases h1 : c = '"'
  · subst h1
    have hE : Lean.Json.escapeAux "" '"' = "\\\"" := by
      unfold Lean.Json.escapeAux
      rw [ite_eq_left rfl, String.empty_append]
    rw [hE] at h ⊢
    have h0 : p.Splits t₁ (String.singleton '\\' ++ (String.singleton '"' ++ t₂)) :=
      splits_of_toList h rfl (by simp)
    exact ⟨_, splits_of_toList h0.next.next (by simp) rfl,
      strCore_backslash h0 (escapedChar_simple h0.next (Or.inl ⟨rfl, rfl⟩))⟩
  by_cases h2 : c = '\\'
  · subst h2
    have hE : Lean.Json.escapeAux "" '\\' = "\\\\" := by
      unfold Lean.Json.escapeAux
      rw [ite_eq_right h1, ite_eq_left rfl, String.empty_append]
    rw [hE] at h ⊢
    have h0 : p.Splits t₁ (String.singleton '\\' ++ (String.singleton '\\' ++ t₂)) :=
      splits_of_toList h rfl (by simp)
    exact ⟨_, splits_of_toList h0.next.next (by simp) rfl,
      strCore_backslash h0 (escapedChar_simple h0.next (Or.inr (Or.inl ⟨rfl, rfl⟩)))⟩
  by_cases h3 : c = '\n'
  · subst h3
    have hE : Lean.Json.escapeAux "" '\n' = "\\n" := by
      unfold Lean.Json.escapeAux
      rw [ite_eq_right h1, ite_eq_right h2, ite_eq_left rfl, String.empty_append]
    rw [hE] at h ⊢
    have h0 : p.Splits t₁ (String.singleton '\\' ++ (String.singleton 'n' ++ t₂)) :=
      splits_of_toList h rfl (by simp)
    exact ⟨_, splits_of_toList h0.next.next (by simp) rfl,
      strCore_backslash h0 (escapedChar_simple h0.next (Or.inr (Or.inr (Or.inl ⟨rfl, rfl⟩))))⟩
  by_cases h4 : c = '\x0d'
  · subst h4
    have hE : Lean.Json.escapeAux "" '\x0d' = "\\r" := by
      unfold Lean.Json.escapeAux
      rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, ite_eq_left rfl, String.empty_append]
    rw [hE] at h ⊢
    have h0 : p.Splits t₁ (String.singleton '\\' ++ (String.singleton 'r' ++ t₂)) :=
      splits_of_toList h rfl (by simp)
    exact ⟨_, splits_of_toList h0.next.next (by simp) rfl,
      strCore_backslash h0 (escapedChar_simple h0.next (Or.inr (Or.inr (Or.inr ⟨rfl, rfl⟩))))⟩
  by_cases h5 : 0x0020 ≤ c.val ∧ c.val ≤ 0x10ffff
  · have hE : Lean.Json.escapeAux "" c = String.singleton c := by
      unfold Lean.Json.escapeAux
      rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, ite_eq_right h4, ite_eq_left h5,
        String.push_eq_append, String.empty_append]
    rw [hE] at h ⊢
    exact ⟨_, h.next, strCore_plain h ⟨h5.1, h1, h2⟩⟩
  have hc : c.toNat < 32 := by
    have hlt : ¬(0x0020 : UInt32) ≤ c.val := fun h => h5 ⟨h, val_le_max c⟩
    rw [UInt32.le_iff_toNat_le] at hlt
    have : (0x0020 : UInt32).toNat = 32 := rfl
    show c.val.toNat < 32
    omega
  have hE : Lean.Json.escapeAux "" c = "\\u" ++ String.singleton (Nat.digitChar (c.toNat / 4096)) ++
      String.singleton (Nat.digitChar ((c.toNat % 4096) / 256)) ++
      String.singleton (Nat.digitChar ((c.toNat % 256) / 16)) ++
      String.singleton (Nat.digitChar (c.toNat % 16)) := by
    unfold Lean.Json.escapeAux
    rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, ite_eq_right h4, ite_eq_right h5]
    simp only [String.push_eq_append, String.empty_append]
  rw [hE] at h ⊢
  have h0 : p.Splits t₁ (String.singleton '\\' ++ (String.singleton 'u' ++
      (String.singleton (Nat.digitChar (c.toNat / 4096)) ++
      (String.singleton (Nat.digitChar ((c.toNat % 4096) / 256)) ++
      (String.singleton (Nat.digitChar ((c.toNat % 256) / 16)) ++
      (String.singleton (Nat.digitChar (c.toNat % 16)) ++ t₂)))))) :=
    splits_of_toList h rfl (by simp)
  refine ⟨_, splits_of_toList h0.next.next.next.next.next.next (by simp) rfl, ?_⟩
  apply strCore_backslash h0
  unfold Lean.Json.Parser.escapedChar
  rw [bind_of_success (any_eq h0.next)]
  simp only [bind_of_success (hexChar_digitChar (by omega) h0.next.next rfl),
    bind_of_success (hexChar_digitChar (by omega) h0.next.next.next rfl),
    bind_of_success (hexChar_digitChar (by omega) h0.next.next.next.next rfl),
    bind_of_success (hexChar_digitChar (by omega) h0.next.next.next.next.next rfl)]
  obtain ⟨hlt, hval⟩ := hexDigits_val c.toNat hc
  rw [dite_eq_left hlt, pure_apply]
  congr 1
  exact Char.val_inj.1 (UInt32.toNat_inj.1 hval)

private theorem strCore_foldl (l : List Char) :
    ∀ (acc : String) {s : String} {p : s.Pos} {t₁ t₂ : String},
    p.Splits t₁ (l.foldl Lean.Json.escapeAux "" ++ ("\"" ++ t₂)) →
    ∃ q : s.Pos, q.Splits (t₁ ++ (l.foldl Lean.Json.escapeAux "" ++ "\"")) t₂ ∧
      strCore acc ⟨s, p⟩ = .success ⟨s, q⟩ (acc ++ String.ofList l) := by
  induction l with
  | nil =>
    intro acc s p t₁ t₂ h
    have h' : p.Splits t₁ (String.singleton '"' ++ t₂) := splits_of_toList h rfl (by simp)
    refine ⟨_, splits_of_toList h'.next (by simp) rfl, ?_⟩
    rw [strCore_quote h']
    simp
  | cons c l ih =>
    intro acc s p t₁ t₂ h
    rw [List.foldl_cons, foldl_escapeAux l (Lean.Json.escapeAux "" c)] at h ⊢
    have h' : p.Splits t₁ (Lean.Json.escapeAux "" c ++
        (l.foldl Lean.Json.escapeAux "" ++ ("\"" ++ t₂))) :=
      splits_of_toList h rfl (by simp)
    obtain ⟨q, hq, hrun⟩ := strCore_escapeAux c acc h'
    obtain ⟨q', hq', hrun'⟩ := ih (acc.push c) hq
    refine ⟨q', splits_of_toList hq' (by simp) rfl, ?_⟩
    rw [hrun, hrun', String.ofList_cons, String.push_eq_append, String.append_assoc]

private theorem renderString_eq (u : String) :
    Lean.Json.renderString u = "\"" ++ Lean.Json.escape u ++ "\"" := by
  simp only [Lean.Json.renderString, Lean.Json.escape, String.foldl_eq_foldl_toList,
    String.empty_append]
  by_cases hn : Lean.Json.needEscape u = true
  · rw [ite_eq_left hn, ite_eq_left hn, foldl_escapeAux _ "\""]
  · rw [ite_eq_right hn, ite_eq_right hn]

private theorem strCore_escape {u : String} {s : String} {p : s.Pos} {t₁ t₂ : String}
    (h : p.Splits t₁ (Lean.Json.escape u ++ ("\"" ++ t₂))) :
    ∃ q : s.Pos, q.Splits (t₁ ++ (Lean.Json.escape u ++ "\"")) t₂ ∧
      strCore "" ⟨s, p⟩ = .success ⟨s, q⟩ u := by
  have he : Lean.Json.escape u = u.toList.foldl Lean.Json.escapeAux "" := by
    simp only [Lean.Json.escape, String.foldl_eq_foldl_toList, String.empty_append]
    by_cases hn : Lean.Json.needEscape u = true
    · rw [ite_eq_left hn]
    · rw [ite_eq_right hn, foldl_escapeAux_plain _ _ (plain_of_needEscape (by simpa using hn)),
        String.empty_append, String.ofList_toList]
  rw [he] at h ⊢
  obtain ⟨q, hq, hrun⟩ := strCore_foldl u.toList "" h
  rw [String.ofList_toList, String.empty_append] at hrun
  exact ⟨q, hq, hrun⟩

end Strings

section Values

open DecimalJson.Parser (value)

private theorem head_of_eq {t : String} {c : Char} {P : Char → Prop}
    (ht : t.toList.head? = some c) (hc : P c) : ∀ x ∈ t.toList.head?, P x := by
  intro x hx
  rw [ht, Option.mem_def, Option.some.injEq] at hx
  exact hx ▸ hc

private theorem follows_of_head {t : String} {c : Char} (ht : t.toList.head? = some c)
    (hc : c = ',' ∨ c = ']' ∨ c = '}') : Follows t :=
  head_of_eq ht hc

private theorem not_ws_follows {t : String} (ht : Follows t) :
    ∀ c ∈ t.toList.head?, ¬(c = '\t' ∨ c = '\n' ∨ c = '\r' ∨ c = ' ') :=
  ht.head (by decide) (by decide) (by decide)

private theorem head_append {X Y : String} {c : Char} (h : X.toList.head? = some c) :
    (X ++ Y).toList.head? = some c := by
  simp [String.toList_append, List.head?_append, h]

private theorem splits_head {s : String} {p : s.Pos} {t₁ X t₂ : String} {c : Char}
    (hX : X.toList.head? = some c) (h : p.Splits t₁ (X ++ t₂)) :
    p.Splits t₁ (String.singleton c ++ String.ofList (X.toList.tail ++ t₂.toList)) := by
  obtain ⟨L, hL⟩ := List.head?_eq_some_iff.1 hX
  exact splits_of_toList h rfl (by simp [hL])

private theorem length_le_utf8ByteSize (u : String) : u.length ≤ u.utf8ByteSize := by
  obtain ⟨l, rfl⟩ : ∃ l, u = String.ofList l := ⟨u.toList, String.ofList_toList.symm⟩
  induction l with
  | nil => simp
  | cons c l ih =>
    rw [String.ofList_cons, String.length_append, String.utf8ByteSize_append,
      String.length_singleton, String.utf8ByteSize_singleton]
    have := c.utf8Size_pos
    omega

private theorem intercalate_cons_eq (a : String) (l : List String) :
    ∃ R, ",".intercalate (a :: l) = a ++ R := by
  cases l with
  | nil => exact ⟨"", by simp⟩
  | cons b l => exact ⟨"," ++ ",".intercalate (b :: l), by simp [String.append_assoc]⟩

private theorem head_intercalate {a : String} {l : List String} {Y : String} {c : Char}
    (h : a.toList.head? = some c) : (",".intercalate (a :: l) ++ Y).toList.head? = some c := by
  obtain ⟨R, hR⟩ := intercalate_cons_eq a l
  rw [hR, String.append_assoc]
  exact head_append h

private theorem render_bool (b : Bool) :
    Value.render (.bool b) = if b = true then "true" else "false" := by
  rw [Value.render]
  cases b <;> rfl

private theorem render_num_head (d : Decimal) :
    ∃ c, (Decimal.render d).toList.head? = some c ∧ (c = '-' ∨ c.isDigit = true) := by
  obtain ⟨neg, m, e⟩ := d
  by_cases hm : m = 0
  · subst hm
    cases neg
    · exact ⟨'0', by simp [Decimal.render], Or.inr (by decide)⟩
    · exact ⟨'-', by simp [Decimal.render], Or.inl rfl⟩
  · obtain ⟨c, cs, hds, -⟩ := toDigits_head m hm
    have hcd : c.isDigit = true :=
      Nat.isDigit_of_mem_toDigits (b := 10) (n := m) (by decide) (by decide) (by simp [hds])
    by_cases he : e = 0
    · subst he
      cases neg
      · exact ⟨c, by rw [render_exp_zero hm]; simp [hds], Or.inr hcd⟩
      · exact ⟨'-', by rw [render_exp_zero hm]; simp, Or.inl rfl⟩
    · cases neg
      · exact ⟨c, by rw [render_exp hm he]; simp [hds], Or.inr hcd⟩
      · exact ⟨'-', by rw [render_exp hm he]; simp, Or.inl rfl⟩

private theorem start_facts {c : Char}
    (hc : c = '[' ∨ c = '{' ∨ c = '"' ∨ c = 'n' ∨ c = 't' ∨ c = 'f' ∨ c = '-' ∨
      c.isDigit = true) :
    ¬(c = '\t' ∨ c = '\n' ∨ c = '\r' ∨ c = ' ') ∧ (c == ']') = false := by
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | hc
  any_goals decide
  refine ⟨?_, beq_of_isDigit hc (by decide)⟩
  rintro (rfl | rfl | rfl | rfl) <;> exact absurd hc (by decide)

private theorem render_head {v : Value} (hv : v.Canonical) :
    ∃ c, v.render.toList.head? = some c ∧
      (c = '[' ∨ c = '{' ∨ c = '"' ∨ c = 'n' ∨ c = 't' ∨ c = 'f' ∨ c = '-' ∨
        c.isDigit = true) := by
  cases hv with
  | null => exact ⟨'n', by rw [Value.render]; simp, by decide⟩
  | bool b =>
    cases b
    · exact ⟨'f', by rw [render_bool]; simp, by decide⟩
    · exact ⟨'t', by rw [render_bool]; simp, by decide⟩
  | num d _ =>
    obtain ⟨c, hc, h⟩ := render_num_head d
    refine ⟨c, by rw [Value.render]; exact hc, ?_⟩
    rcases h with h | h
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl h))))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr h))))))
  | str u => exact ⟨'"', by rw [Value.render, renderString_eq]; simp, by decide⟩
  | arr xs _ => exact ⟨'[', by rw [Value.render]; simp, by decide⟩
  | obj fields _ _ => exact ⟨'{', by rw [Value.render]; simp, by decide⟩

/-- Reading `v.render` with `value n` gives `v` when the text is shorter than the fuel. -/
private def ReadsBack (n : Nat) : Prop :=
  ∀ v : Value, v.Canonical → v.render.length < n →
    ∀ {s : String} {p : s.Pos} {t₁ t₂ : String}, p.Splits t₁ (v.render ++ t₂) → Follows t₂ →
      ∃ q : s.Pos, q.Splits (t₁ ++ v.render) t₂ ∧ value n ⟨s, p⟩ = .success ⟨s, q⟩ v

private theorem arrayItems_render {N : Nat} (H : ∀ m < N, ReadsBack m) :
    ∀ (xs : List Value) (x : Value) (acc : Array Value) (m : Nat), m ≤ N →
      (∀ y ∈ x :: xs, y.Canonical) →
      (",".intercalate ((x :: xs).map Value.render) ++ "]").length < m →
      ∀ {s : String} {p : s.Pos} {t₁ t₂ : String},
        p.Splits t₁ ((",".intercalate ((x :: xs).map Value.render) ++ "]") ++ t₂) →
        Follows t₂ →
        ∃ q : s.Pos, q.Splits (t₁ ++ (",".intercalate ((x :: xs).map Value.render) ++ "]")) t₂ ∧
          Parser.arrayItems m acc ⟨s, p⟩ = .success ⟨s, q⟩ (acc ++ (x :: xs).toArray) := by
  intro xs
  induction xs with
  | nil =>
    intro x acc m hm hc hlen s p t₁ t₂ h ht
    obtain _ | m := m
    · exact absurd hlen (Nat.not_lt_zero _)
    simp only [List.map_cons, List.map_nil, String.intercalate_singleton] at hlen h ⊢
    have hlen' : x.render.length < m := by
      have : "]".length = 1 := rfl
      simp only [String.length_append] at hlen
      omega
    have hx : p.Splits t₁ (x.render ++ ("]" ++ t₂)) := h.of_eq rfl (by rw [String.append_assoc])
    obtain ⟨q, hq, hrun⟩ := H m (by omega) x (hc x (by simp)) hlen' hx
      (follows_of_head (c := ']') (by simp) (Or.inr (Or.inl rfl)))
    have hq' : q.Splits (t₁ ++ x.render) (String.singleton ']' ++ t₂) :=
      splits_of_toList hq rfl (by simp)
    refine ⟨q.next hq'.ne_endPos_of_singleton, splits_of_toList hq'.next (by simp) rfl, ?_⟩
    rw [Parser.arrayItems]
    simp only [bind_of_success hrun, bind_of_success (any_eq hq'), Char.reduceBEq, ↓reduceIte,
      bind_of_success (ws_eq hq'.next (not_ws_follows ht)), pure_apply]
    congr 1
  | cons y ys ih =>
    intro x acc m hm hc hlen s p t₁ t₂ h ht
    obtain _ | m := m
    · exact absurd hlen (Nat.not_lt_zero _)
    have e1 : ",".intercalate ((x :: y :: ys).map Value.render) ++ "]" =
        x.render ++ ("," ++ (",".intercalate ((y :: ys).map Value.render) ++ "]")) := by
      simp only [List.map_cons, String.intercalate_cons_cons, String.append_assoc]
    rw [e1] at hlen h ⊢
    have hlen' : x.render.length < m ∧
        (",".intercalate ((y :: ys).map Value.render) ++ "]").length < m := by
      have : ",".length = 1 := rfl
      simp only [String.length_append] at hlen ⊢
      omega
    have hx : p.Splits t₁ (x.render ++
        ("," ++ (",".intercalate ((y :: ys).map Value.render) ++ "]" ++ t₂))) :=
      h.of_eq rfl (by simp only [String.append_assoc])
    obtain ⟨q, hq, hrun⟩ := H m (by omega) x (hc x (by simp)) hlen'.1 hx
      (follows_of_head (c := ',') (by simp) (Or.inl rfl))
    have hq' : q.Splits (t₁ ++ x.render) (String.singleton ',' ++
        (",".intercalate ((y :: ys).map Value.render) ++ "]" ++ t₂)) :=
      splits_of_toList hq rfl (by simp)
    obtain ⟨c, hc1, hc2⟩ := render_head (hc y (by simp))
    obtain ⟨R, hR⟩ := intercalate_cons_eq y.render (ys.map Value.render)
    have hhead : (",".intercalate ((y :: ys).map Value.render) ++ "]" ++ t₂).toList.head? =
        some c := by
      simp only [List.map_cons, hR, String.append_assoc]
      exact head_append hc1
    obtain ⟨q', hq'', hrun'⟩ := ih y (acc.push x) m (by omega)
      (fun z hz => hc z (List.mem_cons_of_mem x hz)) hlen'.2 hq'.next ht
    refine ⟨q', splits_of_toList hq'' (by simp) rfl, ?_⟩
    rw [Parser.arrayItems]
    simp only [bind_of_success hrun, bind_of_success (any_eq hq'), Char.reduceBEq, ↓reduceIte,
      Bool.false_eq_true, bind_of_success (ws_eq hq'.next (head_of_eq hhead (start_facts hc2).1)),
      hrun']
    congr 1
    exact Array.toList_inj.1 (by simp)

private theorem field_steps {k : String} {v : Value} (hv : v.Canonical) {s : String} {p : s.Pos}
    {t₁ t₂ : String}
    (h : p.Splits t₁ (String.singleton '"' ++ (Lean.Json.escape k ++ (String.singleton '"' ++
      (String.singleton ':' ++ (v.render ++ t₂)))))) :
    ∃ q₁ q₂ q₃ : s.Pos,
      Lean.Json.Parser.lookahead (fun c => c == '"') "\"" ⟨s, p⟩ = .success ⟨s, p⟩ () ∧
      (skip : Parser Unit) ⟨s, p⟩ = .success ⟨s, q₁⟩ () ∧
      strCore "" ⟨s, q₁⟩ = .success ⟨s, q₂⟩ k ∧
      ws ⟨s, q₂⟩ = .success ⟨s, q₂⟩ () ∧
      Lean.Json.Parser.lookahead (fun c => c == ':') ":" ⟨s, q₂⟩ = .success ⟨s, q₂⟩ () ∧
      (skip : Parser Unit) ⟨s, q₂⟩ = .success ⟨s, q₃⟩ () ∧
      ws ⟨s, q₃⟩ = .success ⟨s, q₃⟩ () ∧
      q₃.Splits (t₁ ++ ("\"" ++ Lean.Json.escape k ++ "\"" ++ ":")) (v.render ++ t₂) := by
  obtain ⟨q₂, hq₂, hrun⟩ := strCore_escape (splits_of_toList h.next rfl (by simp) :
    (p.next h.ne_endPos_of_singleton).Splits (t₁ ++ String.singleton '"')
      (Lean.Json.escape k ++ ("\"" ++ (String.singleton ':' ++ (v.render ++ t₂)))))
  have hq₂' : q₂.Splits (t₁ ++ String.singleton '"' ++ (Lean.Json.escape k ++ "\""))
      (String.singleton ':' ++ (v.render ++ t₂)) := hq₂
  obtain ⟨c, hc1, hc2⟩ := render_head hv
  refine ⟨_, q₂, _, lookahead_eq _ "\"" h (by decide), skip_eq h, hrun,
    ws_eq hq₂' (head_singleton_append (by decide)), lookahead_eq _ ":" hq₂' (by decide),
    skip_eq hq₂', ws_eq hq₂'.next (head_of_eq (head_append hc1) (start_facts hc2).1),
    splits_of_toList hq₂'.next (by simp) rfl⟩

private theorem objectFields_render {N : Nat} (H : ∀ m < N, ReadsBack m) :
    ∀ (fs : List (String × Value)) (f : String × Value) (acc : Array (String × Value))
      (m : Nat), m ≤ N → ((acc.toList ++ f :: fs).map Prod.fst).Nodup →
      (∀ g ∈ f :: fs, g.2.Canonical) →
      (",".intercalate ((f :: fs).map fun g => Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++
        "}").length < m →
      ∀ {s : String} {p : s.Pos} {t₁ t₂ : String},
        p.Splits t₁ ((",".intercalate ((f :: fs).map fun g =>
          Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}") ++ t₂) →
        Follows t₂ →
        ∃ q : s.Pos, q.Splits (t₁ ++ (",".intercalate ((f :: fs).map fun g =>
            Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}")) t₂ ∧
          Parser.objectFields m acc ⟨s, p⟩ = .success ⟨s, q⟩ (acc ++ (f :: fs).toArray) := by
  intro fs
  induction fs with
  | nil =>
    intro f acc m hm hnd hc hlen s p t₁ t₂ h ht
    obtain ⟨k, v⟩ := f
    obtain _ | m := m
    · exact absurd hlen (Nat.not_lt_zero _)
    have hv := hc (k, v) (by simp)
    simp only [List.map_cons, List.map_nil, String.intercalate_singleton] at hlen h ⊢
    rw [renderString_eq k] at hlen h ⊢
    have hlen' : v.render.length < m := by
      have : "}".length = 1 := rfl
      simp only [String.length_append] at hlen
      omega
    have h' : p.Splits t₁ (String.singleton '"' ++ (Lean.Json.escape k ++ (String.singleton '"' ++
        (String.singleton ':' ++ (v.render ++ ("}" ++ t₂)))))) :=
      splits_of_toList h rfl (by simp)
    obtain ⟨q₁, q₂, q₃, e1, e2, e3, e4, e5, e6, e7, hq₃⟩ := field_steps hv h'
    have hany : (acc.any fun g => g.1 == k) = false := by
      rw [Array.any_eq_false']
      intro g hg hgk
      rw [List.map_append, List.nodup_append] at hnd
      exact hnd.2.2 g.1 (List.mem_map_of_mem (Array.mem_toList_iff.2 hg)) k (by simp)
        (beq_iff_eq.1 hgk)
    obtain ⟨q, hq, hrun⟩ := H m (by omega) v hv hlen' hq₃
      (follows_of_head (c := '}') (by simp) (Or.inr (Or.inr rfl)))
    have hq' : q.Splits (t₁ ++ ("\"" ++ Lean.Json.escape k ++ "\"" ++ ":") ++ v.render)
        (String.singleton '}' ++ t₂) :=
      splits_of_toList hq rfl (by simp)
    refine ⟨q.next hq'.ne_endPos_of_singleton, splits_of_toList hq'.next (by simp) rfl, ?_⟩
    rw [Parser.objectFields]
    simp only [bind_of_success e1, bind_of_success e2, bind_of_success e3, bind_of_success e4,
      bind_of_success e5, bind_of_success e6, bind_of_success e7, hany, Bool.false_eq_true,
      ↓reduceIte, bind_of_success hrun, bind_of_success (any_eq hq'), Char.reduceBEq,
      bind_of_success (ws_eq hq'.next (not_ws_follows ht)), pure_apply]
    congr 1
  | cons f' fs ih =>
    intro f acc m hm hnd hc hlen s p t₁ t₂ h ht
    obtain ⟨k, v⟩ := f
    obtain _ | m := m
    · exact absurd hlen (Nat.not_lt_zero _)
    have hv := hc (k, v) (by simp)
    have e0 : ",".intercalate (((k, v) :: f' :: fs).map fun g =>
          Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}" =
        "\"" ++ Lean.Json.escape k ++ "\"" ++ ":" ++ v.render ++
          ("," ++ (",".intercalate ((f' :: fs).map fun g =>
          Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}")) := by
      simp only [List.map_cons, String.intercalate_cons_cons, renderString_eq k,
        String.append_assoc]
    rw [e0] at hlen h ⊢
    have hlen' : v.render.length < m ∧ (",".intercalate ((f' :: fs).map fun g =>
        Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}").length < m := by
      have : ",".length = 1 := rfl
      simp only [String.length_append] at hlen ⊢
      omega
    have h' : p.Splits t₁ (String.singleton '"' ++ (Lean.Json.escape k ++ (String.singleton '"' ++
        (String.singleton ':' ++ (v.render ++ ("," ++ (",".intercalate ((f' :: fs).map fun g =>
          Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}" ++ t₂))))))) :=
      splits_of_toList h rfl (by simp)
    obtain ⟨q₁, q₂, q₃, e1, e2, e3, e4, e5, e6, e7, hq₃⟩ := field_steps hv h'
    have hany : (acc.any fun g => g.1 == k) = false := by
      rw [Array.any_eq_false']
      intro g hg hgk
      rw [List.map_append, List.nodup_append] at hnd
      exact hnd.2.2 g.1 (List.mem_map_of_mem (Array.mem_toList_iff.2 hg)) k (by simp)
        (beq_iff_eq.1 hgk)
    obtain ⟨q, hq, hrun⟩ := H m (by omega) v hv hlen'.1 hq₃
      (follows_of_head (c := ',') (by simp) (Or.inl rfl))
    have hq' : q.Splits (t₁ ++ ("\"" ++ Lean.Json.escape k ++ "\"" ++ ":") ++ v.render)
        (String.singleton ',' ++ (",".intercalate ((f' :: fs).map fun g =>
          Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}" ++ t₂)) :=
      splits_of_toList hq rfl (by simp)
    have hhead : (",".intercalate ((f' :: fs).map fun g =>
        Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}" ++ t₂).toList.head? =
        some '"' := by
      rw [String.append_assoc, List.map_cons]
      exact head_intercalate (by simp [renderString_eq])
    have hnd' : (((acc.push (k, v)).toList ++ f' :: fs).map Prod.fst).Nodup := by
      simpa [Array.toList_push, List.append_assoc] using hnd
    obtain ⟨q', hq'', hrun'⟩ := ih f' (acc.push (k, v)) m (by omega) hnd'
      (fun g hg => hc g (List.mem_cons_of_mem _ hg)) hlen'.2 hq'.next ht
    refine ⟨q', splits_of_toList hq'' (by simp) rfl, ?_⟩
    rw [Parser.objectFields]
    simp only [bind_of_success e1, bind_of_success e2, bind_of_success e3, bind_of_success e4,
      bind_of_success e5, bind_of_success e6, bind_of_success e7, hany, Bool.false_eq_true,
      ↓reduceIte, bind_of_success hrun, bind_of_success (any_eq hq'), Char.reduceBEq,
      bind_of_success (ws_eq hq'.next (head_of_eq hhead (by decide))), hrun']
    congr 1
    exact Array.toList_inj.1 (by simp)

private theorem readsBack (n : Nat) : ReadsBack n := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
  intro v hv hlen s p t₁ t₂ h ht
  obtain _ | n := n
  · exact absurd hlen (Nat.not_lt_zero _)
  cases hv with
  | null =>
    rw [Value.render] at h ⊢
    have h1 : p.Splits t₁ (String.singleton 'n' ++ ("ull" ++ t₂)) :=
      splits_of_toList h rfl (by simp)
    refine ⟨_, splits_nextn h, ?_⟩
    rw [value]
    simp only [bind_of_success (peek!_eq h1), Char.reduceBEq, Bool.false_eq_true, ↓reduceIte,
      bind_of_success (skipString_eq h),
      bind_of_success (ws_eq (splits_nextn h) (not_ws_follows ht)), pure_apply]
  | bool b =>
    cases b
    · have e : Value.render (.bool false) = "false" := by rw [render_bool]; rfl
      rw [e] at h ⊢
      have h1 : p.Splits t₁ (String.singleton 'f' ++ ("alse" ++ t₂)) :=
        splits_of_toList h rfl (by simp)
      refine ⟨_, splits_nextn h, ?_⟩
      rw [value]
      simp only [bind_of_success (peek!_eq h1), Char.reduceBEq, Bool.false_eq_true, ↓reduceIte,
        bind_of_success (skipString_eq h),
        bind_of_success (ws_eq (splits_nextn h) (not_ws_follows ht)), pure_apply]
    · have e : Value.render (.bool true) = "true" := by rw [render_bool]; rfl
      rw [e] at h ⊢
      have h1 : p.Splits t₁ (String.singleton 't' ++ ("rue" ++ t₂)) :=
        splits_of_toList h rfl (by simp)
      refine ⟨_, splits_nextn h, ?_⟩
      rw [value]
      simp only [bind_of_success (peek!_eq h1), Char.reduceBEq, Bool.false_eq_true, ↓reduceIte,
        bind_of_success (skipString_eq h),
        bind_of_success (ws_eq (splits_nextn h) (not_ws_follows ht)), pure_apply]
  | num d hd =>
    rw [Value.render] at h ⊢
    obtain ⟨c, hc1, hc2⟩ := render_num_head d
    obtain ⟨q, hq, hrun⟩ := decimal_render hd h ht
    refine ⟨q, hq, ?_⟩
    have h1 := splits_head hc1 h
    have hb : (c == '[') = false ∧ (c == '{') = false ∧ (c == '"') = false ∧
        (c == 'f') = false ∧ (c == 't') = false ∧ (c == 'n') = false ∧
        (c == '-' || c.isDigit) = true := by
      rcases hc2 with rfl | hc2
      · decide
      · exact ⟨beq_of_isDigit hc2 (by decide), beq_of_isDigit hc2 (by decide),
          beq_of_isDigit hc2 (by decide), beq_of_isDigit hc2 (by decide),
          beq_of_isDigit hc2 (by decide), beq_of_isDigit hc2 (by decide), by simp [hc2]⟩
    obtain ⟨b1, b2, b3, b4, b5, b6, b7⟩ := hb
    rw [value]
    simp only [bind_of_success (peek!_eq h1), b1, b2, b3, b4, b5, b6, b7, Bool.false_eq_true,
      ↓reduceIte, bind_of_success hrun, bind_of_success (ws_eq hq (not_ws_follows ht)),
      pure_apply]
  | str u =>
    rw [Value.render, renderString_eq] at h ⊢
    have h1 : p.Splits t₁ (String.singleton '"' ++ (Lean.Json.escape u ++ ("\"" ++ t₂))) :=
      splits_of_toList h rfl (by simp)
    obtain ⟨q, hq, hrun⟩ := strCore_escape h1.next
    refine ⟨q, splits_of_toList hq (by simp) rfl, ?_⟩
    rw [value]
    simp only [bind_of_success (peek!_eq h1), Char.reduceBEq, Bool.false_eq_true, ↓reduceIte,
      bind_of_success (skip_eq h1), bind_of_success hrun,
      bind_of_success (ws_eq hq (not_ws_follows ht)), pure_apply]
  | arr xs hxs =>
    obtain ⟨l, rfl⟩ : ∃ l : List Value, xs = l.toArray := ⟨xs.toList, rfl⟩
    rw [Value.render] at h hlen ⊢
    simp only [Array.toList_map] at h hlen ⊢
    cases l with
    | nil =>
      simp only [List.map_nil, String.intercalate_nil] at h ⊢
      have h1 : p.Splits t₁ (String.singleton '[' ++ (String.singleton ']' ++ t₂)) :=
        splits_of_toList h rfl (by simp)
      refine ⟨_, splits_of_toList h1.next.next (by simp) rfl, ?_⟩
      rw [value]
      simp only [bind_of_success (peek!_eq h1), Char.reduceBEq, ↓reduceIte,
        bind_of_success (skip_eq h1),
        bind_of_success (ws_eq h1.next (head_singleton_append (by decide))),
        bind_of_success (nextIs_cons (· == ']') h1.next), bind_of_success (skip_eq h1.next),
        bind_of_success (ws_eq h1.next.next (not_ws_follows ht)), pure_apply]
    | cons x xs =>
      have h1 : p.Splits t₁ (String.singleton '[' ++
          ((",".intercalate ((x :: xs).map Value.render) ++ "]") ++ t₂)) :=
        splits_of_toList h rfl (by simp)
      obtain ⟨c, hc1, hc2⟩ := render_head (hxs x (by simp))
      have hhead : ((",".intercalate ((x :: xs).map Value.render) ++ "]") ++ t₂).toList.head? =
          some c := by
        rw [String.append_assoc, List.map_cons]
        exact head_intercalate hc1
      have hlen' : (",".intercalate ((x :: xs).map Value.render) ++ "]").length < n := by
        have : "[".length = 1 := rfl
        simp only [String.length_append] at hlen ⊢
        omega
      obtain ⟨q, hq, hrun⟩ := arrayItems_render (N := n) (fun m hm => ih m (by omega)) xs x #[] n
        le_rfl (fun y hy => hxs y (by simpa using hy)) hlen' h1.next ht
      refine ⟨q, splits_of_toList hq (by simp) rfl, ?_⟩
      rw [value]
      simp only [bind_of_success (peek!_eq h1), Char.reduceBEq, ↓reduceIte,
        bind_of_success (skip_eq h1),
        bind_of_success (ws_eq h1.next (head_of_eq hhead (start_facts hc2).1)),
        bind_of_success (nextIs_false (· == ']') h1.next (head_of_eq hhead (start_facts hc2).2)),
        Bool.false_eq_true, bind_of_success hrun, pure_apply, Array.empty_append]
  | obj fields hk hfs =>
    obtain ⟨l, rfl⟩ : ∃ l : List (String × Value), fields = l.toArray := ⟨fields.toList, rfl⟩
    rw [Value.render] at h hlen ⊢
    simp only [Array.toList_map] at h hlen ⊢
    cases l with
    | nil =>
      simp only [List.map_nil, String.intercalate_nil] at h ⊢
      have h1 : p.Splits t₁ (String.singleton '{' ++ (String.singleton '}' ++ t₂)) :=
        splits_of_toList h rfl (by simp)
      refine ⟨_, splits_of_toList h1.next.next (by simp) rfl, ?_⟩
      rw [value]
      simp only [bind_of_success (peek!_eq h1), Char.reduceBEq, Bool.false_eq_true, ↓reduceIte,
        bind_of_success (skip_eq h1),
        bind_of_success (ws_eq h1.next (head_singleton_append (by decide))),
        bind_of_success (nextIs_cons (· == '}') h1.next), bind_of_success (skip_eq h1.next),
        bind_of_success (ws_eq h1.next.next (not_ws_follows ht)), pure_apply]
    | cons f fs =>
      have h1 : p.Splits t₁ (String.singleton '{' ++
          ((",".intercalate ((f :: fs).map fun g =>
            Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}") ++ t₂)) :=
        splits_of_toList h rfl (by simp)
      have hhead : ((",".intercalate ((f :: fs).map fun g =>
          Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}") ++ t₂).toList.head? =
          some '"' := by
        rw [String.append_assoc, List.map_cons]
        exact head_intercalate (by simp [renderString_eq])
      have hlen' : (",".intercalate ((f :: fs).map fun g =>
          Lean.Json.renderString g.1 ++ ":" ++ g.2.render) ++ "}").length < n := by
        have : "{".length = 1 := rfl
        simp only [String.length_append] at hlen ⊢
        omega
      obtain ⟨q, hq, hrun⟩ := objectFields_render (N := n) (fun m hm => ih m (by omega)) fs f #[]
        n le_rfl (by simpa using hk)
        (fun g hg => hfs g (by simpa using hg)) hlen' h1.next ht
      refine ⟨q, splits_of_toList hq (by simp) rfl, ?_⟩
      rw [value]
      simp only [bind_of_success (peek!_eq h1), Char.reduceBEq, Bool.false_eq_true, ↓reduceIte,
        bind_of_success (skip_eq h1),
        bind_of_success (ws_eq h1.next (head_of_eq hhead (by decide))),
        bind_of_success (nextIs_false (· == '}') h1.next (head_of_eq hhead (by decide))),
        bind_of_success hrun, pure_apply, Array.empty_append]

end Values

/-- `parse v.render = .ok v` for every value `v` with `Value.Canonical v`: the keys of each object
are pairwise distinct and each decimal with significand `0` has exponent `-1` when negative and
`0` otherwise. -/
theorem parse_render {v : Value} (hv : v.Canonical) : parse v.render = .ok v := by
  have h0 : v.render.startPos.Splits "" (v.render ++ "") :=
    (String.splits_startPos v.render).of_eq rfl (by simp)
  obtain ⟨c, hc1, hc2⟩ := render_head hv
  obtain ⟨q, hq, hrun⟩ := readsBack (2 * v.render.utf8ByteSize + 2) v hv
    (by have := length_le_utf8ByteSize v.render; omega) h0 (fun c hc => by simp at hc)
  unfold parse Std.Internal.Parsec.String.Parser.run
  rw [bind_of_success (ws_eq h0 (head_of_eq (head_append hc1) (start_facts hc2).1)),
    bind_of_success hrun, bind_of_success (eof_eq hq)]
  rfl

end TorchLean.Certificate.DecimalJson
