/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.ODE.ArithParse
public import Batteries.Data.String.Lemmas
import all Init.Data.String.Slice
import all Init.Data.String.TakeDrop

/-!
# The language of `ArithParse.parse`

We prove four facts about `ArithParse.parse`: its fuel budget suffices, it reads the printed text of
every well-formed tree with a value back as that value, every value it gives is the value of some
well-formed tree, and whitespace between tokens does not change its result. The proofs read a cursor
at the byte offset of a prefix `l` of its source as the list of characters after `l`:
`TextCursor.peek_eq_head?`, `TextCursor.bump_eq` and `TextCursor.remainingFuel_eq` give `peek`,
`bump` and `remainingFuel` on that list, and `TextCursor.takeWhile_eq`, `TextCursor.skipWs_eq_self`,
`TextCursor.parseNat_repr` and `TextCursor.parseFloatBounds_eq` give the lexer's steps.

The main theorems, for every `g : Grammar E`:

* `parse_eq_of_le`: `parse g` gives the same result with every fuel of at least
  `5 * TextCursor.remainingFuel`, so its budget of `16 * TextCursor.remainingFuel + 32` never
  runs out. The `*_eq_of_le` theorems give the bound for each parsing function.
* `parse_print`: `Ast` holds the trees of the grammar documented in
  `TorchLeanExamples.Verification.ODE.ArithParse`, `Ast.eval` gives a tree's value in `g` and
  `Ast.print` its text with the fewest parentheses that precedence and left association need.
  `parse g t.print` is the value of every `Ast.WellFormed` tree `t` that has one.
* `exists_eval_eq_of_parse_eq_ok`: every value `parse g` gives is the value of a well-formed tree.
* `parse_append_whitespace`: inserting whitespace between two strings leaves the result of
  `parse g` unchanged when the last character of the first and the first character of the second
  are not both letters, digits, `_` or `.`.

A literal's value is `g.const lo hi` for the binary64 enclosure `(lo, hi)` that
`TextCursor.parseFloatBounds` reads from its text, so no theorem here depends on decimal rounding.
Exponents in well-formed trees have at most 256 digits, the most `TextCursor.parseNat` accepts.
-/

@[expose] public section

namespace TorchLean.Examples.TextCursor

open String (utf8Len)
open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

/-- A cursor that peeks `c` sits at the byte offset of a prefix `l` of its source, and `c` follows
that prefix. -/
theorem exists_of_peek_eq_some {cursor : TextCursor} {c : Char} (h : peek cursor = some c) :
    ∃ l r, cursor.source.toList = l ++ c :: r ∧ cursor.position = ⟨utf8Len l⟩ := by
  suffices key : ∀ (cs : List Char) (i : String.Pos.Raw),
      String.Pos.Raw.utf8GetAux? cs i cursor.position = some c →
      ∃ l r, cs = l ++ c :: r ∧ cursor.position.byteIdx = i.byteIdx + utf8Len l by
    obtain ⟨l, r, h₁, h₂⟩ := key cursor.source.toList 0 h
    exact ⟨l, r, h₁, String.Pos.Raw.ext (by simpa using h₂)⟩
  intro cs
  induction cs with
  | nil => intro i h; simp [String.Pos.Raw.utf8GetAux?] at h
  | cons d cs ih =>
    intro i h
    simp only [String.Pos.Raw.utf8GetAux?] at h
    split at h
    · next hi =>
      obtain rfl : d = c := Option.some.inj h
      exact ⟨[], cs, rfl, by simp [hi]⟩
    · obtain ⟨l, r, rfl, hl⟩ := ih _ h
      refine ⟨d :: l, r, rfl, ?_⟩
      simp only [String.Pos.Raw.byteIdx_add_char] at hl
      simp only [String.utf8Len_cons]
      omega

/-- Bumping a cursor past the character it peeks lowers its remaining fuel. -/
theorem remainingFuel_bump_lt {cursor : TextCursor} {c : Char} (h : peek cursor = some c) :
    remainingFuel (bump cursor) < remainingFuel cursor := by
  obtain ⟨l, r, hs, hp⟩ := exists_of_peek_eq_some h
  obtain ⟨s, p⟩ := cursor
  simp only at hs hp
  obtain rfl : s = String.ofList (l ++ c :: r) := by rw [← hs, String.ofList_toList]
  subst hp
  have := Char.utf8Size_pos c
  simp only [remainingFuel, bump, String.next_of_valid, String.rawEndPos_ofList,
    String.utf8Len_append, String.utf8Len_cons]
  omega

/-- The cursor after `takeWhile` has at most the remaining fuel of the cursor before it. -/
theorem remainingFuel_takeWhile_le (p : Char → Bool) (acc : String) (cursor : TextCursor) :
    remainingFuel (takeWhile p acc cursor).2 ≤ remainingFuel cursor := by
  suffices key : ∀ fuel acc cursor,
      remainingFuel (takeWhile.go p fuel acc cursor).2 ≤ remainingFuel cursor from key _ _ _
  intro fuel
  induction fuel with
  | zero => intro acc cursor; exact Nat.le_refl _
  | succ fuel ih =>
    intro acc cursor
    simp only [takeWhile.go]
    cases hpk : peek cursor with
    | none => exact Nat.le_refl _
    | some c =>
      simp only
      split
      · exact Nat.le_trans (ih _ _) (Nat.le_of_lt (remainingFuel_bump_lt hpk))
      · exact Nat.le_refl _

/-- Skipping whitespace never raises the remaining fuel. -/
theorem remainingFuel_skipWs_le (cursor : TextCursor) :
    remainingFuel (skipWs cursor) ≤ remainingFuel cursor :=
  remainingFuel_takeWhile_le _ _ _

/-- The cursor after a successful `parseNat` has at most the remaining fuel of the one before. -/
theorem remainingFuel_le_of_parseNat {cursor cursor' : TextCursor} {n : Nat}
    (h : parseNat cursor = .ok (n, cursor')) : remainingFuel cursor' ≤ remainingFuel cursor := by
  simp only [parseNat] at h
  split at h
  · cases h
    exact Nat.le_trans (remainingFuel_takeWhile_le _ _ _) (remainingFuel_skipWs_le _)
  · cases h

/-- What `parseFloatBounds` does after its scans: it checks the text
`sign ++ integerText ++ fractionText` with `Data.parseFloatString`, rounds the decimal with digits
`integerText ++ fractionDigits` down and up to binary64, and pairs the two bounds with `cursor`. -/
private def boundsOf (sign integerText fractionText fractionDigits : String)
    (cursor : TextCursor) : Except String ((Float × Float) × TextCursor) := do
  let _ ← Data.parseFloatString "number"
    (sign ++ integerText ++ fractionText)
  let numerator := Data.digitsToNat
    ((integerText ++ fractionDigits).toList.filterMap Data.digitVal?)
  let denominator := 10 ^ fractionDigits.length
  let bound (mode : Model.IEEERoundingMode) : Float :=
    ExecFloat.Binary.toFloat (ExecFloat.Binary.ofModel
      (Model.Internal.Directed.roundRat FloatFormat.binary64 mode (sign == "-") numerator
        denominator) : FloatLib.Floats.Binary64)
  let lo := bound .down
  let hi := bound .up
  if !(lo.isFinite && hi.isFinite) then
    .error "number outside the finite binary64 range"
  pure ((lo, hi), cursor)

private theorem boundsOf_cursor (sign integerText fractionText fractionDigits : String)
    (cursor cursor' : TextCursor) :
    boundsOf sign integerText fractionText fractionDigits cursor' =
      (boundsOf sign integerText fractionText fractionDigits cursor).map
        fun q => (q.1, cursor') := by
  unfold boundsOf
  cases Data.parseFloatString "number" (sign ++ integerText ++ fractionText) with
  | error e => rfl
  | ok x =>
    dsimp only
    split <;> rfl

/-- The cursor after a successful `parseFloatBounds` has at most the remaining fuel of the one
before. -/
theorem remainingFuel_le_of_parseFloatBounds {cursor cursor' : TextCursor} {b : Float × Float}
    (h : parseFloatBounds cursor = .ok (b, cursor')) :
    remainingFuel cursor' ≤ remainingFuel cursor := by
  have hs := remainingFuel_skipWs_le cursor
  simp only [parseFloatBounds, bind, Except.bind, pure, Except.pure] at h
  repeat' (split at h)
  all_goals try (cases h; done)
  all_goals obtain ⟨-, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
  all_goals dsimp only at *
  · have := remainingFuel_bump_lt (c := '.') ‹_›
    have := remainingFuel_bump_lt (c := '-') ‹_›
    have := remainingFuel_takeWhile_le (fun char => char.isDigit) "" (bump (skipWs cursor))
    have := remainingFuel_takeWhile_le (fun char => char.isDigit) ""
      (bump (takeWhile (fun char => char.isDigit) "" (bump (skipWs cursor))).snd)
    omega
  · have := remainingFuel_bump_lt (c := '-') ‹_›
    have := remainingFuel_takeWhile_le (fun char => char.isDigit) "" (bump (skipWs cursor))
    omega
  · have := remainingFuel_bump_lt (c := '.') ‹_›
    have := remainingFuel_takeWhile_le (fun char => char.isDigit) "" (skipWs cursor)
    have := remainingFuel_takeWhile_le (fun char => char.isDigit) ""
      (bump (takeWhile (fun char => char.isDigit) "" (skipWs cursor)).snd)
    omega
  · exact Nat.le_trans (remainingFuel_takeWhile_le _ _ _) hs

private theorem isWhitespace_eq_false_of_isDigit {c : Char} (h : c.isDigit = true) :
    c.isWhitespace = false := by
  cases hw : c.isWhitespace
  · rfl
  · simp only [Char.isWhitespace, Bool.or_eq_true, decide_eq_true_eq] at hw
    rcases hw with ((rfl | rfl) | rfl) | rfl <;> revert h <;> decide

private theorem trimAscii_toString_eq {s : String} (h : ∀ c ∈ s.toList, c.isWhitespace = false) :
    s.trimAscii.toString = s := by
  have h1 : s.toSlice.startsWith Char.isWhitespace = false := by
    rw [String.Slice.startsWith_bool_eq_head?, String.copy_toSlice]
    cases hs : s.toList with
    | nil => rfl
    | cons c cs => simp [h c (by simp [hs])]
  have h2 : s.toSlice.endsWith Char.isWhitespace = false := by
    rw [String.Slice.endsWith_bool_eq_getLast?, String.copy_toSlice]
    cases hs : s.toList.getLast? with
    | none => rfl
    | some c => simp [h c (List.mem_of_getLast? hs)]
  show ((s.toSlice.dropWhile Char.isWhitespace).dropEndWhile Char.isWhitespace).toString = s
  rw [String.Slice.dropWhile_eq_self h1, String.Slice.dropEndWhile_eq_self h2]
  exact String.copy_toSlice

private theorem takeDigits_go_of_isDigit (acc : List Nat) {cs : List Char}
    (h : ∀ c ∈ cs, c.isDigit = true) :
    Data.takeDigits.go acc cs =
      (acc.reverse ++ cs.map (fun c => c.toNat - '0'.toNat), []) := by
  induction cs generalizing acc with
  | nil => simp [Data.takeDigits.go]
  | cons c cs ih =>
    have hc := h c (by simp)
    simp only [Data.takeDigits.go, Data.digitVal?, hc,
      ite_true]
    rw [ih _ (fun d hd => h d (by simp [hd]))]
    simp

private theorem parseNatValue_repr {n : Nat} (hn : n < 10 ^ 256) :
    Data.NPY.parseNatValue n.repr = some n := by
  have hd : ∀ c ∈ n.repr.toList, c.isDigit = true := fun c hc =>
    Nat.isDigit_of_mem_toDigits (b := 10) (by decide) (by decide) (by rwa [Nat.toList_repr] at hc)
  have htrim : n.repr.trimAscii.toString = n.repr :=
    trimAscii_toString_eq fun c hc => isWhitespace_eq_false_of_isDigit (hd c hc)
  have hlen : n.repr.length ≤ 256 := (Nat.length_repr_le_iff (by decide)).2 hn
  have hne : n.repr.toList ≠ [] := by rw [Nat.toList_repr]; exact Nat.toDigits_ne_nil
  simp only [Data.NPY.parseNatValue, htrim,
    Data.maxNumericCellChars, Data.takeDigits,
    takeDigits_go_of_isDigit [] hd]
  rw [ite_eq_right (by simp; omega)]
  simp only [List.reverse_nil, List.nil_append, List.isEmpty_nil, List.isEmpty_map,
    Bool.not_eq_true', Bool.true_and]
  rw [ite_eq_left (by simp)]
  congr 1
  have key : Data.digitsToNat
      (n.repr.toList.map fun c => c.toNat - '0'.toNat) = Nat.ofDigitChars 10 n.repr.toList 0 := by
    simp only [Data.digitsToNat, List.foldl_map, Nat.ofDigitChars_eq_foldl]
    congr
    funext a c
    omega
  rw [key, Nat.toList_repr, Nat.ofDigitChars_ten_toDigits]

/-- At the byte offset of a prefix `l` of its source, a cursor peeks the head of the rest. -/
theorem peek_eq_head? {s : String} {l r : List Char} (h : s.toList = l ++ r) :
    peek ⟨s, ⟨utf8Len l⟩⟩ = r.head? := by
  obtain rfl : s = String.ofList (l ++ r) := by rw [← h, String.ofList_toList]
  exact String.get?_of_valid l r

/-- Bumping a cursor at the byte offset of `l` moves it past the next character `c`. -/
theorem bump_eq {s : String} {l r : List Char} {c : Char} (h : s.toList = l ++ c :: r) :
    bump ⟨s, ⟨utf8Len l⟩⟩ = ⟨s, ⟨utf8Len (l ++ [c])⟩⟩ := by
  obtain rfl : s = String.ofList (l ++ c :: r) := by rw [← h, String.ofList_toList]
  simp only [bump, String.next_of_valid, String.utf8Len_append, String.utf8Len_cons,
    String.utf8Len_nil, Nat.zero_add]

/-- The remaining fuel at the byte offset of `l` is one more than the UTF-8 size of the rest. -/
theorem remainingFuel_eq {s : String} {l r : List Char} (h : s.toList = l ++ r) :
    remainingFuel ⟨s, ⟨utf8Len l⟩⟩ = utf8Len r + 1 := by
  obtain rfl : s = String.ofList (l ++ r) := by rw [← h, String.ofList_toList]
  simp only [remainingFuel, String.rawEndPos_ofList, String.utf8Len_append]
  omega

private theorem length_le_utf8Len (w : List Char) : w.length ≤ utf8Len w := by
  induction w with
  | nil => simp
  | cons c w ih =>
    simp only [List.length_cons, String.utf8Len_cons]
    have := Char.utf8Size_pos c
    omega

/-- `takeWhile` consumes a run `w` of characters satisfying `p` that the rest does not continue. -/
theorem takeWhile_eq (p : Char → Bool) (acc : String) {s : String} {l w r : List Char}
    (h : s.toList = l ++ w ++ r) (hw : ∀ c ∈ w, p c = true) (hr : ∀ c ∈ r.head?, p c = false) :
    takeWhile p acc ⟨s, ⟨utf8Len l⟩⟩ = (acc ++ String.ofList w, ⟨s, ⟨utf8Len (l ++ w)⟩⟩) := by
  suffices key : ∀ w fuel acc l, s.toList = l ++ w ++ r → (∀ c ∈ w, p c = true) →
      w.length < fuel → takeWhile.go p fuel acc ⟨s, ⟨utf8Len l⟩⟩ =
        (acc ++ String.ofList w, ⟨s, ⟨utf8Len (l ++ w)⟩⟩) by
    refine key w _ acc l h hw ?_
    rw [remainingFuel_eq (r := w ++ r) (by simpa using h), String.utf8Len_append]
    have := length_le_utf8Len w
    omega
  intro w
  induction w with
  | nil =>
    intro fuel acc l h _ hf
    obtain ⟨k, rfl⟩ : ∃ k, fuel = k + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    rw [takeWhile.go, peek_eq_head? (r := r) (by simpa using h)]
    cases hrh : r.head? with
    | none => simp
    | some c => simp [hr c (by simp [hrh])]
  | cons c w ih =>
    intro fuel acc l h hw hf
    obtain ⟨k, rfl⟩ : ∃ k, fuel = k + 1 := ⟨fuel - 1, by simp at hf; omega⟩
    rw [takeWhile.go, peek_eq_head? (r := c :: (w ++ r)) (by simpa using h)]
    simp only [List.head?_cons, hw c (by simp), ite_true]
    rw [bump_eq (r := w ++ r) (by simpa using h),
      ih k (acc.push c) (l ++ [c]) (by simpa using h) (fun d hd => hw d (by simp [hd]))
        (by simp at hf; omega)]
    simp only [List.append_assoc, List.singleton_append]
    congr 1
    apply String.toList_inj.mp
    simp

/-- `skipWs` leaves a cursor in place when the rest does not start with whitespace. -/
theorem skipWs_eq_self {s : String} {l r : List Char} (h : s.toList = l ++ r)
    (hr : ∀ c ∈ r.head?, c.isWhitespace = false) : skipWs ⟨s, ⟨utf8Len l⟩⟩ = ⟨s, ⟨utf8Len l⟩⟩ := by
  have := takeWhile_eq Char.isWhitespace "" (w := []) (by simpa using h) (by simp) hr
  simp only [skipWs, this, List.append_nil]

/-- `parseNat` reads the decimal digits of `n < 10 ^ 256` back as `n`. -/
theorem parseNat_repr {s : String} {l r : List Char} {n : Nat}
    (h : s.toList = l ++ n.repr.toList ++ r) (hn : n < 10 ^ 256)
    (hr : ∀ c ∈ r.head?, c.isDigit = false) :
    parseNat ⟨s, ⟨utf8Len l⟩⟩ = .ok (n, ⟨s, ⟨utf8Len (l ++ n.repr.toList)⟩⟩) := by
  have hd : ∀ c ∈ n.repr.toList, c.isDigit = true := fun c hc =>
    Nat.isDigit_of_mem_toDigits (b := 10) (by decide) (by decide) (by rwa [Nat.toList_repr] at hc)
  obtain ⟨d, ds, hds⟩ := List.exists_cons_of_ne_nil
    (show n.repr.toList ≠ [] by rw [Nat.toList_repr]; exact Nat.toDigits_ne_nil)
  have h1 : skipWs ⟨s, ⟨utf8Len l⟩⟩ = ⟨s, ⟨utf8Len l⟩⟩ := by
    refine skipWs_eq_self (r := n.repr.toList ++ r) (by simpa using h) ?_
    intro c hc
    rw [hds, List.cons_append, List.head?_cons, Option.mem_some_iff] at hc
    subst hc
    exact isWhitespace_eq_false_of_isDigit (hd _ (by simp [hds]))
  simp only [parseNat, h1, takeWhile_eq Char.isDigit "" h hd hr, String.empty_append,
    String.ofList_toList, parseNatValue_repr hn]
  rfl

private theorem parseFloatBounds_lit {s : String} {l i F r : List Char}
    (h : s.toList = l ++ (i ++ F) ++ r) (hi : i ≠ []) (hid : ∀ c ∈ i, c.isDigit = true)
    (hF : F = [] ∨ ∃ f, F = '.' :: f ∧ ∀ c ∈ f, c.isDigit = true)
    (hr : ∀ c ∈ r.head?, c.isDigit = false ∧ c ≠ '.') :
    parseFloatBounds ⟨s, ⟨utf8Len l⟩⟩ = boundsOf "" (String.ofList i) (String.ofList F)
      (String.ofList F.tail) ⟨s, ⟨utf8Len (l ++ (i ++ F))⟩⟩ := by
  rw [← List.append_assoc] at h ⊢
  obtain ⟨d, i', rfl⟩ := List.exists_cons_of_ne_nil hi
  have hd := hid d (by simp)
  have h1 : skipWs ⟨s, ⟨utf8Len l⟩⟩ = ⟨s, ⟨utf8Len l⟩⟩ :=
    skipWs_eq_self (r := d :: i' ++ F ++ r) (by simp [h])
      (by simp [isWhitespace_eq_false_of_isDigit hd])
  have h2 : peek ⟨s, ⟨utf8Len l⟩⟩ = some d := by
    rw [peek_eq_head? (r := d :: i' ++ F ++ r) (by simp [h])]
    rfl
  have hfr : ∀ c ∈ (F ++ r).head?, c.isDigit = false := by
    rcases hF with rfl | ⟨f', rfl, -⟩
    · intro c hc; exact (hr c (by simpa using hc)).1
    · intro c hc; simp at hc; subst hc; decide
  have h3 := takeWhile_eq (fun char => char.isDigit) "" (s := s) (l := l) (w := d :: i')
    (r := F ++ r) (by simp [h]) hid hfr
  unfold parseFloatBounds
  simp only [bind, Except.bind, pure, Except.pure, h1, h2]
  split
  · rename_i heq
    simp only [Option.some.injEq] at heq
    subst heq
    exact absurd hd (by decide)
  · have hne : String.ofList (d :: i') ≠ "" := fun he => by
      have := congrArg String.toList he
      simp at this
    simp only [h3, String.empty_append, hne, ite_false]
    rw [peek_eq_head? (r := F ++ r) (by simp [h])]
    rcases hF with rfl | ⟨f', rfl, hf'⟩
    · simp only [List.nil_append]
      cases hrh : r.head? with
      | none =>
        simp only [boundsOf, bind, Except.bind, pure, Except.pure, String.ofList_nil,
          List.tail_nil, List.append_nil, String.empty_append, String.append_empty]
      | some c =>
        have hc : c ≠ '.' := (hr c (by simp [hrh])).2
        simp only [hc, Option.some.injEq, imp_false, not_false_eq_true, boundsOf, bind,
          Except.bind, pure, Except.pure, String.ofList_nil, List.tail_nil, List.append_nil,
          String.empty_append, String.append_empty]
    · simp only [List.cons_append, List.head?_cons]
      rw [bump_eq (c := '.') (r := f' ++ r) (by simp [h]),
        takeWhile_eq (fun char => char.isDigit) "" (l := l ++ d :: i' ++ ['.']) (w := f') (r := r)
          (by simp [h]) hf' (fun c hc => (hr c hc).1)]
      have hstr : "." ++ String.ofList f' = String.ofList ('.' :: f') :=
        String.toList_inj.mp (by simp)
      simp only [boundsOf, bind, Except.bind, pure, Except.pure, String.empty_append, hstr,
        List.tail_cons, List.append_assoc, List.cons_append, List.nil_append]

/-- `parseFloatBounds` reads a nonempty run of digits `i`, alone or followed by `.` and the digits
`f`, that no digit or `.` follows, as the enclosure it reads from that text alone. -/
theorem parseFloatBounds_eq {s : String} {l X i f r : List Char} {b : Float × Float}
    {cursor : TextCursor} (h : s.toList = l ++ X ++ r) (hi : i ≠ [])
    (hd : ∀ c ∈ i ++ f, c.isDigit = true) (hX : X = i ∨ X = i ++ '.' :: f)
    (hr : ∀ c ∈ r.head?, c.isDigit = false ∧ c ≠ '.')
    (hx : parseFloatBounds { source := String.ofList X } = .ok (b, cursor)) :
    parseFloatBounds ⟨s, ⟨utf8Len l⟩⟩ = .ok (b, ⟨s, ⟨utf8Len (l ++ X)⟩⟩) := by
  obtain ⟨F, rfl, hF⟩ : ∃ F, X = i ++ F ∧
      (F = [] ∨ ∃ f', F = '.' :: f' ∧ ∀ c ∈ f', c.isDigit = true) := by
    rcases hX with rfl | rfl
    · exact ⟨[], by simp, Or.inl rfl⟩
    · exact ⟨'.' :: f, rfl, Or.inr ⟨f, rfl, fun c hc => hd c (by simp [hc])⟩⟩
  have hid : ∀ c ∈ i, c.isDigit = true := fun c hc => hd c (by simp [hc])
  have hst : ({ source := String.ofList (i ++ F) } : TextCursor) =
      ⟨String.ofList (i ++ F), ⟨utf8Len []⟩⟩ := rfl
  rw [hst, parseFloatBounds_lit (s := String.ofList (i ++ F)) (l := []) (r := []) (by simp) hi
    hid hF (by simp)] at hx
  rw [parseFloatBounds_lit h hi hid hF hr,
    boundsOf_cursor _ _ _ _ ⟨String.ofList (i ++ F), ⟨utf8Len ([] ++ (i ++ F))⟩⟩, hx]
  rfl

private theorem isWhitespace_eq_false_of_isAlpha {c : Char} (h : c.isAlpha = true) :
    c.isWhitespace = false := by
  cases hw : c.isWhitespace
  · rfl
  · simp only [Char.isWhitespace, Bool.or_eq_true, decide_eq_true_eq] at hw
    rcases hw with ((rfl | rfl) | rfl) | rfl <;> revert h <;> decide

private theorem isDigit_eq_false_of_isAlpha {c : Char} (h : c.isAlpha = true) :
    c.isDigit = false := by
  cases hd : c.isDigit
  · rfl
  · have h1 := Char.isDigit_iff_toNat.mp hd
    simp only [Char.isAlpha, Char.isUpper, Char.isLower, Bool.or_eq_true, Bool.and_eq_true,
      decide_eq_true_eq, ge_iff_le, UInt32.le_iff_toNat_le, Char.toNat_val] at h
    have h0 : '0'.toNat = 48 := rfl
    have h9 : '9'.toNat = 57 := rfl
    have hA : 'A'.toNat = 65 := rfl
    have hZ : 'Z'.toNat = 90 := rfl
    have ha : 'a'.toNat = 97 := rfl
    have hz : 'z'.toNat = 122 := rfl
    omega

private theorem forall_mem_head?_cons {p : Char → Prop} {c : Char} {r : List Char} (h : p c) :
    ∀ d ∈ (c :: r).head?, p d := by
  intro d hd
  simp only [List.head?_cons, Option.mem_def, Option.some.injEq] at hd
  exact hd ▸ h

private theorem skipWs_eq_of_peek {st : TextCursor} {c : Char} (h : peek st = some c)
    (hc : c.isWhitespace = false) : skipWs st = st := by
  simp only [skipWs, takeWhile, remainingFuel, takeWhile.go, h, hc, Bool.false_eq_true, ite_false]

private theorem takeWhile_go_fst (p : Char → Bool) (fuel : Nat) (acc : String) (st : TextCursor) :
    ∃ w : List Char, (takeWhile.go p fuel acc st).1 = acc ++ String.ofList w ∧
      ∀ d ∈ w, p d = true := by
  induction fuel generalizing acc st with
  | zero => exact ⟨[], by simp [takeWhile.go], by simp⟩
  | succ fuel ih =>
    simp only [takeWhile.go]
    cases hpk : peek st with
    | none => exact ⟨[], by simp, by simp⟩
    | some c =>
      simp only
      split
      · rename_i hc
        obtain ⟨w, hw, hp⟩ := ih (acc.push c) (bump st)
        refine ⟨c :: w, ?_, ?_⟩
        · rw [hw]
          apply String.toList_inj.mp
          simp
        · intro d hd
          simp only [List.mem_cons] at hd
          rcases hd with rfl | hd
          exacts [hc, hp d hd]
      · exact ⟨[], by simp, by simp⟩

private theorem takeWhile_fst (p : Char → Bool) (acc : String) (st : TextCursor) :
    ∃ w : List Char, (takeWhile p acc st).1 = acc ++ String.ofList w ∧ ∀ d ∈ w, p d = true :=
  takeWhile_go_fst p _ acc st

private theorem takeWhile_fst_of_peek (p : Char → Bool) {st : TextCursor} {c : Char}
    (h : peek st = some c) (hc : p c = true) :
    ∃ w : List Char, (takeWhile p "" st).1.toList = c :: w ∧ ∀ d ∈ w, p d = true := by
  simp only [takeWhile, remainingFuel, takeWhile.go, h, hc, ite_true]
  obtain ⟨w, hw, hp⟩ := takeWhile_go_fst p _ ("".push c) (bump st)
  exact ⟨w, by rw [hw]; simp, hp⟩

private theorem takeDigits_go_lt (acc : List Nat) (hacc : ∀ d ∈ acc, d < 10) (cs : List Char) :
    (Data.takeDigits.go acc cs).1.length ≤ acc.length + cs.length ∧
      ∀ d ∈ (Data.takeDigits.go acc cs).1, d < 10 := by
  induction cs generalizing acc with
  | nil =>
    simp only [Data.takeDigits.go, List.length_reverse, List.length_nil,
      Nat.add_zero, Nat.le_refl, List.mem_reverse, true_and]
    exact hacc
  | cons c cs ih =>
    simp only [Data.takeDigits.go]
    split
    · rename_i d hd
      simp only [Data.digitVal?] at hd
      split at hd
      · rename_i hc
        obtain rfl := Option.some.inj hd
        have h1 := Char.isDigit_iff_toNat.mp hc
        have h0 : '0'.toNat = 48 := rfl
        have h9 : '9'.toNat = 57 := rfl
        obtain ⟨ih1, ih2⟩ := ih ((c.toNat - '0'.toNat) :: acc) (by
          intro d hd
          simp only [List.mem_cons] at hd
          rcases hd with rfl | hd
          · omega
          · exact hacc d hd)
        simp only [List.length_cons] at ih1 ⊢
        exact ⟨by omega, ih2⟩
      · cases hd
    · simp only [List.length_reverse, List.length_cons, List.mem_reverse]
      exact ⟨by omega, hacc⟩

private theorem foldl_digits_lt (ds : List Nat) (hds : ∀ d ∈ ds, d < 10) (a : Nat) :
    ds.foldl (fun acc d => acc * 10 + d) a < (a + 1) * 10 ^ ds.length := by
  induction ds generalizing a with
  | nil => simp
  | cons d ds ih =>
    simp only [List.foldl_cons, List.length_cons]
    have hd := hds d (by simp)
    have := ih (fun d hd => hds d (by simp [hd])) (a * 10 + d)
    calc _ < (a * 10 + d + 1) * 10 ^ ds.length := this
      _ ≤ ((a + 1) * 10) * 10 ^ ds.length := Nat.mul_le_mul_right _ (by omega)
      _ = (a + 1) * 10 ^ (ds.length + 1) := by rw [Nat.pow_succ, Nat.mul_assoc, Nat.mul_comm 10]

private theorem lt_of_parseNatValue {s : String} {n : Nat}
    (h : Data.NPY.parseNatValue s = some n) : n < 10 ^ 256 := by
  simp only [Data.NPY.parseNatValue,
    Data.maxNumericCellChars, Data.takeDigits] at h
  obtain ⟨h1, h2⟩ := takeDigits_go_lt [] (by simp) s.trimAscii.toString.toList
  rcases hq : Data.takeDigits.go [] s.trimAscii.toString.toList with
    ⟨ds, rest⟩
  simp only [hq, List.length_nil, Nat.zero_add, String.length_toList] at h h1 h2
  split at h
  · cases h
  · rename_i hlen
    simp only [Bool.or_eq_true, List.isEmpty_iff, gt_iff_lt, decide_eq_true_eq, not_or,
      Nat.not_lt] at hlen
    cases hc : (rest.isEmpty && !ds.isEmpty)
    · simp [hc] at h
    · simp only [hc, ite_true, Option.some.injEq] at h
      subst h
      have := foldl_digits_lt ds h2 0
      simp only [Nat.zero_add, Nat.one_mul] at this
      exact Nat.lt_of_lt_of_le this (Nat.pow_le_pow_right (by decide) (by omega))

private theorem lt_of_parseNat {st st' : TextCursor} {n : Nat} (h : parseNat st = .ok (n, st')) :
    n < 10 ^ 256 := by
  simp only [parseNat] at h
  split at h
  · rename_i v hv
    cases h
    exact lt_of_parseNatValue hv
  · cases h

private theorem exists_of_boundsOf {I u v : String} {i F : List Char}
    {cursor cursor' : TextCursor} {b : Float × Float}
    (h : boundsOf "" I u v cursor = .ok (b, cursor'))
    (hI : I.toList = i) (hu : u.toList = F) (hv : v.toList = F.tail) (hi : i ≠ [])
    (hid : ∀ c ∈ i, c.isDigit = true)
    (hF : F = [] ∨ ∃ f, F = '.' :: f ∧ ∀ c ∈ f, c.isDigit = true) :
    ∃ cursor'', parseFloatBounds { source := String.ofList (i ++ F) } = .ok (b, cursor'') := by
  obtain rfl : I = String.ofList i := by rw [← hI, String.ofList_toList]
  obtain rfl : u = String.ofList F := by rw [← hu, String.ofList_toList]
  obtain rfl : v = String.ofList F.tail := by rw [← hv, String.ofList_toList]
  refine ⟨⟨String.ofList (i ++ F), ⟨utf8Len ([] ++ (i ++ F))⟩⟩, ?_⟩
  have hst : ({ source := String.ofList (i ++ F) } : TextCursor) =
      ⟨String.ofList (i ++ F), ⟨utf8Len []⟩⟩ := rfl
  rw [hst, parseFloatBounds_lit (s := String.ofList (i ++ F)) (l := []) (r := []) (by simp) hi
    hid hF (by simp), boundsOf_cursor _ _ _ _ cursor, h]
  rfl

private theorem exists_of_parseFloatBounds {st st' : TextCursor} {c : Char} {b : Float × Float}
    (hpk : peek st = some c) (hc : (c.isDigit || c = '.') = true)
    (h : parseFloatBounds st = .ok (b, st')) :
    ∃ text : String, (∃ i f : List Char, i ≠ [] ∧ (∀ d ∈ i ++ f, d.isDigit = true) ∧
        (text.toList = i ∨ text.toList = i ++ '.' :: f)) ∧
      ∃ cursor, parseFloatBounds { source := text } = .ok (b, cursor) := by
  have hws : c.isWhitespace = false := by
    simp only [Bool.or_eq_true, decide_eq_true_eq] at hc
    rcases hc with hc | rfl
    · exact isWhitespace_eq_false_of_isDigit hc
    · decide
  have hneg : c ≠ '-' := by
    rintro rfl
    revert hc
    decide
  simp only [parseFloatBounds, bind, Except.bind, pure, Except.pure,
    skipWs_eq_of_peek hpk hws] at h
  split at h
  · rename_i heq
    rw [hpk, Option.some.injEq] at heq
    exact absurd heq hneg
  dsimp only at h
  obtain ⟨w, hw, hwd⟩ := takeWhile_fst (fun char => char.isDigit) "" st
  split at h
  · cases h
  rename_i hne
  have hw0 : w ≠ [] := by
    rintro rfl
    exact hne (by simpa using hw)
  by_cases hdot : peek (takeWhile (fun char => char.isDigit) "" st).2 = some '.'
  · obtain ⟨f, hf, hfd⟩ := takeWhile_fst (fun char => char.isDigit) ""
      (bump (takeWhile (fun char => char.isDigit) "" st).2)
    have hstr : "." ++ String.ofList f = String.ofList ('.' :: f) :=
      String.toList_inj.mp (by simp)
    simp only [hdot, hw, hf, String.empty_append, hstr] at h
    refine ⟨String.ofList (w ++ '.' :: f), ⟨w, f, hw0, ?_, Or.inr (by simp)⟩, ?_⟩
    · intro d hd
      rcases List.mem_append.mp hd with hd | hd
      exacts [hwd d hd, hfd d hd]
    · exact exists_of_boundsOf (I := String.ofList w) (u := String.ofList ('.' :: f))
        (v := String.ofList f)
        (by simp only [boundsOf, bind, Except.bind, pure, Except.pure, String.empty_append]
            exact h)
        (by simp) (by simp) (by simp) hw0 hwd (Or.inr ⟨f, rfl, hfd⟩)
  · simp only [hw, String.empty_append, String.append_empty] at h
    refine ⟨String.ofList w, ⟨w, [], hw0, by simpa using hwd, Or.inl (by simp)⟩, ?_⟩
    have := exists_of_boundsOf (I := String.ofList w) (u := "") (v := "") (F := [])
      (by simp only [boundsOf, bind, Except.bind, pure, Except.pure, String.empty_append,
            String.append_empty]
          exact h)
      (by simp) (by simp) (by simp) hw0 hwd (Or.inl rfl)
    simpa only [List.append_nil] using this

end TorchLean.Examples.TextCursor

namespace TorchLean.Examples.ArithParse

section

open TextCursor (remainingFuel peek bump skipWs)

variable {E : Type}

private theorem bind_eq_ok {ε α β : Type} {m : Except ε α} {k : α → Except ε β} {b : β}
    (h : m >>= k = .ok b) : ∃ a, m = .ok a ∧ k a = .ok b := by
  cases m with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

private theorem bind_congr_ok {ε α β : Type} {m : Except ε α} {k₁ k₂ : α → Except ε β}
    (h : ∀ a, m = .ok a → k₁ a = k₂ a) : m >>= k₁ = m >>= k₂ := by
  cases m with
  | error e => rfl
  | ok a => exact h a rfl

/-- Each parsing function returns a cursor with at most the remaining fuel of its input. -/
private theorem remainingFuel_le_of_ok (g : Grammar E) (fuel : Nat) :
    (∀ st e st', primaryFuel g fuel st = .ok (e, st') → remainingFuel st' ≤ remainingFuel st) ∧
    (∀ st e st', factorFuel g fuel st = .ok (e, st') → remainingFuel st' ≤ remainingFuel st) ∧
    (∀ st e st', unaryFuel g fuel st = .ok (e, st') → remainingFuel st' ≤ remainingFuel st) ∧
    (∀ acc st e st', termFuel.loop g fuel acc st = .ok (e, st') →
      remainingFuel st' ≤ remainingFuel st) ∧
    (∀ st e st', termFuel g fuel st = .ok (e, st') → remainingFuel st' ≤ remainingFuel st) ∧
    (∀ acc st e st', exprFuel.loop g fuel acc st = .ok (e, st') →
      remainingFuel st' ≤ remainingFuel st) ∧
    (∀ st e st', exprFuel g fuel st = .ok (e, st') → remainingFuel st' ≤ remainingFuel st) := by
  induction fuel with
  | zero =>
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro st e st' h; rw [primaryFuel.eq_1] at h; cases h
    · intro st e st' h; rw [factorFuel.eq_1] at h; cases h
    · intro st e st' h; rw [unaryFuel.eq_1] at h; cases h
    · intro acc st e st' h; rw [termFuel.loop.eq_1] at h; cases h; exact Nat.le_refl _
    · intro st e st' h; rw [termFuel.eq_1] at h; cases h
    · intro acc st e st' h; rw [exprFuel.loop.eq_1] at h; cases h; exact Nat.le_refl _
    · intro st e st' h; rw [exprFuel.eq_1] at h; cases h
  | succ k ih =>
    obtain ⟨ihP, ihFa, ihU, ihLT, ihT, ihLE, ihE⟩ := ih
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro st e st' h
      rw [primaryFuel.eq_2] at h
      have hs := TextCursor.remainingFuel_skipWs_le st
      split at h
      · obtain ⟨⟨e₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
        have h₁' := ihE _ _ _ h₁
        have hb := TextCursor.remainingFuel_bump_lt (c := '(') ‹_›
        dsimp only at h
        split at h
        · cases h
          have := TextCursor.remainingFuel_bump_lt (c := ')') ‹_›
          have := TextCursor.remainingFuel_skipWs_le st₁
          omega
        · cases h
      · split at h
        · obtain ⟨⟨⟨x, y⟩, st₁⟩, h₁, h⟩ := bind_eq_ok h
          cases h
          have := TextCursor.remainingFuel_le_of_parseFloatBounds h₁
          omega
        · split at h
          · split at h
            rename_i id st₁ heq
            have ht : remainingFuel st₁ ≤ remainingFuel (skipWs st) := by
              have := congrArg (fun q => remainingFuel q.2) heq
              simp only at this
              rw [← this]
              exact TextCursor.remainingFuel_takeWhile_le _ _ _
            have hs₁ := TextCursor.remainingFuel_skipWs_le st₁
            dsimp only at h
            split at h
            · obtain ⟨⟨a, st₂⟩, h₂, h⟩ := bind_eq_ok h
              have := ihE _ _ _ h₂
              have := TextCursor.remainingFuel_bump_lt (c := '(') ‹_›
              dsimp only at h
              split at h
              · split at h
                · cases h
                  have := TextCursor.remainingFuel_bump_lt (c := ')') ‹_›
                  have := TextCursor.remainingFuel_skipWs_le st₂
                  omega
                · cases h
              · cases h
            · split at h
              · cases h; omega
              · cases h
          · cases h
      · cases h
    · intro st e st' h
      rw [factorFuel.eq_2] at h
      obtain ⟨⟨b, st₁⟩, h₁, h⟩ := bind_eq_ok h
      have := ihP _ _ _ h₁
      have := TextCursor.remainingFuel_skipWs_le st₁
      dsimp only at h
      split at h
      · obtain ⟨⟨m, st₂⟩, h₂, h⟩ := bind_eq_ok h
        dsimp only at h
        cases h
        have := TextCursor.remainingFuel_le_of_parseNat h₂
        have := TextCursor.remainingFuel_bump_lt (c := '^') ‹_›
        omega
      · cases h; omega
    · intro st e st' h
      rw [unaryFuel.eq_2] at h
      have hs := TextCursor.remainingFuel_skipWs_le st
      split at h
      · obtain ⟨⟨e₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
        cases h
        have := ihU _ _ _ h₁
        have := TextCursor.remainingFuel_bump_lt (c := '-') ‹_›
        omega
      · have := ihFa _ _ _ h; omega
    · intro acc st e st' h
      rw [termFuel.loop.eq_2] at h
      have hs := TextCursor.remainingFuel_skipWs_le st
      split at h
      · obtain ⟨⟨f₂, st₂⟩, h₁, h⟩ := bind_eq_ok h
        have := ihU _ _ _ h₁
        have := ihLT _ _ _ _ h
        have := TextCursor.remainingFuel_bump_lt (c := '*') ‹_›
        omega
      · obtain ⟨⟨f₂, st₂⟩, h₁, h⟩ := bind_eq_ok h
        have := ihU _ _ _ h₁
        have := ihLT _ _ _ _ h
        have := TextCursor.remainingFuel_bump_lt (c := '/') ‹_›
        omega
      · cases h; omega
    · intro st e st' h
      rw [termFuel.eq_2] at h
      obtain ⟨⟨f₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
      have := ihU _ _ _ h₁
      have := ihLT _ _ _ _ h
      omega
    · intro acc st e st' h
      rw [exprFuel.loop.eq_2] at h
      have hs := TextCursor.remainingFuel_skipWs_le st
      split at h
      · obtain ⟨⟨t₂, st₂⟩, h₁, h⟩ := bind_eq_ok h
        have := ihT _ _ _ h₁
        have := ihLE _ _ _ _ h
        have := TextCursor.remainingFuel_bump_lt (c := '+') ‹_›
        omega
      · obtain ⟨⟨t₂, st₂⟩, h₁, h⟩ := bind_eq_ok h
        have := ihT _ _ _ h₁
        have := ihLE _ _ _ _ h
        have := TextCursor.remainingFuel_bump_lt (c := '-') ‹_›
        omega
      · cases h; omega
    · intro st e st' h
      rw [exprFuel.eq_2] at h
      obtain ⟨⟨t₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
      have := ihT _ _ _ h₁
      have := ihLE _ _ _ _ h
      omega

/-- Above the bounds below, the fuel given to a parsing function does not change its result. -/
private theorem eq_of_le_fuel (g : Grammar E) (n : Nat) : ∀ st : TextCursor, remainingFuel st ≤ n →
    (∀ f₁ f₂, 5 * remainingFuel st ≤ f₁ + 4 → 5 * remainingFuel st ≤ f₂ + 4 →
      primaryFuel g f₁ st = primaryFuel g f₂ st) ∧
    (∀ f₁ f₂, 5 * remainingFuel st ≤ f₁ + 3 → 5 * remainingFuel st ≤ f₂ + 3 →
      factorFuel g f₁ st = factorFuel g f₂ st) ∧
    (∀ f₁ f₂, 5 * remainingFuel st ≤ f₁ + 2 → 5 * remainingFuel st ≤ f₂ + 2 →
      unaryFuel g f₁ st = unaryFuel g f₂ st) ∧
    (∀ f₁ f₂ acc, 5 * remainingFuel st ≤ f₁ + 4 → 5 * remainingFuel st ≤ f₂ + 4 →
      termFuel.loop g f₁ acc st = termFuel.loop g f₂ acc st) ∧
    (∀ f₁ f₂, 5 * remainingFuel st ≤ f₁ + 1 → 5 * remainingFuel st ≤ f₂ + 1 →
      termFuel g f₁ st = termFuel g f₂ st) ∧
    (∀ f₁ f₂ acc, 5 * remainingFuel st ≤ f₁ + 4 → 5 * remainingFuel st ≤ f₂ + 4 →
      exprFuel.loop g f₁ acc st = exprFuel.loop g f₂ acc st) ∧
    (∀ f₁ f₂, 5 * remainingFuel st ≤ f₁ → 5 * remainingFuel st ≤ f₂ →
      exprFuel g f₁ st = exprFuel g f₂ st) := by
  have hpos : ∀ st : TextCursor, 1 ≤ remainingFuel st := fun _ => Nat.le_add_left _ _
  have hM := remainingFuel_le_of_ok g
  induction n with
  | zero => intro st hst; have := hpos st; omega
  | succ n ih =>
    have hP : ∀ st : TextCursor, remainingFuel st ≤ n + 1 → ∀ f₁ f₂,
        5 * remainingFuel st ≤ f₁ + 4 → 5 * remainingFuel st ≤ f₂ + 4 →
        primaryFuel g f₁ st = primaryFuel g f₂ st := by
      intro st hst f₁ f₂ h₁ h₂
      have := hpos st
      obtain ⟨k₁, rfl⟩ : ∃ k, f₁ = k + 1 := ⟨f₁ - 1, by omega⟩
      obtain ⟨k₂, rfl⟩ : ∃ k, f₂ = k + 1 := ⟨f₂ - 1, by omega⟩
      have hs := TextCursor.remainingFuel_skipWs_le st
      rw [primaryFuel.eq_2, primaryFuel.eq_2]
      split
      · have hb := TextCursor.remainingFuel_bump_lt (c := '(') ‹_›
        rw [(ih (bump (skipWs st)) (by omega)).2.2.2.2.2.2 k₁ k₂ (by omega) (by omega)]
      · split
        · rfl
        · split
          · split
            rename_i id st₁ heq
            have ht : remainingFuel st₁ ≤ remainingFuel (skipWs st) := by
              have := congrArg (fun q => remainingFuel q.2) heq
              simp only at this
              rw [← this]
              exact TextCursor.remainingFuel_takeWhile_le _ _ _
            have hs₁ := TextCursor.remainingFuel_skipWs_le st₁
            dsimp only
            split
            · have hb := TextCursor.remainingFuel_bump_lt (c := '(') ‹_›
              rw [(ih (bump (skipWs st₁)) (by omega)).2.2.2.2.2.2 k₁ k₂ (by omega) (by omega)]
            · rfl
          · rfl
      · rfl
    have hFa : ∀ st : TextCursor, remainingFuel st ≤ n + 1 → ∀ f₁ f₂,
        5 * remainingFuel st ≤ f₁ + 3 → 5 * remainingFuel st ≤ f₂ + 3 →
        factorFuel g f₁ st = factorFuel g f₂ st := by
      intro st hst f₁ f₂ h₁ h₂
      have := hpos st
      obtain ⟨k₁, rfl⟩ : ∃ k, f₁ = k + 1 := ⟨f₁ - 1, by omega⟩
      obtain ⟨k₂, rfl⟩ : ∃ k, f₂ = k + 1 := ⟨f₂ - 1, by omega⟩
      rw [factorFuel.eq_2, factorFuel.eq_2, hP st hst k₁ k₂ (by omega) (by omega)]
    have hU : ∀ st : TextCursor, remainingFuel st ≤ n + 1 → ∀ f₁ f₂,
        5 * remainingFuel st ≤ f₁ + 2 → 5 * remainingFuel st ≤ f₂ + 2 →
        unaryFuel g f₁ st = unaryFuel g f₂ st := by
      intro st hst f₁ f₂ h₁ h₂
      have := hpos st
      obtain ⟨k₁, rfl⟩ : ∃ k, f₁ = k + 1 := ⟨f₁ - 1, by omega⟩
      obtain ⟨k₂, rfl⟩ : ∃ k, f₂ = k + 1 := ⟨f₂ - 1, by omega⟩
      have hs := TextCursor.remainingFuel_skipWs_le st
      rw [unaryFuel.eq_2, unaryFuel.eq_2]
      split
      · have hb := TextCursor.remainingFuel_bump_lt (c := '-') ‹_›
        rw [(ih (bump (skipWs st)) (by omega)).2.2.1 k₁ k₂ (by omega) (by omega)]
      · exact hFa (skipWs st) (by omega) k₁ k₂ (by omega) (by omega)
    have hLT : ∀ st : TextCursor, remainingFuel st ≤ n + 1 → ∀ f₁ f₂ acc,
        5 * remainingFuel st ≤ f₁ + 4 → 5 * remainingFuel st ≤ f₂ + 4 →
        termFuel.loop g f₁ acc st = termFuel.loop g f₂ acc st := by
      intro st hst f₁ f₂ acc h₁ h₂
      have := hpos st
      obtain ⟨k₁, rfl⟩ : ∃ k, f₁ = k + 1 := ⟨f₁ - 1, by omega⟩
      obtain ⟨k₂, rfl⟩ : ∃ k, f₂ = k + 1 := ⟨f₂ - 1, by omega⟩
      have hs := TextCursor.remainingFuel_skipWs_le st
      rw [termFuel.loop.eq_2, termFuel.loop.eq_2]
      split
      · have hb := TextCursor.remainingFuel_bump_lt (c := '*') ‹_›
        rw [(ih (bump (skipWs st)) (by omega)).2.2.1 k₁ k₂ (by omega) (by omega)]
        refine bind_congr_ok fun ⟨f, st₂⟩ hres => ?_
        have := (hM k₂).2.2.1 _ _ _ hres
        exact (ih st₂ (by omega)).2.2.2.1 k₁ k₂ _ (by omega) (by omega)
      · have hb := TextCursor.remainingFuel_bump_lt (c := '/') ‹_›
        rw [(ih (bump (skipWs st)) (by omega)).2.2.1 k₁ k₂ (by omega) (by omega)]
        refine bind_congr_ok fun ⟨f, st₂⟩ hres => ?_
        have := (hM k₂).2.2.1 _ _ _ hres
        exact (ih st₂ (by omega)).2.2.2.1 k₁ k₂ _ (by omega) (by omega)
      · rfl
    have hT : ∀ st : TextCursor, remainingFuel st ≤ n + 1 → ∀ f₁ f₂,
        5 * remainingFuel st ≤ f₁ + 1 → 5 * remainingFuel st ≤ f₂ + 1 →
        termFuel g f₁ st = termFuel g f₂ st := by
      intro st hst f₁ f₂ h₁ h₂
      have := hpos st
      obtain ⟨k₁, rfl⟩ : ∃ k, f₁ = k + 1 := ⟨f₁ - 1, by omega⟩
      obtain ⟨k₂, rfl⟩ : ∃ k, f₂ = k + 1 := ⟨f₂ - 1, by omega⟩
      rw [termFuel.eq_2, termFuel.eq_2, hU st hst k₁ k₂ (by omega) (by omega)]
      refine bind_congr_ok fun ⟨f, st₁⟩ hres => ?_
      have := (hM k₂).2.2.1 _ _ _ hres
      exact hLT st₁ (by omega) k₁ k₂ f (by omega) (by omega)
    have hLE : ∀ st : TextCursor, remainingFuel st ≤ n + 1 → ∀ f₁ f₂ acc,
        5 * remainingFuel st ≤ f₁ + 4 → 5 * remainingFuel st ≤ f₂ + 4 →
        exprFuel.loop g f₁ acc st = exprFuel.loop g f₂ acc st := by
      intro st hst f₁ f₂ acc h₁ h₂
      have := hpos st
      obtain ⟨k₁, rfl⟩ : ∃ k, f₁ = k + 1 := ⟨f₁ - 1, by omega⟩
      obtain ⟨k₂, rfl⟩ : ∃ k, f₂ = k + 1 := ⟨f₂ - 1, by omega⟩
      have hs := TextCursor.remainingFuel_skipWs_le st
      rw [exprFuel.loop.eq_2, exprFuel.loop.eq_2]
      split
      · have hb := TextCursor.remainingFuel_bump_lt (c := '+') ‹_›
        rw [(ih (bump (skipWs st)) (by omega)).2.2.2.2.1 k₁ k₂ (by omega) (by omega)]
        refine bind_congr_ok fun ⟨t, st₂⟩ hres => ?_
        have := (hM k₂).2.2.2.2.1 _ _ _ hres
        exact (ih st₂ (by omega)).2.2.2.2.2.1 k₁ k₂ _ (by omega) (by omega)
      · have hb := TextCursor.remainingFuel_bump_lt (c := '-') ‹_›
        rw [(ih (bump (skipWs st)) (by omega)).2.2.2.2.1 k₁ k₂ (by omega) (by omega)]
        refine bind_congr_ok fun ⟨t, st₂⟩ hres => ?_
        have := (hM k₂).2.2.2.2.1 _ _ _ hres
        exact (ih st₂ (by omega)).2.2.2.2.2.1 k₁ k₂ _ (by omega) (by omega)
      · rfl
    have hE : ∀ st : TextCursor, remainingFuel st ≤ n + 1 → ∀ f₁ f₂,
        5 * remainingFuel st ≤ f₁ → 5 * remainingFuel st ≤ f₂ →
        exprFuel g f₁ st = exprFuel g f₂ st := by
      intro st hst f₁ f₂ h₁ h₂
      have := hpos st
      obtain ⟨k₁, rfl⟩ : ∃ k, f₁ = k + 1 := ⟨f₁ - 1, by omega⟩
      obtain ⟨k₂, rfl⟩ : ∃ k, f₂ = k + 1 := ⟨f₂ - 1, by omega⟩
      rw [exprFuel.eq_2, exprFuel.eq_2, hT st hst k₁ k₂ (by omega) (by omega)]
      refine bind_congr_ok fun ⟨t, st₁⟩ hres => ?_
      have := (hM k₂).2.2.2.2.1 _ _ _ hres
      exact hLE st₁ (by omega) k₁ k₂ t (by omega) (by omega)
    intro st hst
    exact ⟨hP st hst, hFa st hst, hU st hst, hLT st hst, hT st hst, hLE st hst, hE st hst⟩

/-- `primaryFuel` gives one result for every fuel `f` with `5 * remainingFuel st ≤ f + 4`. -/
theorem primaryFuel_eq_of_le (g : Grammar E) {fuel₁ fuel₂ : Nat} {st : TextCursor}
    (h₁ : 5 * remainingFuel st ≤ fuel₁ + 4) (h₂ : 5 * remainingFuel st ≤ fuel₂ + 4) :
    primaryFuel g fuel₁ st = primaryFuel g fuel₂ st :=
  (eq_of_le_fuel g _ st (Nat.le_refl _)).1 _ _ h₁ h₂

/-- `factorFuel` gives one result for every fuel `f` with `5 * remainingFuel st ≤ f + 3`. -/
theorem factorFuel_eq_of_le (g : Grammar E) {fuel₁ fuel₂ : Nat} {st : TextCursor}
    (h₁ : 5 * remainingFuel st ≤ fuel₁ + 3) (h₂ : 5 * remainingFuel st ≤ fuel₂ + 3) :
    factorFuel g fuel₁ st = factorFuel g fuel₂ st :=
  (eq_of_le_fuel g _ st (Nat.le_refl _)).2.1 _ _ h₁ h₂

/-- `unaryFuel` gives one result for every fuel `f` with `5 * remainingFuel st ≤ f + 2`. -/
theorem unaryFuel_eq_of_le (g : Grammar E) {fuel₁ fuel₂ : Nat} {st : TextCursor}
    (h₁ : 5 * remainingFuel st ≤ fuel₁ + 2) (h₂ : 5 * remainingFuel st ≤ fuel₂ + 2) :
    unaryFuel g fuel₁ st = unaryFuel g fuel₂ st :=
  (eq_of_le_fuel g _ st (Nat.le_refl _)).2.2.1 _ _ h₁ h₂

/-- `termFuel.loop` gives one result for every fuel `f` with `5 * remainingFuel st ≤ f + 4`. -/
theorem termFuel.loop_eq_of_le (g : Grammar E) {fuel₁ fuel₂ : Nat} (acc : E) {st : TextCursor}
    (h₁ : 5 * remainingFuel st ≤ fuel₁ + 4) (h₂ : 5 * remainingFuel st ≤ fuel₂ + 4) :
    termFuel.loop g fuel₁ acc st = termFuel.loop g fuel₂ acc st :=
  (eq_of_le_fuel g _ st (Nat.le_refl _)).2.2.2.1 _ _ acc h₁ h₂

/-- `termFuel` gives one result for every fuel `f` with `5 * remainingFuel st ≤ f + 1`. -/
theorem termFuel_eq_of_le (g : Grammar E) {fuel₁ fuel₂ : Nat} {st : TextCursor}
    (h₁ : 5 * remainingFuel st ≤ fuel₁ + 1) (h₂ : 5 * remainingFuel st ≤ fuel₂ + 1) :
    termFuel g fuel₁ st = termFuel g fuel₂ st :=
  (eq_of_le_fuel g _ st (Nat.le_refl _)).2.2.2.2.1 _ _ h₁ h₂

/-- `exprFuel.loop` gives one result for every fuel `f` with `5 * remainingFuel st ≤ f + 4`. -/
theorem exprFuel.loop_eq_of_le (g : Grammar E) {fuel₁ fuel₂ : Nat} (acc : E) {st : TextCursor}
    (h₁ : 5 * remainingFuel st ≤ fuel₁ + 4) (h₂ : 5 * remainingFuel st ≤ fuel₂ + 4) :
    exprFuel.loop g fuel₁ acc st = exprFuel.loop g fuel₂ acc st :=
  (eq_of_le_fuel g _ st (Nat.le_refl _)).2.2.2.2.2.1 _ _ acc h₁ h₂

/-- `exprFuel` gives one result for every fuel `f` with `5 * remainingFuel st ≤ f`. -/
theorem exprFuel_eq_of_le (g : Grammar E) {fuel₁ fuel₂ : Nat} {st : TextCursor}
    (h₁ : 5 * remainingFuel st ≤ fuel₁) (h₂ : 5 * remainingFuel st ≤ fuel₂) :
    exprFuel g fuel₁ st = exprFuel g fuel₂ st :=
  (eq_of_le_fuel g _ st (Nat.le_refl _)).2.2.2.2.2.2 _ _ h₁ h₂

end

variable {E : Type}

/-- The budget of `parse` is at least the bound of `exprFuel_eq_of_le`, so `parse` gives
the result of `exprFuel` with any fuel at or above five times the remaining fuel. -/
theorem parse_eq_of_le (g : Grammar E) (source : String) {fuel : Nat}
    (h : 5 * TextCursor.remainingFuel { source := source } ≤ fuel) :
    parse g source = (do
      let (e, st1) ← exprFuel g fuel { source := source }
      match TextCursor.peek (TextCursor.skipWs st1) with
      | none => .ok e
      | some c => .error s!"unexpected trailing input near '{c}'") := by
  unfold parse
  dsimp only
  rw [exprFuel_eq_of_le g (by omega) h]
  rfl

/-- Expression trees of the language `parse` reads: decimal literals kept as their text,
identifiers, calls with one argument, negation, the four operators, and natural powers. -/
inductive Ast where
  /-- A decimal literal, kept as its text. -/
  | num (text : String)
  /-- An identifier. -/
  | ident (name : String)
  /-- The function `name` applied to `arg`. -/
  | call (name : String) (arg : Ast)
  /-- The negation `-a`. -/
  | neg (a : Ast)
  /-- The sum `a + b`. -/
  | add (a b : Ast)
  /-- The difference `a - b`. -/
  | sub (a b : Ast)
  /-- The product `a * b`. -/
  | mul (a b : Ast)
  /-- The quotient `a / b`. -/
  | div (a b : Ast)
  /-- The power `base ^ n`. -/
  | pow (base : Ast) (n : Nat)

namespace Ast

/-- The value of a tree in the language of `g`: a literal is `g.const lo hi` for the enclosure
`(lo, hi)` that `TextCursor.parseFloatBounds` reads from its text, and an unknown identifier or
function, or `/` and calls when `g` lacks them, give none. -/
def eval (g : Grammar E) : Ast → Option E
  | num text => (TextCursor.parseFloatBounds { source := text }).toOption.map fun r =>
      g.const r.1.1 r.1.2
  | ident name => g.ident? name
  | call name a => g.call?.bind fun f => (a.eval g).bind (f name)
  | neg a => (a.eval g).map g.neg
  | add a b => (a.eval g).bind fun x => (b.eval g).map (g.add x)
  | sub a b => (a.eval g).bind fun x => (b.eval g).map (g.sub x)
  | mul a b => (a.eval g).bind fun x => (b.eval g).map (g.mul x)
  | div a b => g.div?.bind fun f => (a.eval g).bind fun x => (b.eval g).map (f x)
  | pow a n => (a.eval g).map (g.pow · n)

/-- How tightly the top of a tree binds: `0` for `+` and `-`, `1` for `*` and `/`, `2` for a
negation, `3` for a power and `4` for literals, identifiers and calls. -/
def prec : Ast → Nat
  | add .. | sub .. => 0
  | mul .. | div .. => 1
  | neg _ => 2
  | pow .. => 3
  | num _ | ident _ | call .. => 4

/-- The text of a tree without spaces, with an operand in parentheses exactly when its `prec` is
below what its place needs: `+`, `-`, `*` and `/` group to the left, and the base of `^` is a
literal, an identifier, a call or a parenthesized tree. -/
def print : Ast → String
  | num text => text
  | ident name => name
  | call name a => name ++ "(" ++ a.print ++ ")"
  | neg a => "-" ++ if a.prec < 2 then "(" ++ a.print ++ ")" else a.print
  | add a b => a.print ++ "+" ++ if b.prec < 1 then "(" ++ b.print ++ ")" else b.print
  | sub a b => a.print ++ "-" ++ if b.prec < 1 then "(" ++ b.print ++ ")" else b.print
  | mul a b => (if a.prec < 1 then "(" ++ a.print ++ ")" else a.print) ++ "*" ++
      if b.prec < 2 then "(" ++ b.print ++ ")" else b.print
  | div a b => (if a.prec < 1 then "(" ++ a.print ++ ")" else a.print) ++ "/" ++
      if b.prec < 2 then "(" ++ b.print ++ ")" else b.print
  | pow a n => (if a.prec < 4 then "(" ++ a.print ++ ")" else a.print) ++ "^" ++ n.repr

/-- Trees whose literals are a nonempty run of digits, alone or followed by `.` and digits, whose
names are an ASCII letter followed by letters, digits and `_`, and whose exponents have at most
256 digits. -/
def WellFormed : Ast → Prop
  | num text => ∃ i f : List Char, i ≠ [] ∧ (∀ c ∈ i ++ f, c.isDigit = true) ∧
      (text.toList = i ∨ text.toList = i ++ '.' :: f)
  | ident name => ∃ c cs, name.toList = c :: cs ∧ c.isAlpha = true ∧
      ∀ d ∈ cs, d.isAlpha = true ∨ d.isDigit = true ∨ d = '_'
  | call name a => (∃ c cs, name.toList = c :: cs ∧ c.isAlpha = true ∧
      ∀ d ∈ cs, d.isAlpha = true ∨ d.isDigit = true ∨ d = '_') ∧ a.WellFormed
  | neg a => a.WellFormed
  | add a b | sub a b | mul a b | div a b => a.WellFormed ∧ b.WellFormed
  | pow a n => a.WellFormed ∧ n < 10 ^ 256

end Ast

section

open TextCursor (remainingFuel peek bump skipWs)
open String (utf8Len)

/-- `primaryFuel` reads `X` as `v` and stops after it, in every source where the end or one of
`)+-*/^` follows `X`, with fuel `n` such that `5 * (size + 1) ≤ n + 4`. -/
private def PrimaryReads (g : Grammar E) (X : List Char) (v : E) : Prop :=
  ∀ (s : String) (l r : List Char) (n : Nat), s.toList = l ++ X ++ r →
    (∀ c ∈ r.head?, c ∈ [')', '+', '-', '*', '/', '^']) → 5 * (utf8Len (X ++ r) + 1) ≤ n + 4 →
    primaryFuel g n ⟨s, ⟨utf8Len l⟩⟩ = .ok (v, ⟨s, ⟨utf8Len (l ++ X)⟩⟩)

/-- `factorFuel` reads `X` as `v` and stops after it, in every source where the end or one of
`)+-*/` follows `X`, with fuel `n` such that `5 * (size + 1) ≤ n + 3`. -/
private def FactorReads (g : Grammar E) (X : List Char) (v : E) : Prop :=
  ∀ (s : String) (l r : List Char) (n : Nat), s.toList = l ++ X ++ r →
    (∀ c ∈ r.head?, c ∈ [')', '+', '-', '*', '/']) → 5 * (utf8Len (X ++ r) + 1) ≤ n + 3 →
    factorFuel g n ⟨s, ⟨utf8Len l⟩⟩ = .ok (v, ⟨s, ⟨utf8Len (l ++ X)⟩⟩)

/-- `unaryFuel` reads `X` as `v` and stops after it, in every source where the end or one of
`)+-*/` follows `X`, with fuel `n` such that `5 * (size + 1) ≤ n + 2`. -/
private def UnaryReads (g : Grammar E) (X : List Char) (v : E) : Prop :=
  ∀ (s : String) (l r : List Char) (n : Nat), s.toList = l ++ X ++ r →
    (∀ c ∈ r.head?, c ∈ [')', '+', '-', '*', '/']) → 5 * (utf8Len (X ++ r) + 1) ≤ n + 2 →
    unaryFuel g n ⟨s, ⟨utf8Len l⟩⟩ = .ok (v, ⟨s, ⟨utf8Len (l ++ X)⟩⟩)

/-- `termFuel` reads `X` as `v` and continues as its loop after `X` with enough fuel for the
rest, in every source where the end or one of `)+-*/` follows `X`. -/
private def TermReads (g : Grammar E) (X : List Char) (v : E) : Prop :=
  ∀ (s : String) (l r : List Char) (n : Nat), s.toList = l ++ X ++ r →
    (∀ c ∈ r.head?, c ∈ [')', '+', '-', '*', '/']) → 5 * (utf8Len (X ++ r) + 1) ≤ n + 1 →
    ∃ m, 5 * (utf8Len r + 1) ≤ m + 4 ∧
      termFuel g n ⟨s, ⟨utf8Len l⟩⟩ = termFuel.loop g m v ⟨s, ⟨utf8Len (l ++ X)⟩⟩

/-- `exprFuel` reads `X` as `v` and continues as its loop after `X` with enough fuel for the
rest, in every source where the end or one of `)+-` follows `X`. -/
private def ExprReads (g : Grammar E) (X : List Char) (v : E) : Prop :=
  ∀ (s : String) (l r : List Char) (n : Nat), s.toList = l ++ X ++ r →
    (∀ c ∈ r.head?, c ∈ [')', '+', '-']) → 5 * (utf8Len (X ++ r) + 1) ≤ n →
    ∃ m, 5 * (utf8Len r + 1) ≤ m + 4 ∧
      exprFuel g n ⟨s, ⟨utf8Len l⟩⟩ = exprFuel.loop g m v ⟨s, ⟨utf8Len (l ++ X)⟩⟩

private theorem termFuel.loop_eq_ok (g : Grammar E) (m : Nat) (acc : E) {s : String}
    {L r : List Char} (h : s.toList = L ++ r) (hr : ∀ c ∈ r.head?, c ∈ [')', '+', '-']) :
    termFuel.loop g m acc ⟨s, ⟨utf8Len L⟩⟩ = .ok (acc, ⟨s, ⟨utf8Len L⟩⟩) := by
  cases m with
  | zero => rw [termFuel.loop.eq_1]
  | succ m =>
    rw [termFuel.loop.eq_2]
    simp only [TextCursor.skipWs_eq_self h (fun c hc =>
      (by decide : ∀ c ∈ [')', '+', '-'], c.isWhitespace = false) c (hr c hc)),
      TextCursor.peek_eq_head? h]
    cases r with
    | nil => rfl
    | cons c r =>
      have hc := hr c rfl
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      rcases hc with rfl | rfl | rfl <;> rfl

private theorem exprFuel.loop_eq_ok (g : Grammar E) (m : Nat) (acc : E) {s : String}
    {L r : List Char} (h : s.toList = L ++ r) (hr : ∀ c ∈ r.head?, c ∈ [')']) :
    exprFuel.loop g m acc ⟨s, ⟨utf8Len L⟩⟩ = .ok (acc, ⟨s, ⟨utf8Len L⟩⟩) := by
  cases m with
  | zero => rw [exprFuel.loop.eq_1]
  | succ m =>
    rw [exprFuel.loop.eq_2]
    simp only [TextCursor.skipWs_eq_self h (fun c hc =>
      (by decide : ∀ c ∈ [')'], c.isWhitespace = false) c (hr c hc)),
      TextCursor.peek_eq_head? h]
    cases r with
    | nil => rfl
    | cons c r =>
      have hc := hr c rfl
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      subst hc
      rfl

private theorem FactorReads.of_primary {g : Grammar E} {X : List Char} {v : E}
    (hX : PrimaryReads g X v) : FactorReads g X v := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    rw [factorFuel.eq_2, hX s l r n h (fun c hc => (by decide : ∀ c ∈ [')', '+', '-', '*', '/'],
      c ∈ [')', '+', '-', '*', '/', '^']) c (hr c hc)) (by omega)]
    simp only [bind, Except.bind, TextCursor.skipWs_eq_self h (fun c hc => (by decide :
      ∀ c ∈ [')', '+', '-', '*', '/'], c.isWhitespace = false) c (hr c hc)),
      TextCursor.peek_eq_head? h]
    cases r with
    | nil => rfl
    | cons c r =>
      have hc := hr c rfl
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      rcases hc with rfl | rfl | rfl | rfl | rfl <;> rfl

private theorem UnaryReads.of_factor {g : Grammar E} {X : List Char} {v : E}
    (hX : FactorReads g X v) (hh : ∃ c X', X = c :: X' ∧ c ≠ '-' ∧ c.isWhitespace = false) :
    UnaryReads g X v := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    obtain ⟨c, X', rfl, hc1, hc2⟩ := hh
    have h0 : s.toList = l ++ (c :: X' ++ r) := by rw [h, List.append_assoc]
    rw [unaryFuel.eq_2]
    simp only [TextCursor.skipWs_eq_self h0 (TextCursor.forall_mem_head?_cons hc2),
      TextCursor.peek_eq_head? h0, List.cons_append, List.head?_cons]
    split
    · rename_i heq
      simp only [Option.some.injEq] at heq
      exact absurd heq hc1
    · exact hX s l r n h hr (by omega)

private theorem TermReads.of_unary {g : Grammar E} {X : List Char} {v : E}
    (hX : UnaryReads g X v) : TermReads g X v := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    refine ⟨n, ?_, ?_⟩
    · rw [String.utf8Len_append] at hn
      omega
    · rw [termFuel.eq_2, hX s l r n h hr (by omega)]
      rfl

private theorem ExprReads.of_term {g : Grammar E} {X : List Char} {v : E}
    (hX : TermReads g X v) : ExprReads g X v := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    obtain ⟨m, -, hm⟩ := hX s l r n h (fun c hc => (by decide : ∀ c ∈ [')', '+', '-'],
      c ∈ [')', '+', '-', '*', '/']) c (hr c hc)) (by omega)
    refine ⟨n, ?_, ?_⟩
    · rw [String.utf8Len_append] at hn
      omega
    · rw [exprFuel.eq_2, hm, termFuel.loop_eq_ok g m v h hr]
      rfl

private theorem PrimaryReads.paren {g : Grammar E} {X : List Char} {v : E}
    (hX : ExprReads g X v) : PrimaryReads g ('(' :: (X ++ [')'])) v := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    have h0 : s.toList = l ++ '(' :: (X ++ ')' :: r) := by rw [h]; simp
    have h1 : s.toList = l ++ ['('] ++ X ++ ')' :: r := by rw [h]; simp
    obtain ⟨m, -, hm⟩ := hX s (l ++ ['(']) (')' :: r) n h1 (by simp) (by
      simp only [String.utf8Len_append, String.utf8Len_cons, String.utf8Len_nil,
        List.cons_append] at hn ⊢
      have := Char.utf8Size_pos '('
      omega)
    rw [primaryFuel.eq_2]
    simp only [TextCursor.skipWs_eq_self h0 (TextCursor.forall_mem_head?_cons (by decide)),
      TextCursor.peek_eq_head? h0, List.head?_cons]
    rw [TextCursor.bump_eq h0, hm, exprFuel.loop_eq_ok g m v h1 (by simp)]
    simp only [bind, Except.bind, TextCursor.skipWs_eq_self h1
      (TextCursor.forall_mem_head?_cons (by decide)), TextCursor.peek_eq_head? h1,
      List.head?_cons]
    rw [TextCursor.bump_eq h1]
    simp

private theorem PrimaryReads.num (g : Grammar E) {X i f : List Char} {x y : Float}
    {cursor : TextCursor} (hi : i ≠ []) (hd : ∀ c ∈ i ++ f, c.isDigit = true)
    (hX : X = i ∨ X = i ++ '.' :: f)
    (hx : TextCursor.parseFloatBounds { source := String.ofList X } = .ok ((x, y), cursor)) :
    PrimaryReads g X (g.const x y) := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    obtain ⟨d, i', rfl⟩ := List.exists_cons_of_ne_nil hi
    have hdd : d.isDigit = true := hd d (by simp)
    obtain ⟨X', rfl⟩ : ∃ X', X = d :: X' := by rcases hX with rfl | rfl <;> exact ⟨_, rfl⟩
    have h0 : s.toList = l ++ (d :: X' ++ r) := by rw [h, List.append_assoc]
    rw [primaryFuel.eq_2]
    simp only [TextCursor.skipWs_eq_self h0
        (TextCursor.forall_mem_head?_cons (TextCursor.isWhitespace_eq_false_of_isDigit hdd)),
      TextCursor.peek_eq_head? h0, List.cons_append, List.head?_cons]
    split
    · rename_i heq
      simp only [Option.some.injEq] at heq
      subst heq
      exact absurd hdd (by decide)
    · rename_i c hne heq
      simp only [Option.some.injEq] at heq
      subst heq
      simp only [hdd, ↓reduceIte]
      rw [TextCursor.parseFloatBounds_eq h hi hd hX (fun c hc => (by decide :
        ∀ c ∈ [')', '+', '-', '*', '/', '^'], c.isDigit = false ∧ c ≠ '.') c (hr c hc)) hx]
      rfl
    · rename_i heq
      cases heq

private theorem PrimaryReads.ident {g : Grammar E} {name : String} {c : Char} {cs : List Char}
    {v : E} (hname : name.toList = c :: cs) (hc : c.isAlpha = true)
    (hcs : ∀ d ∈ cs, d.isAlpha = true ∨ d.isDigit = true ∨ d = '_')
    (hv : g.ident? name = some v) : PrimaryReads g name.toList v := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    rw [hname] at h ⊢
    have h0 : s.toList = l ++ (c :: cs ++ r) := by rw [h, List.append_assoc]
    have hname' : String.ofList (c :: cs) = name := by rw [← hname, String.ofList_toList]
    have hw : ∀ d ∈ c :: cs, (d.isAlpha || d.isDigit || decide (d = '_')) = true := by
      intro d hd
      rcases List.mem_cons.mp hd with rfl | hd
      · simp [hc]
      · rcases hcs d hd with h' | h' | h' <;> simp [h']
    rw [primaryFuel.eq_2]
    simp only [TextCursor.skipWs_eq_self h0
        (TextCursor.forall_mem_head?_cons (TextCursor.isWhitespace_eq_false_of_isAlpha hc)),
      TextCursor.peek_eq_head? h0, List.cons_append, List.head?_cons]
    split
    · rename_i heq
      simp only [Option.some.injEq] at heq
      subst heq
      exact absurd hc (by decide)
    · rename_i c' hne heq
      simp only [Option.some.injEq] at heq
      subst heq
      rw [TextCursor.takeWhile_eq (fun c => c.isAlpha || c.isDigit || c = '_') "" h hw
        (fun d hd => (by decide : ∀ d ∈ [')', '+', '-', '*', '/', '^'],
          (d.isAlpha || d.isDigit || decide (d = '_')) = false) d (hr d hd))]
      simp only [TextCursor.isDigit_eq_false_of_isAlpha hc, hc,
        Bool.false_eq_true, ↓reduceIte, String.empty_append, hname', hv,
        TextCursor.skipWs_eq_self h (fun d hd => (by decide : ∀ d ∈ [')', '+', '-', '*', '/', '^'],
          d.isWhitespace = false) d (hr d hd)), TextCursor.peek_eq_head? h]
      cases r with
      | nil => cases g.call? <;> rfl
      | cons d r =>
        have hd := hr d rfl
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hd
        rcases hd with rfl | rfl | rfl | rfl | rfl | rfl <;> cases g.call? <;> rfl
    · rename_i heq
      cases heq

private theorem PrimaryReads.call {g : Grammar E} {name : String} {c : Char} {cs A : List Char}
    {f : String → E → Option E} {va v : E} (hname : name.toList = c :: cs)
    (hc : c.isAlpha = true) (hcs : ∀ d ∈ cs, d.isAlpha = true ∨ d.isDigit = true ∨ d = '_')
    (hg : g.call? = some f) (hA : ExprReads g A va) (hv : f name va = some v) :
    PrimaryReads g (name.toList ++ '(' :: (A ++ [')'])) v := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    rw [hname] at h ⊢
    have h0 : s.toList = l ++ (c :: (cs ++ '(' :: (A ++ ')' :: r))) := by rw [h]; simp
    have h1 : s.toList = l ++ c :: cs ++ '(' :: (A ++ ')' :: r) := by rw [h]; simp
    have h2 : s.toList = l ++ c :: cs ++ ['('] ++ A ++ ')' :: r := by rw [h]; simp
    have hname' : String.ofList (c :: cs) = name := by rw [← hname, String.ofList_toList]
    have hw : ∀ d ∈ c :: cs, (d.isAlpha || d.isDigit || decide (d = '_')) = true := by
      intro d hd
      rcases List.mem_cons.mp hd with rfl | hd
      · simp [hc]
      · rcases hcs d hd with h' | h' | h' <;> simp [h']
    obtain ⟨m, -, hm⟩ := hA s (l ++ c :: cs ++ ['(']) (')' :: r) n h2 (by simp) (by
      simp only [String.utf8Len_append, String.utf8Len_cons, String.utf8Len_nil] at hn ⊢
      have := Char.utf8Size_pos c
      have := Char.utf8Size_pos '('
      omega)
    rw [primaryFuel.eq_2]
    simp only [TextCursor.skipWs_eq_self h0
        (TextCursor.forall_mem_head?_cons (TextCursor.isWhitespace_eq_false_of_isAlpha hc)),
      TextCursor.peek_eq_head? h0, List.head?_cons]
    split
    · rename_i heq
      simp only [Option.some.injEq] at heq
      subst heq
      exact absurd hc (by decide)
    · rename_i c' hne heq
      simp only [Option.some.injEq] at heq
      subst heq
      rw [TextCursor.takeWhile_eq (fun c => c.isAlpha || c.isDigit || c = '_') "" h1 hw
        (TextCursor.forall_mem_head?_cons (by decide))]
      simp only [TextCursor.isDigit_eq_false_of_isAlpha hc, hc,
        Bool.false_eq_true, ↓reduceIte, String.empty_append, hname', hg,
        TextCursor.skipWs_eq_self h1 (TextCursor.forall_mem_head?_cons (by decide)),
        TextCursor.peek_eq_head? h1, List.head?_cons]
      rw [TextCursor.bump_eq h1, hm, exprFuel.loop_eq_ok g m va h2 (by simp)]
      simp only [bind, Except.bind, TextCursor.skipWs_eq_self h2
        (TextCursor.forall_mem_head?_cons (by decide)), TextCursor.peek_eq_head? h2,
        List.head?_cons, hv]
      rw [TextCursor.bump_eq h2]
      simp
    · rename_i heq
      cases heq

private theorem UnaryReads.neg {g : Grammar E} {B : List Char} {vb : E}
    (hB : UnaryReads g B vb) : UnaryReads g ('-' :: B) (g.neg vb) := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    have h0 : s.toList = l ++ '-' :: (B ++ r) := by rw [h]; simp
    have h1 : s.toList = l ++ ['-'] ++ B ++ r := by rw [h]; simp
    rw [unaryFuel.eq_2]
    simp only [TextCursor.skipWs_eq_self h0 (TextCursor.forall_mem_head?_cons (by decide)),
      TextCursor.peek_eq_head? h0, List.head?_cons]
    rw [TextCursor.bump_eq h0, hB s (l ++ ['-']) r n h1 hr (by
      simp only [String.utf8Len_append, String.utf8Len_cons] at hn ⊢
      have := Char.utf8Size_pos '-'
      omega)]
    simp [bind, Except.bind]

private theorem FactorReads.pow {g : Grammar E} {A : List Char} {va : E} {k : Nat}
    (hA : PrimaryReads g A va) (hk : k < 10 ^ 256) :
    FactorReads g (A ++ '^' :: k.repr.toList) (g.pow va k) := by
  intro s l r n h hr hn
  cases n with
  | zero => omega
  | succ n =>
    have h1 : s.toList = l ++ A ++ '^' :: (k.repr.toList ++ r) := by rw [h]; simp
    have h2 : s.toList = l ++ A ++ ['^'] ++ k.repr.toList ++ r := by rw [h]; simp
    rw [factorFuel.eq_2, hA s l ('^' :: (k.repr.toList ++ r)) n h1 (by simp) (by
      simp only [String.utf8Len_append, String.utf8Len_cons] at hn ⊢
      omega)]
    simp only [bind, Except.bind, TextCursor.skipWs_eq_self h1
        (TextCursor.forall_mem_head?_cons (by decide)),
      TextCursor.peek_eq_head? h1, List.head?_cons]
    rw [TextCursor.bump_eq h1, TextCursor.parseNat_repr h2 hk (fun c hc => (by decide :
      ∀ c ∈ [')', '+', '-', '*', '/'], c.isDigit = false) c (hr c hc))]
    simp

private theorem ExprReads.add {g : Grammar E} {A B : List Char} {va vb : E}
    (hA : ExprReads g A va) (hB : TermReads g B vb) :
    ExprReads g (A ++ '+' :: B) (g.add va vb) := by
  intro s l r n h hr hn
  have h1 : s.toList = l ++ A ++ '+' :: (B ++ r) := by rw [h]; simp
  have h2 : s.toList = l ++ A ++ ['+'] ++ B ++ r := by rw [h]; simp
  obtain ⟨m, hm, he⟩ := hA s l ('+' :: (B ++ r)) n h1 (by simp) (by
    simp only [String.utf8Len_append, String.utf8Len_cons] at hn ⊢
    omega)
  simp only [String.utf8Len_append, String.utf8Len_cons] at hm
  have := Char.utf8Size_pos '+'
  obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  obtain ⟨m2, -, hb⟩ := hB s (l ++ A ++ ['+']) r m h2
    (fun c hc => (by decide : ∀ c ∈ [')', '+', '-'], c ∈ [')', '+', '-', '*', '/']) c (hr c hc))
    (by rw [String.utf8Len_append]; omega)
  refine ⟨m, by omega, ?_⟩
  rw [he, exprFuel.loop.eq_2]
  simp only [TextCursor.skipWs_eq_self h1 (TextCursor.forall_mem_head?_cons (by decide)),
    TextCursor.peek_eq_head? h1, List.head?_cons]
  rw [TextCursor.bump_eq h1, hb, termFuel.loop_eq_ok g m2 vb h2 hr]
  simp [bind, Except.bind]

private theorem ExprReads.sub {g : Grammar E} {A B : List Char} {va vb : E}
    (hA : ExprReads g A va) (hB : TermReads g B vb) :
    ExprReads g (A ++ '-' :: B) (g.sub va vb) := by
  intro s l r n h hr hn
  have h1 : s.toList = l ++ A ++ '-' :: (B ++ r) := by rw [h]; simp
  have h2 : s.toList = l ++ A ++ ['-'] ++ B ++ r := by rw [h]; simp
  obtain ⟨m, hm, he⟩ := hA s l ('-' :: (B ++ r)) n h1 (by simp) (by
    simp only [String.utf8Len_append, String.utf8Len_cons] at hn ⊢
    omega)
  simp only [String.utf8Len_append, String.utf8Len_cons] at hm
  have := Char.utf8Size_pos '-'
  obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  obtain ⟨m2, -, hb⟩ := hB s (l ++ A ++ ['-']) r m h2
    (fun c hc => (by decide : ∀ c ∈ [')', '+', '-'], c ∈ [')', '+', '-', '*', '/']) c (hr c hc))
    (by rw [String.utf8Len_append]; omega)
  refine ⟨m, by omega, ?_⟩
  rw [he, exprFuel.loop.eq_2]
  simp only [TextCursor.skipWs_eq_self h1 (TextCursor.forall_mem_head?_cons (by decide)),
    TextCursor.peek_eq_head? h1, List.head?_cons]
  rw [TextCursor.bump_eq h1, hb, termFuel.loop_eq_ok g m2 vb h2 hr]
  simp [bind, Except.bind]

private theorem TermReads.mul {g : Grammar E} {A B : List Char} {va vb : E}
    (hA : TermReads g A va) (hB : UnaryReads g B vb) :
    TermReads g (A ++ '*' :: B) (g.mul va vb) := by
  intro s l r n h hr hn
  have h1 : s.toList = l ++ A ++ '*' :: (B ++ r) := by rw [h]; simp
  have h2 : s.toList = l ++ A ++ ['*'] ++ B ++ r := by rw [h]; simp
  obtain ⟨m, hm, he⟩ := hA s l ('*' :: (B ++ r)) n h1 (by simp) (by
    simp only [String.utf8Len_append, String.utf8Len_cons] at hn ⊢
    omega)
  simp only [String.utf8Len_append, String.utf8Len_cons] at hm
  have := Char.utf8Size_pos '*'
  obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  have hb := hB s (l ++ A ++ ['*']) r m h2 hr (by rw [String.utf8Len_append]; omega)
  refine ⟨m, by omega, ?_⟩
  rw [he, termFuel.loop.eq_2]
  simp only [TextCursor.skipWs_eq_self h1 (TextCursor.forall_mem_head?_cons (by decide)),
    TextCursor.peek_eq_head? h1, List.head?_cons]
  rw [TextCursor.bump_eq h1, hb]
  simp [bind, Except.bind]

private theorem TermReads.div {g : Grammar E} {A B : List Char} {va vb : E} {f : E → E → E}
    (hg : g.div? = some f) (hA : TermReads g A va) (hB : UnaryReads g B vb) :
    TermReads g (A ++ '/' :: B) (f va vb) := by
  intro s l r n h hr hn
  have h1 : s.toList = l ++ A ++ '/' :: (B ++ r) := by rw [h]; simp
  have h2 : s.toList = l ++ A ++ ['/'] ++ B ++ r := by rw [h]; simp
  obtain ⟨m, hm, he⟩ := hA s l ('/' :: (B ++ r)) n h1 (by simp) (by
    simp only [String.utf8Len_append, String.utf8Len_cons] at hn ⊢
    omega)
  simp only [String.utf8Len_append, String.utf8Len_cons] at hm
  have := Char.utf8Size_pos '/'
  obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  have hb := hB s (l ++ A ++ ['/']) r m h2 hr (by rw [String.utf8Len_append]; omega)
  refine ⟨m, by omega, ?_⟩
  rw [he, termFuel.loop.eq_2]
  simp only [TextCursor.skipWs_eq_self h1 (TextCursor.forall_mem_head?_cons (by decide)),
    TextCursor.peek_eq_head? h1, List.head?_cons, hg]
  rw [TextCursor.bump_eq h1, hb]
  simp [bind, Except.bind]

private theorem reads_of_paren {g : Grammar E} {X : List Char} {v : E} (hX : ExprReads g X v) :
    PrimaryReads g ('(' :: (X ++ [')'])) v ∧ FactorReads g ('(' :: (X ++ [')'])) v ∧
      UnaryReads g ('(' :: (X ++ [')'])) v ∧ TermReads g ('(' :: (X ++ [')'])) v := by
  have hP := PrimaryReads.paren hX
  have hF := FactorReads.of_primary hP
  have hU := UnaryReads.of_factor hF ⟨'(', _, rfl, by decide, by decide⟩
  exact ⟨hP, hF, hU, TermReads.of_unary hU⟩

private theorem reads_of_factor {g : Grammar E} {X : List Char} {v : E} (hX : FactorReads g X v)
    (hh : ∃ c X', X = c :: X' ∧ c ≠ '-' ∧ c.isWhitespace = false) :
    UnaryReads g X v ∧ TermReads g X v ∧ ExprReads g X v := by
  have hU := UnaryReads.of_factor hX hh
  have hT := TermReads.of_unary hU
  exact ⟨hU, hT, ExprReads.of_term hT⟩

/-- The text of `t` at a place that needs `Ast.prec` at least `k`. -/
private def paren (k : Nat) (t : Ast) : List Char :=
  if t.prec < k then '(' :: (t.print.toList ++ [')']) else t.print.toList

private theorem toList_paren (k : Nat) (t : Ast) :
    (if t.prec < k then "(" ++ t.print ++ ")" else t.print).toList = paren k t := by
  unfold paren
  split <;> simp

private theorem reads_print (g : Grammar E) (t : Ast) (ht : t.WellFormed) {v : E}
    (hv : t.eval g = some v) :
    PrimaryReads g (paren 4 t) v ∧ FactorReads g (paren 3 t) v ∧ UnaryReads g (paren 2 t) v ∧
      TermReads g (paren 1 t) v ∧ ExprReads g t.print.toList v ∧
      ∃ c X, paren 4 t = c :: X ∧ c ≠ '-' ∧ c.isWhitespace = false := by
  induction t generalizing v with
  | num text =>
    obtain ⟨i, f, hi, hd, hX⟩ := ht
    simp only [Ast.eval, Option.map_eq_some_iff] at hv
    obtain ⟨⟨⟨x, y⟩, cursor⟩, hx, rfl⟩ := hv
    have hx' : TextCursor.parseFloatBounds { source := text } = .ok ((x, y), cursor) := by
      cases h : TextCursor.parseFloatBounds { source := text } <;>
        simp_all [Except.toOption]
    have hP : PrimaryReads g text.toList (g.const x y) :=
      PrimaryReads.num g hi hd hX (by rwa [String.ofList_toList])
    have hh : ∃ c X, text.toList = c :: X ∧ c ≠ '-' ∧ c.isWhitespace = false := by
      obtain ⟨d, i', rfl⟩ := List.exists_cons_of_ne_nil hi
      have hdd := hd d (by simp)
      have hd' : d ≠ '-' := by
        rintro rfl
        exact absurd hdd (by decide)
      rcases hX with hX | hX
      · exact ⟨d, i', hX, hd', TextCursor.isWhitespace_eq_false_of_isDigit hdd⟩
      · exact ⟨d, _, by rw [hX]; rfl, hd', TextCursor.isWhitespace_eq_false_of_isDigit hdd⟩
    obtain ⟨hU, hT, hE⟩ := reads_of_factor (FactorReads.of_primary hP) hh
    simp only [paren, Ast.prec, Ast.print, Nat.lt_irrefl, ↓reduceIte]
    exact ⟨hP, FactorReads.of_primary hP, hU, hT, hE, hh⟩
  | ident name =>
    obtain ⟨c, cs, hname, hc, hcs⟩ := ht
    have hP := PrimaryReads.ident (g := g) hname hc hcs hv
    have hh : ∃ c' X, name.toList = c' :: X ∧ c' ≠ '-' ∧ c'.isWhitespace = false := by
      refine ⟨c, cs, hname, ?_, TextCursor.isWhitespace_eq_false_of_isAlpha hc⟩
      rintro rfl
      exact absurd hc (by decide)
    obtain ⟨hU, hT, hE⟩ := reads_of_factor (FactorReads.of_primary hP) hh
    simp only [paren, Ast.prec, Ast.print, Nat.lt_irrefl, ↓reduceIte]
    exact ⟨hP, FactorReads.of_primary hP, hU, hT, hE, hh⟩
  | call name a ih =>
    obtain ⟨⟨c, cs, hname, hc, hcs⟩, ha⟩ := ht
    simp only [Ast.eval, Option.bind_eq_some_iff] at hv
    obtain ⟨f, hg, va, hva, hv⟩ := hv
    obtain ⟨-, -, -, -, hEa, -⟩ := ih ha hva
    have hpr : (Ast.call name a).print.toList =
        name.toList ++ '(' :: (a.print.toList ++ [')']) := by
      simp [Ast.print]
    have hP := PrimaryReads.call hname hc hcs hg hEa hv
    rw [← hpr] at hP
    have hh : ∃ c' X, (Ast.call name a).print.toList = c' :: X ∧ c' ≠ '-' ∧
        c'.isWhitespace = false := by
      refine ⟨c, cs ++ '(' :: (a.print.toList ++ [')']), by rw [hpr, hname]; rfl, ?_,
        TextCursor.isWhitespace_eq_false_of_isAlpha hc⟩
      rintro rfl
      exact absurd hc (by decide)
    obtain ⟨hU, hT, hE⟩ := reads_of_factor (FactorReads.of_primary hP) hh
    simp only [paren, Ast.prec, Nat.lt_irrefl, ↓reduceIte]
    exact ⟨hP, FactorReads.of_primary hP, hU, hT, hE, hh⟩
  | neg a ih =>
    simp only [Ast.eval, Option.map_eq_some_iff] at hv
    obtain ⟨va, hva, rfl⟩ := hv
    obtain ⟨-, -, hUa, -, -, -⟩ := ih ht hva
    have hpr : (Ast.neg a).print.toList = '-' :: paren 2 a := by
      simp only [Ast.print, String.toList_append, toList_paren]
      simp
    have hU := UnaryReads.neg hUa
    rw [← hpr] at hU
    have hT := TermReads.of_unary hU
    have hE := ExprReads.of_term hT
    obtain ⟨hP', hF', -, -⟩ := reads_of_paren hE
    simp only [paren, Ast.prec]
    exact ⟨hP', hF', hU, hT, hE, '(', _, rfl, by decide, by decide⟩
  | add a b iha ihb =>
    obtain ⟨ha, hb⟩ := ht
    simp only [Ast.eval, Option.bind_eq_some_iff, Option.map_eq_some_iff] at hv
    obtain ⟨va, hva, vb, hvb, rfl⟩ := hv
    obtain ⟨-, -, -, -, hEa, -⟩ := iha ha hva
    obtain ⟨-, -, -, hTb, -, -⟩ := ihb hb hvb
    have hpr : (Ast.add a b).print.toList = a.print.toList ++ '+' :: paren 1 b := by
      simp only [Ast.print, String.toList_append, toList_paren]
      simp
    have hE := ExprReads.add hEa hTb
    rw [← hpr] at hE
    obtain ⟨hP, hF, hU, hT⟩ := reads_of_paren hE
    simp only [paren, Ast.prec]
    exact ⟨hP, hF, hU, hT, hE, '(', _, rfl, by decide, by decide⟩
  | sub a b iha ihb =>
    obtain ⟨ha, hb⟩ := ht
    simp only [Ast.eval, Option.bind_eq_some_iff, Option.map_eq_some_iff] at hv
    obtain ⟨va, hva, vb, hvb, rfl⟩ := hv
    obtain ⟨-, -, -, -, hEa, -⟩ := iha ha hva
    obtain ⟨-, -, -, hTb, -, -⟩ := ihb hb hvb
    have hpr : (Ast.sub a b).print.toList = a.print.toList ++ '-' :: paren 1 b := by
      simp only [Ast.print, String.toList_append, toList_paren]
      simp
    have hE := ExprReads.sub hEa hTb
    rw [← hpr] at hE
    obtain ⟨hP, hF, hU, hT⟩ := reads_of_paren hE
    simp only [paren, Ast.prec]
    exact ⟨hP, hF, hU, hT, hE, '(', _, rfl, by decide, by decide⟩
  | mul a b iha ihb =>
    obtain ⟨ha, hb⟩ := ht
    simp only [Ast.eval, Option.bind_eq_some_iff, Option.map_eq_some_iff] at hv
    obtain ⟨va, hva, vb, hvb, rfl⟩ := hv
    obtain ⟨-, -, -, hTa, -, -⟩ := iha ha hva
    obtain ⟨-, -, hUb, -, -, -⟩ := ihb hb hvb
    have hpr : (Ast.mul a b).print.toList = paren 1 a ++ '*' :: paren 2 b := by
      simp only [Ast.print, String.toList_append, toList_paren]
      simp
    have hT := TermReads.mul hTa hUb
    rw [← hpr] at hT
    have hE := ExprReads.of_term hT
    obtain ⟨hP, hF, hU, -⟩ := reads_of_paren hE
    simp only [paren, Ast.prec]
    exact ⟨hP, hF, hU, hT, hE, '(', _, rfl, by decide, by decide⟩
  | div a b iha ihb =>
    obtain ⟨ha, hb⟩ := ht
    simp only [Ast.eval, Option.bind_eq_some_iff, Option.map_eq_some_iff] at hv
    obtain ⟨f, hg, va, hva, vb, hvb, rfl⟩ := hv
    obtain ⟨-, -, -, hTa, -, -⟩ := iha ha hva
    obtain ⟨-, -, hUb, -, -, -⟩ := ihb hb hvb
    have hpr : (Ast.div a b).print.toList = paren 1 a ++ '/' :: paren 2 b := by
      simp only [Ast.print, String.toList_append, toList_paren]
      simp
    have hT := TermReads.div hg hTa hUb
    rw [← hpr] at hT
    have hE := ExprReads.of_term hT
    obtain ⟨hP, hF, hU, -⟩ := reads_of_paren hE
    simp only [paren, Ast.prec]
    exact ⟨hP, hF, hU, hT, hE, '(', _, rfl, by decide, by decide⟩
  | pow a k ih =>
    obtain ⟨ha, hk⟩ := ht
    simp only [Ast.eval, Option.map_eq_some_iff] at hv
    obtain ⟨va, hva, rfl⟩ := hv
    obtain ⟨hPa, -, -, -, -, ⟨c, X, hcX, hc1, hc2⟩⟩ := ih ha hva
    have hpr : (Ast.pow a k).print.toList = paren 4 a ++ '^' :: k.repr.toList := by
      simp only [Ast.print, String.toList_append, toList_paren]
      simp
    have hF := FactorReads.pow (k := k) hPa hk
    rw [← hpr] at hF
    obtain ⟨hU, hT, hE⟩ := reads_of_factor hF
      ⟨c, X ++ '^' :: k.repr.toList, by rw [hpr, hcX]; rfl, hc1, hc2⟩
    obtain ⟨hP, -, -, -⟩ := reads_of_paren hE
    simp only [paren, Ast.prec]
    exact ⟨hP, hF, hU, hT, hE, '(', _, rfl, by decide, by decide⟩

/-- `e` is the value in `g` of a well-formed tree. -/
private def Valued (g : Grammar E) (e : E) : Prop := ∃ t : Ast, t.WellFormed ∧ t.eval g = some e

private theorem valued_of_ok (g : Grammar E) (fuel : Nat) :
    (∀ st e st', primaryFuel g fuel st = .ok (e, st') → Valued g e) ∧
    (∀ st e st', factorFuel g fuel st = .ok (e, st') → Valued g e) ∧
    (∀ st e st', unaryFuel g fuel st = .ok (e, st') → Valued g e) ∧
    (∀ acc st e st', Valued g acc → termFuel.loop g fuel acc st = .ok (e, st') → Valued g e) ∧
    (∀ st e st', termFuel g fuel st = .ok (e, st') → Valued g e) ∧
    (∀ acc st e st', Valued g acc → exprFuel.loop g fuel acc st = .ok (e, st') → Valued g e) ∧
    (∀ st e st', exprFuel g fuel st = .ok (e, st') → Valued g e) := by
  induction fuel with
  | zero =>
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro st e st' h; rw [primaryFuel.eq_1] at h; cases h
    · intro st e st' h; rw [factorFuel.eq_1] at h; cases h
    · intro st e st' h; rw [unaryFuel.eq_1] at h; cases h
    · intro acc st e st' hacc h; rw [termFuel.loop.eq_1] at h; cases h; exact hacc
    · intro st e st' h; rw [termFuel.eq_1] at h; cases h
    · intro acc st e st' hacc h; rw [exprFuel.loop.eq_1] at h; cases h; exact hacc
    · intro st e st' h; rw [exprFuel.eq_1] at h; cases h
  | succ k ih =>
    obtain ⟨ihP, ihFa, ihU, ihLT, ihT, ihLE, ihE⟩ := ih
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro st e st' h
      rw [primaryFuel.eq_2] at h
      split at h
      · obtain ⟨⟨e₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
        dsimp only at h
        split at h
        · cases h
          exact ihE _ _ _ h₁
        · cases h
      · rename_i c _ hpk
        split at h
        · rename_i hc
          obtain ⟨⟨⟨x, y⟩, st₁⟩, h₁, h⟩ := bind_eq_ok h
          cases h
          obtain ⟨text, hw, cursor, hx⟩ := TextCursor.exists_of_parseFloatBounds hpk
            (by simp [hc]) h₁
          exact ⟨.num text, hw, by simp [Ast.eval, hx, Except.toOption]⟩
        · split at h
          · rename_i hc
            split at h
            rename_i id st₁ heq
            have hname : ∃ c cs, id.toList = c :: cs ∧ c.isAlpha = true ∧
                ∀ d ∈ cs, d.isAlpha = true ∨ d.isDigit = true ∨ d = '_' := by
              obtain ⟨w, hw, hwd⟩ := TextCursor.takeWhile_fst_of_peek
                (fun c => c.isAlpha || c.isDigit || c = '_') hpk (by simp [hc])
              rw [heq] at hw
              refine ⟨c, w, hw, hc, fun d hd => ?_⟩
              have := hwd d hd
              simpa [or_assoc] using this
            dsimp only at h
            split at h
            · obtain ⟨⟨a, st₂⟩, h₂, h⟩ := bind_eq_ok h
              obtain ⟨ta, hta, hva⟩ := ihE _ _ _ h₂
              dsimp only at h
              split at h
              · split at h
                · cases h
                  rename_i hcall _ _ _ _ hf
                  exact ⟨.call id ta, ⟨hname, hta⟩, by simp [Ast.eval, hcall, hva, hf]⟩
                · cases h
              · cases h
            · split at h
              · cases h
                rename_i hv
                exact ⟨.ident id, hname, hv⟩
              · cases h
          · cases h
      · cases h
    · intro st e st' h
      rw [factorFuel.eq_2] at h
      obtain ⟨⟨b, st₁⟩, h₁, h⟩ := bind_eq_ok h
      obtain ⟨tb, htb, hvb⟩ := ihP _ _ _ h₁
      dsimp only at h
      split at h
      · obtain ⟨⟨m, st₂⟩, h₂, h⟩ := bind_eq_ok h
        dsimp only at h
        cases h
        exact ⟨.pow tb m, ⟨htb, TextCursor.lt_of_parseNat h₂⟩, by simp [Ast.eval, hvb]⟩
      · cases h
        exact ⟨tb, htb, hvb⟩
    · intro st e st' h
      rw [unaryFuel.eq_2] at h
      split at h
      · obtain ⟨⟨e₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
        cases h
        obtain ⟨t, ht, hv⟩ := ihU _ _ _ h₁
        exact ⟨.neg t, ht, by simp [Ast.eval, hv]⟩
      · exact ihFa _ _ _ h
    · intro acc st e st' hacc h
      rw [termFuel.loop.eq_2] at h
      obtain ⟨ta, hta, hva⟩ := hacc
      split at h
      · obtain ⟨⟨f₂, st₂⟩, h₁, h⟩ := bind_eq_ok h
        obtain ⟨tb, htb, hvb⟩ := ihU _ _ _ h₁
        exact ihLT _ _ _ _ ⟨.mul ta tb, ⟨hta, htb⟩, by simp [Ast.eval, hva, hvb]⟩ h
      · rename_i f _ hg
        obtain ⟨⟨f₂, st₂⟩, h₁, h⟩ := bind_eq_ok h
        obtain ⟨tb, htb, hvb⟩ := ihU _ _ _ h₁
        exact ihLT _ _ _ _ ⟨.div ta tb, ⟨hta, htb⟩, by simp [Ast.eval, hg, hva, hvb]⟩ h
      · cases h
        exact ⟨ta, hta, hva⟩
    · intro st e st' h
      rw [termFuel.eq_2] at h
      obtain ⟨⟨f₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
      exact ihLT _ _ _ _ (ihU _ _ _ h₁) h
    · intro acc st e st' hacc h
      rw [exprFuel.loop.eq_2] at h
      obtain ⟨ta, hta, hva⟩ := hacc
      split at h
      · obtain ⟨⟨t₂, st₂⟩, h₁, h⟩ := bind_eq_ok h
        obtain ⟨tb, htb, hvb⟩ := ihT _ _ _ h₁
        exact ihLE _ _ _ _ ⟨.add ta tb, ⟨hta, htb⟩, by simp [Ast.eval, hva, hvb]⟩ h
      · obtain ⟨⟨t₂, st₂⟩, h₁, h⟩ := bind_eq_ok h
        obtain ⟨tb, htb, hvb⟩ := ihT _ _ _ h₁
        exact ihLE _ _ _ _ ⟨.sub ta tb, ⟨hta, htb⟩, by simp [Ast.eval, hva, hvb]⟩ h
      · cases h
        exact ⟨ta, hta, hva⟩
    · intro st e st' h
      rw [exprFuel.eq_2] at h
      obtain ⟨⟨t₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
      exact ihLE _ _ _ _ (ihT _ _ _ h₁) h

open TextCursor (takeWhile parseNat parseFloatBounds)

/-- Letters, digits, `_` and `.`: the characters that continue an identifier or a literal. -/
private def tokenChar (c : Char) : Bool := c.isAlpha || c.isDigit || c = '_' || c = '.'

private theorem isWhitespace_eq_false_of_tokenChar {c : Char} (h : tokenChar c = true) :
    c.isWhitespace = false := by
  cases hw : c.isWhitespace
  · rfl
  · simp only [Char.isWhitespace, Bool.or_eq_true, decide_eq_true_eq] at hw
    rcases hw with ((rfl | rfl) | rfl) | rfl <;> revert h <;> decide

private theorem exists_run (p : Char → Bool) (r : List Char) :
    ∃ u v, r = u ++ v ∧ (∀ c ∈ u, p c = true) ∧ ∀ c ∈ v.head?, p c = false := by
  refine ⟨r.takeWhile p, r.dropWhile p, List.takeWhile_append_dropWhile.symm,
    List.all_eq_true.1 List.all_takeWhile, fun c hc => ?_⟩
  have := List.head?_dropWhile_not p r
  rw [Option.mem_def] at hc
  rw [hc] at this
  exact this

/-- `s` lists as `L₁ ++ L₂` and `s'` as `L₁ ++ W ++ L₂`, where `W` is whitespace and `L₁` does not
end with a token character when `L₂` starts with one. -/
private structure Insertion (s s' : String) (L₁ W L₂ : List Char) : Prop where
  /-- The characters of `s`. -/
  hs : s.toList = L₁ ++ L₂
  /-- The characters of `s'`. -/
  hs' : s'.toList = L₁ ++ W ++ L₂
  /-- `W` is whitespace. -/
  hW : ∀ c ∈ W, c.isWhitespace = true
  /-- No token runs across the end of `L₁`. -/
  hb : ∀ a ∈ L₁.getLast?, ∀ b ∈ L₂.head?, tokenChar a = false ∨ tokenChar b = false

/-- `c` into `s` and `c'` into `s'` mark the same place: the same offset in `L₁` (up to its end),
or the same offset in `L₂` after `L₁` and after `L₁ ++ W`. -/
private def Shifted (s s' : String) (L₁ W L₂ : List Char) (c c' : TextCursor) : Prop :=
  (∃ l r, L₁ = l ++ r ∧ c = ⟨s, ⟨utf8Len l⟩⟩ ∧ c' = ⟨s', ⟨utf8Len l⟩⟩) ∨
    ∃ m r, L₂ = m ++ r ∧ c = ⟨s, ⟨utf8Len (L₁ ++ m)⟩⟩ ∧ c' = ⟨s', ⟨utf8Len (L₁ ++ W ++ m)⟩⟩

/-- `Shifted` cursors that see the same rest up to `W`: strictly inside `L₁`, or in `L₂`. -/
private def Aligned (s s' : String) (L₁ W L₂ : List Char) (c c' : TextCursor) : Prop :=
  (∃ l x r, L₁ = l ++ x :: r ∧ c = ⟨s, ⟨utf8Len l⟩⟩ ∧ c' = ⟨s', ⟨utf8Len l⟩⟩) ∨
    ∃ m r, L₂ = m ++ r ∧ c = ⟨s, ⟨utf8Len (L₁ ++ m)⟩⟩ ∧ c' = ⟨s', ⟨utf8Len (L₁ ++ W ++ m)⟩⟩

/-- `Aligned` cursors, or both cursors at the end of `L₁` when `L₂` does not start with a token
character. -/
private def TokenSafe (s s' : String) (L₁ W L₂ : List Char) (c c' : TextCursor) : Prop :=
  Aligned s s' L₁ W L₂ c c' ∨
    c = ⟨s, ⟨utf8Len L₁⟩⟩ ∧ c' = ⟨s', ⟨utf8Len L₁⟩⟩ ∧ ∀ b ∈ L₂.head?, tokenChar b = false

/-- Two parser results agree: the same error, or the same value at `Shifted` cursors. -/
private inductive ResRel (s s' : String) (L₁ W L₂ : List Char) {α : Type} :
    Except String (α × TextCursor) → Except String (α × TextCursor) → Prop
  /-- The same value at `Shifted` cursors. -/
  | ok (a : α) {d d' : TextCursor} :
      Shifted s s' L₁ W L₂ d d' → ResRel s s' L₁ W L₂ (.ok (a, d)) (.ok (a, d'))
  /-- The same error. -/
  | error (e : String) : ResRel s s' L₁ W L₂ (.error e) (.error e)

variable {s s' : String} {L₁ W L₂ : List Char}

private theorem shifted_of_aligned {c c' : TextCursor} (h : Aligned s s' L₁ W L₂ c c') :
    Shifted s s' L₁ W L₂ c c' := by
  rcases h with ⟨l, x, r, h1, h2, h3⟩ | h
  · exact Or.inl ⟨l, x :: r, h1, h2, h3⟩
  · exact Or.inr h

private theorem shifted_of_tokenSafe {c c' : TextCursor} (h : TokenSafe s s' L₁ W L₂ c c') :
    Shifted s s' L₁ W L₂ c c' := by
  rcases h with h | ⟨h1, h2, -⟩
  · exact shifted_of_aligned h
  · exact Or.inl ⟨L₁, [], by simp, h1, h2⟩

private theorem tokenChar_head?_append (I : Insertion s s' L₁ W L₂)
    (h : ∀ b ∈ L₂.head?, tokenChar b = false) : ∀ b ∈ (W ++ L₂).head?, tokenChar b = false := by
  have hW := I.hW
  cases W with
  | nil => simpa using h
  | cons w₀ W =>
    intro b hb
    simp only [List.cons_append, List.head?_cons, Option.mem_def, Option.some.injEq] at hb
    subst hb
    cases ht : tokenChar w₀
    · rfl
    · have := isWhitespace_eq_false_of_tokenChar ht
      rw [hW w₀ (by simp)] at this
      cases this

private theorem peek_eq_of_aligned (I : Insertion s s' L₁ W L₂) {c c' : TextCursor}
    (h : Aligned s s' L₁ W L₂ c c') : peek c' = peek c := by
  have hs := I.hs
  have hs' := I.hs'
  rcases h with ⟨l, x, r, rfl, rfl, rfl⟩ | ⟨m, r, rfl, rfl, rfl⟩
  · have h1 := TextCursor.peek_eq_head? (s := s) (l := l) (r := x :: (r ++ L₂)) (by simp [hs])
    have h2 := TextCursor.peek_eq_head? (s := s') (l := l) (r := x :: (r ++ W ++ L₂))
      (by simp [hs'])
    simp only [h1, h2, List.head?_cons]
  · have h1 := TextCursor.peek_eq_head? (s := s) (l := L₁ ++ m) (r := r) (by simp [hs])
    have h2 := TextCursor.peek_eq_head? (s := s') (l := L₁ ++ W ++ m) (r := r) (by simp [hs'])
    rw [h1, h2]

private theorem bump_shifted (I : Insertion s s' L₁ W L₂) {c c' : TextCursor} {ch : Char}
    (h : Aligned s s' L₁ W L₂ c c') (hc : peek c = some ch) :
    Shifted s s' L₁ W L₂ (bump c) (bump c') := by
  have hs := I.hs
  have hs' := I.hs'
  rcases h with ⟨l, x, r, rfl, rfl, rfl⟩ | ⟨m, r, rfl, rfl, rfl⟩
  · rw [TextCursor.bump_eq (s := s) (l := l) (c := x) (r := r ++ L₂) (by simp [hs]),
      TextCursor.bump_eq (s := s') (l := l) (c := x) (r := r ++ W ++ L₂) (by simp [hs'])]
    exact Or.inl ⟨l ++ [x], r, by simp, rfl, rfl⟩
  · cases r with
    | nil =>
      rw [TextCursor.peek_eq_head? (s := s) (l := L₁ ++ m) (r := []) (by simp [hs])] at hc
      cases hc
    | cons y r =>
      rw [TextCursor.bump_eq (s := s) (l := L₁ ++ m) (c := y) (r := r) (by simp [hs]),
        TextCursor.bump_eq (s := s') (l := L₁ ++ W ++ m) (c := y) (r := r) (by simp [hs'])]
      exact Or.inr ⟨m ++ [y], r, by simp, by simp, by simp⟩

private theorem bump_tokenSafe (I : Insertion s s' L₁ W L₂) {c c' : TextCursor} {ch : Char}
    (h : Aligned s s' L₁ W L₂ c c') (hc : peek c = some ch) (ht : tokenChar ch = true) :
    TokenSafe s s' L₁ W L₂ (bump c) (bump c') := by
  have hs := I.hs
  have hs' := I.hs'
  rcases h with ⟨l, x, r, rfl, rfl, rfl⟩ | ⟨m, r, rfl, rfl, rfl⟩
  · have hx : x = ch := by
      rw [TextCursor.peek_eq_head? (s := s) (l := l) (r := x :: (r ++ L₂)) (by simp [hs])] at hc
      simpa using hc
    subst hx
    rw [TextCursor.bump_eq (s := s) (l := l) (c := x) (r := r ++ L₂) (by simp [hs]),
      TextCursor.bump_eq (s := s') (l := l) (c := x) (r := r ++ W ++ L₂) (by simp [hs'])]
    cases r with
    | cons y r => exact Or.inl (Or.inl ⟨l ++ [x], y, r, by simp, rfl, rfl⟩)
    | nil =>
      refine Or.inr ⟨rfl, rfl, fun b hb => ?_⟩
      exact (I.hb x (by simp) b hb).resolve_left (by simp [ht])
  · cases r with
    | nil =>
      rw [TextCursor.peek_eq_head? (s := s) (l := L₁ ++ m) (r := []) (by simp [hs])] at hc
      cases hc
    | cons y r =>
      rw [TextCursor.bump_eq (s := s) (l := L₁ ++ m) (c := y) (r := r) (by simp [hs]),
        TextCursor.bump_eq (s := s') (l := L₁ ++ W ++ m) (c := y) (r := r) (by simp [hs'])]
      exact Or.inl (Or.inr ⟨m ++ [y], r, by simp, by simp, by simp⟩)

private theorem skipWs_shifted (I : Insertion s s' L₁ W L₂) {c c' : TextCursor}
    (h : Shifted s s' L₁ W L₂ c c') : Aligned s s' L₁ W L₂ (skipWs c) (skipWs c') := by
  have hs := I.hs
  have hs' := I.hs'
  have hW := I.hW
  rcases h with ⟨l, r, rfl, rfl, rfl⟩ | ⟨m, r, rfl, rfl, rfl⟩
  · obtain ⟨u, v, rfl, hu, hv⟩ := exists_run Char.isWhitespace r
    cases v with
    | cons y v =>
      have e1 := TextCursor.takeWhile_eq Char.isWhitespace "" (s := s) (l := l) (w := u)
        (r := y :: (v ++ L₂)) (by simp [hs]) hu hv
      have e2 := TextCursor.takeWhile_eq Char.isWhitespace "" (s := s') (l := l) (w := u)
        (r := y :: (v ++ W ++ L₂)) (by simp [hs']) hu hv
      simp only [skipWs, e1, e2]
      exact Or.inl ⟨l ++ u, y, v, by simp, rfl, rfl⟩
    | nil =>
      obtain ⟨u₂, v₂, hL₂, hu₂, hv₂⟩ := exists_run Char.isWhitespace L₂
      have e1 := TextCursor.takeWhile_eq Char.isWhitespace "" (s := s) (l := l) (w := u ++ u₂)
        (r := v₂) (by simp [hs, hL₂])
        (fun c hc => (List.mem_append.1 hc).elim (hu c) (hu₂ c)) hv₂
      have e2 := TextCursor.takeWhile_eq Char.isWhitespace "" (s := s') (l := l)
        (w := u ++ W ++ u₂) (r := v₂) (by simp [hs', hL₂])
        (fun c hc => by
          simp only [List.mem_append] at hc
          rcases hc with (hc | hc) | hc
          exacts [hu c hc, hW c hc, hu₂ c hc]) hv₂
      simp only [skipWs, e1, e2]
      exact Or.inr ⟨u₂, v₂, hL₂, by simp, by simp⟩
  · obtain ⟨u, v, rfl, hu, hv⟩ := exists_run Char.isWhitespace r
    have e1 := TextCursor.takeWhile_eq Char.isWhitespace "" (s := s) (l := L₁ ++ m) (w := u)
      (r := v) (by simp [hs]) hu hv
    have e2 := TextCursor.takeWhile_eq Char.isWhitespace "" (s := s') (l := L₁ ++ W ++ m) (w := u)
      (r := v) (by simp [hs']) hu hv
    simp only [skipWs, e1, e2]
    exact Or.inr ⟨m ++ u, v, by simp, by simp, by simp⟩

private theorem takeWhile_tokenSafe (I : Insertion s s' L₁ W L₂) (p : Char → Bool)
    (hp : ∀ ch, p ch = true → tokenChar ch = true) (acc : String) {c c' : TextCursor}
    (h : TokenSafe s s' L₁ W L₂ c c') : ∃ t d d', takeWhile p acc c = (t, d) ∧
      takeWhile p acc c' = (t, d') ∧ TokenSafe s s' L₁ W L₂ d d' := by
  have hs := I.hs
  have hs' := I.hs'
  have hpf : ∀ b, tokenChar b = false → p b = false := fun b hb => by
    cases hpb : p b
    · rfl
    · rw [hp b hpb] at hb
      cases hb
  rcases h with (⟨l, x, r, rfl, rfl, rfl⟩ | ⟨m, r, rfl, rfl, rfl⟩) | ⟨rfl, rfl, hL₂⟩
  · obtain ⟨u, v, hxr, hu, hv⟩ := exists_run p (x :: r)
    cases v with
    | cons y v =>
      have e1 := TextCursor.takeWhile_eq p acc (s := s) (l := l) (w := u) (r := y :: (v ++ L₂))
        (by rw [hs, hxr]; simp) hu hv
      have e2 := TextCursor.takeWhile_eq p acc (s := s') (l := l) (w := u)
        (r := y :: (v ++ W ++ L₂)) (by rw [hs', hxr]; simp) hu hv
      exact ⟨_, _, _, e1, e2, Or.inl (Or.inl ⟨l ++ u, y, v, by rw [hxr]; simp, rfl, rfl⟩)⟩
    | nil =>
      rw [List.append_nil] at hxr
      subst hxr
      have hL₂ : ∀ b ∈ L₂.head?, tokenChar b = false := by
        obtain ⟨l', z, hz⟩ : ∃ l' z, x :: r = l' ++ [z] := by
          rcases List.eq_nil_or_concat (x :: r) with h | ⟨l', z, h⟩
          · cases h
          · exact ⟨l', z, by simpa using h⟩
        have hzt : tokenChar z = true := hp z (hu z (by rw [hz]; simp))
        intro b hb
        exact (I.hb z (by rw [hz]; simp) b hb).resolve_left (by simp [hzt])
      have e1 := TextCursor.takeWhile_eq p acc (s := s) (l := l) (w := x :: r) (r := L₂)
        hs hu (fun b hb => hpf b (hL₂ b hb))
      have e2 := TextCursor.takeWhile_eq p acc (s := s') (l := l) (w := x :: r) (r := W ++ L₂)
        (by simp [hs']) hu (fun b hb => hpf b (tokenChar_head?_append I hL₂ b hb))
      exact ⟨_, _, _, e1, e2, Or.inr ⟨rfl, rfl, hL₂⟩⟩
  · obtain ⟨u, v, rfl, hu, hv⟩ := exists_run p r
    have e1 := TextCursor.takeWhile_eq p acc (s := s) (l := L₁ ++ m) (w := u) (r := v)
      (by simp [hs]) hu hv
    have e2 := TextCursor.takeWhile_eq p acc (s := s') (l := L₁ ++ W ++ m) (w := u) (r := v)
      (by simp [hs']) hu hv
    exact ⟨_, _, _, e1, e2, Or.inl (Or.inr ⟨m ++ u, v, by simp, by simp, by simp⟩)⟩
  · have e1 := TextCursor.takeWhile_eq p acc (s := s) (l := L₁) (w := []) (r := L₂)
      (by simp [hs]) (by simp) (fun b hb => hpf b (hL₂ b hb))
    have e2 := TextCursor.takeWhile_eq p acc (s := s') (l := L₁) (w := []) (r := W ++ L₂)
      (by simp [hs']) (by simp) (fun b hb => hpf b (tokenChar_head?_append I hL₂ b hb))
    simp only [List.append_nil] at e1 e2
    exact ⟨_, _, _, e1, e2, Or.inr ⟨rfl, rfl, hL₂⟩⟩

private theorem aligned_of_peek (I : Insertion s s' L₁ W L₂) {c c' : TextCursor} {ch : Char}
    (h : TokenSafe s s' L₁ W L₂ c c') (hc : peek c = some ch) (ht : tokenChar ch = true) :
    Aligned s s' L₁ W L₂ c c' := by
  rcases h with h | ⟨rfl, rfl, hL₂⟩
  · exact h
  · rw [TextCursor.peek_eq_head? (s := s) (l := L₁) (r := L₂) I.hs] at hc
    rw [hL₂ ch hc] at ht
    cases ht

private theorem aligned_of_peek' (I : Insertion s s' L₁ W L₂) {c c' : TextCursor} {ch : Char}
    (h : TokenSafe s s' L₁ W L₂ c c') (hc : peek c' = some ch) (ht : tokenChar ch = true) :
    Aligned s s' L₁ W L₂ c c' := by
  rcases h with h | ⟨rfl, rfl, hL₂⟩
  · exact h
  · rw [TextCursor.peek_eq_head? (s := s') (l := L₁) (r := W ++ L₂) (by simp [I.hs'])] at hc
    rw [tokenChar_head?_append I hL₂ ch hc] at ht
    cases ht

private theorem resRel_bind {α β : Type} {x x' : Except String (α × TextCursor)}
    {k k' : α × TextCursor → Except String (β × TextCursor)} (h : ResRel s s' L₁ W L₂ x x')
    (hk : ∀ a d d', Shifted s s' L₁ W L₂ d d' → ResRel s s' L₁ W L₂ (k (a, d)) (k' (a, d'))) :
    ResRel s s' L₁ W L₂ (x >>= k) (x' >>= k') := by
  cases h with
  | ok a hd => exact hk a _ _ hd
  | error e => exact ResRel.error e

private theorem parseNat_shifted (I : Insertion s s' L₁ W L₂) {c c' : TextCursor}
    (h : Shifted s s' L₁ W L₂ c c') : ResRel s s' L₁ W L₂ (parseNat c) (parseNat c') := by
  obtain ⟨t, d, d', e1, e2, hd⟩ := takeWhile_tokenSafe I Char.isDigit
    (fun ch hch => by simp only [tokenChar, hch, Bool.or_true, Bool.true_or]) ""
    (Or.inl (skipWs_shifted I h))
  simp only [parseNat, e1, e2]
  split
  · exact ResRel.ok _ (shifted_of_tokenSafe hd)
  · exact ResRel.error _

private theorem parseFloatBounds_shifted (I : Insertion s s' L₁ W L₂) {c c' : TextCursor}
    {ch : Char} (h : Aligned s s' L₁ W L₂ c c') (hc : peek c = some ch)
    (ht : (ch.isDigit || ch = '.') = true) :
    ResRel s s' L₁ W L₂ (parseFloatBounds c) (parseFloatBounds c') := by
  have hc' : peek c' = some ch := (peek_eq_of_aligned I h).trans hc
  have htok : tokenChar ch = true := by
    simp only [Bool.or_eq_true, decide_eq_true_eq] at ht
    rcases ht with ht | rfl
    · simp only [tokenChar, ht, Bool.or_true, Bool.true_or]
    · decide
  have hws := isWhitespace_eq_false_of_tokenChar htok
  have hneg : ch ≠ '-' := by
    rintro rfl
    revert ht
    decide
  have hdig : ∀ d : Char, d.isDigit = true → tokenChar d = true := fun d hd => by
    simp only [tokenChar, hd, Bool.or_true, Bool.true_or]
  simp only [parseFloatBounds, bind, Except.bind, pure, Except.pure,
    TextCursor.skipWs_eq_of_peek hc hws, TextCursor.skipWs_eq_of_peek hc' hws, hc, hc']
  split
  · rename_i heq
    simp only [Option.some.injEq] at heq
    exact absurd heq hneg
  · obtain ⟨t, d, d', e1, e2, hd⟩ := takeWhile_tokenSafe I (fun char => char.isDigit) hdig ""
      (Or.inl h)
    simp only [e1, e2]
    split
    · exact ResRel.error _
    · by_cases hdot : peek d = some '.'
      · have hA := aligned_of_peek I hd hdot (by decide)
        have hdot' : peek d' = some '.' := (peek_eq_of_aligned I hA).trans hdot
        obtain ⟨f, e, e', g1, g2, he⟩ := takeWhile_tokenSafe I (fun char => char.isDigit) hdig ""
          (bump_tokenSafe I hA hdot (by decide))
        simp only [hdot, hdot', g1, g2]
        repeat' (first
          | exact ResRel.error _
          | exact ResRel.ok _ (shifted_of_tokenSafe he)
          | split)
      · have hdot' : ¬ peek d' = some '.' := fun h' =>
          hdot ((peek_eq_of_aligned I (aligned_of_peek' I hd h' (by decide))).symm.trans h')
        simp only
        repeat' (first
          | exact ResRel.error _
          | exact ResRel.ok _ (shifted_of_tokenSafe hd)
          | split)

/-- With the same fuel, each parsing function gives the same error, or the same value at `Shifted`
cursors, from `Shifted` cursors. -/
private theorem resRel_fuel (g : Grammar E) (I : Insertion s s' L₁ W L₂) (fuel : Nat) :
    (∀ c c', Shifted s s' L₁ W L₂ c c' →
      ResRel s s' L₁ W L₂ (primaryFuel g fuel c) (primaryFuel g fuel c')) ∧
    (∀ c c', Shifted s s' L₁ W L₂ c c' →
      ResRel s s' L₁ W L₂ (factorFuel g fuel c) (factorFuel g fuel c')) ∧
    (∀ c c', Shifted s s' L₁ W L₂ c c' →
      ResRel s s' L₁ W L₂ (unaryFuel g fuel c) (unaryFuel g fuel c')) ∧
    (∀ acc c c', Shifted s s' L₁ W L₂ c c' →
      ResRel s s' L₁ W L₂ (termFuel.loop g fuel acc c) (termFuel.loop g fuel acc c')) ∧
    (∀ c c', Shifted s s' L₁ W L₂ c c' →
      ResRel s s' L₁ W L₂ (termFuel g fuel c) (termFuel g fuel c')) ∧
    (∀ acc c c', Shifted s s' L₁ W L₂ c c' →
      ResRel s s' L₁ W L₂ (exprFuel.loop g fuel acc c) (exprFuel.loop g fuel acc c')) ∧
    (∀ c c', Shifted s s' L₁ W L₂ c c' →
      ResRel s s' L₁ W L₂ (exprFuel g fuel c) (exprFuel g fuel c')) := by
  have hid : ∀ ch : Char, (ch.isAlpha || ch.isDigit || ch = '_') = true → tokenChar ch = true :=
    fun ch hch => by simp only [tokenChar, hch, Bool.true_or]
  induction fuel with
  | zero =>
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro c c' _; rw [primaryFuel.eq_1, primaryFuel.eq_1]; exact ResRel.error _
    · intro c c' _; rw [factorFuel.eq_1, factorFuel.eq_1]; exact ResRel.error _
    · intro c c' _; rw [unaryFuel.eq_1, unaryFuel.eq_1]; exact ResRel.error _
    · intro acc c c' h; rw [termFuel.loop.eq_1, termFuel.loop.eq_1]; exact ResRel.ok _ h
    · intro c c' _; rw [termFuel.eq_1, termFuel.eq_1]; exact ResRel.error _
    · intro acc c c' h; rw [exprFuel.loop.eq_1, exprFuel.loop.eq_1]; exact ResRel.ok _ h
    · intro c c' _; rw [exprFuel.eq_1, exprFuel.eq_1]; exact ResRel.error _
  | succ n ih =>
    obtain ⟨ihP, ihF, ihU, ihTL, ihT, ihEL, ihE⟩ := ih
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · intro c c' h
      rw [primaryFuel.eq_2, primaryFuel.eq_2]
      have ha := skipWs_shifted I h
      rw [peek_eq_of_aligned I ha]
      split
      · refine resRel_bind (ihE _ _ (bump_shifted I ha ‹_›)) fun e d d' hd => ?_
        have ha₂ := skipWs_shifted I hd
        try dsimp only
        rw [peek_eq_of_aligned I ha₂]
        split
        · exact ResRel.ok e (bump_shifted I ha₂ ‹_›)
        · exact ResRel.error _
      · split
        · exact resRel_bind (parseFloatBounds_shifted I ha ‹_›
            (by rw [Bool.or_eq_true]; exact .inl ‹_›)) fun ⟨x, y⟩ d d' hd =>
            ResRel.ok _ hd
        · split
          · obtain ⟨t, d, d', e1, e2, hd⟩ := takeWhile_tokenSafe I
              (fun c => c.isAlpha || c.isDigit || c = '_') hid "" (Or.inl ha)
            simp only [e1, e2]
            have ha₂ := skipWs_shifted I (shifted_of_tokenSafe hd)
            rw [peek_eq_of_aligned I ha₂]
            split
            · refine resRel_bind (ihE _ _ (bump_shifted I ha₂ ‹_›)) fun a e e' he => ?_
              have ha₃ := skipWs_shifted I he
              try dsimp only
              rw [peek_eq_of_aligned I ha₃]
              split
              · split
                · exact ResRel.ok _ (bump_shifted I ha₃ ‹_›)
                · exact ResRel.error _
              · exact ResRel.error _
            · split
              · exact ResRel.ok _ (shifted_of_aligned ha₂)
              · exact ResRel.error _
          · exact ResRel.error _
      · exact ResRel.error _
    · intro c c' h
      rw [factorFuel.eq_2, factorFuel.eq_2]
      refine resRel_bind (ihP _ _ h) fun b d d' hd => ?_
      have ha := skipWs_shifted I hd
      try dsimp only
      rw [peek_eq_of_aligned I ha]
      split
      · exact resRel_bind (parseNat_shifted I (bump_shifted I ha ‹_›)) fun k e e' he =>
          ResRel.ok _ he
      · exact ResRel.ok _ (shifted_of_aligned ha)
    · intro c c' h
      rw [unaryFuel.eq_2, unaryFuel.eq_2]
      have ha := skipWs_shifted I h
      rw [peek_eq_of_aligned I ha]
      split
      · exact resRel_bind (ihU _ _ (bump_shifted I ha ‹_›)) fun e d d' hd => ResRel.ok _ hd
      · exact ihF _ _ (shifted_of_aligned ha)
    · intro acc c c' h
      rw [termFuel.loop.eq_2, termFuel.loop.eq_2]
      have ha := skipWs_shifted I h
      rw [peek_eq_of_aligned I ha]
      split
      · exact resRel_bind (ihU _ _ (bump_shifted I ha ‹_›)) fun f d d' hd => ihTL _ _ _ hd
      · exact resRel_bind (ihU _ _ (bump_shifted I ha ‹_›)) fun f d d' hd => ihTL _ _ _ hd
      · exact ResRel.ok _ (shifted_of_aligned ha)
    · intro c c' h
      rw [termFuel.eq_2, termFuel.eq_2]
      exact resRel_bind (ihU _ _ h) fun f d d' hd => ihTL _ _ _ hd
    · intro acc c c' h
      rw [exprFuel.loop.eq_2, exprFuel.loop.eq_2]
      have ha := skipWs_shifted I h
      rw [peek_eq_of_aligned I ha]
      split
      · exact resRel_bind (ihT _ _ (bump_shifted I ha ‹_›)) fun t d d' hd => ihEL _ _ _ hd
      · exact resRel_bind (ihT _ _ (bump_shifted I ha ‹_›)) fun t d d' hd => ihEL _ _ _ hd
      · exact ResRel.ok _ (shifted_of_aligned ha)
    · intro c c' h
      rw [exprFuel.eq_2, exprFuel.eq_2]
      exact resRel_bind (ihT _ _ h) fun t d d' hd => ihEL _ _ _ hd

end

open String (utf8Len)

/-- `parse` reads the printed text of a well-formed tree back as the tree's value, for every tree
whose value in `g` is defined. -/
theorem parse_print (g : Grammar E) {t : Ast} (ht : t.WellFormed) {v : E}
    (hv : t.eval g = some v) : parse g t.print = .ok v := by
  obtain ⟨-, -, -, -, hE, -⟩ := reads_print g t ht hv
  have h0 : t.print.toList = [] ++ t.print.toList ++ [] := by simp
  have hst : ({ source := t.print } : TextCursor) = ⟨t.print, ⟨utf8Len []⟩⟩ := rfl
  unfold parse
  simp only [bind, Except.bind]
  rw [hst, TextCursor.remainingFuel_eq (l := []) (r := t.print.toList) (by simp)]
  obtain ⟨m, -, hm⟩ := hE t.print [] [] (16 * (utf8Len t.print.toList + 1) + 32) h0 (by simp)
    (by simp only [List.append_nil]; omega)
  rw [hm, exprFuel.loop_eq_ok g m v (r := []) h0 (by simp)]
  simp only [TextCursor.skipWs_eq_self (r := []) h0 (by simp), TextCursor.peek_eq_head? h0,
    List.head?_nil]

/-- Every value `parse` gives is the value of a well-formed tree. -/
theorem exists_eval_eq_of_parse_eq_ok (g : Grammar E) {source : String} {e : E}
    (h : parse g source = .ok e) : ∃ t : Ast, t.WellFormed ∧ t.eval g = some e := by
  unfold parse at h
  obtain ⟨⟨e₁, st₁⟩, h₁, h⟩ := bind_eq_ok h
  dsimp only at h
  split at h
  · cases h
    exact (valued_of_ok g _).2.2.2.2.2.2 _ _ _ h₁
  · cases h

/-- Inserting whitespace `w` between `s₁` and `s₂` does not change what `parse` gives when the last
character of `s₁` and the first of `s₂` are not both letters, digits, `_` or `.`. -/
theorem parse_append_whitespace (g : Grammar E) (s₁ w s₂ : String)
    (hw : ∀ c ∈ w.toList, c.isWhitespace = true)
    (hb : ∀ a ∈ s₁.toList.getLast?, ∀ b ∈ s₂.toList.head?,
      (a.isAlpha || a.isDigit || a = '_' || a = '.') = false ∨
        (b.isAlpha || b.isDigit || b = '_' || b = '.') = false) :
    parse g (s₁ ++ w ++ s₂) = parse g (s₁ ++ s₂) := by
  have I : Insertion (s₁ ++ s₂) (s₁ ++ w ++ s₂) s₁.toList w.toList s₂.toList :=
    ⟨by simp, by simp, hw, hb⟩
  have e₁ : TextCursor.remainingFuel { source := s₁ ++ s₂ } =
      utf8Len (s₁.toList ++ s₂.toList) + 1 := TextCursor.remainingFuel_eq (l := []) (by simp)
  have e₂ : TextCursor.remainingFuel { source := s₁ ++ w ++ s₂ } =
      utf8Len (s₁.toList ++ w.toList ++ s₂.toList) + 1 :=
    TextCursor.remainingFuel_eq (l := []) (by simp)
  simp only [String.utf8Len_append] at e₁ e₂
  obtain ⟨F, hF⟩ : ∃ F, F = 16 * TextCursor.remainingFuel { source := s₁ ++ w ++ s₂ } + 32 :=
    ⟨_, rfl⟩
  rw [parse_eq_of_le g (s₁ ++ w ++ s₂) (fuel := F) (by omega),
    parse_eq_of_le g (s₁ ++ s₂) (fuel := F) (by omega)]
  have h0 : Shifted (s₁ ++ s₂) (s₁ ++ w ++ s₂) s₁.toList w.toList s₂.toList
      { source := s₁ ++ s₂ } { source := s₁ ++ w ++ s₂ } :=
    Or.inl ⟨[], s₁.toList, rfl, rfl, rfl⟩
  have key := (resRel_fuel g I F).2.2.2.2.2.2 _ _ h0
  revert key
  generalize exprFuel g F { source := s₁ ++ s₂ } = x
  generalize exprFuel g F { source := s₁ ++ w ++ s₂ } = x'
  intro key
  cases key with
  | error e => rfl
  | ok a hd =>
    simp only [bind, Except.bind]
    rw [peek_eq_of_aligned I (skipWs_shifted I hd)]

end TorchLean.Examples.ArithParse
