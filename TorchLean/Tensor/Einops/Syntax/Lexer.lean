/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Einops.Syntax.Diagnostic
public import TorchLean.Tensor.Einops.Syntax.UnicodeData

/-!
# Einops lexer

The lexer separates punctuation from axis words while preserving Unicode
scalar offsets. Its identifier policy `IdentifierPolicy.python` reproduces the
CPython 3.12 Unicode predicates used by einops v0.8.2.

Locations count Unicode scalar values, so a multibyte character advances the
offset by one. Both `...` and `…` produce an ellipsis token, with their original
three-character and one-character spans respectively.
-/

public section

namespace TorchLean.Einops

/-- Identifier operations required by the lexer and parser. -/
structure IdentifierPolicy where
  /-- Whether a character may continue the current axis word. -/
  isWordChar : Char → Bool
  /-- Whether a complete word is a valid named axis. -/
  isIdentifier : String → Bool
  /-- Whether a complete word denotes a nonnegative decimal integer. -/
  isDecimal : String → Bool
  /-- Convert a validated decimal word to its natural-number value. -/
  toNat? : String → Option Nat

namespace IdentifierPolicy

/-- Match Python's complete Unicode identifier predicate. -/
private def isPythonIdentifier (text : String) : Bool :=
  match text.toList with
  | [] => false
  | first :: rest =>
      PythonUnicode.hasFlag first PythonUnicode.identifierStartFlag &&
        rest.all fun char => PythonUnicode.hasFlag char PythonUnicode.identifierContinueFlag

/-- Match Python's Unicode decimal-string predicate. -/
private def isPythonDecimal (text : String) : Bool :=
  !text.isEmpty &&
    text.toList.all fun char => PythonUnicode.hasFlag char PythonUnicode.decimalFlag

/-- Convert a nonempty Python decimal string to an arbitrary-precision natural. -/
private def pythonDecimalToNat? (text : String) : Option Nat :=
  match text.toList with
  | [] => none
  | characters => go characters 0
where
  go : List Char → Nat → Option Nat
    | [], value => some value
    | char :: rest, value => do
        let digit ← PythonUnicode.decimalValue? char
        go rest (10 * value + digit)

/--
Match the word, identifier, and decimal predicates used by einops v0.8.2 on
CPython 3.12.13 with Unicode 15.0.0.

Word characters follow `str.isalnum()` plus underscore. Complete words then
follow `str.isidentifier()`, while anonymous dimensions follow
`str.isdecimal()` and Python's arbitrary-precision decimal conversion.
-/
def python : IdentifierPolicy where
  isWordChar := fun char =>
    char == '_' || PythonUnicode.hasFlag char PythonUnicode.alphanumericFlag
  isIdentifier := isPythonIdentifier
  isDecimal := isPythonDecimal
  toNat? := pythonDecimalToNat?

/-- The Unicode table places every ASCII digit in the ASCII decimal run. -/
private theorem lookup_of_isDigit (character : Char)
    (hDigit : character.isDigit = true) :
    PythonUnicode.lookup? character = some ⟨0x30, 0x39, 13⟩ := by
  have key : ∀ codepoint, codepoint < 58 → 48 ≤ codepoint →
      PythonUnicode.lookup? (Char.ofNat codepoint) = some ⟨0x30, 0x39, 13⟩ := by
    decide +kernel
  simp only [Char.isDigit_iff_toNat] at hDigit
  simp at hDigit
  have hLookup := key character.toNat (by omega) (by omega)
  rwa [Char.ofNat_toNat] at hLookup

/-- Each lexer-relevant Unicode property holds for an ASCII digit. -/
private theorem hasFlag_of_isDigit (character : Char)
    (hDigit : character.isDigit = true) (flag : UInt8)
    (hFlag : flag ∈ [PythonUnicode.alphanumericFlag, PythonUnicode.identifierContinueFlag,
      PythonUnicode.decimalFlag]) :
    PythonUnicode.hasFlag character flag = true := by
  unfold PythonUnicode.hasFlag
  rw [lookup_of_isDigit character hDigit]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hFlag
  rcases hFlag with (rfl | rfl | rfl) <;> decide

/-- Every ASCII decimal digit is a word character under the Python policy. -/
theorem python_isWordChar_of_isDigit (character : Char)
    (hDigit : character.isDigit = true) :
    python.isWordChar character = true := by
  simp [python,
    hasFlag_of_isDigit character hDigit PythonUnicode.alphanumericFlag (by simp)]

/-- The Python policy accepts the decimal notation `toString value` of every natural number. -/
theorem python_isDecimal_toString (value : Nat) :
    python.isDecimal (toString value) = true := by
  simp only [python, Nat.toString_eq_repr]
  unfold isPythonDecimal
  simp only [Bool.and_eq_true]
  constructor
  · simp
  · rw [List.all_eq_true]
    intro character hCharacter
    apply hasFlag_of_isDigit character
    · apply Nat.isDigit_of_mem_toDigits (b := 10) (by decide) (by decide)
      simpa using hCharacter
    · simp

/--
The decimal notation `toString value` of a natural number is a nonempty lexer word under the
Python identifier policy.
-/
theorem python_word_toString (value : Nat) :
    (toString value).toList ≠ [] ∧
      ∀ character ∈ (toString value).toList,
        python.isWordChar character = true := by
  simp only [Nat.toString_eq_repr]
  constructor
  · simp
  · intro character hCharacter
    apply python_isWordChar_of_isDigit character
    apply Nat.isDigit_of_mem_toDigits (b := 10) (by decide) (by decide)
    simpa using hCharacter

/-- The Unicode decimal table maps an ASCII digit to its ordinary numeric value. -/
private theorem decimalValue_of_isDigit (character : Char)
    (hDigit : character.isDigit = true) :
    PythonUnicode.decimalValue? character =
      some (character.toNat - '0'.toNat) := by
  unfold PythonUnicode.decimalValue?
  rw [lookup_of_isDigit character hDigit]
  simp [PythonUnicode.decimalFlag]
  decide

/-- The decimal scanner agrees with `Nat.ofDigitChars` on ASCII digit lists. -/
private theorem pythonDecimalToNat_go_eq (characters : List Char)
    (initial : Nat)
    (hDigits : ∀ character ∈ characters, character.isDigit = true) :
    pythonDecimalToNat?.go characters initial =
      some (Nat.ofDigitChars 10 characters initial) := by
  induction characters generalizing initial with
  | nil =>
      simp only [pythonDecimalToNat?.go, Nat.ofDigitChars_nil]
  | cons character characters induction =>
      have hCharacter := hDigits character (by simp)
      have hRest :
          ∀ remaining ∈ characters, remaining.isDigit = true :=
        fun remaining hRemaining =>
          hDigits remaining (by simp [hRemaining])
      simp only [pythonDecimalToNat?.go,
        decimalValue_of_isDigit character hCharacter]
      simp
      rw [Nat.ofDigitChars_cons]
      exact
        induction (10 * initial + (character.toNat - '0'.toNat)) hRest

/-- The Python policy decodes the decimal notation `toString value` back to `value`. -/
theorem python_toNat?_toString (value : Nat) :
    python.toNat? (toString value) = some value := by
  simp only [python, Nat.toString_eq_repr]
  unfold pythonDecimalToNat?
  rw [Nat.toList_repr]
  split
  · rename_i hEmpty
    exact (Nat.toDigits_ne_nil hEmpty).elim
  · rw [pythonDecimalToNat_go_eq]
    · simp
    · intro character hCharacter
      exact
        Nat.isDigit_of_mem_toDigits (b := 10) (by decide) (by decide)
          hCharacter

end IdentifierPolicy

/-- Tokens shared by the three pattern grammars. -/
inductive TokenKind where
  /-- An identifier or decimal axis literal. -/
  | word (text : String)
  /-- The opening delimiter of a composite axis. -/
  | leftParen
  /-- The closing delimiter of a composite axis. -/
  | rightParen
  /-- The ellipsis token, written `...` or `…`. -/
  | ellipsis
  /-- The `->` token separating input and output expressions. -/
  | arrow
  /-- The comma separating einsum operands. -/
  | comma
  /-- The packed-axis marker used by `pack` and `unpack`. -/
  | star
deriving Repr, DecidableEq

/-- A token with its source location. -/
abbrev Token := Located TokenKind

/-- Which source characters count as inter-token whitespace. -/
structure LexicalWhitespace where
  /-- Decide whether a source character separates adjacent tokens. -/
  isWhitespace : Char → Bool

namespace LexicalWhitespace

/-- Transformation parsing in einops v0.8.2 treats only spaces as whitespace. -/
def transformation : LexicalWhitespace where
  isWhitespace := fun char => char == ' '

/--
Packing accepts spaces, tabs, carriage returns, and newlines.

The predicate is Lean's `Char.isWhitespace`, so other Python whitespace
characters such as the nonbreaking space do not separate tokens.
-/
def pack : LexicalWhitespace where
  isWhitespace := Char.isWhitespace

end LexicalWhitespace

/--
Finish the pending reversed word, if any, and prepend its located token to the
reverse token stream.
-/
private def Syntax.emitWord (start stop : Nat) (wordRev : List Char)
    (tokensRev : List Token) : List Token :=
  match wordRev with
  | [] => tokensRev
  | _ =>
      let text := String.ofList wordRev.reverse
      ⟨.word text, Span.between start stop⟩ :: tokensRev

/-- Emitting a nonempty pending word prepends exactly one located word token. -/
private theorem Syntax.emitWord_of_ne_nil (start stop : Nat)
    (wordRev : List Char) (tokensRev : List Token)
    (hNonempty : wordRev ≠ []) :
    emitWord start stop wordRev tokensRev =
      ⟨.word (String.ofList wordRev.reverse),
        Span.between start stop⟩ :: tokensRev := by
  cases wordRev with
  | nil => exact (hNonempty rfl).elim
  | cons => rfl

/-- Construct the one-character diagnostic used for an unknown source symbol. -/
private def Syntax.unknownCharacter (offset : Nat) : Diagnostic ParseError :=
  { code := .unknownCharacter
    message := "unknown character in einops pattern"
    span := ⟨offset, 1⟩ }

/--
Scan source characters while accumulating the current word and token stream in
reverse order.
-/
private def Syntax.lexChars (policy : IdentifierPolicy) (whitespace : LexicalWhitespace) :
    Nat → Nat → List Char → List Char → List Token → Result (List Token)
  | offset, wordStart, [], wordRev, tokensRev =>
      .ok (emitWord wordStart offset wordRev tokensRev).reverse
  | offset, wordStart, char :: rest, wordRev, tokensRev =>
      if policy.isWordChar char then
        let start := if wordRev.isEmpty then offset else wordStart
        lexChars policy whitespace (offset + 1) start rest (char :: wordRev) tokensRev
      else
        let tokensRev := emitWord wordStart offset wordRev tokensRev
        if whitespace.isWhitespace char then
          lexChars policy whitespace (offset + 1) (offset + 1) rest [] tokensRev
        else
          match char, rest with
          | '(', rest =>
              lexChars policy whitespace (offset + 1) (offset + 1) rest []
                (⟨.leftParen, ⟨offset, 1⟩⟩ :: tokensRev)
          | ')', rest =>
              lexChars policy whitespace (offset + 1) (offset + 1) rest []
                (⟨.rightParen, ⟨offset, 1⟩⟩ :: tokensRev)
          | ',', rest =>
              lexChars policy whitespace (offset + 1) (offset + 1) rest []
                (⟨.comma, ⟨offset, 1⟩⟩ :: tokensRev)
          | '*', rest =>
              let joinedToWord :=
                !wordRev.isEmpty ||
                  match rest with
                  | next :: _ => policy.isWordChar next
                  | [] => false
              if joinedToWord then
                .error (unknownCharacter offset)
              else
                lexChars policy whitespace (offset + 1) (offset + 1) rest []
                  (⟨.star, ⟨offset, 1⟩⟩ :: tokensRev)
          | '-', '>' :: rest =>
              lexChars policy whitespace (offset + 2) (offset + 2) rest []
                (⟨.arrow, ⟨offset, 2⟩⟩ :: tokensRev)
          | '.', '.' :: '.' :: rest =>
              let joinedToWord :=
                !wordRev.isEmpty ||
                  match rest with
                  | next :: _ => policy.isWordChar next
                  | [] => false
              if joinedToWord then
                .error (unknownCharacter offset)
              else
                lexChars policy whitespace (offset + 3) (offset + 3) rest []
                  (⟨.ellipsis, ⟨offset, 3⟩⟩ :: tokensRev)
          | '\u2026', rest =>
              let joinedToWord :=
                !wordRev.isEmpty ||
                  match rest with
                  | next :: _ => policy.isWordChar next
                  | [] => false
              if joinedToWord then
                .error (unknownCharacter offset)
              else
                lexChars policy whitespace (offset + 1) (offset + 1) rest []
                  (⟨.ellipsis, ⟨offset, 1⟩⟩ :: tokensRev)
          | _, _ => .error (unknownCharacter offset)

/-- Tokenize an einops pattern using the selected identifier and whitespace policies. -/
def Syntax.lex (source : String) (policy := IdentifierPolicy.python)
    (whitespace := LexicalWhitespace.transformation) : Result (List Token) :=
  lexChars policy whitespace 0 0 source.toList [] []

/--
A word token satisfies `policy` when its text is nonempty and consists of `policy` word
characters. Every other token satisfies it.
-/
private def Syntax.TokenWordValid (policy : IdentifierPolicy) (token : Token) : Prop :=
  match token.value with
  | .word text =>
      text.toList ≠ [] ∧
        ∀ character ∈ text.toList, policy.isWordChar character = true
  | _ => True

/-- Flushing a valid pending word preserves validity of every accumulated token. -/
private theorem Syntax.emitWord_words_valid (policy : IdentifierPolicy)
    (start stop : Nat) (wordRev : List Char) (tokensRev : List Token)
    (hWordRev :
      ∀ character ∈ wordRev, policy.isWordChar character = true)
    (hTokensRev :
      ∀ token ∈ tokensRev, TokenWordValid policy token) :
    ∀ token ∈ emitWord start stop wordRev tokensRev,
      TokenWordValid policy token := by
  intro token hToken
  cases wordRev with
  | nil =>
      exact hTokensRev token hToken
  | cons first rest =>
      simp only [emitWord, List.mem_cons] at hToken
      rcases hToken with rfl | hToken
      · simp only [TokenWordValid, String.toList_ofList]
        constructor
        · simp
        · intro character hCharacter
          exact hWordRev character (List.mem_reverse.mp hCharacter)
      · exact hTokensRev token hToken

/-- The character scanner preserves the word-validity invariant on successful runs. -/
private theorem Syntax.lexChars_words_valid (policy : IdentifierPolicy)
    (whitespace : LexicalWhitespace) (offset wordStart : Nat)
    (characters wordRev : List Char) (tokensRev : List Token)
    (hWordRev :
      ∀ character ∈ wordRev, policy.isWordChar character = true)
    (hTokensRev :
      ∀ token ∈ tokensRev, TokenWordValid policy token) :
    match
      lexChars policy whitespace offset wordStart characters wordRev tokensRev
    with
    | .error _ => True
    | .ok tokens => ∀ token ∈ tokens, TokenWordValid policy token := by
  fun_induction
    lexChars policy whitespace offset wordStart characters wordRev tokensRev
  all_goals
    simp_all +zetaDelta [List.mem_reverse]
  all_goals
    first
    | exact
        emitWord_words_valid policy _ _ _ _ hWordRev hTokensRev
    | apply_assumption
      all_goals
        first
        | exact
            emitWord_words_valid policy _ _ _ _ hWordRev hTokensRev
        | trivial

/--
Every word emitted by a successful lexer run is nonempty and consists entirely
of characters accepted by that run's identifier policy.
-/
theorem Syntax.word_valid_of_lex_eq_ok (source : String) (policy : IdentifierPolicy)
    (whitespace : LexicalWhitespace) {tokens : List Token}
    (hLex : lex source policy whitespace = .ok tokens)
    {token : Token} (hToken : token ∈ tokens) {text : String}
    (hWord : token.value = .word text) :
    text.toList ≠ [] ∧
      ∀ character ∈ text.toList,
        policy.isWordChar character = true := by
  have hValid :=
    lexChars_words_valid policy whitespace 0 0 source.toList [] []
      (by simp) (by simp)
  unfold lex at hLex
  rw [hLex] at hValid
  simpa only [TokenWordValid, hWord] using hValid token hToken

/-- Scanning additional word characters only extends the pending reversed word. -/
private theorem Syntax.lexChars_all_word_append (policy : IdentifierPolicy)
    (whitespace : LexicalWhitespace) (characters : List Char)
    (suffix : List Char)
    (offset wordStart : Nat) (wordRev : List Char) (tokensRev : List Token)
    (hWordRev : wordRev ≠ [])
    (hCharacters :
      ∀ character ∈ characters, policy.isWordChar character = true) :
    lexChars policy whitespace offset wordStart (characters ++ suffix)
        wordRev tokensRev =
      lexChars policy whitespace (offset + characters.length) wordStart suffix
        (characters.reverse ++ wordRev) tokensRev := by
  induction characters generalizing offset wordRev with
  | nil =>
      simp only [List.length_nil, Nat.add_zero, List.reverse_nil,
        List.nil_append]
  | cons character characters induction =>
      have hCharacter := hCharacters character (by simp)
      have hRemaining :
          ∀ remaining ∈ characters,
            policy.isWordChar remaining = true :=
        fun remaining hRemaining =>
          hCharacters remaining (by simp [hRemaining])
      have hWordRevEmpty : wordRev.isEmpty = false := by
        cases wordRev with
        | nil => exact (hWordRev rfl).elim
        | cons => rfl
      simp only [List.cons_append, lexChars, hCharacter, ite_true]
      simp only [hWordRevEmpty, Bool.false_eq_true, ite_false]
      rw [induction (offset := offset + 1)
        (wordRev := character :: wordRev) (by simp) hRemaining]
      simp [Nat.add_assoc]
      rw [Nat.add_comm 1 characters.length]

/--
A nonempty string consisting entirely of `policy` word characters lexes as one word whose
span covers the whole string.

The parser, not the lexer, checks identifiers and decimal literals.
-/
theorem Syntax.lex_word (policy : IdentifierPolicy)
    (whitespace : LexicalWhitespace) (text : String)
    (hNonempty : text.toList ≠ [])
    (hCharacters :
      ∀ character ∈ text.toList,
        policy.isWordChar character = true) :
    lex text policy whitespace =
      .ok [⟨.word text, ⟨0, text.toList.length⟩⟩] := by
  unfold lex
  cases hText : text.toList with
  | nil => exact (hNonempty hText).elim
  | cons first rest =>
      rw [lexChars]
      have hFirst : policy.isWordChar first = true :=
        hCharacters first (by simp [hText])
      simp only [hFirst, List.isEmpty_nil, ↓reduceIte]
      rw [show rest = rest ++ [] by simp]
      rw [lexChars_all_word_append policy whitespace rest [] 1 0 [first] []
        (by simp)]
      · simp only [lexChars, List.length_cons, Except.ok.injEq]
        rw [emitWord_of_ne_nil 0 (1 + rest.length)
          (rest.reverse ++ [first]) [] (by simp)]
        simp only [List.reverse_append, List.reverse_singleton,
          List.reverse_reverse, List.singleton_append, List.reverse_singleton]
        congr 3
        · apply String.ext
          simp [hText]
        · simp [Nat.add_comm]
      · intro character hCharacter
        exact hCharacters character (by simp [hText, hCharacter])

end TorchLean.Einops
