/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor.Internal.Syntax.Diagnostic
public import NN.Tensor.Internal.Syntax.UnicodeData

/-!
# Einops lexer

The lexer separates punctuation from axis words while preserving Unicode
scalar offsets. Its default identifier policy reproduces the CPython 3.12
Unicode predicates used by einops v0.8.2. An explicit ASCII policy remains
available for applications that deliberately restrict their pattern format.
-/

public section

namespace TorchLean.Tensor.Internal.Syntax

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

/-- ASCII identifier predicate used by the restricted lexer policy. -/
private def isAsciiIdentifier (text : String) : Bool :=
  match text.toList with
  | [] => false
  | first :: rest =>
      (first.isAlpha || first == '_') &&
        rest.all fun char => char.isAlphanum || char == '_'

/-- ASCII decimal predicate used for anonymous numeric axes. -/
private def isAsciiDecimal (text : String) : Bool :=
  !text.isEmpty && text.toList.all Char.isDigit

/-- Restrict axis names and anonymous dimensions to ASCII characters. -/
def ascii : IdentifierPolicy where
  isWordChar := fun char => char.isAlphanum || char == '_'
  isIdentifier := isAsciiIdentifier
  isDecimal := isAsciiDecimal
  toNat? := String.toNat?

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
def pythonUnicode : IdentifierPolicy where
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
theorem pythonUnicode_isWordChar_of_isDigit (character : Char)
    (hDigit : character.isDigit = true) :
    pythonUnicode.isWordChar character = true := by
  simp [pythonUnicode,
    hasFlag_of_isDigit character hDigit PythonUnicode.alphanumericFlag (by simp)]

/-- Canonical decimal notation is recognized by the Python policy. -/
theorem pythonUnicode_isDecimal_toString (value : Nat) :
    pythonUnicode.isDecimal (toString value) = true := by
  simp only [pythonUnicode, Nat.toString_eq_repr]
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
Canonical natural-number notation is a nonempty lexer word under the Python
identifier policy.
-/
theorem pythonUnicode_word_toString (value : Nat) :
    (toString value).toList ≠ [] ∧
      ∀ character ∈ (toString value).toList,
        pythonUnicode.isWordChar character = true := by
  simp only [Nat.toString_eq_repr]
  constructor
  · simp
  · intro character hCharacter
    apply pythonUnicode_isWordChar_of_isDigit character
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

/-- Canonical decimal notation decodes to the natural number it renders. -/
theorem pythonUnicode_toNat?_toString (value : Nat) :
    pythonUnicode.toNat? (toString value) = some value := by
  simp only [pythonUnicode, Nat.toString_eq_repr]
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
  /-- The `...` token. -/
  | ellipsis
  /-- The `->` token separating input and output expressions. -/
  | arrow
  /-- The comma separating einsum operands. -/
  | comma
  /-- The packed-axis marker used by `pack` and `unpack`. -/
  | star
deriving Repr, DecidableEq

namespace TokenKind

/-- Canonical source text for one lexer token. -/
def render : TokenKind → String
  | .word text => text
  | .leftParen => "("
  | .rightParen => ")"
  | .ellipsis => "..."
  | .arrow => "->"
  | .comma => ","
  | .star => "*"

/-- Choose canonical whitespace between two rendered tokens. -/
private def separator (left right : TokenKind) : String :=
  match left, right with
  | .leftParen, _ | _, .rightParen | _, .comma => ""
  | _, _ => " "

/--
Render a token sequence with conventional einops spacing.

Parentheses remain tight, commas have no preceding space, and every other
token boundary receives one space. Empty operands are represented by
consecutive commas or by a comma adjacent to the arrow.
-/
def renderSequence : List TokenKind → String
  | [] => ""
  | token :: tokens =>
      token.render ++
        match tokens with
        | [] => ""
        | next :: _ => separator token next ++ renderSequence tokens

end TokenKind

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

/-- Packing uses Python's whitespace splitting behavior. -/
def general : LexicalWhitespace where
  isWhitespace := Char.isWhitespace

end LexicalWhitespace

/--
Finish the pending reversed word, if any, and prepend its located token to the
reverse token stream.
-/
private def emitWord (start stop : Nat) (wordRev : List Char)
    (tokensRev : List Token) : List Token :=
  match wordRev with
  | [] => tokensRev
  | _ =>
      let text := String.ofList wordRev.reverse
      ⟨.word text, Span.between start stop⟩ :: tokensRev

/-- Emitting a nonempty pending word prepends exactly one located word token. -/
private theorem emitWord_of_ne_nil (start stop : Nat)
    (wordRev : List Char) (tokensRev : List Token)
    (hNonempty : wordRev ≠ []) :
    emitWord start stop wordRev tokensRev =
      ⟨.word (String.ofList wordRev.reverse),
        Span.between start stop⟩ :: tokensRev := by
  cases wordRev with
  | nil => exact (hNonempty rfl).elim
  | cons => rfl

/-- Construct the one-character diagnostic used for an unknown source symbol. -/
private def unknownCharacter (offset : Nat) : Diagnostic :=
  { code := .unknownCharacter
    message := "unknown character in einops pattern"
    span := ⟨offset, 1⟩ }

/--
Scan source characters while accumulating the current word and token stream in
reverse order.
-/
private def lexChars (policy : IdentifierPolicy) (whitespace : LexicalWhitespace) :
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
def lex (source : String) (policy := IdentifierPolicy.pythonUnicode)
    (whitespace := LexicalWhitespace.transformation) : Result (List Token) :=
  lexChars policy whitespace 0 0 source.toList [] []

/-- Every word token emitted by the lexer satisfies the selected word policy. -/
private def tokenWordValid (policy : IdentifierPolicy) (token : Token) : Prop :=
  match token.value with
  | .word text =>
      text.toList ≠ [] ∧
        ∀ character ∈ text.toList, policy.isWordChar character = true
  | _ => True

/-- Flushing a valid pending word preserves validity of every accumulated token. -/
private theorem emitWord_words_valid (policy : IdentifierPolicy)
    (start stop : Nat) (wordRev : List Char) (tokensRev : List Token)
    (hWordRev :
      ∀ character ∈ wordRev, policy.isWordChar character = true)
    (hTokensRev :
      ∀ token ∈ tokensRev, tokenWordValid policy token) :
    ∀ token ∈ emitWord start stop wordRev tokensRev,
      tokenWordValid policy token := by
  intro token hToken
  cases wordRev with
  | nil =>
      exact hTokensRev token hToken
  | cons first rest =>
      simp only [emitWord, List.mem_cons] at hToken
      rcases hToken with rfl | hToken
      · simp only [tokenWordValid, String.toList_ofList]
        constructor
        · simp
        · intro character hCharacter
          exact hWordRev character (List.mem_reverse.mp hCharacter)
      · exact hTokensRev token hToken

/-- The character scanner preserves the word-validity invariant on successful runs. -/
private theorem lexChars_words_valid (policy : IdentifierPolicy)
    (whitespace : LexicalWhitespace) (offset wordStart : Nat)
    (characters wordRev : List Char) (tokensRev : List Token)
    (hWordRev :
      ∀ character ∈ wordRev, policy.isWordChar character = true)
    (hTokensRev :
      ∀ token ∈ tokensRev, tokenWordValid policy token) :
    match
      lexChars policy whitespace offset wordStart characters wordRev tokensRev
    with
    | .error _ => True
    | .ok tokens => ∀ token ∈ tokens, tokenWordValid policy token := by
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
theorem word_valid_of_lex_eq_ok (source : String) (policy : IdentifierPolicy)
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
  simpa only [tokenWordValid, hWord] using hValid token hToken

/-- Scanning additional word characters only extends the pending reversed word. -/
private theorem lexChars_all_word_append (policy : IdentifierPolicy)
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

/-- Scanning a complete nonempty word leaves that word pending before a suffix. -/
private theorem lexChars_word_append (policy : IdentifierPolicy)
    (whitespace : LexicalWhitespace) (text : String) (suffix : List Char)
    (offset : Nat) (tokensRev : List Token)
    (hNonempty : text.toList ≠ [])
    (hCharacters :
      ∀ character ∈ text.toList, policy.isWordChar character = true) :
    lexChars policy whitespace offset offset (text.toList ++ suffix) []
        tokensRev =
      lexChars policy whitespace (offset + text.toList.length) offset suffix
        text.toList.reverse tokensRev := by
  cases hText : text.toList with
  | nil => exact (hNonempty hText).elim
  | cons first rest =>
      have hFirst : policy.isWordChar first = true :=
        hCharacters first (by simp [hText])
      simp only [List.cons_append, lexChars, hFirst,
        List.isEmpty_nil, ↓reduceIte]
      rw [lexChars_all_word_append policy whitespace rest suffix
        (offset + 1) offset [first] tokensRev (by simp)]
      · simp [Nat.add_assoc, Nat.add_comm 1]
      · intro character hCharacter
        exact hCharacters character (by simp [hText, hCharacter])

/--
A nonempty string consisting entirely of word characters lexes as one word.

This is the compositional lexer fact used by canonical pattern rendering;
identifier and decimal validation remain the parser's responsibility.
-/
theorem lex_word (policy : IdentifierPolicy)
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

/-- Structural punctuation and spaces are never Python-policy word characters. -/
private theorem pythonUnicode_not_word_delimiter (character : Char)
    (hCharacter :
      character ∈ [' ', '(', ')', ',', '*', '-', '.', '\u2026']) :
    IdentifierPolicy.pythonUnicode.isWordChar character = false := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hCharacter
  rcases hCharacter with
    (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl) <;> decide +kernel

/-- Transformation lexing treats only the ordinary space delimiter as whitespace. -/
private theorem transformation_whitespace_delimiter (character : Char)
    (_hCharacter :
      character ∈ [' ', '(', ')', ',', '*', '-', '.', '\u2026']) :
    LexicalWhitespace.transformation.isWhitespace character =
      (character == ' ') := by
  rfl

/-- General lexing agrees with ordinary-space behavior on structural delimiters. -/
private theorem general_whitespace_delimiter (character : Char)
    (hCharacter :
      character ∈ [' ', '(', ')', ',', '*', '-', '.', '\u2026']) :
    LexicalWhitespace.general.isWhitespace character =
      (character == ' ') := by
  simp only [LexicalWhitespace.general]
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hCharacter
  rcases hCharacter with
    (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl) <;> decide

/-- Scanning one rendered token emits it and advances to the remaining rendering. -/
private theorem lexChars_renderSequence_cons
    (whitespace : LexicalWhitespace) (token : TokenKind)
    (tokens : List TokenKind) (offset : Nat) (tokensRev : List Token)
    (hWhitespace :
      ∀ character ∈ [' ', '(', ')', ',', '*', '-', '.', '\u2026'],
        whitespace.isWhitespace character = (character == ' '))
    (hWord :
      ∀ text, token = .word text →
        text.toList ≠ [] ∧
          ∀ character ∈ text.toList,
            IdentifierPolicy.pythonUnicode.isWordChar character = true) :
    ∃ nextOffset tokenSpan,
      lexChars IdentifierPolicy.pythonUnicode whitespace offset offset
          (TokenKind.renderSequence (token :: tokens)).toList [] tokensRev =
        lexChars IdentifierPolicy.pythonUnicode whitespace nextOffset nextOffset
          (TokenKind.renderSequence tokens).toList []
          (⟨token, tokenSpan⟩ :: tokensRev) := by
  cases token with
  | word text =>
      have hText := hWord text rfl
      have hTextReverse : text.toList.reverse ≠ [] :=
        List.reverse_ne_nil_iff.mpr hText.1
      simp only [TokenKind.renderSequence, TokenKind.render,
        String.toList_append]
      rw [lexChars_word_append IdentifierPolicy.pythonUnicode whitespace
        text _ offset tokensRev hText.1 hText.2]
      cases tokens with
      | nil =>
          refine
            ⟨offset + text.toList.length,
              Span.between offset (offset + text.toList.length), ?_⟩
          simp [TokenKind.renderSequence, lexChars, emitWord]
      | cons next rest =>
          refine
            ⟨offset + text.toList.length +
                (TokenKind.separator (.word text) next).toList.length,
              Span.between offset (offset + text.toList.length), ?_⟩
          cases next <;>
            simp [TokenKind.renderSequence, TokenKind.render,
              TokenKind.separator, String.toList_append, lexChars, emitWord,
              pythonUnicode_not_word_delimiter, hWhitespace, Nat.add_assoc]
  | leftParen =>
      cases tokens with
      | nil =>
          refine ⟨offset + 1, ⟨offset, 1⟩, ?_⟩
          simp [TokenKind.renderSequence, TokenKind.render, lexChars, emitWord,
            pythonUnicode_not_word_delimiter, hWhitespace]
      | cons next rest =>
          refine
            ⟨offset + 1 +
                (TokenKind.separator .leftParen next).toList.length,
              ⟨offset, 1⟩, ?_⟩
          cases next <;>
            simp [TokenKind.renderSequence, TokenKind.render,
              TokenKind.separator, String.toList_append, lexChars, emitWord,
              pythonUnicode_not_word_delimiter, hWhitespace, Nat.add_assoc]
  | rightParen =>
      cases tokens with
      | nil =>
          refine ⟨offset + 1, ⟨offset, 1⟩, ?_⟩
          simp [TokenKind.renderSequence, TokenKind.render, lexChars, emitWord,
            pythonUnicode_not_word_delimiter, hWhitespace]
      | cons next rest =>
          refine
            ⟨offset + 1 +
                (TokenKind.separator .rightParen next).toList.length,
              ⟨offset, 1⟩, ?_⟩
          cases next <;>
            simp [TokenKind.renderSequence, TokenKind.render,
              TokenKind.separator, String.toList_append, lexChars, emitWord,
              pythonUnicode_not_word_delimiter, hWhitespace, Nat.add_assoc]
  | ellipsis =>
      cases tokens with
      | nil =>
          refine ⟨offset + 3, ⟨offset, 3⟩, ?_⟩
          simp [TokenKind.renderSequence, TokenKind.render, lexChars, emitWord,
            pythonUnicode_not_word_delimiter, hWhitespace]
      | cons next rest =>
          refine
            ⟨offset + 3 +
                (TokenKind.separator .ellipsis next).toList.length,
              ⟨offset, 3⟩, ?_⟩
          cases next <;>
            simp [TokenKind.renderSequence, TokenKind.render,
              TokenKind.separator, String.toList_append, lexChars, emitWord,
              pythonUnicode_not_word_delimiter, hWhitespace, Nat.add_assoc]
  | arrow =>
      cases tokens with
      | nil =>
          refine ⟨offset + 2, ⟨offset, 2⟩, ?_⟩
          simp [TokenKind.renderSequence, TokenKind.render, lexChars, emitWord,
            pythonUnicode_not_word_delimiter, hWhitespace]
      | cons next rest =>
          refine
            ⟨offset + 2 +
                (TokenKind.separator .arrow next).toList.length,
              ⟨offset, 2⟩, ?_⟩
          cases next <;>
            simp [TokenKind.renderSequence, TokenKind.render,
              TokenKind.separator, String.toList_append, lexChars, emitWord,
              pythonUnicode_not_word_delimiter, hWhitespace, Nat.add_assoc]
  | comma =>
      cases tokens with
      | nil =>
          refine ⟨offset + 1, ⟨offset, 1⟩, ?_⟩
          simp [TokenKind.renderSequence, TokenKind.render, lexChars, emitWord,
            pythonUnicode_not_word_delimiter, hWhitespace]
      | cons next rest =>
          refine
            ⟨offset + 1 +
                (TokenKind.separator .comma next).toList.length,
              ⟨offset, 1⟩, ?_⟩
          cases next <;>
            simp [TokenKind.renderSequence, TokenKind.render,
              TokenKind.separator, String.toList_append, lexChars, emitWord,
              pythonUnicode_not_word_delimiter, hWhitespace, Nat.add_assoc]
  | star =>
      cases tokens with
      | nil =>
          refine ⟨offset + 1, ⟨offset, 1⟩, ?_⟩
          simp [TokenKind.renderSequence, TokenKind.render, lexChars, emitWord,
            pythonUnicode_not_word_delimiter, hWhitespace]
      | cons next rest =>
          refine
            ⟨offset + 1 +
                (TokenKind.separator .star next).toList.length,
              ⟨offset, 1⟩, ?_⟩
          cases next <;>
            simp [TokenKind.renderSequence, TokenKind.render,
              TokenKind.separator, String.toList_append, lexChars, emitWord,
              pythonUnicode_not_word_delimiter, hWhitespace, Nat.add_assoc]

/-- Lexing a canonical rendered sequence recovers its token kinds in order. -/
private theorem lexChars_renderSequence (whitespace : LexicalWhitespace)
    (tokens : List TokenKind) (offset : Nat) (tokensRev : List Token)
    (hWhitespace :
      ∀ character ∈ [' ', '(', ')', ',', '*', '-', '.', '\u2026'],
        whitespace.isWhitespace character = (character == ' '))
    (hWords :
      ∀ text, .word text ∈ tokens →
        text.toList ≠ [] ∧
          ∀ character ∈ text.toList,
            IdentifierPolicy.pythonUnicode.isWordChar character = true) :
    (lexChars IdentifierPolicy.pythonUnicode whitespace offset offset
      (TokenKind.renderSequence tokens).toList [] tokensRev).map
        (List.map Located.value) =
      .ok (tokensRev.reverse.map Located.value ++ tokens) := by
  induction tokens generalizing offset tokensRev with
  | nil =>
      simp [TokenKind.renderSequence, lexChars, emitWord, Except.map]
  | cons token tokens induction =>
      have hToken :
          ∀ text, token = .word text →
            text.toList ≠ [] ∧
              ∀ character ∈ text.toList,
                IdentifierPolicy.pythonUnicode.isWordChar character = true :=
        fun text hText => hWords text (by simp [hText])
      have hRemaining :
          ∀ text, .word text ∈ tokens →
            text.toList ≠ [] ∧
              ∀ character ∈ text.toList,
                IdentifierPolicy.pythonUnicode.isWordChar character = true :=
        fun text hText => hWords text (by simp [hText])
      obtain ⟨nextOffset, tokenSpan, hStep⟩ :=
        lexChars_renderSequence_cons whitespace token tokens offset tokensRev
          hWhitespace hToken
      rw [hStep, induction nextOffset (⟨token, tokenSpan⟩ :: tokensRev)
        hRemaining]
      simp [List.map_reverse, List.append_assoc]

namespace TokenKind

/--
Canonical token rendering is a right inverse of the transformation lexer once
source spans are erased.

The word hypothesis is necessary because `TokenKind.word` is also available to
clients constructing token lists directly, whereas words produced by `lex`
always satisfy it.
-/
theorem lex_renderSequence_transformation (tokens : List TokenKind)
    (hWords :
      ∀ text, .word text ∈ tokens →
        text.toList ≠ [] ∧
          ∀ character ∈ text.toList,
            IdentifierPolicy.pythonUnicode.isWordChar character = true) :
    (lex (renderSequence tokens) IdentifierPolicy.pythonUnicode
      LexicalWhitespace.transformation).map (List.map Located.value) =
        .ok tokens := by
  unfold lex
  simpa using
    lexChars_renderSequence LexicalWhitespace.transformation tokens 0 []
      transformation_whitespace_delimiter hWords

/--
Canonical token rendering is also a right inverse of the general-whitespace
lexer used by `pack` and `unpack`.
-/
theorem lex_renderSequence_general (tokens : List TokenKind)
    (hWords :
      ∀ text, .word text ∈ tokens →
        text.toList ≠ [] ∧
          ∀ character ∈ text.toList,
            IdentifierPolicy.pythonUnicode.isWordChar character = true) :
    (lex (renderSequence tokens) IdentifierPolicy.pythonUnicode
      LexicalWhitespace.general).map (List.map Located.value) =
        .ok tokens := by
  unfold lex
  simpa using
    lexChars_renderSequence LexicalWhitespace.general tokens 0 []
      general_whitespace_delimiter hWords

end TokenKind

end TorchLean.Tensor.Internal.Syntax
