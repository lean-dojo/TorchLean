/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Text.Unicode
public import NN.API.Json -- shake: keep
public import NN.API.Text.Tokenizer -- shake: keep

/-!
# GPT-2 Byte-Pair Encoding

Lean-native support for GPT-2-style byte-level BPE tokenizers.

This module lives in `NN.API.Text` rather than a model file: any Transformer, diffusion LM, or
verifier that wants GPT-2-compatible tokenization should share the same implementation.
The implementation parses the standard `vocab.json` and `merges.txt` files directly in Lean.

The pre-tokenizer implements the GPT-2 regex shape:

`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

The Unicode `\p{L}`, `\p{N}`, and `\s` predicates are supplied by `NN.API.Text.Unicode`, rather
than Lean's ASCII-oriented `Char.isAlpha` / `Char.isDigit` helpers.
-/

@[expose] public section

namespace TorchLean
namespace text
namespace GPT2BPE

open Lean

namespace Internal

/-! ## Data -/

/-- One token-to-id entry from GPT-2's `vocab.json`. -/
structure VocabularyEntry where
  /-- Token spelling after GPT-2 byte-to-unicode escaping. -/
  token : String
  /-- Token id. -/
  id : Nat
deriving Repr, DecidableEq

/-- One ranked merge from GPT-2's `merges.txt`. Lower rank is applied earlier. -/
structure MergeRank where
  /-- Left symbol. -/
  left : String
  /-- Right symbol. -/
  right : String
  /-- Merge priority. -/
  rank : Nat
deriving Repr, DecidableEq

/-- Loaded GPT-2 BPE tokenizer. -/
structure Tokenizer where
  /-- Token vocabulary as loaded from `vocab.json`. -/
  vocabulary : Array VocabularyEntry
  /-- Ranked merge table from `merges.txt`. -/
  merges : Array MergeRank
  /-- Fast token-to-id lookup derived from `vocabulary`. -/
  tokenIds : Std.HashMap String Nat := Std.HashMap.emptyWithCapacity
  /-- Fast id-to-token lookup derived from `vocabulary`. -/
  tokensById : Std.HashMap Nat String := Std.HashMap.emptyWithCapacity
  /-- Fast pair-to-rank lookup derived from `merges`. -/
  mergeRanks : Std.HashMap (String × String) Nat := Std.HashMap.emptyWithCapacity

/-- One merge that is currently applicable to a BPE symbol sequence. -/
structure MergeCandidate where
  left : String
  right : String
  rank : Nat

/-! ## Byte Escaping -/

/-- Bytes that GPT-2 leaves at their visible Unicode code points. -/
def bytesVisible : Array Nat :=
  (Array.range 94).map (fun i => i + 33)

/-- First Latin-1 byte range kept visible by GPT-2 byte escaping. -/
def bytesLatin1A : Array Nat :=
  (Array.range 12).map (fun i => i + 161)

/-- Second Latin-1 byte range kept visible by GPT-2 byte escaping. -/
def bytesLatin1B : Array Nat :=
  (Array.range 82).map (fun i => i + 174)

/-- Bytes that do not need synthetic code points in GPT-2 byte escaping. -/
def baseBytes : Array Nat :=
  bytesVisible ++ bytesLatin1A ++ bytesLatin1B

/-- Boolean membership test used while constructing the byte escape table. -/
def containsNat (xs : Array Nat) (x : Nat) : Bool :=
  xs.any (fun y => y == x)

/-- GPT-2 byte-to-Unicode code-point table for all 256 byte values. -/
def byteCodeTable : Array Nat := Id.run do
  let mut codes := Array.replicate 256 0
  for b in baseBytes do
    codes := codes.set! b b
  let mut next := 0
  for b in Array.range 256 do
    if !Internal.containsNat baseBytes b then
      codes := codes.set! b (256 + next)
      next := next + 1
  return codes

/-- GPT-2 byte-to-unicode escape for one byte. -/
def byteToChar (b : UInt8) : Char :=
  Char.ofNat (Array.getD byteCodeTable b.toNat b.toNat)

/-- Shared inverse byte-escape lookup, constructed once for the fixed GPT-2 byte alphabet. -/
def charByteTable : Std.HashMap Char UInt8 :=
  (Array.range 256).foldl
    (fun table byte => table.insert (byteToChar (UInt8.ofNat byte)) (UInt8.ofNat byte)) {}

/-- Inverse of `byteToChar`, used when decoding BPE token strings back to UTF-8. -/
def charToByte? (c : Char) : Option UInt8 :=
  charByteTable[c]?

/-- Reversible GPT-2 byte-to-unicode escape for a string fragment. -/
def byteEncode (s : String) : String :=
  String.ofList ((s.toUTF8.toList).map byteToChar)

/-- Decode GPT-2 byte-to-unicode escaped text back into a UTF-8 string. -/
def byteDecode? (s : String) : Option String := do
  let bytes ← s.toList.foldl
    (fun acc? c => do
      let acc ← acc?
      let b ← charToByte? c
      some (acc.push b))
    (some ByteArray.empty)
  String.fromUTF8? bytes

/-! ## Pre-tokenization -/

/-- Character classes used by the GPT-2 pre-tokenizer branches. -/
inductive RegexClass where
  | letter
  | number
  | other
deriving DecidableEq

/-- Character predicate for GPT-2's non-whitespace, non-letter, non-number regex branch. -/
def isRegexOther (c : Char) : Bool :=
  !Unicode.isRegexWhitespace c && !Unicode.isLetter c && !Unicode.isNumber c

/-- Test whether a character belongs to one of the GPT-2 regex classes. -/
def matchesRegexClass (cls : RegexClass) (c : Char) : Bool :=
  match cls with
  | .letter => Unicode.isLetter c
  | .number => Unicode.isNumber c
  | .other => isRegexOther c

/-- Consume one GPT-2 contraction token such as `'s` or `'ll`, if present. -/
def consumeContraction? : List Char → Option (String × List Char)
  | List.cons '\'' (List.cons 's' rest) => some ("'s", rest)
  | List.cons '\'' (List.cons 't' rest) => some ("'t", rest)
  | List.cons '\'' (List.cons 'm' rest) => some ("'m", rest)
  | List.cons '\'' (List.cons 'd' rest) => some ("'d", rest)
  | List.cons '\'' (List.cons 'r' (List.cons 'e' rest)) => some ("'re", rest)
  | List.cons '\'' (List.cons 'v' (List.cons 'e' rest)) => some ("'ve", rest)
  | List.cons '\'' (List.cons 'l' (List.cons 'l' rest)) => some ("'ll", rest)
  | _ => none

/-- Consume one GPT-2 letter/number/other run, allowing a leading ASCII space. -/
def consumeClassRun? (cls : RegexClass) : List Char → Option (String × List Char)
  | List.cons ' ' (List.cons c rest) =>
      if matchesRegexClass cls c then
        let (body, rest') := (c :: rest).span (matchesRegexClass cls)
        some (String.ofList (' ' :: body), rest')
      else
        none
  | List.cons c rest =>
      if matchesRegexClass cls c then
        let (body, rest') := (c :: rest).span (matchesRegexClass cls)
        some (String.ofList body, rest')
      else
        none
  | List.nil => none

/--
Consume the GPT-2 branch `\s+(?!\S)`.

Python's regex engine greedily takes a whitespace run but may backtrack so the negative lookahead
sees either end-of-input or another whitespace character.  For a whitespace run before a non-space
token, this consumes all but the final whitespace; the final ASCII space can then attach to the next
letter/number/punctuation branch, matching GPT-2's standard token boundaries.
-/
def consumeLookaheadWhitespace?
    (xs : List Char) : Option (String × List Char) :=
  let (run, rest) := xs.span Unicode.isRegexWhitespace
  match run, rest with
  | List.nil, _ => none
  | _, List.nil => some (String.ofList run, [])
  | List.cons _ List.nil, _ => none
  | _, _ =>
      match run.reverse with
      | List.nil => none
      | List.cons last revPrefix => some (String.ofList revPrefix.reverse, last :: rest)

/-- Consume a plain whitespace run when the lookahead-sensitive branch did not apply. -/
def consumeWhitespaceRun? (xs : List Char) : Option (String × List Char) :=
  let (run, rest) := xs.span Unicode.isRegexWhitespace
  if run.isEmpty then none else some (String.ofList run, rest)

/--
Fuel-bounded worker for GPT-2 regex pre-token fragments before byte escaping and BPE merges.

The branch order mirrors GPT-2's tokenizer regex exactly: contractions, optional-space letter runs,
optional-space number runs, optional-space non-space/non-letter/non-number runs, whitespace not
followed by non-space, and finally a plain whitespace run. The fuel argument keeps this definition
total; `pretokenize` supplies enough fuel for the whole input.
-/
def pretokenizeWithFuel : Nat → List Char → List String
  | 0, _ => []
  | _fuel + 1, List.nil => []
  | fuel + 1, xs =>
      match consumeContraction? xs with
      | some (tok, rest) => tok :: pretokenizeWithFuel fuel rest
      | none =>
          match consumeClassRun? .letter xs with
          | some (tok, rest) => tok :: pretokenizeWithFuel fuel rest
          | none =>
              match consumeClassRun? .number xs with
              | some (tok, rest) => tok :: pretokenizeWithFuel fuel rest
              | none =>
                  match consumeClassRun? .other xs with
                  | some (tok, rest) => tok :: pretokenizeWithFuel fuel rest
                  | none =>
                      match consumeLookaheadWhitespace? xs with
                      | some (tok, rest) => tok :: pretokenizeWithFuel fuel rest
                      | none =>
                          match consumeWhitespaceRun? xs with
                          | some (tok, rest) => tok :: pretokenizeWithFuel fuel rest
                          | none => []

/-- Split a string into GPT-2-style pre-token fragments. -/
def pretokenize (s : String) : List String :=
  let cs := s.toList
  pretokenizeWithFuel (cs.length + 1) cs

/-! ## BPE Merging -/

/-- Look up a token id in a loaded tokenizer. -/
def tokenId? (tokenizer : Tokenizer) (token : String) : Option Nat :=
  tokenizer.tokenIds[token]?

/-- Look up the token spelling for a token id. -/
def tokenString? (tokenizer : Tokenizer) (id : Nat) : Option String :=
  tokenizer.tokensById[id]?

/-- Look up the merge rank for an adjacent pair of BPE symbols. -/
def mergeRank? (tokenizer : Tokenizer) (left right : String) : Option Nat :=
  tokenizer.mergeRanks[(left, right)]?

/-- Find the lowest-ranked merge currently available in a symbol list. -/
def bestMerge? (tokenizer : Tokenizer) : List String → Option MergeCandidate
  | left :: right :: rest =>
      let here := (mergeRank? tokenizer left right).map fun rank =>
        { left, right, rank }
      let tail := bestMerge? tokenizer (right :: rest)
      match here, tail with
      | none, candidate => candidate
      | candidate, none => candidate
      | some first, some second =>
          if first.rank ≤ second.rank then some first else some second
  | _ => none

/-- Apply one BPE merge everywhere it appears in the current symbol list. -/
def applyMerge (targetLeft targetRight : String) : List String → List String
  | left :: right :: rest =>
      if left == targetLeft && right == targetRight then
        (left ++ right) :: applyMerge targetLeft targetRight rest
      else
        left :: applyMerge targetLeft targetRight (right :: rest)
  | symbols => symbols

/-- Fuel-bounded BPE merge loop for a single escaped pre-token fragment. -/
def bpeLoop (tokenizer : Tokenizer) : Nat → List String → List String
  | 0, symbols => symbols
  | fuel + 1, symbols =>
      match bestMerge? tokenizer symbols with
      | none => symbols
      | some candidate =>
          bpeLoop tokenizer fuel
            (applyMerge candidate.left candidate.right symbols)

/-- Apply BPE to one pre-tokenized fragment. -/
def encodeFragment (tokenizer : Tokenizer) (fragment : String) :
    Except String (Array Nat) := do
  let escaped := byteEncode fragment
  let pieces := bpeLoop tokenizer escaped.length (escaped.toList.map String.singleton)
  let ids ← List.mapM (fun piece =>
    match tokenId? tokenizer piece with
    | some id => pure id
    | none => throw s!"BPE piece is absent from vocabulary: {repr piece}") pieces
  pure ids.toArray

/-- Build the token-to-id lookup table stored in a loaded GPT-2 BPE tokenizer. -/
def tokenIdsFromVocabulary (vocabulary : Array VocabularyEntry) :
    Std.HashMap String Nat :=
  vocabulary.foldl
    (fun tokenIds entry => tokenIds.insert entry.token entry.id)
    Std.HashMap.emptyWithCapacity

/-- Build the id-to-token lookup table stored in a loaded GPT-2 BPE tokenizer. -/
def tokensByIdFromVocabulary (vocabulary : Array VocabularyEntry) :
    Std.HashMap Nat String :=
  vocabulary.foldl
    (fun tokensById entry => tokensById.insert entry.id entry.token)
    Std.HashMap.emptyWithCapacity

/-- Build the pair-to-rank lookup table stored in a loaded GPT-2 BPE tokenizer. -/
def mergeRanksFromMerges (merges : Array MergeRank) :
    Std.HashMap (String × String) Nat :=
  merges.foldl
    (fun mergeRanks merge =>
      mergeRanks.insert (merge.left, merge.right) merge.rank)
    Std.HashMap.emptyWithCapacity

/-- Assemble a tokenizer and its lookup maps from parsed GPT-2 vocabulary and merge tables. -/
def buildTokenizer (vocabulary : Array VocabularyEntry)
    (merges : Array MergeRank) : Tokenizer :=
  { vocabulary
    merges := merges
    tokenIds := tokenIdsFromVocabulary vocabulary
    tokensById := tokensByIdFromVocabulary vocabulary
    mergeRanks := mergeRanksFromMerges merges }

end Internal

/--
Loaded GPT-2 BPE tokenizer.

The representation is intentionally hidden.  In particular, callers cannot construct a tokenizer
whose lookup tables disagree with its vocabulary or merge list; use `load`.
-/
structure Tokenizer where
  private mk ::
  private representation : Internal.Tokenizer
  private isCanonical :
    representation =
      Internal.buildTokenizer representation.vocabulary representation.merges

namespace Tokenizer.Internal

/-- Wrap a canonical GPT-2 BPE representation at the public API boundary. -/
opaque create (representation : GPT2BPE.Internal.Tokenizer)
    (isCanonical :
      representation =
        GPT2BPE.Internal.buildTokenizer
          representation.vocabulary representation.merges) :
    Tokenizer :=
  ⟨representation, isCanonical⟩

/-- Reveal the canonical tokenizer representation only to implementation code. -/
opaque view (tokenizer : Tokenizer) : GPT2BPE.Internal.Tokenizer :=
  match tokenizer with
  | ⟨representation, _⟩ => representation

end Tokenizer.Internal

/-- Number of tokens in a loaded GPT-2 vocabulary. -/
def Tokenizer.vocabularySize (tokenizer : Tokenizer) : Nat :=
  (Tokenizer.Internal.view tokenizer).vocabulary.size

/-- Encode text using the loaded GPT-2 BPE files. -/
def encode (tokenizer : Tokenizer) (text : String) : Except String (Array Nat) := do
  let representation := Tokenizer.Internal.view tokenizer
  let encoded ← (Internal.pretokenize text).mapM (fun fragment =>
    Internal.encodeFragment representation fragment)
  pure (encoded.foldl (· ++ ·) #[])

/-- Decode GPT-2 BPE ids back to text. -/
def decode (tokenizer : Tokenizer) (tokens : Array Nat) : Except String String := do
  let representation := Tokenizer.Internal.view tokenizer
  let escaped ← tokens.toList.mapM (fun token =>
    match Internal.tokenString? representation token with
    | some text => pure text
    | none => throw s!"BPE token id is absent from vocabulary: {token}")
  match Internal.byteDecode? (String.join escaped) with
  | some s => pure s
  | none => throw "BPE decoded bytes were not valid UTF-8"

/-! ## File Loading -/

/-!
The standard GPT-2 `vocab.json` is a single flat JSON object from token strings to numeric ids.
Using Lean's fully general JSON object parser is convenient but slow for interactive examples
because it builds a 50k-entry tree before we immediately flatten it again.  The small parser below
recognizes exactly the JSON shape used by GPT-2 vocab files and decodes JSON string escapes,
including `\uXXXX` escapes for byte-to-unicode code points.
-/

namespace Internal

/-- Read a character for the specialized `vocab.json` parser, using NUL past the input boundary. -/
def charAtOrNull (cs : Array Char) (i : Nat) : Char :=
  cs.getD i '\x00'

/-- Skip JSON whitespace in the specialized GPT-2 vocabulary parser. -/
def skipJsonWs (cs : Array Char) (i : Nat) : Nat :=
  Id.run do
    let mut j := i
    while j < cs.size &&
        (charAtOrNull cs j == ' ' || charAtOrNull cs j == '\n' ||
          charAtOrNull cs j == '\r' || charAtOrNull cs j == '\t') do
      j := j + 1
    return j

/-- Interpret one hexadecimal digit from a JSON unicode escape. -/
def hexVal? (c : Char) : Option Nat :=
  let n := c.toNat
  if 48 ≤ n && n ≤ 57 then
    some (n - 48)
  else if 65 ≤ n && n ≤ 70 then
    some (10 + n - 65)
  else if 97 ≤ n && n ≤ 102 then
    some (10 + n - 97)
  else
    none

/-- Parse four hexadecimal digits starting at `i`. -/
def parseHex4? (cs : Array Char) (i : Nat) : Option Nat := do
  let a ← hexVal? (charAtOrNull cs i)
  let b ← hexVal? (charAtOrNull cs (i + 1))
  let c ← hexVal? (charAtOrNull cs (i + 2))
  let d ← hexVal? (charAtOrNull cs (i + 3))
  some (((a * 16 + b) * 16 + c) * 16 + d)

/-- Combine a JSON UTF-16 surrogate pair into one Unicode code point. -/
def combineSurrogate (hi lo : Nat) : Nat :=
  0x10000 + ((hi - 0xD800) * 0x400) + (lo - 0xDC00)

/-- Fuel-bounded worker for JSON string parsing with escape handling. -/
def parseJsonStringWithFuel (cs : Array Char) : Nat → Nat → List Char →
    Except String (String × Nat)
  | 0, _, _ => throw "vocab.json: string parser exhausted fuel"
  | fuel + 1, i, acc =>
      if i ≥ cs.size then
        throw "vocab.json: unterminated JSON string"
      else
        let c := charAtOrNull cs i
        if c == '"' then
          pure (String.ofList acc.reverse, i + 1)
        else if c == '\\' then
          let j := i + 1
          if j ≥ cs.size then
            throw "vocab.json: unterminated JSON escape"
          else
            match charAtOrNull cs j with
            | '"' => parseJsonStringWithFuel cs fuel (j + 1) ('"' :: acc)
            | '\\' => parseJsonStringWithFuel cs fuel (j + 1) ('\\' :: acc)
            | '/' => parseJsonStringWithFuel cs fuel (j + 1) ('/' :: acc)
            | 'b' => parseJsonStringWithFuel cs fuel (j + 1) ('\x08' :: acc)
            | 'f' => parseJsonStringWithFuel cs fuel (j + 1) ('\x0c' :: acc)
            | 'n' => parseJsonStringWithFuel cs fuel (j + 1) ('\n' :: acc)
            | 'r' => parseJsonStringWithFuel cs fuel (j + 1) ('\r' :: acc)
            | 't' => parseJsonStringWithFuel cs fuel (j + 1) ('\t' :: acc)
            | 'u' =>
                match parseHex4? cs (j + 1) with
                | none => throw "vocab.json: invalid unicode escape"
                | some hi =>
                    let afterHi := j + 5
                    if 0xD800 ≤ hi && hi ≤ 0xDBFF &&
                        afterHi + 5 < cs.size &&
                        charAtOrNull cs afterHi == '\\' && charAtOrNull cs (afterHi + 1) == 'u' then
                      match parseHex4? cs (afterHi + 2) with
                      | some lo =>
                          if 0xDC00 ≤ lo && lo ≤ 0xDFFF then
                            parseJsonStringWithFuel cs fuel (afterHi + 6)
                              (Char.ofNat (combineSurrogate hi lo) :: acc)
                          else
                            throw "vocab.json: invalid low surrogate"
                      | none => throw "vocab.json: invalid low surrogate escape"
                    else if 0xD800 ≤ hi && hi ≤ 0xDFFF then
                      throw "vocab.json: unpaired unicode surrogate"
                    else
                      parseJsonStringWithFuel cs fuel afterHi (Char.ofNat hi :: acc)
            | esc => throw s!"vocab.json: unsupported escape \\{esc}"
        else if c.toNat < 32 then
          throw "vocab.json: unescaped control character in string"
        else
          parseJsonStringWithFuel cs fuel (i + 1) (c :: acc)

/-- Parse a JSON string beginning at index `i`. -/
def parseJsonStringAt (cs : Array Char) (i : Nat) : Except String (String × Nat) := do
  if charAtOrNull cs i != '"' then
    throw "vocab.json: expected JSON string"
  parseJsonStringWithFuel cs (cs.size - i + 1) (i + 1) []

/-- Parse a natural-number literal beginning at index `i`. -/
def parseNatAt (cs : Array Char) (i : Nat) : Except String (Nat × Nat) := do
  let mut j := i
  let mut n := 0
  let mut seen := false
  while j < cs.size do
    let c := charAtOrNull cs j
    if '0' ≤ c && c ≤ '9' then
      n := n * 10 + (c.toNat - '0'.toNat)
      j := j + 1
      seen := true
    else
      break
  if seen then
    if j > i + 1 && charAtOrNull cs i == '0' then
      throw "vocab.json: leading zero in token id"
    else
      pure (n, j)
  else
    throw "vocab.json: expected natural number"

/-- Finish the object only when its closing brace is followed by JSON whitespace. -/
def finishVocabulary (cs : Array Char) (closing : Nat) (entries : Array VocabularyEntry) :
    Except String (Array VocabularyEntry) :=
  if skipJsonWs cs (closing + 1) == cs.size then
    pure entries
  else
    throw "vocab.json: unexpected content after object"

/-- Fuel-bounded loop for the specialized GPT-2 `vocab.json` object parser. -/
def parseVocabularyTextLoop (cs : Array Char) :
    Nat → Nat → Array VocabularyEntry → Except String (Array VocabularyEntry)
  | 0, _, _ => throw "vocab.json: parser exhausted fuel"
  | fuel + 1, i, acc => do
      let i := skipJsonWs cs i
      if i ≥ cs.size then
        throw "vocab.json: unexpected end of file"
      else if charAtOrNull cs i == '}' then
        finishVocabulary cs i acc
      else
        let (tok, i) ← parseJsonStringAt cs i
        let i := skipJsonWs cs i
        if charAtOrNull cs i != ':' then
          throw "vocab.json: expected ':'"
        else
          let i := skipJsonWs cs (i + 1)
          let (id, i) ← parseNatAt cs i
          let i := skipJsonWs cs i
          let acc := acc.push { token := tok, id := id }
          if charAtOrNull cs i == ',' then
            if charAtOrNull cs (skipJsonWs cs (i + 1)) == '}' then
              throw "vocab.json: trailing comma"
            else
              parseVocabularyTextLoop cs fuel (i + 1) acc
          else if charAtOrNull cs i == '}' then
            finishVocabulary cs i acc
          else
            throw "vocab.json: expected ',' or '}'"

/-- Parse GPT-2 `vocab.json`, requiring unique tokens and contiguous token ids from zero. -/
def parseVocabularyText (s : String) : Except String (Array VocabularyEntry) := do
  let cs := s.toList.toArray
  let i := skipJsonWs cs 0
  if charAtOrNull cs i != '{' then
    throw "vocab.json: expected top-level object"
  let vocabulary ← parseVocabularyTextLoop cs (cs.size + 1) (i + 1) #[]
  let mut tokens : Std.HashMap String Nat := {}
  let mut ids : Std.HashMap Nat String := {}
  for entry in vocabulary do
    if tokens.contains entry.token then
      throw s!"vocab.json: duplicate token {repr entry.token}"
    if ids.contains entry.id then
      throw s!"vocab.json: duplicate token id {entry.id}"
    unless entry.id < vocabulary.size do
      throw s!"vocab.json: token id {entry.id} is outside vocabulary size {vocabulary.size}"
    tokens := tokens.insert entry.token entry.id
    ids := ids.insert entry.id entry.token
  pure vocabulary

/-- Parse one `merges.txt` line, skipping the version header and blank lines. -/
def parseMergeLine (rank : Nat) (line : String) : Except String (Option MergeRank) :=
  let s := line.trimAscii.toString
  if s.isEmpty || String.isPrefixOf "#version:" s then
    pure none
  else
    let fields :=
      (s.split fun c => c = ' ' || c = '\t').toList.map (·.toString) |>.filter (· ≠ "")
    match fields with
    | List.cons a (List.cons b List.nil) =>
        pure (some { left := a, right := b, rank := rank })
    | _ => throw
        s!"merges.txt line {rank + 1}: expected two whitespace-separated symbols"

/-- Parse GPT-2 `merges.txt`, retaining hash-prefixed symbols and rejecting malformed pairs. -/
def parseMerges (s : String) : Except String (Array MergeRank) := do
  let lines := s.splitOn "\n"
  let parsed ← (List.zip (List.range lines.length) lines).mapM
    (fun (rank, line) => parseMergeLine rank line)
  pure (parsed.filterMap id).toArray

end Internal

/--
Load GPT-2 BPE files directly in Lean. Vocabulary tokens and ids must be unique, with ids
covering `0 .. vocabularySize - 1`.

Set `progress := true` to print progress for larger `vocab.json` and `merges.txt` assets. The
optional `label` prefixes those messages.
-/
def load (vocabularyFile mergesFile : System.FilePath)
    (progress : Bool := false) (label : String := "GPT2BPE") :
    IO Tokenizer := do
  if progress then
    IO.eprintln
      s!"{label}: loading BPE tokenizer vocabulary={vocabularyFile} merges={mergesFile}"
    IO.eprintln s!"{label}: reading BPE vocab.json"
  let vocabularyText ← IO.FS.readFile vocabularyFile
  if progress then
    IO.eprintln
      s!"{label}: parsing BPE vocab.json chars={vocabularyText.length}"
  let vocabulary ←
    match Internal.parseVocabularyText vocabularyText with
    | .ok parsed => pure parsed
    | .error message => throw <| IO.userError message
  if progress then
    IO.eprintln s!"{label}: parsed BPE vocabulary entries={vocabulary.size}"
    IO.eprintln s!"{label}: reading BPE merges.txt"
  let mergesText ← IO.FS.readFile mergesFile
  let merges ←
    match Internal.parseMerges mergesText with
    | .ok parsed => pure parsed
    | .error message => throw <| IO.userError message
  if progress then
    IO.eprintln s!"{label}: parsed BPE merges={merges.size}"
    IO.eprintln s!"{label}: building BPE lookup maps"
  let representation := Internal.buildTokenizer vocabulary merges
  if progress then
    IO.eprintln <|
      s!"{label}: loaded BPE tokenizer vocabulary={representation.vocabulary.size} " ++
      s!"merges={representation.merges.size}"
  pure (Tokenizer.Internal.create representation rfl)

end GPT2BPE

end text
end TorchLean
