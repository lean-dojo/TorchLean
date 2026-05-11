/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Json
public import NN.API.Text
public import NN.API.Text.Unicode
public import Std.Data.HashMap

/-!
# GPT-2 Byte-Pair Encoding

Lean-native support for GPT-2-style byte-level BPE tokenizers.

This module is intentionally in `NN.API.Text` rather than a model file: any Transformer, diffusion
LM, or verifier that wants GPT-2-compatible tokenization should share the same implementation.
The implementation parses the standard `vocab.json` and `merges.txt` files directly in Lean.

The pre-tokenizer implements the GPT-2 regex shape:

`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

The Unicode `\p{L}`, `\p{N}`, and `\s` predicates are supplied by `NN.API.Text.Unicode`, rather
than Lean's ASCII-oriented `Char.isAlpha` / `Char.isDigit` helpers.
-/

@[expose] public section

namespace NN
namespace API
namespace text
namespace Gpt2Bpe

open Lean

/-! ## Data -/

/-- One token-to-id entry from GPT-2's `vocab.json`. -/
structure VocabEntry where
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
  vocab : Array VocabEntry
  /-- Ranked merge table from `merges.txt`. -/
  merges : Array MergeRank
  /-- Fast token-to-id lookup derived from `vocab`. -/
  vocabMap : Std.HashMap String Nat := Std.HashMap.emptyWithCapacity
  /-- Fast id-to-token lookup derived from `vocab`. -/
  idMap : Std.HashMap Nat String := Std.HashMap.emptyWithCapacity
  /-- Fast pair-to-rank lookup derived from `merges`. -/
  mergeMap : Std.HashMap (String × String) Nat := Std.HashMap.emptyWithCapacity

/-! ## Byte Escaping -/

def bytesVisible : List Nat :=
  (List.range 94).map (fun i => i + 33)

def bytesLatin1A : List Nat :=
  (List.range 12).map (fun i => i + 161)

def bytesLatin1B : List Nat :=
  (List.range 82).map (fun i => i + 174)

def baseBytes : List Nat :=
  bytesVisible ++ bytesLatin1A ++ bytesLatin1B

def containsNat (xs : List Nat) (x : Nat) : Bool :=
  xs.any (fun y => y == x)

def byteCodeTable : Array Nat := Id.run do
  let mut codes := Array.replicate 256 0
  for b in baseBytes do
    codes := codes.set! b b
  let mut next := 0
  for b in List.range 256 do
    if !containsNat baseBytes b then
      codes := codes.set! b (256 + next)
      next := next + 1
  return codes

/-- GPT-2 byte-to-unicode escape for one byte. -/
def byteToChar (b : UInt8) : Char :=
  Char.ofNat (Array.getD byteCodeTable b.toNat b.toNat)

/-- Inverse of `byteToChar`, used when decoding BPE token strings back to UTF-8. -/
def charToByte? (c : Char) : Option UInt8 :=
  (List.range 256).findSome? (fun b =>
    if Char.ofNat (Array.getD byteCodeTable b b) = c then
      some (UInt8.ofNat b)
    else
      none)

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

inductive RegexClass where
  | letter
  | number
  | other
deriving DecidableEq

def isRegexOther (c : Char) : Bool :=
  !Unicode.isRegexWhitespace c && !Unicode.isLetter c && !Unicode.isNumber c

def matchesRegexClass (cls : RegexClass) (c : Char) : Bool :=
  match cls with
  | .letter => Unicode.isLetter c
  | .number => Unicode.isNumber c
  | .other => isRegexOther c

def takeWhileChars (p : Char → Bool) : List Char → List Char × List Char
  | [] => ([], [])
  | c :: cs =>
      if p c then
        let (pre, rest) := takeWhileChars p cs
        (c :: pre, rest)
      else
        ([], c :: cs)

def consumeContraction? : List Char → Option (String × List Char)
  | '\'' :: 's' :: rest => some ("'s", rest)
  | '\'' :: 't' :: rest => some ("'t", rest)
  | '\'' :: 'm' :: rest => some ("'m", rest)
  | '\'' :: 'd' :: rest => some ("'d", rest)
  | '\'' :: 'r' :: 'e' :: rest => some ("'re", rest)
  | '\'' :: 'v' :: 'e' :: rest => some ("'ve", rest)
  | '\'' :: 'l' :: 'l' :: rest => some ("'ll", rest)
  | _ => none

def consumeClassRun? (cls : RegexClass) : List Char → Option (String × List Char)
  | ' ' :: c :: rest =>
      if matchesRegexClass cls c then
        let (body, rest') := takeWhileChars (matchesRegexClass cls) (c :: rest)
        some (String.ofList (' ' :: body), rest')
      else
        none
  | c :: rest =>
      if matchesRegexClass cls c then
        let (body, rest') := takeWhileChars (matchesRegexClass cls) (c :: rest)
        some (String.ofList body, rest')
      else
        none
  | [] => none

/--
Consume the GPT-2 branch `\s+(?!\S)`.

Python's regex engine greedily takes a whitespace run but may backtrack so the negative lookahead
sees either end-of-input or another whitespace character.  For a whitespace run before a non-space
token, this consumes all but the final whitespace; the final ASCII space can then attach to the next
letter/number/punctuation branch, matching GPT-2's standard token boundaries.
-/
def consumeWhitespaceNotFollowedByNonspace? (xs : List Char) : Option (String × List Char) :=
  let (run, rest) := takeWhileChars Unicode.isRegexWhitespace xs
  match run, rest with
  | [], _ => none
  | _, [] => some (String.ofList run, [])
  | [_], _ => none
  | _, _ =>
      match run.reverse with
      | [] => none
      | last :: revPrefix => some (String.ofList revPrefix.reverse, last :: rest)

def consumeWhitespaceRun? (xs : List Char) : Option (String × List Char) :=
  let (run, rest) := takeWhileChars Unicode.isRegexWhitespace xs
  if run.isEmpty then none else some (String.ofList run, rest)

/--
Fuel-bounded worker for GPT-2 regex pre-token fragments before byte escaping and BPE merges.

The branch order mirrors GPT-2's tokenizer regex exactly: contractions, optional-space letter runs,
optional-space number runs, optional-space non-space/non-letter/non-number runs, whitespace not
followed by non-space, and finally a plain whitespace run. The fuel argument keeps this definition
total and should never be exhausted when called by `pretokenize`.
-/
def pretokenizeAux : Nat → List Char → List String
  | 0, _ => []
  | _fuel + 1, [] => []
  | fuel + 1, xs =>
      match consumeContraction? xs with
      | some (tok, rest) => tok :: pretokenizeAux fuel rest
      | none =>
          match consumeClassRun? .letter xs with
          | some (tok, rest) => tok :: pretokenizeAux fuel rest
          | none =>
              match consumeClassRun? .number xs with
              | some (tok, rest) => tok :: pretokenizeAux fuel rest
              | none =>
                  match consumeClassRun? .other xs with
                  | some (tok, rest) => tok :: pretokenizeAux fuel rest
                  | none =>
                      match consumeWhitespaceNotFollowedByNonspace? xs with
                      | some (tok, rest) => tok :: pretokenizeAux fuel rest
                      | none =>
                          match consumeWhitespaceRun? xs with
                          | some (tok, rest) => tok :: pretokenizeAux fuel rest
                          | none => []

/-- Split a string into GPT-2-style pre-token fragments. -/
def pretokenize (s : String) : List String :=
  let cs := s.toList
  pretokenizeAux (cs.length + 1) cs

/-! ## BPE Merging -/

def vocabId? (tok : Tokenizer) (s : String) : Option Nat :=
  tok.vocabMap[s]?

def tokenString? (tok : Tokenizer) (id : Nat) : Option String :=
  tok.idMap[id]?

def mergeRank? (tok : Tokenizer) (a b : String) : Option Nat :=
  tok.mergeMap[(a, b)]?

def bestMerge? (tok : Tokenizer) : List String → Option (String × String × Nat)
  | a :: b :: rest =>
      let here := (mergeRank? tok a b).map (fun r => (a, b, r))
      let tail := bestMerge? tok (b :: rest)
      match here, tail with
      | none, t => t
      | h, none => h
      | some x, some y => if x.2.2 ≤ y.2.2 then some x else some y
  | _ => none

def applyMerge (target : String × String) : List String → List String
  | a :: b :: rest =>
      if a == target.1 && b == target.2 then
        (a ++ b) :: applyMerge target rest
      else
        a :: applyMerge target (b :: rest)
  | xs => xs

def bpeLoop (tok : Tokenizer) : Nat → List String → List String
  | 0, symbols => symbols
  | fuel + 1, symbols =>
      match bestMerge? tok symbols with
      | none => symbols
      | some (a, b, _) => bpeLoop tok fuel (applyMerge (a, b) symbols)

/-- Apply BPE to one pre-tokenized fragment. -/
def encodeFragment (tok : Tokenizer) (fragment : String) : Except String (List Nat) := do
  let escaped := byteEncode fragment
  let pieces := bpeLoop tok escaped.length (escaped.toList.map String.singleton)
  List.mapM (fun p =>
    match vocabId? tok p with
    | some id => pure id
    | none => NN.API.Json.fail s!"BPE piece is absent from vocab: {repr p}") pieces

/-- Encode text using the loaded GPT-2 BPE files. -/
def encode (tok : Tokenizer) (text : String) : Except String (List Nat) := do
  let encoded ← (pretokenize text).mapM (fun fragment => encodeFragment tok fragment)
  pure (encoded.foldr (· ++ ·) [])

/-- Decode GPT-2 BPE ids back to text. -/
def decode? (tok : Tokenizer) (ids : List Nat) : Except String String := do
  let escaped ← ids.mapM (fun id =>
    match tokenString? tok id with
    | some s => pure s
    | none => NN.API.Json.fail s!"BPE token id is absent from vocab: {id}")
  match byteDecode? (String.join escaped) with
  | some s => pure s
  | none => NN.API.Json.fail "BPE decoded bytes were not valid UTF-8"

/-- Total display-oriented decoder: invalid ids/UTF-8 decode to an empty string. -/
def decodeD (tok : Tokenizer) (ids : List Nat) : String :=
  match decode? tok ids with
  | .ok s => s
  | .error _ => ""

/-- Adapt a loaded GPT-2 BPE tokenizer to the generic text-tokenizer interface. -/
def asTextTokenizer (tok : Tokenizer) : NN.API.text.Tokenizer where
  vocabSize := tok.vocab.size
  encode := fun s =>
    match encode tok s with
    | .ok ids => ids
    | .error _ => []
  decode := decodeD tok

/-! ## File Loading -/

/-- Parse GPT-2 `vocab.json` as an array of `(token, id)` entries. -/
def parseVocab (j : Json) : Except String (Array VocabEntry) := do
  let o ← NN.API.Json.expectObjE "vocab.json" j
  let entries := Std.TreeMap.Raw.toList o
  entries.toArray.mapM (fun (tok, idJ) => do
    let id ← NN.API.Json.expectNatE s!"vocab id for {repr tok}" idJ
    pure { token := tok, id := id })

/-!
The standard GPT-2 `vocab.json` is a single flat JSON object from token strings to numeric ids.
Using Lean's fully general JSON object parser is convenient but slow for interactive examples
because it builds a 50k-entry tree before we immediately flatten it again.  The small parser below
recognizes exactly the JSON shape used by GPT-2 vocab files and decodes JSON string escapes,
including `\uXXXX` escapes for byte-to-unicode code points.
-/

def charAtD (cs : Array Char) (i : Nat) : Char :=
  cs.getD i '\x00'

def skipJsonWs (cs : Array Char) (i : Nat) : Nat :=
  Id.run do
    let mut j := i
    while j < cs.size &&
        (charAtD cs j == ' ' || charAtD cs j == '\n' ||
          charAtD cs j == '\r' || charAtD cs j == '\t') do
      j := j + 1
    return j

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

def parseHex4? (cs : Array Char) (i : Nat) : Option Nat := do
  let a ← hexVal? (charAtD cs i)
  let b ← hexVal? (charAtD cs (i + 1))
  let c ← hexVal? (charAtD cs (i + 2))
  let d ← hexVal? (charAtD cs (i + 3))
  some (((a * 16 + b) * 16 + c) * 16 + d)

def combineSurrogate (hi lo : Nat) : Nat :=
  0x10000 + ((hi - 0xD800) * 0x400) + (lo - 0xDC00)

def parseJsonStringAux (cs : Array Char) : Nat → Nat → List Char →
    Except String (String × Nat)
  | 0, _, _ => NN.API.Json.fail "vocab.json: string parser exhausted fuel"
  | fuel + 1, i, acc =>
      if i ≥ cs.size then
        NN.API.Json.fail "vocab.json: unterminated JSON string"
      else
        let c := charAtD cs i
        if c == '"' then
          pure (String.ofList acc.reverse, i + 1)
        else if c == '\\' then
          let j := i + 1
          if j ≥ cs.size then
            NN.API.Json.fail "vocab.json: unterminated JSON escape"
          else
            match charAtD cs j with
            | '"' => parseJsonStringAux cs fuel (j + 1) ('"' :: acc)
            | '\\' => parseJsonStringAux cs fuel (j + 1) ('\\' :: acc)
            | '/' => parseJsonStringAux cs fuel (j + 1) ('/' :: acc)
            | 'b' => parseJsonStringAux cs fuel (j + 1) ('\x08' :: acc)
            | 'f' => parseJsonStringAux cs fuel (j + 1) ('\x0c' :: acc)
            | 'n' => parseJsonStringAux cs fuel (j + 1) ('\n' :: acc)
            | 'r' => parseJsonStringAux cs fuel (j + 1) ('\r' :: acc)
            | 't' => parseJsonStringAux cs fuel (j + 1) ('\t' :: acc)
            | 'u' =>
                match parseHex4? cs (j + 1) with
                | none => NN.API.Json.fail "vocab.json: invalid unicode escape"
                | some hi =>
                    let afterHi := j + 5
                    if 0xD800 ≤ hi && hi ≤ 0xDBFF &&
                        afterHi + 5 < cs.size &&
                        charAtD cs afterHi == '\\' && charAtD cs (afterHi + 1) == 'u' then
                      match parseHex4? cs (afterHi + 2) with
                      | some lo =>
                          if 0xDC00 ≤ lo && lo ≤ 0xDFFF then
                            parseJsonStringAux cs fuel (afterHi + 6)
                              (Char.ofNat (combineSurrogate hi lo) :: acc)
                          else
                            NN.API.Json.fail "vocab.json: invalid low surrogate"
                      | none => NN.API.Json.fail "vocab.json: invalid low surrogate escape"
                    else
                      parseJsonStringAux cs fuel afterHi (Char.ofNat hi :: acc)
            | esc => NN.API.Json.fail s!"vocab.json: unsupported escape \\{esc}"
        else
          parseJsonStringAux cs fuel (i + 1) (c :: acc)

def parseJsonStringAt (cs : Array Char) (i : Nat) : Except String (String × Nat) := do
  if charAtD cs i != '"' then
    NN.API.Json.fail "vocab.json: expected JSON string"
  parseJsonStringAux cs (cs.size - i + 1) (i + 1) []

def parseNatAt (cs : Array Char) (i : Nat) : Except String (Nat × Nat) := do
  let mut j := i
  let mut n := 0
  let mut seen := false
  while j < cs.size do
    let c := charAtD cs j
    if '0' ≤ c && c ≤ '9' then
      n := n * 10 + (c.toNat - '0'.toNat)
      j := j + 1
      seen := true
    else
      break
  if seen then
    pure (n, j)
  else
    NN.API.Json.fail "vocab.json: expected natural number"

def parseVocabTextLoop (cs : Array Char) : Nat → Nat → Array VocabEntry →
    Except String (Array VocabEntry)
  | 0, _, _ => NN.API.Json.fail "vocab.json: parser exhausted fuel"
  | fuel + 1, i, acc => do
      let i := skipJsonWs cs i
      if i ≥ cs.size then
        NN.API.Json.fail "vocab.json: unexpected end of file"
      else if charAtD cs i == '}' then
        pure acc
      else
        let (tok, i) ← parseJsonStringAt cs i
        let i := skipJsonWs cs i
        if charAtD cs i != ':' then
          NN.API.Json.fail "vocab.json: expected ':'"
        else
          let i := skipJsonWs cs (i + 1)
          let (id, i) ← parseNatAt cs i
          let i := skipJsonWs cs i
          let acc := acc.push { token := tok, id := id }
          if charAtD cs i == ',' then
            parseVocabTextLoop cs fuel (i + 1) acc
          else if charAtD cs i == '}' then
            pure acc
          else
            NN.API.Json.fail "vocab.json: expected ',' or '}'"

/-- Parse GPT-2 `vocab.json` directly from text. -/
def parseVocabText (s : String) : Except String (Array VocabEntry) := do
  let cs := s.toList.toArray
  let i := skipJsonWs cs 0
  if charAtD cs i != '{' then
    NN.API.Json.fail "vocab.json: expected top-level object"
  parseVocabTextLoop cs (cs.size + 1) (i + 1) #[]

def vocabMapOf (vocab : Array VocabEntry) : Std.HashMap String Nat :=
  vocab.foldl (fun acc e => acc.insert e.token e.id) Std.HashMap.emptyWithCapacity

def idMapOf (vocab : Array VocabEntry) : Std.HashMap Nat String :=
  vocab.foldl (fun acc e => acc.insert e.id e.token) Std.HashMap.emptyWithCapacity

def mergeMapOf (merges : Array MergeRank) : Std.HashMap (String × String) Nat :=
  merges.foldl (fun acc m => acc.insert (m.left, m.right) m.rank) Std.HashMap.emptyWithCapacity

def parseMergeLine? (rank : Nat) (line : String) : Option MergeRank :=
  let s := line.trimAscii.toString
  if s.isEmpty || String.isPrefixOf "#" s then
    none
  else
    match s.splitOn " " with
    | [a, b] => some { left := a, right := b, rank := rank }
    | _ => none

/-- Parse GPT-2 `merges.txt`. Invalid non-comment lines are ignored conservatively. -/
def parseMerges (s : String) : Array MergeRank :=
  let lines := s.splitOn "\n"
  (List.zip (List.range lines.length) lines).foldl
    (fun acc p =>
      match parseMergeLine? p.1 p.2 with
      | some m => acc.push m
      | none => acc)
    #[]

/-- Load GPT-2 BPE files directly in Lean. -/
def load (vocabJson mergesTxt : System.FilePath) : IO Tokenizer := do
  let vocab ←
    match parseVocabText (← IO.FS.readFile vocabJson) with
    | .ok v => pure v
    | .error e => throw <| IO.userError e
  let merges ← parseMerges <$> IO.FS.readFile mergesTxt
  pure { vocab := vocab
         merges := merges
         vocabMap := vocabMapOf vocab
         idMap := idMapOf vocab
         mergeMap := mergeMapOf merges }

end Gpt2Bpe
end text
end API
end NN
