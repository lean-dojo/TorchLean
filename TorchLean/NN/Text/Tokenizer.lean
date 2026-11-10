/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Sample
public import TorchLean.Tensor.Random
public import Std.Data.HashMap.Basic
public import TorchLean.Tensor.Operations

/-!
# Tokenizers and Text Tensors

Byte and character tokenizers, fixed-length token windows, corpus readers and deterministic window
sampling for the text examples.

Language models can keep token ids as `Nat` tensors and gather embedding rows from them, and small
examples can use one-hot tensors of shape `(batchSize × sequenceLength × vocabularySize)`. Either
way the tokens stay separate from the floating-point model parameters at the API boundary.
Next-token samples are built in `TorchLean.NN.Data.CausalLM`, and generation and logit selection
live in `TorchLean.NN.Text.Generation`.
-/

@[expose] public section

namespace TorchLean
namespace Text

open TorchLean.Tensor

-- Runtime lookups use Lean's `Char` hash, because Mathlib's instance is meta-only.
attribute [local instance 2000] instHashableChar

/-! ## Tokenizers -/

/-- A tokenizer: its vocabulary size and an `encode`/`decode` pair between strings and token ids. -/
structure Tokenizer where
  /-- Vocabulary size (token ids are expected to be in `[0, vocabularySize)`). -/
  vocabularySize : Nat
  /-- Encode a string into a variable-length token buffer. -/
  encode : String → Array Nat
  /-- Decode a variable-length token buffer back into a string. -/
  decode : Array Nat → String

namespace Tokenizer

/--
Byte-level UTF-8 tokenizer: each byte is one token in $[0,256)$.

Decoding reduces ids modulo 256 and reads the bytes as UTF-8. A generated stream that is not valid
UTF-8 decodes byte by byte, each byte becoming the character with that code point.

Example:
```lean
-- 256 tokens, with no vocabulary file and nothing to train.
def tokenizer : Text.Tokenizer := Text.Tokenizer.byte

-- A Lean string is valid UTF-8, so decoding its encoding gives back `line`.
def roundTrip (line : String) : String :=
  tokenizer.decode (tokenizer.encode line)
```
-/
def byte : Tokenizer :=
  let decode := fun (ids : Array Nat) =>
    let bytes :=
      ids.foldl (fun acc n => acc.push (UInt8.ofNat (n % 256))) ByteArray.empty
    match String.fromUTF8? bytes with
    | some s => s
    | none => String.ofList (ids.toList.map (fun n => Char.ofNat (n % 256)))
  { vocabularySize := 256
    encode := fun s => s.toByteArray.data.map (fun b => b.toNat)
    decode }

/--
Character-level tokenizer over `alphabet`: `encode` maps each character to its index and `decode`
maps indices back, like the `stoi`/`itos` tables of character-level GPT examples.

A character outside the alphabet encodes to `unknownTokenId`, whose type `Fin alphabet.size` also
proves the alphabet nonempty, and an id at or beyond `alphabet.size` decodes to
`unknownCharacter`. A repeated character encodes to its first index, and the returned encoder
shares one lookup table across calls.

Example:
```lean
-- Ids 0 to 3 stand for 'a', 'b', 'c' and ' '. Any other character encodes to 3, and ids from 4
-- up decode to '?'.
def alphabet : Array Char := #['a', 'b', 'c', ' ']

def tokenizer : Text.Tokenizer :=
  Text.Tokenizer.fromAlphabet alphabet ⟨3, by decide⟩ (unknownCharacter := '?')
```
-/
def fromAlphabet
    (alphabet : Array Char)
    (unknownTokenId : Fin alphabet.size)
    (unknownCharacter : Char := '?') :
    Tokenizer :=
  let vocabularySize := alphabet.size
  let characterIds := (alphabet.foldl
    (fun (state : Nat × Std.HashMap Char Nat) character =>
      let (index, ids) := state
      (index + 1, if ids.contains character then ids else ids.insert character index))
    (0, {})).2
  { vocabularySize := vocabularySize
    encode := fun s =>
      s.toList.toArray.map (fun c =>
        (characterIds[c]?).getD unknownTokenId.val)
    decode := fun ids =>
      String.ofList <|
        ids.toList.map (fun n =>
          alphabet.getD n unknownCharacter) }

end Tokenizer

/-! ## Byte-Corpus Windows -/

/--
Read `length` byte tokens from a raw corpus, starting at byte `offset` and padded with
`paddingTokenId` past the end.

Offsets count bytes, as byte-level causal language modeling needs, so a window can start or end
inside a multi-byte UTF-8 character.
-/
def byteTokenWindow
    (bytes : ByteArray)
    (length : Nat)
    (offset : Nat := 0)
    (paddingTokenId : Nat := 0) :
    Tensor Nat [length] :=
  TorchLean.Tensor.ofFn fun index =>
    match bytes[offset + index.val]? with
    | some byte => byte.toNat
    | none => paddingTokenId

/-! ## Corpus Helpers -/

namespace Corpus

/--
Read a UTF-8 text file, failing when it is missing or empty.

Errors start with `exeName`, and the error for a missing file ends with `missingHint`, which the
examples set to the download or conversion command for their dataset.
-/
def readUTF8File (exeName : String) (path : System.FilePath) (missingHint : String) :
    IO String := do
  if !(← path.pathExists) then
    throw <| IO.userError s!"{exeName}: dataset file not found: {path}\n{missingHint}"
  let text ← IO.FS.readFile path
  if text.isEmpty then
    throw <| IO.userError s!"{exeName}: dataset file is empty: {path}"
  pure text

/--
Read a raw byte corpus, failing when the file is missing, when it has at most `sequenceLength`
bytes or, unless `allowSmallData` is set, when it has fewer than `minimumBytes`.

Corpus trainers set `minimumBytes` to the scale they expect, so a smaller local file needs
`--allow-small-data`.
-/
def readByteFile
    (exeName : String) (path : System.FilePath) (allowSmallData : Bool)
    (minimumBytes sequenceLength : Nat) : IO ByteArray := do
  if !(← path.pathExists) then
    throw <| IO.userError s!"{exeName}: dataset file not found: {path}"
  let bytes ← IO.FS.readBinFile path
  if bytes.size <= sequenceLength then
    throw <| IO.userError
      s!"{exeName}: dataset is too small for a {sequenceLength}-token window"
  if !allowSmallData && bytes.size < minimumBytes then
    throw <| IO.userError (
      s!"{exeName}: corpus is {bytes.size} bytes; real GPU training expects at least " ++
      s!"{minimumBytes} bytes. For bounded local runs, pass --allow-small-data.")
  pure bytes

/--
Parse a text-corpus flag set and return `(text, remainingArgs)`.

Supported forms:
- `--data-file PATH`
- any named alias in `aliases`, such as `("--tiny-shakespeare", path)`
- no data flag, which uses `defaultPath`
-/
def takeUTF8Input
    (exeName : String) (defaultPath : System.FilePath)
    (aliases : List (String × System.FilePath)) (missingHint : String)
    (arguments : List String) : IO (String × List String) :=
  match arguments with
  | List.nil => do
      let text ← readUTF8File exeName defaultPath missingHint
      pure (text, [])
  | List.cons "--data-file" (List.cons path remainingArgs) => do
      let text ← readUTF8File exeName path missingHint
      pure (text, remainingArgs)
  | List.cons "--data-file" List.nil =>
      throw <| IO.userError s!"{exeName}: --data-file expects a path"
  | List.cons argument remainingArgs => do
      match aliases.find? (fun (aliasName, _) => aliasName = argument) with
      | some (_, path) => do
          let text ← readUTF8File exeName path missingHint
          pure (text, remainingArgs)
      | none => do
          let (text, unconsumedArgs) ←
            takeUTF8Input exeName defaultPath aliases missingHint remainingArgs
          pure (text, argument :: unconsumedArgs)

/--
Number of start positions for a next-token window of `sequenceLength + 1` tokens, at least one.

The floor of one gives the offset helpers below a start to pick even on a corpus shorter than one
window. Callers that need a minimum corpus size check it before training.
-/
def usableTokenStarts (tokenCount sequenceLength : Nat) : Nat :=
  Nat.max 1 (tokenCount - sequenceLength)

/-- Start of the `index`-th sliding window over a byte corpus, wrapping around after the last
usable start. -/
def byteOffset (bytes : ByteArray) (index sequenceLength : Nat) : Nat :=
  index % usableTokenStarts bytes.size sequenceLength

/-- Choose deterministic, approximately evenly spaced starts for fixed-width token windows. -/
def evenlySpacedOffsets (tokenCount sequenceLength windowCount : Nat) : Tensor Nat [windowCount] :=
  let usable := usableTokenStarts tokenCount sequenceLength
  let stride := Nat.max 1 (usable / Nat.max 1 windowCount)
  Tensor.ofFn fun index => (index.val * stride) % usable

/--
Deterministic minGPT-style random start offsets for one training batch, one per row.

We derive the key from `(seed, step)` with `Random.keyOf` and draw each row's offset by its
row index, so a run is reproducible without IO randomness.
-/
def randomBatchOffsets
    (tokenCount sequenceLength batchSize seed step : Nat) :
    Tensor Nat [batchSize] :=
  let usable := usableTokenStarts tokenCount sequenceLength
  let key : UInt64 := Random.keyOf seed step
  Tensor.ofFn fun batchIndex => Random.sampleNat key batchIndex.val usable

/--
One deterministic random text batch: row `i` holds the `sequenceLength + 1` tokens starting at the
`i`-th offset of `randomBatchOffsets`, padded with `paddingTokenId` past the end.

The extra token lets the causal-LM helpers cut the input and its shifted target from one row, and
byte, character, BPE and synthetic tokens are batched the same way.
-/
def randomTokenBatch {β : Type} [TorchLean.Storage β] {tokenCount : Nat}
    (tokens : Tensor β [tokenCount])
    (batchSize sequenceLength seed step : Nat)
    (paddingTokenId : β) :
    Tensor β [batchSize, sequenceLength + 1] :=
  let offsetAt :=
    randomBatchOffsets tokenCount sequenceLength batchSize seed step
  TorchLean.Tensor.stack 0 fun batchIndex =>
    Tensor.window tokens (sequenceLength + 1) offsetAt[batchIndex] paddingTokenId

end Corpus

end Text
end TorchLean
