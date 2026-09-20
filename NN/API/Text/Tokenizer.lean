/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.Tensor
public import NN.API.Sample
public import NN.Runtime.Autograd.Model.Metrics
public import NN.Spec.Core.Random
public import Std.Data.HashMap.Basic

/-!
# Tokenizers and Text Tensors

Text and NLP helpers for TorchLean examples.

Language models may keep token ids as `Nat` tensors and gather embedding rows directly. Small
examples can instead use one-hot tensors of shape
`(batchSize × sequenceLength × vocabularySize)`. Both representations
remain separate from floating-point model parameters at the API boundary.

This module provides:
- a tokenizer interface (with a byte-level tokenizer),
- helpers to turn token streams into one-hot tensors,
- “next-token prediction” sample builders used by GPT-style examples,
- display helpers for turning model logits back into readable token predictions.
-/

@[expose] public section

namespace TorchLean
namespace text

open Spec TorchLean.Tensor

/-! ## Tokenizers -/

/-- Tokenizer interface (encode/decode). -/
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

Decoding uses UTF-8 when possible and falls back to byte-wise display for generated streams that
are not valid UTF-8.

Example:
```lean
-- 256 tokens, no vocabulary file, nothing to train: this is where every text example starts.
def tokenizer : text.Tokenizer := text.Tokenizer.byte

-- Encode then decode returns the original string whenever it was valid UTF-8.
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
Build a character-level tokenizer from an explicit alphabet.

The resulting `encode`/`decode` pair has the same role as the `stoi`/`itos` tables used in
character-level GPT examples: `encode` maps characters to ids
`0..alphabet.size-1`, and `decode` maps ids back to characters.

The `unknownTokenId` argument proves that the alphabet is nonempty and identifies the token used for
a character outside the alphabet. Ids outside `[0, alphabet.size)` decode to `unknownCharacter`.
Repeated characters in the alphabet encode to their first index. A lookup table is shared across
calls to the returned encoder.

Example:
```lean
-- The `stoi` and `itos` tables of character-level GPT tutorials, with the nonempty-alphabet
-- requirement carried by the unknown-token index instead of a runtime assertion.
def alphabet : Array Char := #['a', 'b', 'c', ' ']

def tokenizer : text.Tokenizer :=
  text.Tokenizer.fromAlphabet alphabet ⟨3, by decide⟩ (unknownCharacter := '?')
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

/--
Encode a string and pad or truncate it to exactly `sequenceLength` token ids.

Example:
```lean
-- Padded or truncated to the length the model expects, so the result carries a shape rather than
-- a length a caller has to check.
def tokens : Tensor Nat [16] :=
  text.Tokenizer.byte.encodeFixed 16 "hello world"
```
-/
def encodeFixed
    (tokenizer : Tokenizer)
    (sequenceLength : Nat)
    (text : String)
    (paddingTokenId : Nat := 0) :
    Tensor Nat [sequenceLength] :=
  let tokens := tokenizer.encode text
  TorchLean.Tensor.ofFn (fun index => tokens.getD index.val paddingTokenId)

/--
Encode exactly `batchSize` strings, padding or truncating each row to
`sequenceLength` token ids.

Example:
```lean
-- Two prompts become one `[2, 16]` batch, ready for a model with a batch axis in front.
def batch : Tensor Nat [2, 16] :=
  text.Tokenizer.byte.encodeFixedBatch 16 ["hello", "world"]
```
-/
def encodeFixedBatch
    {batchSize : Nat}
    (tokenizer : Tokenizer)
    (sequenceLength : Nat)
    (texts : Tensor String [batchSize])
    (paddingTokenId : Nat := 0) :
    Tensor Nat [batchSize, sequenceLength] :=
  TorchLean.Tensor.stack 0 fun batchIndex =>
    encodeFixed tokenizer sequenceLength texts[batchIndex] paddingTokenId

end Tokenizer

/-! ## Byte-Corpus Windows -/

namespace Internal

/--
Read one byte token from a raw corpus, returning `paddingTokenId` past the end.

This is byte-level rather than BPE-level: examples can train causal language models directly from a
text file without depending on an external tokenizer artifact. GPT-2 BPE support lives in
`NN.API.Text.Bpe`.

Lives in `Internal` on purpose: `byteTokenWindow` below is the only caller, and a padded
single-byte read is not something a user of `text` should have to reason about. (`private` is not an
option here. Every API module is inside `@[expose] public section`, so a private helper cannot be
named from an exposed body; a nested `Internal` namespace is how the rest of the codebase says
"plumbing".)
-/
def byteAtOrPad (bytes : ByteArray) (index : Nat) (paddingTokenId : Nat := 0) : Nat :=
  match bytes[index]? with
  | some b => b.toNat
  | none => paddingTokenId

end Internal

/--
Extract a fixed-length byte-token window from a raw corpus.

`offset` is measured in bytes, as required for byte-level causal language modeling. This avoids
hidden UTF-8 slicing assumptions.
-/
def byteTokenWindow
    (bytes : ByteArray)
    (length : Nat)
    (offset : Nat := 0)
    (paddingTokenId : Nat := 0) :
    Tensor Nat [length] :=
  TorchLean.Tensor.ofFn fun index =>
    Internal.byteAtOrPad bytes (offset + index.val) paddingTokenId

/-! ## Corpus Helpers -/

namespace Corpus

/--
Read a UTF-8 text file with a caller-supplied preparation hint.

The examples pass their executable name and a concrete hint so failures point users to the exact
download or conversion command for that dataset.
-/
def readUtf8File (exeName : String) (path : System.FilePath) (missingHint : String) :
    IO String := do
  if !(← path.pathExists) then
    throw <| IO.userError s!"{exeName}: dataset file not found: {path}\n{missingHint}"
  let text ← IO.FS.readFile path
  if text.isEmpty then
    throw <| IO.userError s!"{exeName}: dataset file is empty: {path}"
  pure text

/--
Read a raw byte corpus and optionally enforce a minimum size.

`allowSmallData` is an explicit override for bounded local runs. Corpus-training commands can set
`minimumBytes` to the scale they expect and require users to acknowledge smaller local files.
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
def takeUtf8Input
    (exeName : String) (defaultPath : System.FilePath)
    (aliases : List (String × System.FilePath)) (missingHint : String)
    (arguments : List String) : IO (String × List String) :=
  match arguments with
  | List.nil => do
      let text ← readUtf8File exeName defaultPath missingHint
      pure (text, [])
  | List.cons "--data-file" (List.cons path remainingArgs) => do
      let text ← readUtf8File exeName path missingHint
      pure (text, remainingArgs)
  | List.cons "--data-file" List.nil =>
      throw <| IO.userError s!"{exeName}: --data-file expects a path"
  | List.cons argument remainingArgs => do
      match aliases.find? (fun (aliasName, _) => aliasName = argument) with
      | some (_, path) => do
          let text ← readUtf8File exeName path missingHint
          pure (text, remainingArgs)
      | none => do
          let (text, unconsumedArgs) ←
            takeUtf8Input exeName defaultPath aliases missingHint remainingArgs
          pure (text, argument :: unconsumedArgs)

/--
Number of legal start positions for a `(sequenceLength + 1)` next-token window.

We return at least one start position so bounded corpora stay total; callers can still enforce a
minimum corpus size before training.
-/
def usableTokenStarts (tokenCount sequenceLength : Nat) : Nat :=
  Nat.max 1 (tokenCount - sequenceLength)

/-- Deterministic sliding-window offset for a byte corpus. -/
def byteOffset (bytes : ByteArray) (index sequenceLength : Nat) : Nat :=
  index % usableTokenStarts bytes.size sequenceLength

/-- Deterministic sliding-window offset for an already-tokenized corpus. -/
def tokenOffset {tokenCount : Nat} (_tokens : Tensor Nat [tokenCount])
    (index sequenceLength : Nat) : Nat :=
  index % usableTokenStarts tokenCount sequenceLength

/-- Choose deterministic, approximately evenly spaced starts for fixed-width token windows. -/
def evenlySpacedOffsets (tokenCount sequenceLength windowCount : Nat) : Tensor Nat [windowCount] :=
  let usable := usableTokenStarts tokenCount sequenceLength
  let stride := Nat.max 1 (usable / Nat.max 1 windowCount)
  Tensor.ofFn fun index => (index.val * stride) % usable

/--
Deterministic minGPT-style random offsets for one training batch.

The result has one corpus start offset per batch row. We derive the
random key from `(seed, step)` and then draw row offsets by the row index, so the run is
reproducible without using ambient IO randomness. This is the text equivalent of a shuffled
`EpochLoader` epoch.
-/
def randomBatchOffsets
    (tokenCount sequenceLength batchSize seed step : Nat) :
    Tensor Nat [batchSize] :=
  let usable := usableTokenStarts tokenCount sequenceLength
  let key : UInt64 := Spec.Random.keyOf seed step
  Tensor.ofFn fun batchIndex => Spec.Random.sampleNat key batchIndex.val usable

/--
Build token windows for one deterministic random text batch.

Each row gets `sequenceLength + 1` ids so downstream causal-LM helpers can form both the input and
shifted target.
Byte, character, BPE, and synthetic tokenizers share the same tensor batching semantics.
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

/--
Choose training-window offsets, biased toward a prompt occurrence when the corpus contains it.

If the prompt is present in the corpus, a portion of the sampled windows covers nearby text. That
keeps generation reports tied to text the model actually saw during training.
-/
def promptAwareOffsets
    (tokenCount sequenceLength windowCount : Nat)
    (promptOffset? : Option Nat) :
    Tensor Nat [windowCount] :=
  let usable := usableTokenStarts tokenCount sequenceLength
  match promptOffset? with
  | none => evenlySpacedOffsets tokenCount sequenceLength windowCount
  | some off =>
      let start := if off > windowCount / 4 then off - windowCount / 4 else 0
      Tensor.ofFn fun index => (start + index.val) % usable

end Corpus

end text
end TorchLean
