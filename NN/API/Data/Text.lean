/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Data.SampleStream
public import NN.API.Text.Tokenizer
public import NN.API.Arithmetic
public import NN.API.Data.Training -- shake: keep
public import NN.Tensor.Conversion -- shake: keep

/-!
# Text Datasets

Causal language-model sample and dataset constructors.
-/

@[expose] public section

namespace TorchLean

namespace Data

namespace CausalLM

/--
Split exact `(sequenceLength + 1)` token windows into causal-LM input and target tensors.

For a window $[t_0,t_1,\ldots,t_{\mathrm{sequenceLength}}]$, the model input is
$[t_0,\ldots,t_{\mathrm{sequenceLength}-1}]$ and the target is
$[t_1,\ldots,t_{\mathrm{sequenceLength}}]$. Corpus readers perform any requested padding before
constructing this tensor, so the split itself cannot invent or discard tokens.
-/
def tokenSample {β : Type} [TorchLean.Storage β]
    (batchShape : Shape) (sequenceLength : Nat)
    (window : Tensor β (batchShape.appendDim (sequenceLength + 1))) :
    Sample.Supervised β (batchShape.appendDim sequenceLength)
      (batchShape.appendDim sequenceLength) := by
  simp only [Shape.appendDim_eq_concat] at window ⊢
  exact
    { input := Tensor.mapLeading batchShape
        (fun row : Tensor β [sequenceLength + 1] => Tensor.ofFn fun position : Fin sequenceLength =>
          row[position.castSucc]) window
      target := Tensor.mapLeading batchShape
        (fun row : Tensor β [sequenceLength + 1] => Tensor.ofFn fun position : Fin sequenceLength =>
          row[position.succ]) window }

/-- One-hot encode every bounded token id along a new final vocabulary dimension. -/
def oneHotInputs
    {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    {shape : Shape} (vocabularySize : Nat) (tokens : Tensor (Fin vocabularySize) shape) :
    Tensor α (shape.appendDim vocabularySize) :=
  TorchLean.Tensor.oneHotIndices (α := α) vocabularySize tokens

/--
Build a one-hot causal-language-model sample over an arbitrary batch shape.

The input and target both append sequence and vocabulary axes to `batchShape`.
-/
def oneHotSample
    {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (batchShape : Shape) (sequenceLength vocabularySize : Nat)
    (tokens : Tensor (Fin vocabularySize) (batchShape.appendDim (sequenceLength + 1))) :
    Sample.Supervised α
      ((batchShape.appendDim sequenceLength).appendDim vocabularySize)
      ((batchShape.appendDim sequenceLength).appendDim vocabularySize) :=
  let sample := tokenSample batchShape sequenceLength tokens
  { input := oneHotInputs vocabularySize sample.input
    target := oneHotInputs vocabularySize sample.target }

/--
Build a batched one-hot causal-language-model sample from a token tensor by choosing one
deterministic `(sequenceLength + 1)` window per batch row.

Use this for GPT-style trainers that keep a tokenized corpus in memory and derive each batch from
the same `(tokens, seed, step)` rule.
-/
def oneHotBatch {tokenCount : Nat}
    {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (batchSize sequenceLength vocabularySize : Nat)
    (tokens : Tensor Nat [tokenCount])
    (seed step : Nat)
    (paddingTokenId : Nat := 0) :
    Except String
      (Sample.Supervised α
        [batchSize, sequenceLength, vocabularySize]
        [batchSize, sequenceLength, vocabularySize]) := do
  let tokenWindows :=
    TorchLean.text.Corpus.randomTokenBatch
      tokens batchSize sequenceLength seed step (paddingTokenId := paddingTokenId)
  let boundedTokens ← TorchLean.Tensor.checkIndices vocabularySize tokenWindows
  pure <| oneHotSample
    (α := α) [batchSize] sequenceLength vocabularySize boundedTokens

/--
Build an indexed-token causal-language-model batch from a tensor corpus.

The result contains the input and next-token target as separate bounded-index tensors. Validation
happens once at this corpus boundary; model code therefore has no out-of-range token case.
-/
def tokenBatch {tokenCount : Nat}
    (vocabularySize batchSize sequenceLength : Nat)
    (tokens : Tensor Nat [tokenCount])
    (seed step : Nat)
    (paddingTokenId : Nat := 0) :
    Except String
      (Sample.Supervised
        (Fin vocabularySize)
        [batchSize, sequenceLength]
        [batchSize, sequenceLength]) := do
  let tokenWindows :=
    TorchLean.text.Corpus.randomTokenBatch
      tokens batchSize sequenceLength seed step (paddingTokenId := paddingTokenId)
  let boundedTokens ← TorchLean.Tensor.checkIndices vocabularySize tokenWindows
  pure (tokenSample [batchSize] sequenceLength boundedTokens)

/--
Build one unbatched one-hot causal-language-model sample from a text corpus string.

This takes one `(sequenceLength + 1)` byte window from the UTF-8 bytes of `text`, converts it to
one-hot input/target matrices, and casts the result into the runtime-selected arithmetic
representation.
-/
def byteSample
    {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (sequenceLength vocabularySize : Nat)
    (encodeToken : Nat → Fin vocabularySize)
    (text : String) :
    Sample.Supervised α [sequenceLength, vocabularySize] [sequenceLength, vocabularySize] :=
  let bytes := text.toUTF8
  let tokens :=
    (TorchLean.text.byteTokenWindow bytes (sequenceLength + 1)).map encodeToken
  oneHotSample (α := α) [] sequenceLength vocabularySize tokens

/--
Build a finite causal-language-model dataset from approximately evenly spaced byte windows.

Offsets are measured in UTF-8 bytes, matching `byteSample`. Short corpora remain total because
`byteTokenWindow` pads beyond the end, while `windowCount = 0` returns an empty dataset.
-/
def byteSamples
    {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (sequenceLength vocabularySize : Nat)
    (encodeToken : Nat → Fin vocabularySize)
    (windowCount : Nat)
    (text : String) (paddingTokenId : Nat := 0) :
    SampleStream
      (Sample.Supervised α
        [sequenceLength, vocabularySize]
        [sequenceLength, vocabularySize]) :=
  let bytes := text.toUTF8
  let offsets :=
    TorchLean.text.Corpus.evenlySpacedOffsets bytes.size sequenceLength windowCount
  SampleStream.fromFunction windowCount fun index =>
    let offset := offsets[index]
    let tokens :=
      (TorchLean.text.byteTokenWindow bytes (sequenceLength + 1) (offset := offset)
        (paddingTokenId := paddingTokenId)).map encodeToken
    oneHotSample (α := α) [] sequenceLength vocabularySize tokens

/--
Build one fixed-batch one-hot causal-language-model sample from a text corpus string by repeating
the same text window across every batch row.
-/
def byteBatch
    {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (batchSize sequenceLength vocabularySize : Nat)
    (encodeToken : Nat → Fin vocabularySize)
    (text : String) :
    Sample.Supervised α
      [batchSize, sequenceLength, vocabularySize]
      [batchSize, sequenceLength, vocabularySize] :=
  let sample :=
    byteSample (α := α) sequenceLength vocabularySize encodeToken text
  { input := Tensor.repeatAxis 0 batchSize sample.input
    target := Tensor.repeatAxis 0 batchSize sample.target }

end CausalLM

end Data

end TorchLean
