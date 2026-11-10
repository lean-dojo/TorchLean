/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Text.Tokenizer
public import TorchLean.NN.Data.Dataset -- shake: keep

/-!
# Text Datasets

Causal language-model samples from token tensors and text. A window of `sequenceLength + 1` tokens
gives an input and a next-token target of length `sequenceLength`. `tokenSample` splits windows,
`tokenBatch` draws seeded batches from a token corpus, and `byteSample`, `byteSamples` and
`byteBatch` build one-hot samples from UTF-8 bytes.
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
    (window : Tensor β (batchShape ++ [sequenceLength + 1])) :
    Sample.Supervised β (batchShape ++ [sequenceLength])
      (batchShape ++ [sequenceLength]) :=
  { input := Tensor.mapLeading batchShape
      (fun row : Tensor β [sequenceLength + 1] => Tensor.ofFn fun position : Fin sequenceLength =>
        row[position.castSucc]) window
    target := Tensor.mapLeading batchShape
      (fun row : Tensor β [sequenceLength + 1] => Tensor.ofFn fun position : Fin sequenceLength =>
        row[position.succ]) window }

/--
Build a one-hot causal-language-model sample over an arbitrary batch shape.

The input and target both append sequence and vocabulary axes to `batchShape`.
-/
def oneHotSample
    {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (batchShape : Shape) (sequenceLength vocabularySize : Nat)
    (tokens : Tensor (Fin vocabularySize) (batchShape ++ [sequenceLength + 1])) :
    Sample.Supervised α
      (batchShape ++ [sequenceLength] ++ [vocabularySize])
      (batchShape ++ [sequenceLength] ++ [vocabularySize]) :=
  let sample := tokenSample batchShape sequenceLength tokens
  { input := Tensor.oneHotIndices vocabularySize sample.input
    target := Tensor.oneHotIndices vocabularySize sample.target }

/--
Draw a causal-language-model batch of token indices from a tensor corpus, with windows chosen by
`seed` and `step` and padded with `paddingTokenId` past the end.

Each drawn token is checked against `vocabularySize` here, so the input and next-token target are
`Fin vocabularySize` tensors and model code has no out-of-range token case.
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
    Text.Corpus.randomTokenBatch
      tokens batchSize sequenceLength seed step (paddingTokenId := paddingTokenId)
  let boundedTokens ← TorchLean.Tensor.checkIndices vocabularySize tokenWindows
  pure (tokenSample [batchSize] sequenceLength boundedTokens)

/--
Build one unbatched one-hot causal-language-model sample from the first `sequenceLength + 1`
UTF-8 bytes of `text`, padded with `0` when the text is shorter. The one-hot input and target
matrices are built in the scalar type `α`, with no `Float` intermediate.
-/
def byteSample
    {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (sequenceLength vocabularySize : Nat)
    (encodeToken : Nat → Fin vocabularySize)
    (text : String) :
    Sample.Supervised α [sequenceLength, vocabularySize] [sequenceLength, vocabularySize] :=
  let bytes := text.toUTF8
  let tokens :=
    (Text.byteTokenWindow bytes (sequenceLength + 1)).map encodeToken
  oneHotSample (α := α) [] sequenceLength vocabularySize tokens

/--
Build an indexed stream of `windowCount` causal-language-model samples from roughly evenly spaced
byte windows of `text`.

Offsets count UTF-8 bytes, as in `byteSample`, and `Text.byteTokenWindow` pads a short corpus
with `paddingTokenId`. Samples are built on access. For supervised training, take `α := Float`
and wrap the stream with `Data.fromStream`.
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
    Text.Corpus.evenlySpacedOffsets bytes.size sequenceLength windowCount
  SampleStream.mk windowCount fun index =>
    let offset := offsets[index]
    let tokens :=
      (Text.byteTokenWindow bytes (sequenceLength + 1) (offset := offset)
        (paddingTokenId := paddingTokenId)).map encodeToken
    oneHotSample (α := α) [] sequenceLength vocabularySize tokens

/--
Repeat the `byteSample` window of `text` across `batchSize` rows to form one batched one-hot
causal-language-model sample.
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
