/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Text.Generation

/-!
# Compiled Docstring Examples: Text

Every `Example:` block in a `TorchLean.text` docstring appears here verbatim, one namespace per
entry point. Nothing in this file runs: the guarantee we want is that the snippet a reader copies
out of the docstring still elaborates, and typechecking gives us exactly that.

`repo_lint.py` compares the fenced `lean` block inside each `Example:` against the namespace body
carrying the matching `doc-example:` marker, so the two cannot drift apart.
-/

@[expose] public section

namespace NN.Tests.API.DocExamples.Text

open TorchLean

-- doc-example: NN/API/Text/Tokenizer.lean :: def byte
namespace ByteTokenizer

-- 256 tokens, no vocabulary file, nothing to train: this is where every text example starts.
def tokenizer : text.Tokenizer := text.Tokenizer.byte

-- Encode then decode returns the original string whenever it was valid UTF-8.
def roundTrip (line : String) : String :=
  tokenizer.decode (tokenizer.encode line)

end ByteTokenizer

-- doc-example: NN/API/Text/Tokenizer.lean :: def fromAlphabet
namespace FromAlphabet

-- The `stoi` and `itos` tables of character-level GPT tutorials, with the nonempty-alphabet
-- requirement carried by the unknown-token index instead of a runtime assertion.
def alphabet : Array Char := #['a', 'b', 'c', ' ']

def tokenizer : text.Tokenizer :=
  text.Tokenizer.fromAlphabet alphabet ⟨3, by decide⟩ (unknownCharacter := '?')

end FromAlphabet

-- doc-example: NN/API/Text/Tokenizer.lean :: def encodeFixed
namespace EncodeFixed

-- Padded or truncated to the length the model expects, so the result carries a shape rather than
-- a length a caller has to check.
def tokens : Tensor Nat [16] :=
  text.Tokenizer.byte.encodeFixed 16 "hello world"

end EncodeFixed

-- doc-example: NN/API/Text/Tokenizer.lean :: def encodeFixedBatch
namespace EncodeFixedBatch

-- Two prompts become one `[2, 16]` batch, ready for a model with a batch axis in front.
def batch : Tensor Nat [2, 16] :=
  text.Tokenizer.byte.encodeFixedBatch 16 ["hello", "world"]

end EncodeFixedBatch

-- doc-example: NN/API/Text/Options.lean :: def tokenWindow
namespace TokenWindow

-- `offset = 0` is the prompt window and `offset = 1` is its next-token target. That pair is the
-- whole of causal language-model supervision.
def prompt : Tensor Nat [8] :=
  text.tokenWindow text.Tokenizer.byte 8 "hello world"

def target : Tensor Nat [8] :=
  text.tokenWindow text.Tokenizer.byte 8 "hello world" (offset := 1)

end TokenWindow

-- doc-example: NN/API/Text/Options.lean :: structure GenerationOptions
namespace Options

-- `topK := 1` is greedy decoding; anything larger samples, and `seed` is what makes that sampling
-- reproducible from one run to the next.
def greedy : text.GenerationOptions :=
  { prompt := "Once upon a time"
    newTokenCount := 64
    temperature := 1.0
    topK := 1
    repeatPenalty := 0.0
    repeatWindow := 0
    seed := 0
    asciiOnly := true }

end Options

-- doc-example: NN/API/Text/Generation.lean :: def topKTokens
namespace TopKTokens

-- Highest scores first. `allowToken` filters rather than masking, so a banned token cannot become
-- selectable just because the masking constant happened to sit above every real score.
def best (scores : Tensor Float [256]) : Tensor (Option (Fin 256)) [8] :=
  text.topKTokens scores 8 (allowToken := fun token => token.val < 128)

end TopKTokens

-- doc-example: NN/API/Text/Generation.lean :: def greedyToken?
namespace GreedyToken

-- `none` means no allowed non-NaN score exists, which is a failure worth seeing rather than a
-- silent fall back to token zero.
def next (scores : Tensor Float [256]) : Option (Fin 256) :=
  text.greedyToken? scores

end GreedyToken

-- doc-example: NN/API/Text/Generation.lean :: def chooseNextToken
namespace ChooseNextToken

-- One decoding policy shared by every text example: repeat penalty first, then greedy or top-k
-- sampling. `counter` keeps each step's randomness distinct while staying reproducible.
def next (scores : Tensor Float [256])
    (options : text.GenerationOptions) (step : Nat) :
    Except String (Fin 256) :=
  text.chooseNextToken scores options (counter := step) (recentTokens := Tensor.full [0] 0)

end ChooseNextToken

-- doc-example: NN/API/Text/Generation.lean :: def autoregressiveTokenIds
namespace Autoregressive

-- The model arrives as a callback: given an exact-length window and the position whose logits to
-- read, return that position's scores. Cropping, padding, penalties, and sampling stay here.
def generate (options : text.GenerationOptions)
    (scoreWindow : Tensor Nat [16] → Fin 16 → IO (Tensor Float [256])) :
    IO (Tensor Nat [(text.Tokenizer.byte.encode options.prompt).size + options.newTokenCount]) :=
  text.autoregressiveTokenIds 16 (paddingTokenId := 0)
    (promptTokens := Tensor.from (text.Tokenizer.byte.encode options.prompt))
    (options := options) (scoreWindow := scoreWindow)

end Autoregressive

-- doc-example: NN/API/Text/Generation.lean :: def formatByteTokens
namespace FormatByteTokens

-- Generated bytes are not always valid UTF-8, so display escapes them instead of guessing:
-- `#[104, 105, 10]` prints as `"hi\n"`.
def display {n : Nat} (tokens : Tensor Nat [n]) : String := text.formatByteTokens tokens

end FormatByteTokens

end NN.Tests.API.DocExamples.Text
