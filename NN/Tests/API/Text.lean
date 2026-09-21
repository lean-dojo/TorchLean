/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Text
public import NN.API.Text.Bpe
public import NN.API.Text.Generation

/-!
# Text API Tests

Regression checks for text parsing, deterministic corpus windows, and floating-point sampling.
-/

@[expose] public section

namespace NN.Tests.API.Text

open TorchLean

def expectEqual {α : Type} [BEq α] [Repr α]
    (label : String) (expected actual : α) : IO Unit := do
  unless expected == actual do
    throw <| IO.userError
      s!"text API check failed: {label}: expected {repr expected}, got {repr actual}"

def checkSelection : IO Unit := do
  let nan : Float := 0.0 / 0.0
  let inf : Float := 1.0 / 0.0
  let scores : Tensor Float [8] := Tensor.from #[nan, -inf, 5, 5, inf, inf, -0.0, 0.0]
  let selected := fun k =>
    ((text.topKTokens scores k).to (Array (Option (Fin 8)))).map (·.map Fin.val)
  expectEqual "top-k keeps stable ties, infinities, signed zeros, and trailing padding"
    #[some 4, some 5, some 2, some 3, some 6, some 7, some 1, none] (selected 20)
  expectEqual "zero top-k output" #[] (selected 0)
  expectEqual "top-k empty vocabulary" (#[] : Array (Option (Fin 0)))
    ((text.topKTokens (Tensor.full [0] 0.0) 10).to (Array (Option (Fin 0))))
  expectEqual "top-k all NaNs pads every slot" (#[none, none] : Array (Option (Fin 3)))
    ((text.topKTokens (Tensor.full [3] nan) 2).to (Array (Option (Fin 3))))
  expectEqual "top-k cutoff breaks ties by smaller token id" #[some 4] (selected 1)
  expectEqual "greedy agrees with the top-k maximum" (some 4)
    ((text.greedyToken? scores).map Fin.val)
  expectEqual "empty vocabulary" (none : Option Nat)
    ((text.greedyToken? (Tensor.full [0] 0.0)).map Fin.val)
  expectEqual "empty allow-list" (none : Option Nat)
    ((text.greedyToken? scores (fun _ => false)).map Fin.val)
  expectEqual "all NaNs" (none : Option Nat)
    ((text.greedyToken? (Tensor.full [3] nan)).map Fin.val)
  expectEqual "filter never substitutes a finite masking sentinel" (some 1)
    ((text.greedyToken? (Tensor.from (#[10.0, -1.0e300] : Array Float))
      (fun token => token.val = 1)).map Fin.val)

  -- A complete independent sort checks heap eviction, not just the first maximum.
  for seed in [:12] do
    let values : Tensor Float [31] := Tensor.ofFn fun token =>
      let draw := Spec.Random.sampleNat (Spec.Random.keyOf seed token.val) 0 13
      if draw = 0 then nan else if draw = 1 then inf else if draw = 2 then -inf
      else Float.ofNat draw - 7.0
    for modulus in [1, 2, 5] do
      let allow := fun token : Fin 31 => token.val % modulus == 0
      let ordered := ((List.finRange 31).filter
        (fun token => allow token && !(values[token]).isNaN)).mergeSort fun left right =>
          values[left] > values[right] ||
            (values[left] == values[right] && left.val ≤ right.val)
      for k in [0, 1, 2, 7, 17, 31, 40] do
        let expected := (List.range (min k 31)).toArray.map
          (fun index => ordered[index]?.map Fin.val)
        let actual := ((text.topKTokens values k allow).to
          (Array (Option (Fin 31)))).map (·.map Fin.val)
        expectEqual s!"heap matches complete sort seed={seed} modulus={modulus} k={k}"
          expected actual

def checkSampling : IO Unit := do
  let nan : Float := 0.0 / 0.0
  let inf : Float := 1.0 / 0.0
  let values : Tensor Float [4] := Tensor.full [4] 2.0
  let sample := fun (scores : Tensor Float [4]) (temperature : Float) (topK seed counter : Nat) =>
    (text.sampleTopKToken? scores temperature topK seed counter).map Fin.val
  for invalid in [0.0, -1.0, nan, inf, -inf] do
    expectEqual "reject invalid sampling temperature" (none : Option Nat)
      (sample values invalid 0 0 0)
  expectEqual "empty sampling vocabulary" (none : Option Nat)
    ((text.sampleTopKToken? (Tensor.full [0] 1.0) 1 0 0 0).map Fin.val)
  expectEqual "sampling rejects all NaNs" (none : Option Nat)
    (sample (Tensor.full [4] nan) 1 0 0 0)
  expectEqual "sampling rejects empty allow-list" (none : Option Nat)
    ((text.sampleTopKToken? values 1 0 0 0 (fun _ => false)).map Fin.val)
  expectEqual "sampling never selects filtered infinities or zero-weight negative infinity" (some 2)
    ((text.sampleTopKToken? (Tensor.from (#[inf, nan, -1.0e300, -inf] : Array Float))
      1 0 0 0 (fun token => token.val ≥ 2)).map Fin.val)
  for k in [0, 2, 4, 9] do
    expectEqual "infinite maximum uses smallest allowed id" (some 1)
      (sample (Tensor.from (#[nan, inf, 10.0, inf] : Array Float)) 1 k 0 0)
    expectEqual "all negative infinities use smallest allowed id" (some 0)
      (sample (Tensor.full [4] (-inf)) 1 k 0 0)
    expectEqual "single allowed negative infinity remains selectable" (some 2)
      ((text.sampleTopKToken? (Tensor.full [4] (-inf)) 1 k 0 0
        (fun token => token.val = 2)).map Fin.val)
  for seed in [:16] do
    for counter in [:4] do
      let draw := Spec.Random.sampleNat (Spec.Random.keyOf seed counter) 0 ((2 : Nat) ^ 32)
      let expected := some (draw / ((2 : Nat) ^ 30))
      expectEqual "full-vocabulary uniform draw uses token-id order" expected
        (sample values 1 0 seed counter)
      expectEqual "oversized cutoff matches full-vocabulary sampling" expected
        (sample values 1 9 seed counter)
      expectEqual "tiny temperature preserves equal huge logits" expected
        (sample (Tensor.full [4] 1.0e300) 1.0e-300 0 seed counter)
      expectEqual "tiny temperature preserves equal negative logits" expected
        (sample (Tensor.full [4] (-1.0e300)) 1.0e-300 0 seed counter)
      expectEqual "top-k uniform draw keeps cutoff ties stable"
        (some (draw / ((2 : Nat) ^ 31))) (sample values 1 2 seed counter)
      expectEqual "filter retains allowed-id order under full sampling"
        (some (1 + 2 * (draw / ((2 : Nat) ^ 31))))
        ((text.sampleTopKToken? values 1 0 seed counter
          (fun token => token.val % 2 = 1)).map Fin.val)
      let u := Float.ofNat draw / Float.ofNat ((2 : Nat) ^ 32)
      let orderedExpected := if u < 1.0 / (1.0 + Float.exp 1.0) then some 0 else some 1
      expectEqual "full sampling follows token-id order for unequal logits" orderedExpected
        ((text.sampleTopKToken? (Tensor.from (#[0.0, 1.0] : Array Float))
          1 0 seed counter).map Fin.val)
      let extremeExpected := if u < 1.0 / (1.0 + Float.exp (-2.0)) then some 0 else some 1
      expectEqual "large temperature handles overflowing unscaled differences"
        extremeExpected
        ((text.sampleTopKToken? (Tensor.from (#[1.0e308, -1.0e308] : Array Float))
          1.0e308 0 seed counter).map Fin.val)
  let options : text.GenerationOptions :=
    { prompt := "", newTokenCount := 1, temperature := 1, topK := 1,
      repeatPenalty := 2, repeatWindow := 4, seed := 0, asciiOnly := false }
  expectEqual "repetition penalty counts occurrences and ignores out-of-vocabulary ids"
    (Except.ok 2 : Except String Nat)
    ((text.chooseNextToken values options 0 (Tensor.from #[0, 0, 1, 99])).map Fin.val)
  for invalid in [-1.0, nan, inf] do
    expectEqual "reject invalid repetition penalty" false
      (text.chooseNextToken values { options with repeatPenalty := invalid } 0
        (Tensor.full [0] 0)).isOk
  expectEqual "greedy decoding ignores unused temperature" (Except.ok 0 : Except String Nat)
    ((text.chooseNextToken values { options with temperature := nan } 0
      (Tensor.full [0] 0)).map Fin.val)
  expectEqual "sampling validates temperature at the generation boundary" false
    (text.chooseNextToken values { options with topK := 0, temperature := nan } 0
      (Tensor.full [0] 0)).isOk

def run : IO Unit := do
  checkSelection
  checkSampling
  let alphabet := text.Tokenizer.fromAlphabet #['a', 'b', 'a', '😀'] ⟨1, by decide⟩
  expectEqual "alphabet lookup keeps first duplicate, Unicode, and unknown ids"
    #[0, 3, 1, 1] (alphabet.encode "a😀?b")
  expectEqual "alphabet decoding preserves positions and rejects invalid ids"
    "aa😀?" (alphabet.decode #[0, 2, 3, 99])
  for byte in [:256] do
    let value := UInt8.ofNat byte
    expectEqual "GPT-2 byte alphabet inverse" (some value)
      (text.GPT2BPE.Internal.charToByte? (text.GPT2BPE.Internal.byteToChar value))
  expectEqual "GPT-2 inverse rejects characters outside byte alphabet" (none : Option UInt8)
    (text.GPT2BPE.Internal.charToByte? '😀')
  expectEqual "GPT-2 byte escape round-trips Unicode text" (some "héllo 😀\n")
    (text.GPT2BPE.Internal.byteDecode? (text.GPT2BPE.Internal.byteEncode "héllo 😀\n"))
  let pretokenize := text.GPT2BPE.Internal.pretokenize
  expectEqual "GPT-2 contractions and character classes"
    ["I", "'m", " café", " １２", "!!"] (pretokenize "I'm café １２!!")
  expectEqual "GPT-2 whitespace lookahead leaves one leading space"
    [" \t", " word", "\n\n"] (pretokenize " \t word\n\n")
  expectEqual "GPT-2 single non-ASCII whitespace stays separate"
    ["a", "\u00a0", "b"] (pretokenize "a\u00a0b")
  let longWord := String.ofList (List.replicate 20000 'a')
  expectEqual "GPT-2 consumes a long character run" [longWord] (pretokenize longWord)
  let parseVocabulary := text.GPT2BPE.Internal.parseVocabularyText
  let specialVocabulary ← IO.ofExcept <| parseVocabulary "{\"<|endoftext|>\":0}"
  let specialTokenizer := text.GPT2BPE.Tokenizer.Internal.create
    (text.GPT2BPE.Internal.buildTokenizer specialVocabulary #[]) rfl
  expectEqual "special token lookup does not pretokenize" (some 0)
    (specialTokenizer.tokenId? "<|endoftext|>")
  expectEqual "missing vocabulary token" (none : Option Nat)
    (specialTokenizer.tokenId? "missing")
  for malformed in ["{\"a\":0} trailing", "{\"a\":0,}",
      "{\"a\":00}", "{\"\\uD800\":0}", "{\"\\uDC00\":0}",
      "{\"a\n\":0}", "{\"a\":0,\"a\":1}", "{\"a\":0,\"b\":0}",
      "{\"a\":1}"] do
    expectEqual s!"reject malformed BPE vocabulary {repr malformed}"
      false (parseVocabulary malformed).isOk
  let vocabulary ← IO.ofExcept
    (parseVocabulary " {\"a\":1,\"\\uD83D\\uDE00\":0} \r\n")
  expectEqual "vocabulary preserves surrogate pairs and source order"
    #["a", "😀"] (vocabulary.map (·.token))
  expectEqual "vocabulary allows reordered contiguous ids"
    #[1, 0] (vocabulary.map (·.id))

  let hashVocabulary ← IO.ofExcept (parseVocabulary "{\"#\":0,\"##\":1,\"####\":2}")
  let hashMerges ← IO.ofExcept
    (text.GPT2BPE.Internal.parseMerges "#version: 0.2\n# #\n## ##\n")
  let hashTokenizer := text.GPT2BPE.Internal.buildTokenizer hashVocabulary hashMerges
  expectEqual "hash-prefixed BPE pairs are merges, not comments"
    #[2] (← IO.ofExcept (text.GPT2BPE.Internal.encodeFragment hashTokenizer "####"))

  let defaults : text.GenerationOptions :=
    { prompt := "", newTokenCount := 1, temperature := 1, topK := 1,
      repeatPenalty := 0, repeatWindow := 0, seed := 0, asciiOnly := true }
  let (unchanged, _) ← IO.ofExcept (text.GenerationOptions.parse "text" [] defaults)
  expectEqual "omitted ASCII flag preserves default" true unchanged.asciiOnly
  for arguments in [["--ascii-only=false"], ["--ascii-only", "false"]] do
    let (overridden, remaining) ←
      IO.ofExcept (text.GenerationOptions.parse "text" arguments defaults)
    expectEqual "explicit false overrides ASCII default" false overridden.asciiOnly
    expectEqual "ASCII flag is consumed" [] remaining

  expectEqual "short corpus remains total"
    1 (text.Corpus.usableTokenStarts 3 4)
  expectEqual "exactly one complete window"
    1 (text.Corpus.usableTokenStarts 5 4)
  expectEqual "include final legal start"
    2 (text.Corpus.usableTokenStarts 6 4)
  expectEqual "all starts in a longer corpus"
    6 (text.Corpus.usableTokenStarts 10 4)

  let bytes : ByteArray := ⟨#[0, 1, 2, 3, 4, 5]⟩
  expectEqual "byte offsets visit both legal starts"
    #[0, 1, 0, 1]
    ((Array.range 4).map fun i => text.Corpus.byteOffset bytes i 4)

  let tokens := #[0, 1, 2, 3, 4, 5]
  expectEqual "token offsets visit both legal starts"
    #[0, 1, 0, 1]
    ((Array.range 4).map fun i => text.Corpus.tokenOffset (Tensor.from tokens) i 4)

  expectEqual "unbiased offsets span the corpus"
    #[0, 2, 4]
    ((text.Corpus.promptAwareOffsets 10 4 3 none).to (Array Nat))

  let byteSamples :=
    Data.CausalLM.byteSamples
      (α := Float) 2 4 (fun token => Fin.ofNat 4 token) 3 "abcdef"
  expectEqual "byte samples preserve the requested window count"
    3 byteSamples.size
  let first := byteSamples.get ⟨0, by decide⟩
  let second := byteSamples.get ⟨1, by decide⟩
  expectEqual "byte samples start at distinct corpus offsets"
    #[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    (first.input.to (Array Float))
  expectEqual "byte samples advance through the corpus"
    #[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    (second.input.to (Array Float))

  IO.println "  text API: passed"

end NN.Tests.API.Text
