/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Sequence.Gpt2

/-!
# GPT-2 cached corpus windows

Compare every input and shifted target with direct UTF-8 byte indexing. Empty, short and
multibyte corpora exercise padding and byte offsets independently of the tokenizer/window helpers.
-/

@[expose] public section

namespace NN.Tests.API.Gpt2Corpus

open TorchLean NN.Examples.Models.Sequence.Gpt2

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"GPT-2 corpus: {label}"

def run : IO Unit := do
  for corpus in ["", "a", "abcd", "abcdefghi", "λ🙂 café\nTorchLean"] do
    let bytes := corpus.toByteArray
    for windows in [0, 1, 2, 7] do
      let samples := samplesFromCorpus corpus windows
      expect "requested window count" (samples.size == windows)
      let usable := max 1 (bytes.size - contextLength)
      let stride := max 1 (usable / max 1 windows)
      for index in List.finRange samples.size do
        let sample := samples.get index
        let offset := (index.val * stride) % usable
        let directByte := fun row position =>
          let start := (offset + row * (contextLength / 2 + 1)) % usable
          match bytes[start + position]? with
          | some value => value.toNat
          | none => 32
        for row in List.finRange batchSize do
          for position in List.finRange contextLength do
            for token in List.finRange vocabularySize do
              let expectedInput : Float :=
                if token.val == directByte row.val position.val then 1 else 0
              let expectedTarget : Float :=
                if token.val == directByte row.val (position.val + 1) then 1 else 0
              expect "input window preserves bytes and padding"
                ((sample.input[row][position][token]).toBits == expectedInput.toBits)
              expect "target is shifted by exactly one byte"
                ((sample.target[row][position][token]).toBits == expectedTarget.toBits)
        let repeated := samples.get index
        expect "repeated sample requests preserve input bits"
          ((Tensor.to repeated.input (Array Float)).map Float.toBits ==
            (Tensor.to sample.input (Array Float)).map Float.toBits)
        expect "repeated sample requests preserve target bits"
          ((Tensor.to repeated.target (Array Float)).map Float.toBits ==
            (Tensor.to sample.target (Array Float)).map Float.toBits)
  IO.println "  GPT-2 cached corpus windows: passed"

end NN.Tests.API.Gpt2Corpus
