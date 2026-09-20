/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# BugZoo: tokenizer/import boundary

Tokenization usually happens outside the tensor graph, which makes it easy for a model import to
silently disagree about vocabulary size, padding, EOS, or special-token IDs. LLM inference-engine
bug studies include tokenizer/config mismatch classes among real production failures:

https://arxiv.org/abs/2506.09713

TorchLean's current contract is focused: once tokens enter the verified fragment, token
IDs can be represented as `Fin vocabularySize`, making out-of-vocabulary IDs unrepresentable.
Bounds alone do not establish that two tokenizers assign the same token to the same ID, or that
padding and end-of-sequence metadata matches the model. Importers must check those agreements.
-/

@[expose] public section

namespace NN.Examples.BugZoo.TokenizerBoundary

/-- The tokenizer metadata that must agree with the model's embedding table. -/
structure TokenizerContract where
  vocabularySize : Nat
  paddingTokenId : Fin vocabularySize
  endOfSequenceTokenId : Fin vocabularySize

/-- A token sequence whose IDs are statically bounded by the vocabulary size. -/
structure TokenSequence (vocabularySize sequenceLength : Nat) where
  tokenAt : Fin sequenceLength → Fin vocabularySize

/-- The padding token is in range by construction. -/
theorem paddingTokenId_isValid (contract : TokenizerContract) :
    contract.paddingTokenId.val < contract.vocabularySize :=
  contract.paddingTokenId.isLt

/-- Every imported token ID is in range by construction. -/
theorem tokenId_isValid {vocabularySize sequenceLength : Nat}
    (sequence : TokenSequence vocabularySize sequenceLength)
    (position : Fin sequenceLength) :
    (sequence.tokenAt position).val < vocabularySize :=
  (sequence.tokenAt position).isLt

end NN.Examples.BugZoo.TokenizerBoundary
