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

TorchLean's current contract is intentionally small: once tokens enter the verified fragment, token
IDs can be represented as `Fin vocabSize`, making out-of-vocabulary IDs unrepresentable.
-/

@[expose] public section

namespace NN.Examples.BugZoo.TokenizerBoundary

/-- The tokenizer metadata that must agree with the model's embedding table. -/
structure TokenizerContract where
  vocabSize : Nat
  padId : Fin vocabSize
  eosId : Fin vocabSize

/-- A token sequence whose IDs are statically bounded by the vocabulary size. -/
structure TokenSeq (vocabSize seqLen : Nat) where
  tokenAt : Fin seqLen → Fin vocabSize

/-- The padding token is in range by construction. -/
theorem padId_in_vocab (contract : TokenizerContract) :
    contract.padId.val < contract.vocabSize :=
  contract.padId.isLt

/-- Every imported token ID is in range by construction. -/
theorem tokenAt_in_vocab {vocabSize seqLen : Nat}
    (seq : TokenSeq vocabSize seqLen) (i : Fin seqLen) :
    (seq.tokenAt i).val < vocabSize :=
  (seq.tokenAt i).isLt

end NN.Examples.BugZoo.TokenizerBoundary
