/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Tactic.Init

/-!
# BugZoo: tokenizer/import boundary

Tokenization usually happens outside the tensor graph, which makes it easy for a model import to
silently disagree about vocabulary size, padding, EOS, or special-token IDs. LLM inference-engine
bug studies include tokenizer/config mismatch classes among real production failures:

https://arxiv.org/abs/2506.09713

TorchLean's contract is focused: once tokens enter the verified fragment, token
IDs can be represented as `Fin vocabularySize`, making out-of-vocabulary IDs unrepresentable.
Bounds alone do not establish that two tokenizers assign the same token to the same ID, or that
padding and end-of-sequence metadata matches the model. Importers must check those agreements.
-/

@[expose] public section

namespace TorchLean.Examples.BugZoo

/-- The tokenizer metadata that must agree with the model's embedding table. -/
structure Tokenizer where
  /-- Number of token IDs in the vocabulary. -/
  vocabularySize : Nat
  /-- ID of the padding token. -/
  paddingTokenId : Fin vocabularySize
  /-- ID of the end-of-sequence token. -/
  endOfSequenceTokenId : Fin vocabularySize

/-- The padding token is in range by construction. -/
theorem Tokenizer.padding_lt_vocabulary (contract : Tokenizer) :
    contract.paddingTokenId.val < contract.vocabularySize :=
  contract.paddingTokenId.isLt

/-- Every imported token ID is in range by construction. -/
theorem Tokenizer.token_lt_vocabulary {vocabularySize sequenceLength : Nat}
    (sequence : Tensor (Fin vocabularySize) [sequenceLength])
    (position : Fin sequenceLength) :
    sequence[position].val < vocabularySize :=
  sequence[position].isLt

end TorchLean.Examples.BugZoo
