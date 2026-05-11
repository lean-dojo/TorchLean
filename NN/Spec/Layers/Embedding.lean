/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Embedding

Spec-layer embedding primitives.

We model embeddings through **single-scalar** one-hot tensors: inputs have the same scalar type `α`
as the embedding matrix, so they compose cleanly with the rest of the tensor language.

If you want index-based embeddings (integer token ids) in runtime graphs, that lives at the
TorchLean/session layer via Nat channels; the spec layer stays purely numeric by default.

References / analogies:
- In most ML frameworks, an embedding table is a matrix `W : (vocab x embedDim)` and an index-based
  lookup returns `W[token_id]`. One-hot embeddings are the equivalent linear map
  `oneHot @ W` (this file).
- Bengio et al., "A Neural Probabilistic Language Model" (2003) for the classic embedding-table
  framing in neural language models.
- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013) for the
  modern word-embedding perspective.
- PyTorch API docs:
  - `torch.nn.Embedding`: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
  - `torch.nn.functional.one_hot`:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-- Standard embedding weight matrix: `vocab × embedDim`. -/
structure EmbeddingSpec (vocab embedDim : Nat) (α : Type) where
  /-- W. -/
  W : Tensor α (.dim vocab (.dim embedDim .scalar))

/--
Embed a batch/sequence of one-hot vectors:

`oneHot : (seqLen × vocab)` and `W : (vocab × embedDim)` gives `(seqLen × embedDim)`.
-/
def embeddingOnehotSpec {vocab embedDim seqLen : Nat}
    (emb : EmbeddingSpec vocab embedDim α)
    (oneHot : Tensor α (.dim seqLen (.dim vocab .scalar))) :
    Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  matMulSpec oneHot emb.W

/-!
## Gradients

`embedding_onehot_spec` is matrix multiplication:

`Y = oneHot @ W`.

So the reverse-mode derivatives are the standard ones:

- `dOneHot = dY @ Wᵀ`
- `dW      = oneHotᵀ @ dY`

Even though "true" one-hot tensors are often treated as non-differentiable in practice, having a
named VJP is useful for:

- treating embeddings as a pure linear map in proofs,
- debugging equivalences (one-hot vs index-based embeddings),
- and keeping this layer consistent with the rest of the spec library.
-/

/-- Backward/VJP for `embedding_onehot_spec`: returns `(dOneHot, dW)`. -/
def embeddingOnehotBackwardSpec {vocab embedDim seqLen : Nat}
    (emb : EmbeddingSpec vocab embedDim α)
    (oneHot : Tensor α (.dim seqLen (.dim vocab .scalar)))
    (dY : Tensor α (.dim seqLen (.dim embedDim .scalar))) :
    (Tensor α (.dim seqLen (.dim vocab .scalar))) × (Tensor α (.dim vocab (.dim embedDim .scalar)))
      :=
  matMulBackwardSpec (α := α) (m := seqLen) (n := vocab) (p := embedDim) oneHot emb.W dY

end Spec
