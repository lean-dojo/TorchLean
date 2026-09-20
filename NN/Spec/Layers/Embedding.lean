/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Embeddings

An embedding is a table whose rows are indexed by a finite vocabulary. The primary definition in
this file uses `Fin vocab`, so an invalid token id cannot be supplied to the mathematical model.
The one-hot presentation is retained as a differentiable linear map and is useful when comparing
an indexed lookup with matrix multiplication.

References / analogies:
- In most ML frameworks, an embedding table is a matrix `weight : (vocab x embedDim)` and an
  index-based lookup returns `weight[token_id]`. One-hot embeddings are the equivalent linear map
  `oneHot @ weight`.
- Bengio et al., "A Neural Probabilistic Language Model" (2003) for the classic embedding-table
  framing in neural language models.
- Mikolov et al., "Efficient Estimation of Word Representations in Vector Space" (2013) for the
  modern word-embedding perspective.
- PyTorch API reference:
  - `torch.nn.Embedding`: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
  - `torch.nn.functional.one_hot`:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- A trainable table with `vocab` rows of width `embedDim`. -/
structure Embedding (vocab embedDim : Nat) (α : Type)
    [TorchLean.Storage α] where
  /-- Embedding table, stored row-major by executable backends. -/
  weight : Tensor α [vocab, embedDim]

namespace Embedding

/--
Look up every finite index in a tensor and append the embedding dimension to its shape.

The index type is `Fin vocab`, rather than `Nat`, so this specification has no out-of-range case.
File and tokenizer boundaries may begin with natural-number identifiers, but executable APIs must
validate them before interpreting them as indices into this table.
-/
def lookup {vocab embedDim : Nat} (embedding : Embedding vocab embedDim α) :
    {shape : Shape} → Tensor (Fin vocab) shape → Tensor α (shape.appendDim embedDim)
  | .scalar, indices => get embedding.weight indices.item
  | .dim _ _, indices =>
      Tensor.dim fun i => lookup embedding (Tensor.unstack indices i)

/--
Embed a batch/sequence of one-hot vectors:

`oneHot : (seqLen × vocab)` and `W : (vocab × embedDim)` gives `(seqLen × embedDim)`.
-/
def oneHot {vocab embedDim seqLen : Nat}
    (embedding : Embedding vocab embedDim α)
    (oneHot : Tensor α [seqLen, vocab]) :
    Tensor α [seqLen, embedDim] :=
  matMulSpec oneHot embedding.weight

/-!
## Gradients

`Embedding.oneHot` is matrix multiplication:

`Y = oneHot @ weight`.

So the reverse-mode derivatives are the standard ones:

- `dOneHot = dY @ Wᵀ`
- `dWeight = oneHotᵀ @ dY`

Even though "true" one-hot tensors are often treated as non-differentiable in practice, having a
named VJP is useful for:

- treating embeddings as a pure linear map in proofs,
- debugging equivalences (one-hot vs index-based embeddings),
- and keeping this layer consistent with the rest of the spec library.
-/

/-- VJP for `Embedding.oneHot`, returning gradients for the input and table. -/
def oneHotVjp {vocab embedDim seqLen : Nat}
    (embedding : Embedding vocab embedDim α)
    (oneHot : Tensor α [seqLen, vocab])
    (dY : Tensor α [seqLen, embedDim]) :
    (Tensor α [seqLen, vocab]) × (Tensor α [vocab, embedDim])
      :=
  matmulBackwardSpec (α := α) (m := seqLen) (n := vocab) (p := embedDim)
    (Shape.CanBroadcastTo.refl .scalar) (Shape.CanBroadcastTo.refl .scalar)
    oneHot embedding.weight dY

end Embedding

end Spec
