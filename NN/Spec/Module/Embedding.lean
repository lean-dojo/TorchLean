/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Embedding
public import NN.Spec.Module.SpecModule

/-!
# Embedding

Module wrappers for spec-layer embeddings.

We intentionally expose the one-hot embedding variant here (purely numeric, no integer indices).

Why one-hot in the spec layer:

- It avoids committing to an "index tensor" representation. In Lean, indices would typically live
  in `Nat`/`Fin`, which is great for proofs, but many numeric backends are scalar-only.
- It keeps the forward definition completely algebraic: an embedding becomes `one_hot @ W`.

In PyTorch terms: the usual API is `nn.Embedding(vocab, embed_dim)` on integer indices. This file
packages the equivalent "one_hot then matmul" semantics.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- One-hot embedding wrapper: `(seqLen,vocab)` → `(seqLen,embedDim)`. -/
def EmbeddingOneHotModuleSpec {vocab embedDim seqLen : Nat}
  (emb : EmbeddingSpec vocab embedDim α) :
  NNModuleSpec α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim embedDim .scalar)) :=
{ forward := fun oneHot => embeddingOnehotSpec (α := α) emb oneHot
  kind := "EmbeddingOneHot"
  export_func := {
    -- PyTorch equivalent is typically `nn.Embedding` on integer indices;
    -- this one-hot version is semantically `one_hot @ W`.
    toPyTorch := s!"EmbeddingOneHot(vocab={vocab}, embedDim={embedDim})"
    dimensions := (vocab, embedDim)
  } }

end Spec
