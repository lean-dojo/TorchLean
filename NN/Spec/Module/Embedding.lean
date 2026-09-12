/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Embedding
public import NN.Spec.Module.Core

/-!
# Embedding

Module wrapper for the differentiable one-hot presentation of an embedding table. The indexed
semantics is `Embedding.lookup`; this wrapper exists for proofs that treat the same table as the
linear map `oneHot @ weight`.
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- One-hot embedding module: `(seqLen, vocab)` to `(seqLen, embedDim)`. -/
def oneHotEmbedding {vocab embedDim seqLen : Nat}
  (embedding : Embedding vocab embedDim α) :
  Spec.Module α ([seqLen, vocab]) ([seqLen, embedDim]) :=
{ forward := fun oneHot => embedding.oneHot oneHot
  kind := "OneHotEmbedding"
  pythonExpr := s!"OneHotEmbedding(vocab={vocab}, embedDim={embedDim})" }

end Spec.Module
