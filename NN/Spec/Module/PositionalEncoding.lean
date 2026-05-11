/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.PositionalEncoding
public import NN.Spec.Module.SpecModule

/-!
# PositionalEncoding

Module wrappers for spec-layer positional encodings.

This is the simplest learnable variant: add a `(seqLen, embedDim)` parameter tensor.

PyTorch equivalent: "learnable positional embedding" that is added to token embeddings. In practice
this is often implemented via `nn.Embedding(seqLen, embedDim)` and an index arange; here we treat
the positional tensor itself as the parameter.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Learnable positional encoding wrapper (adds a `(seqLen,embedDim)` parameter tensor). -/
def PositionalEncodingModuleSpec {seqLen embedDim : Nat}
  (pe : PositionalEncodingSpec seqLen embedDim α) :
  NNModuleSpec α (.dim seqLen (.dim embedDim .scalar)) (.dim seqLen (.dim embedDim .scalar)) :=
{ forward := fun x => addPositionalEncodingSpec (α := α) pe x
  kind := "PositionalEncoding"
  export_func := {
    toPyTorch := s!"PositionalEncoding(seqLen={seqLen}, embedDim={embedDim})"
    dimensions := (seqLen, embedDim)
  } }

end Spec
