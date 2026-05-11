/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# Transformer Model Helpers (API)

Small config-style Transformer constructors for runnable examples.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a single Transformer encoder block over batched token embeddings. -/
structure TransformerEncoderConfig where
  batch : Nat
  seqLen : Nat
  dModel : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
  activation : nn.blocks.Activation := .gelu
deriving Repr

abbrev transformerEncoderShape (cfg : TransformerEncoderConfig) : Shape :=
  shape![cfg.batch, cfg.seqLen, cfg.dModel]

/-- Build one Transformer encoder block with input/output shape `(batch × seqLen × dModel)`. -/
def transformerEncoder (cfg : TransformerEncoderConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (transformerEncoderShape cfg) (transformerEncoderShape cfg)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  nn.transformerEncoderBlock
    { numHeads := cfg.numHeads
      headDim := cfg.headDim
      ffnHidden := cfg.ffnHidden
      activation := cfg.activation
      dropout? := none }

end models
end nn

end API
end NN

