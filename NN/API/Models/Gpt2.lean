/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# GPT-2-Style Model Helpers (API)

This module collects compact, reusable GPT-2-style building blocks for TorchLean examples:

- a single “causal LM over one-hot tokens” model constructor, and
- a small configuration record that keeps the hyperparameter inventory explicit.

These helpers live in the API layer so runnable examples can stay focused on:
data prep, training loops, and text decoding, rather than repeating the same
`embedding → positional embedding → Transformer stack → LayerNorm → linear` boilerplate.

Important scope note:
- This is *not* a pretrained checkpoint loader.
- These are compact example architectures shaped like GPT-2 blocks.
- Tokenizers live under `NN.API.text` / `NN.API.text.Gpt2Bpe`.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/--
Configuration for a small GPT-2-style causal language model over one-hot token inputs.

The model has the common GPT-2 “shape”:

`embedding → learned positional embedding → (masked self-attention + FFN)×layers → LayerNorm → linear`

The input and output shapes are `(batch × seqLen × vocab)` one-hot/logit tensors.
-/
structure CausalOneHotConfig where
  batch : Nat
  seqLen : Nat
  vocab : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
  layers : Nat
  /-- Seed stride used when initializing repeated blocks. -/
  seedStride : Nat := 100
deriving Repr

/-- Transformer width implied by `numHeads * headDim`. -/
def CausalOneHotConfig.dModel (cfg : CausalOneHotConfig) : Nat :=
  cfg.numHeads * cfg.headDim

/-- Input/output tensor shape `(batch × seqLen × vocab)` for a one-hot causal LM. -/
abbrev causalOneHotShape (cfg : CausalOneHotConfig) : Shape :=
  shape![cfg.batch, cfg.seqLen, cfg.vocab]

/--
Build a GPT-2-style causal language model over one-hot tokens.

This is the shared constructor used by the runnable GPT-2 examples. It stays in `nn.M` so it
composes with the rest of the API-layer model-building interface.
-/
def causalTransformerOneHot (cfg : CausalOneHotConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (causalOneHotShape cfg) (causalOneHotShape cfg)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  let dModel := cfg.dModel
  let encCfg : nn.blocks.TransformerEncoderStack :=
    { layers := cfg.layers
      block := { numHeads := cfg.numHeads, headDim := cfg.headDim, ffnHidden := cfg.ffnHidden }
      seedStride := cfg.seedStride }
  nn.sequential![
    nn.embedding cfg.vocab dModel (pfx := NN.Tensor.Shape.Mat cfg.batch cfg.seqLen),
    nn.learnedPositionalEmbedding (batch := cfg.batch) (seqLen := cfg.seqLen) (embedDim := dModel),
    nn.transformerEncoderStack (batch := cfg.batch) (n := cfg.seqLen) (dModel := dModel) encCfg
      (mask := some (text.causalMask cfg.seqLen)),
    nn.layerNorm (batch := cfg.batch) (seqLen := cfg.seqLen) (embedDim := dModel),
    nn.linear dModel cfg.vocab (pfx := NN.Tensor.Shape.Mat cfg.batch cfg.seqLen)
  ]

end models
end nn

end API
end NN
