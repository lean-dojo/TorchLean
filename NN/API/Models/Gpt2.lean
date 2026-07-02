/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.Runtime.Autograd.TorchLean.NN

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

/-- Embedded-token tensor shape `(batch × seqLen × dModel)`. -/
abbrev causalEmbeddingShape (cfg : CausalOneHotConfig) : Shape :=
  shape![cfg.batch, cfg.seqLen, cfg.dModel]

/--
GPT-2-style causal Transformer body after token embeddings have already been computed.

This is the shared body used by both one-hot-token models and indexed-token experiments.  Keeping
it separate avoids duplicating the Transformer stack when callers use a different token
representation: the input boundary changes, while positional embeddings, masked self-attention
blocks, layer norm, and the
language-model head stay the same.
-/
def causalTransformerFromEmbeddings (cfg : CausalOneHotConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (causalEmbeddingShape cfg) (causalOneHotShape cfg)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  let dModel := cfg.dModel
  let encCfg : nn.blocks.TransformerEncoderStack :=
    { layers := cfg.layers
      block := { numHeads := cfg.numHeads, headDim := cfg.headDim, ffnHidden := cfg.ffnHidden }
      seedStride := cfg.seedStride }
  nn.Sequential![
    nn.learnedPositionalEmbedding (batch := cfg.batch) (seqLen := cfg.seqLen) (embedDim := dModel),
    nn.transformerEncoderStack (batch := cfg.batch) (n := cfg.seqLen) (dModel := dModel) encCfg
      (mask := some (text.causalMask cfg.seqLen)),
    nn.layerNorm (batch := cfg.batch) (seqLen := cfg.seqLen) (embedDim := dModel),
    Linear dModel cfg.vocab (pfx := NN.Tensor.Shape.Mat cfg.batch cfg.seqLen)
  ]

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
  nn.embedding cfg.vocab dModel (pfx := NN.Tensor.Shape.Mat cfg.batch cfg.seqLen) >>= fun embed =>
  causalTransformerFromEmbeddings cfg (h_seqLen := h_seqLen) (h_dModel := h_dModel) >>= fun body =>
  pure (embed >>> body)

/--
Scalar loss for causal language modeling with integer token ids.

The public one-hot constructor above is useful for small teaching examples because the input is an
ordinary Float tensor.  File-backed tokenized datasets use the representation found in
language-model training systems: token ids are `Nat`s, the embedding table is a trainable Float
parameter, and the loss gathers the target classes directly instead of building one-hot targets.

`tokens` and `targets` are flattened `(batch * seqLen)` vectors.  This matches the backend gather
ops and keeps dataset storage simple; the embedding helper reshapes gathered rows back to
`(batch, seqLen, dModel)` before running the Transformer body.
-/
def causalTransformerTokenScalarModuleDefWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalOneHotConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalOneHotShape cfg))
    (tokens targets : Spec.Tensor Nat (.dim (cfg.batch * cfg.seqLen) .scalar))
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body) [] :=
  { initParams :=
      .cons (Spec.zeros Float (.dim cfg.vocab (.dim cfg.dModel .scalar))) (initParams body)
    initRequiresGrad := List.replicate (((.dim cfg.vocab (.dim cfg.dModel .scalar)) ::
      paramShapes body).length) true
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
          (ss := ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body) ++
            ([] : List Shape))
          (β := m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α)
            Spec.Shape.scalar))
          (fun args => do
            let (ps, empty) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
                (ss₁ := (.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body)
                (ss₂ := ([] : List Shape)) args
            let .nil := empty
            let .cons tokenEmbedding bodyParams := ps
            let x ← _root_.Runtime.Autograd.TorchLean.F.embeddingBatchSeqNat (m := m) (α := α)
              (vocab := cfg.vocab) (dim := cfg.dModel) (batch := cfg.batch)
              (seqLen := cfg.seqLen) tokenEmbedding tokens
            let logits ← _root_.Runtime.Autograd.TorchLean.NN.Seq.evalParams
              (model := body) (α := α) (m := m) mode bodyParams x
            let logitsRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := .dim cfg.batch (.dim cfg.seqLen (.dim cfg.vocab .scalar)))
              (s₂ := .dim (cfg.batch * cfg.seqLen) (.dim cfg.vocab .scalar))
              logits (by
                simp [_root_.Spec.Shape.size, Nat.mul_assoc])
            _root_.Runtime.Autograd.TorchLean.Loss.crossEntropyRowsNat (m := m) (α := α)
              (rows := cfg.batch * cfg.seqLen) (classes := cfg.vocab)
              logitsRows targets (reduction := reduction)) }

/-- Training-mode wrapper for integer-token causal language modeling. -/
def causalTransformerTokenScalarModuleDef (cfg : CausalOneHotConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalOneHotShape cfg))
    (tokens targets : Spec.Tensor Nat (.dim (cfg.batch * cfg.seqLen) .scalar))
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body) [] :=
  causalTransformerTokenScalarModuleDefWithMode .train cfg body tokens targets
    (reduction := reduction)

abbrev causalTokenLmInputShape (cfg : CausalOneHotConfig) : Shape :=
  .dim (cfg.batch * cfg.seqLen) .scalar

/--
Scalar loss for causal language modeling with per-step float-encoded token ids as inputs.

`xTokens` and `yTokens` are flattened `(batch * seqLen)` float vectors holding integer token ids.
This matches PyTorch `nn.Embedding` + `F.cross_entropy` training and avoids rebuilding the module
each step (Adam state stays on one session).
-/
def causalTransformerTokenLmScalarModuleDefWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalOneHotConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalOneHotShape cfg))
    (initParams : _root_.Runtime.Autograd.Torch.TList Float
      ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body))
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body)
      [causalTokenLmInputShape cfg, causalTokenLmInputShape cfg] :=
  { initParams := initParams
    initRequiresGrad :=
      List.replicate (((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body).length) true
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
          (ss := ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body) ++
            [causalTokenLmInputShape cfg, causalTokenLmInputShape cfg])
          (β := m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α)
            Spec.Shape.scalar))
          (fun args => do
            let (ps, ins) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
                (ss₁ := (.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body)
                (ss₂ := [causalTokenLmInputShape cfg, causalTokenLmInputShape cfg]) args
            let .cons xFloat (.cons yFloat .nil) := ins
            let .cons tokenEmbedding bodyParams := ps
            let tokens ← _root_.Runtime.Autograd.TorchLean.F.floatVecToNatTensor (m := m) (α := α)
              (k := cfg.batch * cfg.seqLen) xFloat
            let targets ← _root_.Runtime.Autograd.TorchLean.F.floatVecToNatTensor (m := m) (α := α)
              (k := cfg.batch * cfg.seqLen) yFloat
            let x ← _root_.Runtime.Autograd.TorchLean.F.embeddingBatchSeqNat (m := m) (α := α)
              (vocab := cfg.vocab) (dim := cfg.dModel) (batch := cfg.batch)
              (seqLen := cfg.seqLen) tokenEmbedding tokens
            let logits ← _root_.Runtime.Autograd.TorchLean.NN.Seq.evalParams
              (model := body) (α := α) (m := m) mode bodyParams x
            let logitsRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := .dim cfg.batch (.dim cfg.seqLen (.dim cfg.vocab .scalar)))
              (s₂ := .dim (cfg.batch * cfg.seqLen) (.dim cfg.vocab .scalar))
              logits (by
                simp [_root_.Spec.Shape.size, Nat.mul_assoc])
            _root_.Runtime.Autograd.TorchLean.Loss.crossEntropyRowsNat (m := m) (α := α)
              (rows := cfg.batch * cfg.seqLen) (classes := cfg.vocab)
              logitsRows targets (reduction := reduction)) }

/-- Training-mode wrapper for float-encoded token-id causal language modeling. -/
def causalTransformerTokenLmScalarModuleDef (cfg : CausalOneHotConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalOneHotShape cfg))
    (initParams : _root_.Runtime.Autograd.Torch.TList Float
      ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body))
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      ((.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body)
      [causalTokenLmInputShape cfg, causalTokenLmInputShape cfg] :=
  causalTransformerTokenLmScalarModuleDefWithMode .train cfg body initParams (reduction := reduction)

end models
end nn

end API
end NN
