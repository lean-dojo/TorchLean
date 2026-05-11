/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Seq2seq
public import NN.Spec.Module.SpecModule

/-!
# Seq2Seq inference wrapper as an `NNModuleSpec`

The Seq2Seq spec model defines encoder/decoder math and differentiable training helpers.
This file provides a small inference-oriented `NNModuleSpec` wrapper so it can be composed/exported.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- Seq2Seq inference module wrapper (one-hot input, greedy decoding). -/
def Seq2SeqModuleSpec {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  (m : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (start_token : Nat)
  (_h1 : srcVocabSize ≠ 0) (h2 : tgtVocabSize ≠ 0) (_h3 : embedDim ≠ 0) (_h4 : hiddenDim ≠ 0)
  (_h5 : srcSeqLen ≠ 0) (h6 : tgtSeqLen ≠ 0) :
  NNModuleSpec α (.dim srcSeqLen (.dim srcVocabSize .scalar)) (.dim tgtSeqLen (.dim tgtVocabSize
    .scalar)) :=
{
  forward := fun src_onehot =>
    let src_embeds := Seq2SeqEmbeddingSpec.forwardOnehot m.src_embedding src_onehot
    let (_encoder_outputs, encoder_hidden) := Seq2SeqRNNEncoderSpec.forward m.encoder src_embeds
      none

    let start_embed :=
      if h : start_token < tgtVocabSize then
        match get m.tgt_embedding.embedding ⟨start_token, h⟩ with
        | Tensor.dim embed_vals => Tensor.dim embed_vals
      else
        Tensor.dim (fun _ => Tensor.scalar (0 : α))

    let (logits, _tokens) :=
      Seq2SeqDecoderSpec.forwardInference m.decoder encoder_hidden start_embed
        m.tgt_embedding.embedding tgtSeqLen h6 h2
    logits,
  kind := "Seq2Seq",
  export_func := {
    toPyTorch :=
      s!"Seq2SeqInference(src_vocab_size={srcVocabSize}, " ++
        s!"tgt_vocab_size={tgtVocabSize}, embed_dim={embedDim}, " ++
        s!"hidden_dim={hiddenDim}, max_tgt_len={tgtSeqLen}, " ++
        s!"start_token={start_token})",
    dimensions := (srcSeqLen, tgtSeqLen * tgtVocabSize)
  }
}

end Spec
