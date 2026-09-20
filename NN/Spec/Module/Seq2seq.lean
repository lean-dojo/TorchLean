/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Seq2seq
public import NN.Spec.Module.Core

/-!
# Seq2Seq inference wrapper as an `Spec.Module`

The Seq2Seq spec model defines encoder/decoder math and differentiable training helpers.
This file provides a small inference-oriented `Spec.Module` wrapper so it can be composed/exported.
-/

@[expose] public section


namespace Spec.Module

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Seq2Seq inference module wrapper (one-hot input, greedy decoding). -/
def seq2seq {srcVocabSize tgtVocabSize embedDim hiddenDim srcSeqLen tgtSeqLen : Nat}
  (m : Seq2SeqSpec α srcVocabSize tgtVocabSize embedDim hiddenDim)
  (startToken : Fin tgtVocabSize) :
  Spec.Module α ([srcSeqLen, srcVocabSize]) ([tgtSeqLen, tgtVocabSize]) :=
{
  forward := fun srcOneHot =>
    let sourceEmbeddings := Seq2SeqEmbeddingSpec.forwardOneHot m.sourceEmbedding srcOneHot
    let (_encoderOutputs, encoderHidden) :=
      Seq2SeqRNNEncoderSpec.forward m.encoder sourceEmbeddings
      none
    let (logits, _tokens) :=
      Seq2SeqDecoderSpec.forwardInference m.decoder encoderHidden m.targetEmbedding.embedding
        startToken tgtSeqLen
    logits,
  kind := "Seq2Seq",
  pythonExpr := s!"Seq2SeqInference(src_vocab_size={srcVocabSize}, " ++
        s!"tgt_vocab_size={tgtVocabSize}, embed_dim={embedDim}, " ++
        s!"hidden_dim={hiddenDim}, max_tgt_len={tgtSeqLen}, " ++
        s!"start_token={startToken})"
}

end Spec.Module
