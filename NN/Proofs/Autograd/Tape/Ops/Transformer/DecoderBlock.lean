/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Attention.MaskedMultiHeadSelfAttention
public import NN.Proofs.Autograd.Tape.Ops.Transformer.EncoderBlock

/-!
# GPT-Style Decoder Block

This module packages the post-norm decoder-block composition theorem.  The attention front half is
supplied as a differentiable residual-pack map; the concrete additive-bias attention core and its
projection/merge composition theorem live in
`NN.Proofs.Autograd.Tape.Ops.Attention.MaskedMultiHeadSelfAttention`.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Transformer

open Spec
open TapeNodes
open DGraph

universe u

noncomputable section

/-!
## Concrete finite-mask decoder-core SSA graph

The concrete graph below starts after the Q/K/V projection split:

`[Q_heads, Kᵀ_heads, V_heads, residual_stream, gamma₁, beta₁, gamma₂, beta₂]`.

It then runs finite-mask split-head attention, merges the head output through a supplied affine map,
adds the residual stream, and applies the two post-norm sublayers.  A separate projection theorem can
feed this graph from a full token/parameter context.
-/

/-- Concrete decoder-core context: masked attention core inputs, residual stream, and two LayerNorm parameter pairs. -/
abbrev ΓDecoderCore (seqLen dModel numHeads headDim : Nat) : List Shape :=
  MultiHeadAttention.ΓMaskedCore seqLen numHeads headDim ++
    [ LayerNorm.MatShape seqLen dModel
    , LayerNorm.VecShape dModel, LayerNorm.VecShape dModel
    , LayerNorm.VecShape dModel, LayerNorm.VecShape dModel
    ]

/-- Saved tensors for the concrete finite-mask decoder-core block. -/
abbrev ssDecoderCore (seqLen dModel numHeads headDim dFF : Nat) : List Shape :=
  MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
    [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
      LayerNorm.MatShape seqLen dModel] ++
    ssSeqFFNResidual seqLen dModel dFF ++
    [LayerNorm.MatShape seqLen dModel]

/-- Residual-stream input in the decoder-core context. -/
def idxDecoderResidualInput {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓDecoderCore seqLen dModel numHeads headDim ++ ss) (LayerNorm.MatShape seqLen dModel) :=
  ⟨⟨3, by simp⟩, by simp⟩

/-- First LayerNorm scale parameter in the decoder-core context. -/
def idxDecoderNorm1Gamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓDecoderCore seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  ⟨⟨4, by simp⟩, by simp⟩

/-- First LayerNorm shift parameter in the decoder-core context. -/
def idxDecoderNorm1Beta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓDecoderCore seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  ⟨⟨5, by simp⟩, by simp⟩

/-- Second LayerNorm scale parameter in the decoder-core context. -/
def idxDecoderNorm2Gamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓDecoderCore seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  ⟨⟨6, by simp⟩, by simp⟩

/-- Second LayerNorm shift parameter in the decoder-core context. -/
def idxDecoderNorm2Beta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓDecoderCore seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  ⟨⟨7, by simp⟩, by simp⟩

/-- Masked attention core while carrying residual and LayerNorm parameters. -/
def decoderMaskedCoreDGraph {seqLen dModel numHeads headDim : Nat}
    (c : ℝ) (bias : Vec (Shape.size (MultiHeadAttention.ScoresShape seqLen numHeads)) := 0) :
    DGraph (ΓDecoderCore seqLen dModel numHeads headDim)
      (MultiHeadAttention.ssMaskedCore seqLen numHeads headDim) :=
  DGraph.weakenContext
    (MultiHeadAttention.maskedCoreDGraph
      (n := seqLen) (numHeads := numHeads) (headDim := headDim) c bias)
    [ LayerNorm.MatShape seqLen dModel
    , LayerNorm.VecShape dModel, LayerNorm.VecShape dModel
    , LayerNorm.VecShape dModel, LayerNorm.VecShape dModel
    ]

/-- Split-head masked attention output after the masked core. -/
def idxDecoderHeadOut {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
      (MultiHeadAttention.HeadsShape seqLen numHeads headDim) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := [MultiHeadAttention.ScoresShape seqLen numHeads,
      MultiHeadAttention.ScoresShape seqLen numHeads,
      MultiHeadAttention.ScoresShape seqLen numHeads,
      MultiHeadAttention.ScoresShape seqLen numHeads])
    (τ := MultiHeadAttention.HeadsShape seqLen numHeads headDim)

/-- Merged attention output after the supplied output projection. -/
def idxDecoderMergedAttention {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel])
      (LayerNorm.MatShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
    (τ := LayerNorm.MatShape seqLen dModel)

/-- Residual input weakened past the merged attention output. -/
def idxDecoderResidualInputAfterMerge {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel])
      (LayerNorm.MatShape seqLen dModel) :=
  _root_.Proofs.Autograd.Idx.weaken
    (idxDecoderResidualInput (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim))
    [LayerNorm.MatShape seqLen dModel]

/-- Residual attention stream `x + masked_attention(x)` before the first LayerNorm. -/
def idxDecoderAttentionResidual {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
      (LayerNorm.MatShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++ [LayerNorm.MatShape seqLen dModel])
    (τ := LayerNorm.MatShape seqLen dModel)

/-- First LayerNorm input triple for the concrete decoder-core graph. -/
def decoderNorm1Inputs {seqLen dModel numHeads headDim : Nat} :
    LayerNorm.Inputs
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
      seqLen dModel :=
  { x := idxDecoderAttentionResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
    gamma := idxDecoderNorm1Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
    beta := idxDecoderNorm1Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel]) }

/-- Decoder graph through the first post-norm masked-attention sublayer. -/
def decoderAfterNorm1Graph {seqLen dModel numHeads headDim : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (c ε₁ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.ScoresShape seqLen numHeads)) := 0) :
    Graph (ΓDecoderCore seqLen dModel numHeads headDim)
      (MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel]) :=
  let g0 := (decoderMaskedCoreDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c bias).g
  let g1 := Graph.snoc g0
    (TapeNodes.affine
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
      (sIn := MultiHeadAttention.HeadsShape seqLen numHeads headDim)
      (sOut := LayerNorm.MatShape seqLen dModel)
      (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      merge mergeBias)
  let g2 := Graph.snoc g1
    (add
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++ [LayerNorm.MatShape seqLen dModel])
      (s := LayerNorm.MatShape seqLen dModel)
      (idxDecoderResidualInputAfterMerge (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      (idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim)))
  Graph.snoc g2
    (LayerNorm.wholeNode
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
      (m := seqLen) (n := dModel)
      (decoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε₁)

/-- First decoder post-norm output. -/
def idxDecoderNorm1Out {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
    (τ := LayerNorm.MatShape seqLen dModel)

/-- First decoder post-norm output weakened through FFN intermediates. -/
def idxDecoderNorm1OutAfterFfn {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
          SeqFFNModelShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  _root_.Proofs.Autograd.Idx.weaken
    (idxDecoderNorm1Out (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim))
    [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
      SeqFFNModelShape seqLen dModel]

/-- First FFN affine output in the concrete decoder graph. -/
def idxDecoderFfnHiddenPre {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF])
      (SeqFFNHiddenShape seqLen dFF) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
        LayerNorm.MatShape seqLen dModel])
    (τ := SeqFFNHiddenShape seqLen dFF)

/-- FFN activation output in the concrete decoder graph. -/
def idxDecoderFfnHiddenAct {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
      (SeqFFNHiddenShape seqLen dFF) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
        LayerNorm.MatShape seqLen dModel, SeqFFNHiddenShape seqLen dFF])
    (τ := SeqFFNHiddenShape seqLen dFF)

/-- FFN projection output in the concrete decoder graph. -/
def idxDecoderFfnProjected {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
          SeqFFNModelShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
        LayerNorm.MatShape seqLen dModel, SeqFFNHiddenShape seqLen dFF,
        SeqFFNHiddenShape seqLen dFF])
    (τ := SeqFFNModelShape seqLen dModel)

/-- FFN residual output before the second decoder LayerNorm. -/
def idxDecoderFfnResidual {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        ssSeqFFNResidual seqLen dModel dFF)
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
        LayerNorm.MatShape seqLen dModel, SeqFFNHiddenShape seqLen dFF,
        SeqFFNHiddenShape seqLen dFF, SeqFFNModelShape seqLen dModel])
    (τ := SeqFFNModelShape seqLen dModel)

/-- Decoder graph through the FFN residual. -/
def decoderFfnResidualGraph {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.ScoresShape seqLen numHeads)) := 0) :
    Graph (ΓDecoderCore seqLen dModel numHeads headDim)
      (MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF) :=
  let g1 := decoderAfterNorm1Graph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) merge mergeBias c ε₁ bias
  let g2 := Graph.snoc g1
    (TapeNodes.affine
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel])
      (sIn := SeqFFNModelShape seqLen dModel) (sOut := SeqFFNHiddenShape seqLen dFF)
      (idxDecoderNorm1Out (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      fc1 b1)
  let g3 := Graph.snoc g2
    (gelu
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ [SeqFFNHiddenShape seqLen dFF])
      (s := SeqFFNHiddenShape seqLen dFF)
      (idxDecoderFfnHiddenPre (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))
  let g4 := Graph.snoc g3
    (TapeNodes.affine
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
      (sIn := SeqFFNHiddenShape seqLen dFF) (sOut := SeqFFNModelShape seqLen dModel)
      (idxDecoderFfnHiddenAct (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      fc2 b2)
  Graph.snoc g4
    (add
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
          SeqFFNModelShape seqLen dModel])
      (s := SeqFFNModelShape seqLen dModel)
      (idxDecoderNorm1OutAfterFfn (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      (idxDecoderFfnProjected (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))

/-- Second LayerNorm input triple for the concrete decoder-core graph. -/
def decoderNorm2Inputs {seqLen dModel numHeads headDim dFF : Nat} :
    LayerNorm.Inputs
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        ssSeqFFNResidual seqLen dModel dFF)
      seqLen dModel :=
  { x := idxDecoderFfnResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
    gamma := idxDecoderNorm2Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
    beta := idxDecoderNorm2Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF) }

/-- Concrete SSA graph for one additive-bias decoder-core block. -/
def decoderCoreGraph {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.ScoresShape seqLen numHeads)) := 0) :
    Graph (ΓDecoderCore seqLen dModel numHeads headDim)
      (ssDecoderCore seqLen dModel numHeads headDim dFF) :=
  Graph.snoc
    (decoderFfnResidualGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      merge mergeBias fc1 b1 fc2 b2 c ε₁ bias)
    (LayerNorm.wholeNode
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (decoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      ε₂)

/-- Pointwise analytic correctness for the decoder graph through the FFN residual. -/
def decoderFfnResidualGraphFDerivCorrectAt {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.ScoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (ΓDecoderCore seqLen dModel numHeads headDim))
    (hNorm1 :
      NodeFDerivCorrectAt
        (LayerNorm.wholeNode
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
            MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
          (m := seqLen) (n := dModel)
          (decoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim))
          ε₁)
        (Graph.evalVec
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
          (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
          (Graph.snoc
            (Graph.snoc
              (decoderMaskedCoreDGraph (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim) c bias).g
              (TapeNodes.affine
                (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
                  MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
                (sIn := MultiHeadAttention.HeadsShape seqLen numHeads headDim)
                (sOut := LayerNorm.MatShape seqLen dModel)
                (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
                  (numHeads := numHeads) (headDim := headDim))
                merge mergeBias))
            (add
              (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
                MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
                [LayerNorm.MatShape seqLen dModel])
              (s := LayerNorm.MatShape seqLen dModel)
              (idxDecoderResidualInputAfterMerge (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim))
              (idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim))))
          xV)) :
    GraphFDerivCorrectAt
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
      (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
      (decoderFfnResidualGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        merge mergeBias fc1 b1 fc2 b2 c ε₁ bias)
      xV := by
  classical
  let dgCore := decoderMaskedCoreDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c bias
  let hgCore : GraphFDerivCorrectAt
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
      (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim) dgCore.g xV :=
    DGraph.graphFDerivCorrectAtOfCorrect dgCore.hg xV
  refine ⟨⟨⟨⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩
  · simpa [decoderAfterNorm1Graph, dgCore] using hgCore
  · exact NodeFDerivCorrect.at
      (TapeNodes.affineFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
        (sIn := MultiHeadAttention.HeadsShape seqLen numHeads headDim)
        (sOut := LayerNorm.MatShape seqLen dModel)
        (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        merge mergeBias)
      _
  · exact NodeFDerivCorrect.at
      (addFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++ [LayerNorm.MatShape seqLen dModel])
        (s := LayerNorm.MatShape seqLen dModel)
        (idxDecoderResidualInputAfterMerge (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        (idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim)))
      _
  · simpa [decoderAfterNorm1Graph, dgCore] using hNorm1
  · exact NodeFDerivCorrect.at
      (TapeNodes.affineFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
          [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
            LayerNorm.MatShape seqLen dModel])
        (sIn := SeqFFNModelShape seqLen dModel) (sOut := SeqFFNHiddenShape seqLen dFF)
        (idxDecoderNorm1Out (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        fc1 b1)
      _
  · exact NodeFDerivCorrect.at
      (geluFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
          [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
            LayerNorm.MatShape seqLen dModel] ++ [SeqFFNHiddenShape seqLen dFF])
        (s := SeqFFNHiddenShape seqLen dFF)
        (idxDecoderFfnHiddenPre (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))
      _
  · exact NodeFDerivCorrect.at
      (TapeNodes.affineFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
          [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
            LayerNorm.MatShape seqLen dModel] ++
          [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
        (sIn := SeqFFNHiddenShape seqLen dFF) (sOut := SeqFFNModelShape seqLen dModel)
        (idxDecoderFfnHiddenAct (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
        fc2 b2)
      _
  · exact NodeFDerivCorrect.at
      (addFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
          [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
            LayerNorm.MatShape seqLen dModel] ++
          [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
            SeqFFNModelShape seqLen dModel])
        (s := SeqFFNModelShape seqLen dModel)
        (idxDecoderNorm1OutAfterFfn (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
        (idxDecoderFfnProjected (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))
      _

/-- Pointwise analytic correctness for the complete concrete decoder-core graph. -/
def decoderCoreGraphFDerivCorrectAt {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.ScoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (ΓDecoderCore seqLen dModel numHeads headDim))
    (hNorm1 :
      NodeFDerivCorrectAt
        (LayerNorm.wholeNode
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
            MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
          (m := seqLen) (n := dModel)
          (decoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim))
          ε₁)
        (Graph.evalVec
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
          (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
          (Graph.snoc
            (Graph.snoc
              (decoderMaskedCoreDGraph (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim) c bias).g
              (TapeNodes.affine
                (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
                  MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
                (sIn := MultiHeadAttention.HeadsShape seqLen numHeads headDim)
                (sOut := LayerNorm.MatShape seqLen dModel)
                (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
                  (numHeads := numHeads) (headDim := headDim))
                merge mergeBias))
            (add
              (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
                MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
                [LayerNorm.MatShape seqLen dModel])
              (s := LayerNorm.MatShape seqLen dModel)
              (idxDecoderResidualInputAfterMerge (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim))
              (idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim))))
          xV))
    (hNorm2 :
      NodeFDerivCorrectAt
        (LayerNorm.wholeNode
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
            MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
              LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
          (m := seqLen) (n := dModel)
          (decoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
          ε₂)
        (Graph.evalVec
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
          (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
              LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
          (decoderFfnResidualGraph (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
            merge mergeBias fc1 b1 fc2 b2 c ε₁ bias)
          xV)) :
    GraphFDerivCorrectAt
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
      (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
      (decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias)
      xV := by
  exact
    ⟨decoderFfnResidualGraphFDerivCorrectAt
      (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
      (dFF := dFF) merge mergeBias fc1 b1 fc2 b2 c ε₁ bias xV hNorm1, hNorm2⟩

/-- End-to-end VJP theorem for the concrete additive-bias decoder-core graph. -/
theorem decoderCore_backpropVec_eq_adjoint_fderiv_at
    {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.ScoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (ΓDecoderCore seqLen dModel numHeads headDim))
    (seedV : CtxVec (ΓDecoderCore seqLen dModel numHeads headDim ++
      ssDecoderCore seqLen dModel numHeads headDim dFF))
    (hNorm1 :
      NodeFDerivCorrectAt
        (LayerNorm.wholeNode
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
            MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
          (m := seqLen) (n := dModel)
          (decoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim))
          ε₁)
        (Graph.evalVec
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
          (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
          (Graph.snoc
            (Graph.snoc
              (decoderMaskedCoreDGraph (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim) c bias).g
              (TapeNodes.affine
                (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
                  MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
                (sIn := MultiHeadAttention.HeadsShape seqLen numHeads headDim)
                (sOut := LayerNorm.MatShape seqLen dModel)
                (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
                  (numHeads := numHeads) (headDim := headDim))
                merge mergeBias))
            (add
              (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
                MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
                [LayerNorm.MatShape seqLen dModel])
              (s := LayerNorm.MatShape seqLen dModel)
              (idxDecoderResidualInputAfterMerge (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim))
              (idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim))))
          xV))
    (hNorm2 :
      NodeFDerivCorrectAt
        (LayerNorm.wholeNode
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
            MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
              LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
          (m := seqLen) (n := dModel)
          (decoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
          ε₂)
        (Graph.evalVec
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
          (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim ++
            [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
              LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
          (decoderFfnResidualGraph (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
            merge mergeBias fc1 b1 fc2 b2 c ε₁ bias)
          xV)) :
    Graph.backpropVec
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
        (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
        (decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
          merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias)
        xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
          (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
          (decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
            merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias))
        xV).adjoint seedV :=
  Graph.backpropVec_eq_adjoint_fderiv_at
    (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
    (g := decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias)
    xV seedV
    (decoderCoreGraphFDerivCorrectAt
      (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
      (dFF := dFF) merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias xV hNorm1 hNorm2)

/--
Projection-to-residual bridge for a GPT-style masked decoder attention sublayer.

The concrete decoder-core graph above starts from already split `Q`, `Kᵀ`, and `V` heads.  This
theorem is the reusable front-end hook for full GPT blocks: any differentiable projection/split
stage may build those heads, and any differentiable merge/residual pack may turn the masked
attention trace into the first LayerNorm input triple `[x + MaskedMHA(x), gamma₁, beta₁]`.

Combine this theorem with `postNormGptDecoderBlock_hasFDerivAt` below to get the full projected
finite-mask decoder-block differentiability statement.
-/
theorem projectedMaskedDecoderAttentionPack_hasFDerivAt
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {seqLen dModel numHeads headDim : Nat}
    (c : ℝ) (bias : Vec (Shape.size (MultiHeadAttention.ScoresShape seqLen numHeads)) := 0)
    (projectPack : E → CtxVec (MultiHeadAttention.ΓMaskedCore seqLen numHeads headDim))
    (DprojectPack : E →L[ℝ] CtxVec (MultiHeadAttention.ΓMaskedCore seqLen numHeads headDim))
    (attentionPack :
      CtxVec (MultiHeadAttention.ΓMaskedCore seqLen numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim) →
        CtxVec (ΓPostNorm seqLen dModel))
    (DattentionPack :
      CtxVec (MultiHeadAttention.ΓMaskedCore seqLen numHeads headDim ++
        MultiHeadAttention.ssMaskedCore seqLen numHeads headDim) →L[ℝ]
        CtxVec (ΓPostNorm seqLen dModel))
    (x : E)
    (hProject : HasFDerivAt projectPack DprojectPack x)
    (hAttentionPack :
      HasFDerivAt attentionPack DattentionPack
        (Graph.evalVec
          (Γ := MultiHeadAttention.ΓMaskedCore seqLen numHeads headDim)
          (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
          (MultiHeadAttention.maskedCoreDGraph
            (n := seqLen) (numHeads := numHeads) (headDim := headDim) c bias).g
          (projectPack x))) :
    HasFDerivAt
      (fun z : E =>
        attentionPack
          (Graph.evalVec
            (Γ := MultiHeadAttention.ΓMaskedCore seqLen numHeads headDim)
            (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
            (MultiHeadAttention.maskedCoreDGraph
              (n := seqLen) (numHeads := numHeads) (headDim := headDim) c bias).g
            (projectPack z)))
      (DattentionPack.comp
        ((fderiv ℝ
          (Graph.evalVec
            (Γ := MultiHeadAttention.ΓMaskedCore seqLen numHeads headDim)
            (ss := MultiHeadAttention.ssMaskedCore seqLen numHeads headDim)
            (MultiHeadAttention.maskedCoreDGraph
              (n := seqLen) (numHeads := numHeads) (headDim := headDim) c bias).g)
          (projectPack x)).comp DprojectPack))
      x :=
  MultiHeadAttention.projectedMaskedAttention_hasFDerivAt
    (n := seqLen) (numHeads := numHeads) (headDim := headDim)
    c bias projectPack DprojectPack attentionPack DattentionPack x hProject hAttentionPack



/--
Fréchet differentiability of a GPT-style post-norm decoder block.

`maskedAttentionPack` builds the first LayerNorm input triple
`[x + MaskedMHA(x), gamma₁, beta₁]`.  Instantiate its differentiability hypothesis with
`MultiHeadAttention.projectedMaskedAttention_hasFDerivAt` when the attention sublayer is built from
the proved finite-mask split-head core.
-/
theorem postNormGptDecoderBlock_hasFDerivAt
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {seqLen dModel : Nat} (ε₁ ε₂ : ℝ)
    (maskedAttentionPack : E → CtxVec (ΓPostNorm seqLen dModel))
    (DmaskedAttentionPack : E →L[ℝ] CtxVec (ΓPostNorm seqLen dModel))
    (ffnPack :
      CtxVec (ΓPostNorm seqLen dModel ++ ssPostNorm seqLen dModel) →
        CtxVec (ΓPostNorm seqLen dModel))
    (DffnPack :
      CtxVec (ΓPostNorm seqLen dModel ++ ssPostNorm seqLen dModel) →L[ℝ]
        CtxVec (ΓPostNorm seqLen dModel))
    (x : E)
    (hMaskedAttentionPack : HasFDerivAt maskedAttentionPack DmaskedAttentionPack x)
    (hNorm1VarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε₁)
            (maskedAttentionPack x)) i)
    (hNorm1StdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε₁)
            (maskedAttentionPack x)) i ≠ 0)
    (hFfnPack :
      HasFDerivAt ffnPack DffnPack
        (Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁)
          (maskedAttentionPack x)))
    (hNorm2VarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε₂)
            (ffnPack
              (Graph.evalVec
                (Γ := ΓPostNorm seqLen dModel)
                (ss := ssPostNorm seqLen dModel)
                (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁)
                (maskedAttentionPack x)))) i)
    (hNorm2StdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε₂)
            (ffnPack
              (Graph.evalVec
                (Γ := ΓPostNorm seqLen dModel)
                (ss := ssPostNorm seqLen dModel)
                (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁)
                (maskedAttentionPack x)))) i ≠ 0) :
    HasFDerivAt
      (fun z : E =>
        Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₂)
          (ffnPack
            (Graph.evalVec
              (Γ := ΓPostNorm seqLen dModel)
              (ss := ssPostNorm seqLen dModel)
              (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁)
              (maskedAttentionPack z))))
      ((fderiv ℝ
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := ssPostNorm seqLen dModel)
            (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₂))
          (ffnPack
            (Graph.evalVec
              (Γ := ΓPostNorm seqLen dModel)
              (ss := ssPostNorm seqLen dModel)
              (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁)
              (maskedAttentionPack x)))).comp
        (DffnPack.comp
          ((fderiv ℝ
            (Graph.evalVec
              (Γ := ΓPostNorm seqLen dModel)
              (ss := ssPostNorm seqLen dModel)
              (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁))
            (maskedAttentionPack x)).comp DmaskedAttentionPack)))
      x :=
  twoSublayerPostNormBlock_hasFDerivAt
    (seqLen := seqLen) (dModel := dModel) ε₁ ε₂
    maskedAttentionPack DmaskedAttentionPack ffnPack DffnPack x
    hMaskedAttentionPack hNorm1VarEpsPos hNorm1StdNe0 hFfnPack hNorm2VarEpsPos hNorm2StdNe0

end

end Transformer
end Autograd
end Proofs
