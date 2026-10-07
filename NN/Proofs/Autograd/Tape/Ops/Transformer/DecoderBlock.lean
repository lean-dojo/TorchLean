/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Attention.MaskedMultiHeadSelfAttention
public import NN.Proofs.Autograd.Tape.Ops.Transformer.PostNorm

/-!
# Post-Norm Decoder Core

This module builds a decoder-core SSA graph with a fixed additive score bias and proves its VJP
theorem (`decoderCore_backpropVec_eq_adjoint_fderiv_at`), and names the post-norm decoder-block
composition theorem (`postNormGptDecoderBlock_hasFDerivAt`).

The concrete graph starts with already projected head tensors. Its merge and FFN affine maps
are fixed arguments, not trainable entries in the differentiated context. The score bias is
finite real data; it does not implement Boolean hard or causal masking. These are exact-real
derivative theorems, not a correspondence theorem for the executable decoder or LibTorch.

The map-level composition theorem accepts differentiable residual-pack maps. The attention
composition theorem in `NN.Proofs.Autograd.Tape.Ops.Attention.MaskedMultiHeadSelfAttention`
can supply such a map when the caller provides the projection and merge derivative certificates.
It does not construct a full decoder from a token/parameter context automatically.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Transformer

open Spec TorchLean
open TapeNodes
open DGraph

noncomputable section

/-!
## Concrete fixed-bias decoder-core SSA graph

The concrete graph below starts after the Q/K/V projection split:

`[Q_heads, Kᵀ_heads, V_heads, residual_stream, gamma₁, beta₁, gamma₂, beta₂]`.

It then adds the fixed bias to the scaled attention scores, merges the head output through a
supplied affine map, adds the residual stream, and applies the two post-norm sublayers.
-/

/-- Concrete decoder-core context: head tensors, residual stream, and two LayerNorm
parameter pairs. -/
abbrev ΓDecoderCore (seqLen dModel numHeads headDim : Nat) : List Shape :=
  DirectReshapeAttention.ΓMaskedCore seqLen numHeads headDim ++
    [ LayerNorm.MatShape seqLen dModel
    , LayerNorm.VecShape dModel, LayerNorm.VecShape dModel
    , LayerNorm.VecShape dModel, LayerNorm.VecShape dModel
    ]

/-- Saved tensors for the concrete fixed-bias decoder-core block. -/
abbrev ssDecoderCore (seqLen dModel numHeads headDim dFF : Nat) : List Shape :=
  DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
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

/-- Fixed-bias attention core while carrying residual and LayerNorm parameters. -/
def decoderMaskedCoreDGraph {seqLen dModel numHeads headDim : Nat}
    (c : ℝ)
    (bias : Vec (Spec.Shape.size (DirectReshapeAttention.ScoresShape seqLen numHeads)) := 0) :
    DGraph (ΓDecoderCore seqLen dModel numHeads headDim)
      (DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim) :=
  DGraph.weakenContext
    (DirectReshapeAttention.maskedCoreDGraph
      (n := seqLen) (numHeads := numHeads) (headDim := headDim) c bias)
    [ LayerNorm.MatShape seqLen dModel
    , LayerNorm.VecShape dModel, LayerNorm.VecShape dModel
    , LayerNorm.VecShape dModel, LayerNorm.VecShape dModel
    ]

/-- Head output after the fixed-bias attention core. -/
def idxDecoderHeadOut {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim)
      (DirectReshapeAttention.HeadsShape seqLen numHeads headDim) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := [DirectReshapeAttention.ScoresShape seqLen numHeads,
      DirectReshapeAttention.ScoresShape seqLen numHeads,
      DirectReshapeAttention.ScoresShape seqLen numHeads,
      DirectReshapeAttention.ScoresShape seqLen numHeads])
    (τ := DirectReshapeAttention.HeadsShape seqLen numHeads headDim)

/-- Merged attention output after the supplied output projection. -/
def idxDecoderMergedAttention {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel])
      (LayerNorm.MatShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim)
    (τ := LayerNorm.MatShape seqLen dModel)

/-- Residual input weakened past the merged attention output. -/
def idxDecoderResidualInputAfterMerge {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel])
      (LayerNorm.MatShape seqLen dModel) :=
  Proofs.Idx.weaken
    (idxDecoderResidualInput (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim))
    [LayerNorm.MatShape seqLen dModel]

/-- Sum of the residual stream and projected attention output before the first LayerNorm. -/
def idxDecoderAttentionResidual {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
      (LayerNorm.MatShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel])
    (τ := LayerNorm.MatShape seqLen dModel)

/-- First LayerNorm input triple for the concrete decoder-core graph. -/
def decoderNorm1Inputs {seqLen dModel numHeads headDim : Nat} :
    LayerNorm.Inputs
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
      seqLen dModel :=
  { x := idxDecoderAttentionResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
    gamma := idxDecoderNorm1Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
    beta := idxDecoderNorm1Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel]) }

/-- Decoder graph through the first post-norm attention sublayer. -/
def decoderAfterNorm1Graph {seqLen dModel numHeads headDim : Nat}
    (merge :
      Vec (Spec.Shape.size (DirectReshapeAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (c ε₁ : ℝ)
    (bias : Vec (Spec.Shape.size (DirectReshapeAttention.ScoresShape seqLen numHeads)) := 0) :
    Graph (ΓDecoderCore seqLen dModel numHeads headDim)
      (DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel]) :=
  let g0 := (decoderMaskedCoreDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c bias).g
  let g1 := Graph.snoc g0
    (TapeNodes.affine
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim)
      (sIn := DirectReshapeAttention.HeadsShape seqLen numHeads headDim)
      (sOut := LayerNorm.MatShape seqLen dModel)
      (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      merge mergeBias)
  let g2 := Graph.snoc g1
    (add
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel])
      (s := LayerNorm.MatShape seqLen dModel)
      (idxDecoderResidualInputAfterMerge (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      (idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim)))
  Graph.snoc g2
    (LayerNorm.wholeNode
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
      (m := seqLen) (n := dModel)
      (decoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε₁)

/-- First decoder post-norm output. -/
def idxDecoderNorm1Out {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
    (τ := LayerNorm.MatShape seqLen dModel)

/-- First decoder post-norm output weakened through FFN intermediates. -/
def idxDecoderNorm1OutAfterFfn {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
          SeqFFNModelShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  Proofs.Idx.weaken
    (idxDecoderNorm1Out (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim))
    [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
      SeqFFNModelShape seqLen dModel]

/-- First FFN affine output in the concrete decoder graph. -/
def idxDecoderFfnHiddenPre {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF])
      (SeqFFNHiddenShape seqLen dFF) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
        LayerNorm.MatShape seqLen dModel])
    (τ := SeqFFNHiddenShape seqLen dFF)

/-- FFN activation output in the concrete decoder graph. -/
def idxDecoderFfnHiddenAct {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
      (SeqFFNHiddenShape seqLen dFF) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
        LayerNorm.MatShape seqLen dModel, SeqFFNHiddenShape seqLen dFF])
    (τ := SeqFFNHiddenShape seqLen dFF)

/-- FFN projection output in the concrete decoder graph. -/
def idxDecoderFfnProjected {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
          SeqFFNModelShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
        LayerNorm.MatShape seqLen dModel, SeqFFNHiddenShape seqLen dFF,
        SeqFFNHiddenShape seqLen dFF])
    (τ := SeqFFNModelShape seqLen dModel)

/-- FFN residual output before the second decoder LayerNorm. -/
def idxDecoderFfnResidual {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        ssSeqFFNResidual seqLen dModel dFF)
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
    (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
      [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
        LayerNorm.MatShape seqLen dModel, SeqFFNHiddenShape seqLen dFF,
        SeqFFNHiddenShape seqLen dFF, SeqFFNModelShape seqLen dModel])
    (τ := SeqFFNModelShape seqLen dModel)

/-- Decoder graph through the FFN residual. -/
def decoderFfnResidualGraph {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Spec.Shape.size (DirectReshapeAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ : ℝ)
    (bias : Vec (Spec.Shape.size (DirectReshapeAttention.ScoresShape seqLen numHeads)) := 0) :
    Graph (ΓDecoderCore seqLen dModel numHeads headDim)
      (DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF) :=
  let g1 := decoderAfterNorm1Graph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) merge mergeBias c ε₁ bias
  let g2 := Graph.snoc g1
    (TapeNodes.affine
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel])
      (sIn := SeqFFNModelShape seqLen dModel) (sOut := SeqFFNHiddenShape seqLen dFF)
      (idxDecoderNorm1Out (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      fc1 b1)
  let g3 := Graph.snoc g2
    (gelu
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ [SeqFFNHiddenShape seqLen dFF])
      (s := SeqFFNHiddenShape seqLen dFF)
      (idxDecoderFfnHiddenPre (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))
  let g4 := Graph.snoc g3
    (TapeNodes.affine
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
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
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
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
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++
        ssSeqFFNResidual seqLen dModel dFF)
      seqLen dModel :=
  { x := idxDecoderFfnResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
    gamma := idxDecoderNorm2Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
    beta := idxDecoderNorm2Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF) }

/-- Concrete SSA graph for one additive-bias decoder-core block. -/
def decoderCoreGraph {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Spec.Shape.size (DirectReshapeAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Spec.Shape.size (DirectReshapeAttention.ScoresShape seqLen numHeads)) := 0) :
    Graph (ΓDecoderCore seqLen dModel numHeads headDim)
      (ssDecoderCore seqLen dModel numHeads headDim dFF) :=
  Graph.snoc
    (decoderFfnResidualGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      merge mergeBias fc1 b1 fc2 b2 c ε₁ bias)
    (LayerNorm.wholeNode
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (decoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      ε₂)

/-- Pointwise analytic correctness for the decoder graph through the FFN residual. -/
def decoderFfnResidualGraphFDerivCorrectAt {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Spec.Shape.size (DirectReshapeAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ : ℝ)
    (bias : Vec (Spec.Shape.size (DirectReshapeAttention.ScoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (ΓDecoderCore seqLen dModel numHeads headDim))
    (hε₁ : 0 < ε₁) :
    GraphFDerivCorrectAt
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
      (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
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
      (ss := DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim) dgCore.g xV :=
    GraphFDerivCorrect.at dgCore.hg xV
  refine ⟨⟨⟨⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩
  · simpa [decoderAfterNorm1Graph, dgCore] using hgCore
  · exact NodeFDerivCorrect.at
      (TapeNodes.affineFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim)
        (sIn := DirectReshapeAttention.HeadsShape seqLen numHeads headDim)
        (sOut := LayerNorm.MatShape seqLen dModel)
        (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        merge mergeBias)
      _
  · exact NodeFDerivCorrect.at
      (addFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
          [LayerNorm.MatShape seqLen dModel])
        (s := LayerNorm.MatShape seqLen dModel)
        (idxDecoderResidualInputAfterMerge (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        (idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim)))
      _
  · exact LayerNorm.wholeNodeFDerivCorrectAt
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel])
      (m := seqLen) (n := dModel)
      (decoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε₁ _ hε₁
  · exact NodeFDerivCorrect.at
      (TapeNodes.affineFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
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
          DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
          [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
            LayerNorm.MatShape seqLen dModel] ++ [SeqFFNHiddenShape seqLen dFF])
        (s := SeqFFNHiddenShape seqLen dFF)
        (idxDecoderFfnHiddenPre (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))
      _
  · exact NodeFDerivCorrect.at
      (TapeNodes.affineFderiv
        (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
          DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
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
          DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
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
      Vec (Spec.Shape.size (DirectReshapeAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Spec.Shape.size (DirectReshapeAttention.ScoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (ΓDecoderCore seqLen dModel numHeads headDim))
    (hε₁ : 0 < ε₁)
    (hε₂ : 0 < ε₂) :
    GraphFDerivCorrectAt
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim)
      (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
      (decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias)
      xV :=
  ⟨decoderFfnResidualGraphFDerivCorrectAt
    (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
    (dFF := dFF) merge mergeBias fc1 b1 fc2 b2 c ε₁ bias xV hε₁,
    LayerNorm.wholeNodeFDerivCorrectAt
      (Γ := ΓDecoderCore seqLen dModel numHeads headDim ++
        DirectReshapeAttention.ssMaskedCore seqLen numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel, LayerNorm.MatShape seqLen dModel,
          LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (decoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      ε₂ _ hε₂⟩

/-- End-to-end VJP theorem for the concrete additive-bias decoder-core graph. -/
theorem decoderCore_backpropVec_eq_adjoint_fderiv_at
    {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Spec.Shape.size (DirectReshapeAttention.HeadsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (mergeBias : Vec (Spec.Shape.size (LayerNorm.MatShape seqLen dModel)))
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Spec.Shape.size (DirectReshapeAttention.ScoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (ΓDecoderCore seqLen dModel numHeads headDim))
    (seedV : CtxVec (ΓDecoderCore seqLen dModel numHeads headDim ++
      ssDecoderCore seqLen dModel numHeads headDim dFF))
    (hε₁ : 0 < ε₁)
    (hε₂ : 0 < ε₂) :
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
      (dFF := dFF) merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias xV hε₁ hε₂)

/--
Post-norm composition theorem under its decoder alias.

This is `twoSublayerPostNormBlock_hasFDerivAt` with the same supplied-map hypotheses.
`DirectReshapeAttention.projectedMaskedAttention_hasFDerivAt` can discharge the attention
map's hypothesis for a fixed-bias core with supplied projection and merge certificates.
Neither theorem constructs Boolean causal masking or proves executable-model correspondence.
-/
alias postNormGptDecoderBlock_hasFDerivAt := twoSublayerPostNormBlock_hasFDerivAt

end

end Transformer
end Autograd
end Proofs
