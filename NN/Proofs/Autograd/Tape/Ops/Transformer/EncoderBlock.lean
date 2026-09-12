/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Transformer.PostNorm

/-!
# Post-Norm Transformer Encoder Block

This module names the full two-sublayer post-norm encoder-block theorem.

The theorem is stated at the block boundary:

`x ↦ LN₂(FFN(LN₁(MHA(x) + x)) + LN₁(MHA(x) + x))`.

The two sublayers already have graph-level VJP theorems in `PostNorm.lean`.  The theorem here is the
composition result that a model proof imports when it needs the whole encoder block as one
differentiable map.  A future lowering pass can still build one concrete SSA graph for the exact
runtime layout; this theorem is the mathematical block contract that such a graph must implement.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Transformer

open Spec TorchLean
open TapeNodes
open DGraph

universe u

noncomputable section

/-!
## Concrete SSA encoder block

The definitions below assemble one executable-style proof graph for a post-norm encoder block.  The
context is:

`[x, Wq, Wk, Wv, Wo, gamma₁, beta₁, gamma₂, beta₂]`.

The FFN affine maps are supplied as fixed sequence-level linear maps, matching the interface used by
`seqFfnResidualDGraph`.
-/

/-- Full encoder-block context: MHA parameters, first LayerNorm affine parameters, second LayerNorm
affine parameters. -/
abbrev ΓEncoderBlock (seqLen dModel numHeads headDim : Nat) : List Shape :=
  ΓMHAWithNorm seqLen dModel numHeads headDim ++
    [LayerNorm.VecShape dModel, LayerNorm.VecShape dModel]

/-- Saved tensors for one post-norm encoder block. -/
abbrev ssEncoderBlock (seqLen dModel numHeads headDim dFF : Nat) : List Shape :=
  ssMHAResidual seqLen dModel numHeads headDim ++
    [LayerNorm.MatShape seqLen dModel] ++
    ssSeqFFNResidual seqLen dModel dFF ++
    [LayerNorm.MatShape seqLen dModel]

/-- First LayerNorm scale parameter in the full encoder-block context. -/
def idxEncoderNorm1Gamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓEncoderBlock seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  ⟨⟨5, by simp [ΓEncoderBlock]⟩, by simp [ΓEncoderBlock]⟩

/-- First LayerNorm shift parameter in the full encoder-block context. -/
def idxEncoderNorm1Beta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓEncoderBlock seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  ⟨⟨6, by simp [ΓEncoderBlock]⟩, by simp [ΓEncoderBlock]⟩

/-- Second LayerNorm scale parameter in the full encoder-block context. -/
def idxEncoderNorm2Gamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓEncoderBlock seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  ⟨⟨7, by simp [ΓEncoderBlock]⟩, by simp [ΓEncoderBlock]⟩

/-- Second LayerNorm shift parameter in the full encoder-block context. -/
def idxEncoderNorm2Beta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓEncoderBlock seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  ⟨⟨8, by simp [ΓEncoderBlock]⟩, by simp [ΓEncoderBlock]⟩

/-- Residual-attention prefix while carrying both LayerNorm parameter pairs. -/
def encoderMhaResidualDGraph {seqLen dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (ΓEncoderBlock seqLen dModel numHeads headDim)
      (ssMHAResidual seqLen dModel numHeads headDim) :=
  DGraph.weakenContext
    (mhaResidualDGraph (n := seqLen) (dModel := dModel) (numHeads := numHeads)
      (headDim := headDim) c)
    [LayerNorm.VecShape dModel, LayerNorm.VecShape dModel,
      LayerNorm.VecShape dModel, LayerNorm.VecShape dModel]

/-- Residual stream `x + MHA(x)` after the attention prefix. -/
def idxEncoderMhaResidual {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      (LayerNorm.MatShape seqLen dModel) :=
  Idx.last (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssMHA seqLen dModel numHeads headDim)
    (τ := MultiHeadAttention.XShape seqLen dModel)

/-- First LayerNorm input triple in the concrete encoder block. -/
def encoderNorm1Inputs {seqLen dModel numHeads headDim : Nat} :
    LayerNorm.Inputs
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      seqLen dModel :=
  { x := idxEncoderMhaResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
    gamma := idxEncoderNorm1Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim)
    beta := idxEncoderNorm1Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim) }

/-- Encoder graph through the first post-norm attention sublayer. -/
def encoderAfterNorm1Graph {seqLen dModel numHeads headDim : Nat} (c ε₁ : ℝ) :
    Graph (ΓEncoderBlock seqLen dModel numHeads headDim)
      (ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel]) :=
  .snoc
    (encoderMhaResidualDGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) c).g
    (LayerNorm.wholeNode
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      (m := seqLen) (n := dModel)
      (encoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε₁)

/-- First post-norm sublayer output. -/
def idxEncoderNorm1Out {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim)
    (τ := LayerNorm.MatShape seqLen dModel)

/-- First post-norm output weakened through the FFN intermediates. -/
def idxEncoderNorm1OutAfterFfn {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
          SeqFFNModelShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  Proofs.Idx.weaken
    (idxEncoderNorm1Out (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim))
    [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF, SeqFFNModelShape seqLen dModel]

/-- First FFN affine output in the full encoder graph. -/
def idxEncoderFfnHiddenPre {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF])
      (SeqFFNHiddenShape seqLen dFF) :=
  Idx.last (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel])
    (τ := SeqFFNHiddenShape seqLen dFF)

/-- FFN activation output in the full encoder graph. -/
def idxEncoderFfnHiddenAct {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
      (SeqFFNHiddenShape seqLen dFF) :=
  Idx.last (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel,
      SeqFFNHiddenShape seqLen dFF])
    (τ := SeqFFNHiddenShape seqLen dFF)

/-- FFN projection output in the full encoder graph. -/
def idxEncoderFfnProjected {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
          SeqFFNModelShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel,
      SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
    (τ := SeqFFNModelShape seqLen dModel)

/-- FFN residual output after the second sublayer residual add. -/
def idxEncoderFfnResidual {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        ssSeqFFNResidual seqLen dModel dFF)
      (SeqFFNModelShape seqLen dModel) :=
  Idx.last (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel,
      SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF, SeqFFNModelShape seqLen dModel])
    (τ := SeqFFNModelShape seqLen dModel)

/-- Graph through the FFN residual part after the first LayerNorm. -/
def encoderFfnResidualGraph {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ : ℝ) :
    Graph (ΓEncoderBlock seqLen dModel numHeads headDim)
      (ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        ssSeqFFNResidual seqLen dModel dFF) :=
  let g1 := encoderAfterNorm1Graph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c ε₁
  let g2 := Graph.snoc g1
    (TapeNodes.affine
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel])
      (sIn := SeqFFNModelShape seqLen dModel) (sOut := SeqFFNHiddenShape seqLen dFF)
      (idxEncoderNorm1Out (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      fc1 b1)
  let g3 := Graph.snoc g2
    (gelu
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF])
      (s := SeqFFNHiddenShape seqLen dFF)
      (idxEncoderFfnHiddenPre (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))
  let g4 := Graph.snoc g3
    (TapeNodes.affine
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
      (sIn := SeqFFNHiddenShape seqLen dFF) (sOut := SeqFFNModelShape seqLen dModel)
      (idxEncoderFfnHiddenAct (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      fc2 b2)
  Graph.snoc g4
    (add
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
          SeqFFNModelShape seqLen dModel])
      (s := SeqFFNModelShape seqLen dModel)
      (idxEncoderNorm1OutAfterFfn (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      (idxEncoderFfnProjected (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))

/-- Second LayerNorm input triple in the concrete encoder block. -/
def encoderNorm2Inputs {seqLen dModel numHeads headDim dFF : Nat} :
    LayerNorm.Inputs
      (ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        ssSeqFFNResidual seqLen dModel dFF)
      seqLen dModel :=
  { x := idxEncoderFfnResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
    gamma := idxEncoderNorm2Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
    beta := idxEncoderNorm2Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF) }

/-- Concrete SSA graph for one full post-norm Transformer encoder block. -/
def encoderBlockGraph {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ) :
    Graph (ΓEncoderBlock seqLen dModel numHeads headDim)
      (ssEncoderBlock seqLen dModel numHeads headDim dFF) :=
  Graph.snoc
    (encoderFfnResidualGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      fc1 b1 fc2 b2 c ε₁)
    (LayerNorm.wholeNode
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
        ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (encoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      ε₂)

/-- Pointwise analytic correctness for the graph through the FFN residual. -/
def encoderFfnResidualGraphFDerivCorrectAt {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ : ℝ)
    (xV : CtxVec (ΓEncoderBlock seqLen dModel numHeads headDim))
    (hε₁ : 0 < ε₁) :
    GraphFDerivCorrectAt
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim ++
        [LayerNorm.MatShape seqLen dModel] ++ ssSeqFFNResidual seqLen dModel dFF)
      (encoderFfnResidualGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        fc1 b1 fc2 b2 c ε₁)
      xV := by
  classical
  let dgMha := encoderMhaResidualDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c
  let hgMha : GraphFDerivCorrectAt
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim) dgMha.g xV :=
    DGraph.graphFDerivCorrectAtOfCorrect dgMha.hg xV
  refine ⟨⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩
  · simpa [encoderAfterNorm1Graph, dgMha] using hgMha
  · simpa [encoderAfterNorm1Graph, dgMha] using
      LayerNorm.wholeNodeFDerivCorrectAt
          (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
            ssMHAResidual seqLen dModel numHeads headDim)
          (m := seqLen) (n := dModel)
          (encoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim))
          ε₁
          (Graph.evalVec
            (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
            (ss := ssMHAResidual seqLen dModel numHeads headDim)
            dgMha.g xV)
          hε₁
  · exact NodeFDerivCorrect.at
      (TapeNodes.affineFderiv
        (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
          ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel])
        (sIn := SeqFFNModelShape seqLen dModel) (sOut := SeqFFNHiddenShape seqLen dFF)
        (idxEncoderNorm1Out (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        fc1 b1)
      _
  · exact NodeFDerivCorrect.at
      (geluFderiv
        (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
          ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
          [SeqFFNHiddenShape seqLen dFF])
        (s := SeqFFNHiddenShape seqLen dFF)
        (idxEncoderFfnHiddenPre (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))
      _
  · exact NodeFDerivCorrect.at
      (TapeNodes.affineFderiv
        (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
          ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
          [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
        (sIn := SeqFFNHiddenShape seqLen dFF) (sOut := SeqFFNModelShape seqLen dModel)
        (idxEncoderFfnHiddenAct (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
        fc2 b2)
      _
  · exact NodeFDerivCorrect.at
      (addFderiv
        (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
          ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
          [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
            SeqFFNModelShape seqLen dModel])
        (s := SeqFFNModelShape seqLen dModel)
        (idxEncoderNorm1OutAfterFfn (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
        (idxEncoderFfnProjected (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)))
      _

/-- Pointwise analytic correctness for the complete concrete encoder-block graph. -/
def encoderBlockGraphFDerivCorrectAt {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ)
    (xV : CtxVec (ΓEncoderBlock seqLen dModel numHeads headDim))
    (hε₁ : 0 < ε₁)
    (hε₂ : 0 < ε₂) :
    GraphFDerivCorrectAt
      (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
      (ss := ssEncoderBlock seqLen dModel numHeads headDim dFF)
      (encoderBlockGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        fc1 b1 fc2 b2 c ε₁ ε₂)
      xV := by
  exact
    ⟨encoderFfnResidualGraphFDerivCorrectAt
      (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
      (dFF := dFF) fc1 b1 fc2 b2 c ε₁ xV hε₁,
      LayerNorm.wholeNodeFDerivCorrectAt
          (Γ := ΓEncoderBlock seqLen dModel numHeads headDim ++
            ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel] ++
            ssSeqFFNResidual seqLen dModel dFF)
          (m := seqLen) (n := dModel)
          (encoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
          ε₂ _ hε₂⟩

/-- End-to-end VJP theorem for the concrete post-norm Transformer encoder-block graph. -/
theorem encoderBlock_backpropVec_eq_adjoint_fderiv_at
    {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Spec.Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Spec.Shape.size (SeqFFNModelShape seqLen dModel)))
    (c ε₁ ε₂ : ℝ)
    (xV : CtxVec (ΓEncoderBlock seqLen dModel numHeads headDim))
    (seedV : CtxVec (ΓEncoderBlock seqLen dModel numHeads headDim ++
      ssEncoderBlock seqLen dModel numHeads headDim dFF))
    (hε₁ : 0 < ε₁)
    (hε₂ : 0 < ε₂) :
    Graph.backpropVec
        (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
        (ss := ssEncoderBlock seqLen dModel numHeads headDim dFF)
        (encoderBlockGraph (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
          fc1 b1 fc2 b2 c ε₁ ε₂)
        xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
          (ss := ssEncoderBlock seqLen dModel numHeads headDim dFF)
          (encoderBlockGraph (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
            fc1 b1 fc2 b2 c ε₁ ε₂))
        xV).adjoint seedV :=
  Graph.backpropVec_eq_adjoint_fderiv_at
    (Γ := ΓEncoderBlock seqLen dModel numHeads headDim)
    (ss := ssEncoderBlock seqLen dModel numHeads headDim dFF)
    (g := encoderBlockGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      fc1 b1 fc2 b2 c ε₁ ε₂)
    xV seedV
    (encoderBlockGraphFDerivCorrectAt
      (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
      (dFF := dFF) fc1 b1 fc2 b2 c ε₁ ε₂ xV hε₁ hε₂)

/--
Fréchet differentiability of a complete post-norm Transformer encoder block.

`attnPack` builds the first LayerNorm input triple from the outer model context.  `ffnPack` builds
the second LayerNorm input triple after the first post-norm sublayer has evaluated.  Both LayerNorm
epsilons must be positive.
-/
theorem postNormEncoderBlock_hasFDerivAt
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {seqLen dModel : Nat} (ε₁ ε₂ : ℝ)
    (attnPack : E → CtxVec (ΓPostNorm seqLen dModel))
    (DattnPack : E →L[ℝ] CtxVec (ΓPostNorm seqLen dModel))
    (ffnPack :
      CtxVec (ΓPostNorm seqLen dModel ++ ssPostNorm seqLen dModel) →
        CtxVec (ΓPostNorm seqLen dModel))
    (DffnPack :
      CtxVec (ΓPostNorm seqLen dModel ++ ssPostNorm seqLen dModel) →L[ℝ]
        CtxVec (ΓPostNorm seqLen dModel))
    (x : E)
    (hAttnPack : HasFDerivAt attnPack DattnPack x)
    (hε₁ : 0 < ε₁)
    (hFfnPack :
      HasFDerivAt ffnPack DffnPack
        (Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁) (attnPack x)))
    (hε₂ : 0 < ε₂) :
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
              (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁) (attnPack z))))
      ((fderiv ℝ
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := ssPostNorm seqLen dModel)
            (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₂))
          (ffnPack
            (Graph.evalVec
              (Γ := ΓPostNorm seqLen dModel)
              (ss := ssPostNorm seqLen dModel)
              (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁) (attnPack x)))).comp
        (DffnPack.comp
          ((fderiv ℝ
            (Graph.evalVec
              (Γ := ΓPostNorm seqLen dModel)
              (ss := ssPostNorm seqLen dModel)
              (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁))
            (attnPack x)).comp DattnPack)))
      x :=
  twoSublayerPostNormBlock_hasFDerivAt
    (seqLen := seqLen) (dModel := dModel) ε₁ ε₂
    attnPack DattnPack ffnPack DffnPack x
    hAttnPack hε₁ hFfnPack hε₂

end

end Transformer
end Autograd
end Proofs
