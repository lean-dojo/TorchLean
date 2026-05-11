/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm
public import NN.Proofs.Autograd.Tape.Ops.Transformer.FeedForward
public import NN.Proofs.Autograd.Tape.Ops.Transformer.ResidualAttention

/-!
# Post-Norm Transformer Sublayers

This file packages the LayerNorm theorem at the interface used by post-norm Transformer blocks.

The preceding files prove the smooth residual components:

* `x ↦ x + MultiHeadSelfAttention(x)`;
* `x ↦ x + W₂ GELU(W₁x + b₁) + b₂`.

This module proves the next runtime-facing boundary:

`residual_stream ↦ LayerNorm(residual_stream, gamma, beta)`.

That is the exact post-norm sublayer shape used by classical Transformer encoder blocks
(`LayerNorm(x + Sublayer(x))`). We deliberately keep this proof factored at the residual-stream
interface. It avoids treating LayerNorm's pointwise domain hypotheses as globally smooth,
and it gives later full-block proofs a clean seam: compose a globally smooth residual graph with
this pointwise post-norm graph once the context-threading adapter for unused parameters is in place.

References:

* Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
* PyTorch `torch.nn.TransformerEncoderLayer`.
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
* PyTorch `torch.nn.LayerNorm`.
  https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Transformer

open Spec

universe u

noncomputable section

/-- A post-norm Transformer stage normalizes a sequence-shaped residual stream. -/
abbrev ΓPostNorm (seqLen dModel : Nat) : List Shape :=
  LayerNorm.ΓLN seqLen dModel

/-- Saved tensors for the LayerNorm part of a post-norm Transformer stage. -/
abbrev ssPostNorm (seqLen dModel : Nat) : List Shape :=
  LayerNorm.ssLayerNorm seqLen dModel

/-- MHA context extended with the affine parameters for the following LayerNorm. -/
abbrev ΓMHAWithNorm (seqLen dModel numHeads headDim : Nat) : List Shape :=
  MultiHeadAttention.ΓMHA seqLen dModel numHeads headDim ++
    [LayerNorm.VecShape dModel, LayerNorm.VecShape dModel]

/-- Residual-MHA plus one whole-node post-norm output. -/
abbrev ssMHAWithPostNorm (seqLen dModel numHeads headDim : Nat) : List Shape :=
  ssMHAResidual seqLen dModel numHeads headDim ++ [LayerNorm.MatShape seqLen dModel]

/-- Gamma parameter for the post-norm LayerNorm, weakened through any saved tensors. -/
def idxMhaPostNormGamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMHAWithNorm seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  _root_.Proofs.Autograd.Idx.weaken
    (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
    (s := LayerNorm.VecShape dModel)
    ⟨⟨5, by simp [ΓMHAWithNorm, MultiHeadAttention.ΓMHA]⟩,
      by simp⟩
    ss

/-- Beta parameter for the post-norm LayerNorm, weakened through any saved tensors. -/
def idxMhaPostNormBeta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (ΓMHAWithNorm seqLen dModel numHeads headDim ++ ss) (LayerNorm.VecShape dModel) :=
  _root_.Proofs.Autograd.Idx.weaken
    (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
    (s := LayerNorm.VecShape dModel)
    ⟨⟨6, by simp [ΓMHAWithNorm, MultiHeadAttention.ΓMHA]⟩,
      by simp⟩
    ss

/-- The residual stream `x + MHA(x)` after the residual-attention prefix graph. -/
def idxMhaResidualForPostNorm {seqLen dModel numHeads headDim : Nat} :
    Idx
      (ΓMHAWithNorm seqLen dModel numHeads headDim ++ ssMHAResidual seqLen dModel numHeads headDim)
      (LayerNorm.MatShape seqLen dModel) :=
  ⟨⟨21, by
      simp [ΓMHAWithNorm, MultiHeadAttention.ΓMHA, ssMHAResidual, MultiHeadAttention.ssMHA]⟩,
    by
      simp [LayerNorm.MatShape, MultiHeadAttention.XShape]⟩

/-- LayerNorm input triple after the residual-MHA prefix has run. -/
def mhaPostNormInputs {seqLen dModel numHeads headDim : Nat} :
    LayerNorm.Inputs
      (ΓMHAWithNorm seqLen dModel numHeads headDim ++ ssMHAResidual seqLen dModel numHeads headDim)
      seqLen dModel :=
  { x := idxMhaResidualForPostNorm (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
    gamma := idxMhaPostNormGamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim)
    beta := idxMhaPostNormBeta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim) }

/--
Residual-MHA graph while carrying the following LayerNorm's affine parameters.

The carried `gamma/beta` are not read by attention; `DGraph.weakenContext` ensures their gradients
from the attention prefix are zero, while still keeping them available to the appended LayerNorm
node.
-/
def mhaResidualWithNormParamsDGraph {seqLen dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (ΓMHAWithNorm seqLen dModel numHeads headDim)
      (ssMHAResidual seqLen dModel numHeads headDim) :=
  DGraph.weakenContext
    (mhaResidualDGraph (n := seqLen) (dModel := dModel) (numHeads := numHeads)
      (headDim := headDim) c)
    [LayerNorm.VecShape dModel, LayerNorm.VecShape dModel]

/--
Single SSA graph for the first post-norm Transformer encoder sublayer:

`LayerNorm(x + MultiHeadSelfAttention(x), gamma, beta)`.

LayerNorm is appended as a whole pointwise node backed by the detailed LayerNorm graph theorem.
-/
def mhaPostNormGraph {seqLen dModel numHeads headDim : Nat} (c ε : ℝ) :
    Graph (ΓMHAWithNorm seqLen dModel numHeads headDim)
      (ssMHAWithPostNorm seqLen dModel numHeads headDim) :=
  .snoc
    (mhaResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) c).g
    (LayerNorm.wholeNode
      (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      (m := seqLen) (n := dModel)
      (mhaPostNormInputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε)

/--
Pointwise correctness for the single-graph residual-MHA post-norm sublayer.
-/
def mhaPostNormGraphFDerivCorrectAt
    {seqLen dModel numHeads headDim : Nat} (c ε : ℝ)
    (xV : CtxVec (ΓMHAWithNorm seqLen dModel numHeads headDim))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := LayerNorm.ΓLN seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := LayerNorm.ΓLN seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε)
            ((LayerNorm.packInputsCLM
              (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim ++
                ssMHAResidual seqLen dModel numHeads headDim)
              (m := seqLen) (n := dModel)
              (mhaPostNormInputs (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim)))
              (Graph.evalVec
                (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
                (ss := ssMHAResidual seqLen dModel numHeads headDim)
                (mhaResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
                  (numHeads := numHeads) (headDim := headDim) c).g xV))) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := LayerNorm.ΓLN seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := LayerNorm.ΓLN seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε)
            ((LayerNorm.packInputsCLM
              (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim ++
                ssMHAResidual seqLen dModel numHeads headDim)
              (m := seqLen) (n := dModel)
              (mhaPostNormInputs (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim)))
              (Graph.evalVec
                (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
                (ss := ssMHAResidual seqLen dModel numHeads headDim)
                (mhaResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
                  (numHeads := numHeads) (headDim := headDim) c).g xV))) i ≠ 0) :
    GraphFDerivCorrectAt
      (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
      (ss := ssMHAWithPostNorm seqLen dModel numHeads headDim)
      (mhaPostNormGraph (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
        (headDim := headDim) c ε)
      xV := by
  classical
  let dgPrefix := mhaResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c
  refine ⟨DGraph.graphFDerivCorrectAtOfCorrect dgPrefix.hg xV, ?_⟩
  exact
    LayerNorm.wholeNodeFDerivCorrectAt
      (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      (m := seqLen) (n := dModel)
      (mhaPostNormInputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε
      (Graph.evalVec
        (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
        (ss := ssMHAResidual seqLen dModel numHeads headDim)
        dgPrefix.g xV)
      hVarEpsPos hStdNe0

/--
End-to-end VJP theorem for the single-graph residual-MHA post-norm sublayer.
-/
theorem mhaPostNorm_backpropVec_eq_adjoint_fderiv_at
    {seqLen dModel numHeads headDim : Nat} (c ε : ℝ)
    (xV : CtxVec (ΓMHAWithNorm seqLen dModel numHeads headDim))
    (seedV :
      CtxVec (ΓMHAWithNorm seqLen dModel numHeads headDim ++
        ssMHAWithPostNorm seqLen dModel numHeads headDim))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := LayerNorm.ΓLN seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := LayerNorm.ΓLN seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε)
            ((LayerNorm.packInputsCLM
              (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim ++
                ssMHAResidual seqLen dModel numHeads headDim)
              (m := seqLen) (n := dModel)
              (mhaPostNormInputs (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim)))
              (Graph.evalVec
                (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
                (ss := ssMHAResidual seqLen dModel numHeads headDim)
                (mhaResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
                  (numHeads := numHeads) (headDim := headDim) c).g xV))) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := LayerNorm.ΓLN seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := LayerNorm.ΓLN seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε)
            ((LayerNorm.packInputsCLM
              (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim ++
                ssMHAResidual seqLen dModel numHeads headDim)
              (m := seqLen) (n := dModel)
              (mhaPostNormInputs (seqLen := seqLen) (dModel := dModel)
                (numHeads := numHeads) (headDim := headDim)))
              (Graph.evalVec
                (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
                (ss := ssMHAResidual seqLen dModel numHeads headDim)
                (mhaResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
                  (numHeads := numHeads) (headDim := headDim) c).g xV))) i ≠ 0) :
    Graph.backpropVec
        (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
        (ss := ssMHAWithPostNorm seqLen dModel numHeads headDim)
        (mhaPostNormGraph (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) c ε)
        xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
          (ss := ssMHAWithPostNorm seqLen dModel numHeads headDim)
          (mhaPostNormGraph (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
            (headDim := headDim) c ε))
        xV).adjoint seedV := by
  exact
    Graph.backpropVec_eq_adjoint_fderiv_at
      (Γ := ΓMHAWithNorm seqLen dModel numHeads headDim)
      (ss := ssMHAWithPostNorm seqLen dModel numHeads headDim)
      (g := mhaPostNormGraph (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
        (headDim := headDim) c ε)
      xV seedV
      (mhaPostNormGraphFDerivCorrectAt (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) c ε xV hVarEpsPos hStdNe0)

/-!
## Sequence feed-forward plus post-norm

This is the second sublayer shape in a post-norm Transformer encoder block:

`LayerNorm(X + FFN(X), gamma, beta)`.

The FFN residual is sequence-shaped, not merely one-token-shaped. Its affine maps are supplied over
flattened sequence tensors, so a shared position-wise implementation or a fused backend can both
instantiate the theorem by exposing their fixed affine maps.
-/

/-- Sequence-FFN context extended with the affine parameters for the following LayerNorm. -/
abbrev ΓSeqFFNWithNorm (seqLen dModel : Nat) : List Shape :=
  ΓSeqFFN seqLen dModel ++ [LayerNorm.VecShape dModel, LayerNorm.VecShape dModel]

/-- Sequence-FFN residual plus one whole-node post-norm output. -/
abbrev ssSeqFFNWithPostNorm (seqLen dModel dFF : Nat) : List Shape :=
  ssSeqFFNResidual seqLen dModel dFF ++ [LayerNorm.MatShape seqLen dModel]

/-- Gamma parameter for the FFN post-norm LayerNorm. -/
def idxSeqFfnPostNormGamma {seqLen dModel : Nat} {ss : List Shape} :
    Idx (ΓSeqFFNWithNorm seqLen dModel ++ ss) (LayerNorm.VecShape dModel) :=
  _root_.Proofs.Autograd.Idx.weaken
    (Γ := ΓSeqFFNWithNorm seqLen dModel)
    (s := LayerNorm.VecShape dModel)
    ⟨⟨1, by simp [ΓSeqFFNWithNorm, ΓSeqFFN]⟩, by simp⟩
    ss

/-- Beta parameter for the FFN post-norm LayerNorm. -/
def idxSeqFfnPostNormBeta {seqLen dModel : Nat} {ss : List Shape} :
    Idx (ΓSeqFFNWithNorm seqLen dModel ++ ss) (LayerNorm.VecShape dModel) :=
  _root_.Proofs.Autograd.Idx.weaken
    (Γ := ΓSeqFFNWithNorm seqLen dModel)
    (s := LayerNorm.VecShape dModel)
    ⟨⟨2, by simp [ΓSeqFFNWithNorm, ΓSeqFFN]⟩, by simp⟩
    ss

/-- The residual stream `X + FFN(X)` after the sequence-FFN prefix graph. -/
def idxSeqFfnResidualForPostNorm {seqLen dModel dFF : Nat} :
    Idx (ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
      (LayerNorm.MatShape seqLen dModel) :=
  idxLast (Γ := ΓSeqFFNWithNorm seqLen dModel)
    (ss := [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
      SeqFFNModelShape seqLen dModel])
    (τ := SeqFFNModelShape seqLen dModel)

/-- LayerNorm input triple after the residual-FFN prefix has run. -/
def seqFfnPostNormInputs {seqLen dModel dFF : Nat} :
    LayerNorm.Inputs
      (ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
      seqLen dModel :=
  { x := idxSeqFfnResidualForPostNorm (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
    gamma := idxSeqFfnPostNormGamma (seqLen := seqLen) (dModel := dModel)
      (ss := ssSeqFFNResidual seqLen dModel dFF)
    beta := idxSeqFfnPostNormBeta (seqLen := seqLen) (dModel := dModel)
      (ss := ssSeqFFNResidual seqLen dModel dFF) }

/-- Sequence-FFN graph while carrying the following LayerNorm's affine parameters. -/
def seqFfnResidualWithNormParamsDGraph {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel))) :
    DGraph (ΓSeqFFNWithNorm seqLen dModel) (ssSeqFFNResidual seqLen dModel dFF) :=
  DGraph.weakenContext
    (seqFfnResidualDGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
      fc1 b1 fc2 b2)
    [LayerNorm.VecShape dModel, LayerNorm.VecShape dModel]

/--
Single SSA graph for the second post-norm Transformer encoder sublayer:

`LayerNorm(X + FFN(X), gamma, beta)`.
-/
def seqFfnPostNormGraph {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (ε : ℝ) :
    Graph (ΓSeqFFNWithNorm seqLen dModel) (ssSeqFFNWithPostNorm seqLen dModel dFF) :=
  .snoc
    (seqFfnResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
      fc1 b1 fc2 b2).g
    (LayerNorm.wholeNode
      (Γ := ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (seqFfnPostNormInputs (seqLen := seqLen) (dModel := dModel) (dFF := dFF))
      ε)

/-- Pointwise correctness for the single-graph residual-FFN post-norm sublayer. -/
def seqFfnPostNormGraphFDerivCorrectAt
    {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (ε : ℝ)
    (xV : CtxVec (ΓSeqFFNWithNorm seqLen dModel))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := LayerNorm.ΓLN seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := LayerNorm.ΓLN seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε)
            ((LayerNorm.packInputsCLM
              (Γ := ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
              (m := seqLen) (n := dModel)
              (seqFfnPostNormInputs (seqLen := seqLen) (dModel := dModel) (dFF := dFF)))
              (Graph.evalVec
                (Γ := ΓSeqFFNWithNorm seqLen dModel)
                (ss := ssSeqFFNResidual seqLen dModel dFF)
                (seqFfnResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
                  (dFF := dFF) fc1 b1 fc2 b2).g xV))) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := LayerNorm.ΓLN seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := LayerNorm.ΓLN seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε)
            ((LayerNorm.packInputsCLM
              (Γ := ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
              (m := seqLen) (n := dModel)
              (seqFfnPostNormInputs (seqLen := seqLen) (dModel := dModel) (dFF := dFF)))
              (Graph.evalVec
                (Γ := ΓSeqFFNWithNorm seqLen dModel)
                (ss := ssSeqFFNResidual seqLen dModel dFF)
                (seqFfnResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
                  (dFF := dFF) fc1 b1 fc2 b2).g xV))) i ≠ 0) :
    GraphFDerivCorrectAt
      (Γ := ΓSeqFFNWithNorm seqLen dModel)
      (ss := ssSeqFFNWithPostNorm seqLen dModel dFF)
      (seqFfnPostNormGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
        fc1 b1 fc2 b2 ε)
      xV := by
  classical
  let dgPrefix := seqFfnResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
    (dFF := dFF) fc1 b1 fc2 b2
  refine ⟨DGraph.graphFDerivCorrectAtOfCorrect dgPrefix.hg xV, ?_⟩
  exact
    LayerNorm.wholeNodeFDerivCorrectAt
      (Γ := ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (seqFfnPostNormInputs (seqLen := seqLen) (dModel := dModel) (dFF := dFF))
      ε
      (Graph.evalVec
        (Γ := ΓSeqFFNWithNorm seqLen dModel)
        (ss := ssSeqFFNResidual seqLen dModel dFF)
        dgPrefix.g xV)
      hVarEpsPos hStdNe0

/-- End-to-end VJP theorem for the single-graph residual-FFN post-norm sublayer. -/
theorem seqFfnPostNorm_backpropVec_eq_adjoint_fderiv_at
    {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (ε : ℝ)
    (xV : CtxVec (ΓSeqFFNWithNorm seqLen dModel))
    (seedV : CtxVec (ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNWithPostNorm seqLen dModel dFF))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := LayerNorm.ΓLN seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := LayerNorm.ΓLN seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε)
            ((LayerNorm.packInputsCLM
              (Γ := ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
              (m := seqLen) (n := dModel)
              (seqFfnPostNormInputs (seqLen := seqLen) (dModel := dModel) (dFF := dFF)))
              (Graph.evalVec
                (Γ := ΓSeqFFNWithNorm seqLen dModel)
                (ss := ssSeqFFNResidual seqLen dModel dFF)
                (seqFfnResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
                  (dFF := dFF) fc1 b1 fc2 b2).g xV))) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := LayerNorm.ΓLN seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := LayerNorm.ΓLN seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε)
            ((LayerNorm.packInputsCLM
              (Γ := ΓSeqFFNWithNorm seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
              (m := seqLen) (n := dModel)
              (seqFfnPostNormInputs (seqLen := seqLen) (dModel := dModel) (dFF := dFF)))
              (Graph.evalVec
                (Γ := ΓSeqFFNWithNorm seqLen dModel)
                (ss := ssSeqFFNResidual seqLen dModel dFF)
                (seqFfnResidualWithNormParamsDGraph (seqLen := seqLen) (dModel := dModel)
                  (dFF := dFF) fc1 b1 fc2 b2).g xV))) i ≠ 0) :
    Graph.backpropVec
        (Γ := ΓSeqFFNWithNorm seqLen dModel)
        (ss := ssSeqFFNWithPostNorm seqLen dModel dFF)
        (seqFfnPostNormGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
          fc1 b1 fc2 b2 ε)
        xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓSeqFFNWithNorm seqLen dModel)
          (ss := ssSeqFFNWithPostNorm seqLen dModel dFF)
          (seqFfnPostNormGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
            fc1 b1 fc2 b2 ε))
        xV).adjoint seedV := by
  exact
    Graph.backpropVec_eq_adjoint_fderiv_at
      (Γ := ΓSeqFFNWithNorm seqLen dModel)
      (ss := ssSeqFFNWithPostNorm seqLen dModel dFF)
      (g := seqFfnPostNormGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
        fc1 b1 fc2 b2 ε)
      xV seedV
      (seqFfnPostNormGraphFDerivCorrectAt (seqLen := seqLen) (dModel := dModel)
        (dFF := dFF) fc1 b1 fc2 b2 ε xV hVarEpsPos hStdNe0)

/--
The post-norm graph itself.

The context is `[residual_stream, gamma, beta]`. For attention this residual stream is
`x + MHA(x)`; for the feed-forward half it is `x + FFN(x)`. The residual computation is proved in
`ResidualAttention`/`FeedForward`; this graph proves the LayerNorm boundary that follows it.
-/
def postNormGraph {seqLen dModel : Nat} (ε : ℝ) :
    Graph (ΓPostNorm seqLen dModel) (ssPostNorm seqLen dModel) :=
  LayerNorm.layerNormGraph (m := seqLen) (n := dModel) ε

/--
Pointwise correctness for the post-norm Transformer boundary.

The two hypotheses are exactly LayerNorm's differentiability side conditions at the runtime point:
the variance-plus-epsilon branch is positive, and the standard deviation denominator is nonzero.
-/
def postNormGraphFderivCorrectAt
    {seqLen dModel : Nat} (ε : ℝ) (xV : CtxVec (ΓPostNorm seqLen dModel))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε) xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε) xV) i ≠ 0) :
    GraphFDerivCorrectAt
      (Γ := ΓPostNorm seqLen dModel)
      (ss := ssPostNorm seqLen dModel)
      (postNormGraph (seqLen := seqLen) (dModel := dModel) ε) xV :=
  LayerNorm.layerNormGraphFderivCorrectAt
    (m := seqLen) (n := dModel) ε xV hVarEpsPos hStdNe0

/--
VJP theorem for the post-norm Transformer boundary.

This is the model-level theorem used after either residual attention or a residual feed-forward
block has produced its sequence-shaped residual stream.
-/
theorem postNorm_backpropVec_eq_adjoint_fderiv_at
    {seqLen dModel : Nat} (ε : ℝ)
    (xV : CtxVec (ΓPostNorm seqLen dModel))
    (seedV : CtxVec (ΓPostNorm seqLen dModel ++ ssPostNorm seqLen dModel))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε) xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε) xV) i ≠ 0) :
    Graph.backpropVec
        (Γ := ΓPostNorm seqLen dModel)
        (ss := ssPostNorm seqLen dModel)
        (postNormGraph (seqLen := seqLen) (dModel := dModel) ε) xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε)) xV).adjoint seedV :=
  LayerNorm.backprop_eq_adjoint_fderiv_layerNorm_at
    (m := seqLen) (n := dModel) ε xV seedV hVarEpsPos hStdNe0

/--
Calculus bridge for a residual block followed by post-norm LayerNorm.

This is the theorem we use to move from separately proved pieces to a whole Transformer sublayer.
Suppose some residual-producing map

`residualPack : E → [residual_stream, gamma, beta]`

is differentiable at `x`. It may come from residual attention, residual feed-forward, or any future
block that produces the same LayerNorm context. If the LayerNorm domain hypotheses hold at
`residualPack x`, then the composed post-norm map

`x ↦ LayerNorm(residualPack x)`

is differentiable, with derivative given by the usual chain rule.

This is intentionally more general than MHA: the same theorem covers Transformer, ViT, GPT-style
blocks, and future residual modules once they expose the residual stream plus affine LayerNorm
parameters.
-/
theorem residualThenPostNorm_hasFDerivAt
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {seqLen dModel : Nat} (ε : ℝ)
    (residualPack : E → CtxVec (ΓPostNorm seqLen dModel))
    (DresidualPack : E →L[ℝ] CtxVec (ΓPostNorm seqLen dModel))
    (x : E)
    (hResidual : HasFDerivAt residualPack DresidualPack x)
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε) (residualPack x)) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε) (residualPack x)) i ≠ 0) :
    HasFDerivAt
      (fun z : E =>
        Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε) (residualPack z))
      ((fderiv ℝ
        (Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε))
        (residualPack x)).comp DresidualPack)
      x := by
  classical
  let g := postNormGraph (seqLen := seqLen) (dModel := dModel) ε
  have hgAt :
      GraphFDerivCorrectAt
        (Γ := ΓPostNorm seqLen dModel)
        (ss := ssPostNorm seqLen dModel)
        g (residualPack x) :=
    postNormGraphFderivCorrectAt
      (seqLen := seqLen) (dModel := dModel) ε (residualPack x) hVarEpsPos hStdNe0
  rcases Graph.hasFDerivAt_evalVec_and_jvp_at
      (Γ := ΓPostNorm seqLen dModel)
      (ss := ssPostNorm seqLen dModel)
      (g := g) (xV := residualPack x) hgAt with
    ⟨Dpost, hDpost, _hJpost⟩
  have hFderiv :
      fderiv ℝ
        (Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel) g)
        (residualPack x) = Dpost := by
    simpa using hDpost.fderiv
  have hComp :
      HasFDerivAt
        (fun z : E =>
          Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := ssPostNorm seqLen dModel) g (residualPack z))
        (Dpost.comp DresidualPack) x :=
    hDpost.comp x hResidual
  rw [hFderiv]
  simpa [g] using hComp

/--
Calculus bridge for a full two-sublayer post-norm Transformer encoder block.

This theorem is deliberately stated at the map level rather than for one concrete SSA graph. A
post-norm encoder block has two domain-sensitive LayerNorms:

1. `attnPack` builds `[x + MHA(x), gamma₁, beta₁]`;
2. after the first LayerNorm evaluation, `ffnPack` builds
   `[norm₁ + FFN(norm₁), gamma₂, beta₂]`;
3. the second LayerNorm produces the block output.

The theorem says that if the two residual-pack maps are differentiable and both LayerNorm calls
satisfy their local denominator hypotheses, then the whole two-sublayer block is differentiable by
ordinary Fréchet chain rule.

The concrete graph-level VJP theorems for each sublayer are `mhaPostNorm_*` and `seqFfnPostNorm_*`.
This bridge is the public mathematical composition point for Transformer, ViT, and GPT-style
post-norm blocks while the final monolithic SSA graph is assembled.
-/
theorem twoSublayerPostNormBlock_hasFDerivAt
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
    (hNorm1VarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε₁) (attnPack x)) i)
    (hNorm1StdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε₁) (attnPack x)) i ≠ 0)
    (hFfnPack :
      HasFDerivAt ffnPack DffnPack
        (Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁) (attnPack x)))
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
                (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁) (attnPack x)))) i)
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
                (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁) (attnPack x)))) i ≠ 0) :
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
      x := by
  classical
  let norm1 :=
    fun z : E =>
      Graph.evalVec
        (Γ := ΓPostNorm seqLen dModel)
        (ss := ssPostNorm seqLen dModel)
        (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁) (attnPack z)
  have hNorm1 :
      HasFDerivAt norm1
        ((fderiv ℝ
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := ssPostNorm seqLen dModel)
            (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁))
          (attnPack x)).comp DattnPack)
        x :=
    residualThenPostNorm_hasFDerivAt
      (seqLen := seqLen) (dModel := dModel) ε₁
      attnPack DattnPack x hAttnPack hNorm1VarEpsPos hNorm1StdNe0
  have hFfnAfterNorm1 :
      HasFDerivAt (fun z : E => ffnPack (norm1 z))
        (DffnPack.comp
          ((fderiv ℝ
            (Graph.evalVec
              (Γ := ΓPostNorm seqLen dModel)
              (ss := ssPostNorm seqLen dModel)
              (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁))
            (attnPack x)).comp DattnPack))
        x :=
    hFfnPack.comp x hNorm1
  exact
    residualThenPostNorm_hasFDerivAt
      (seqLen := seqLen) (dModel := dModel) ε₂
      (fun z : E => ffnPack (norm1 z))
      (DffnPack.comp
        ((fderiv ℝ
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := ssPostNorm seqLen dModel)
            (postNormGraph (seqLen := seqLen) (dModel := dModel) ε₁))
          (attnPack x)).comp DattnPack))
      x hFfnAfterNorm1 hNorm2VarEpsPos hNorm2StdNe0

/--
Named theorem for the post-normalized residual-attention interface.

The residual attention graph proves production of the first input in this context:
`residual_stream = x + MHA(x)`. This theorem proves the LayerNorm pass once that residual stream is
the current tensor.
-/
theorem residualAttentionPostNorm_backpropVec_eq_adjoint_fderiv_at
    {seqLen dModel : Nat} (ε : ℝ)
    (xV : CtxVec (ΓPostNorm seqLen dModel))
    (seedV : CtxVec (ΓPostNorm seqLen dModel ++ ssPostNorm seqLen dModel))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε) xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε) xV) i ≠ 0) :
    Graph.backpropVec
        (Γ := ΓPostNorm seqLen dModel)
        (ss := ssPostNorm seqLen dModel)
        (postNormGraph (seqLen := seqLen) (dModel := dModel) ε) xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε)) xV).adjoint seedV :=
  postNorm_backpropVec_eq_adjoint_fderiv_at
    (seqLen := seqLen) (dModel := dModel) ε xV seedV hVarEpsPos hStdNe0

/--
Named theorem for the post-normalized residual feed-forward interface.

The position-wise FFN proof establishes the smooth residual update. This theorem is the common
LayerNorm boundary used after that update in post-norm encoder blocks.
-/
theorem residualFeedForwardPostNorm_backpropVec_eq_adjoint_fderiv_at
    {seqLen dModel : Nat} (ε : ℝ)
    (xV : CtxVec (ΓPostNorm seqLen dModel))
    (seedV : CtxVec (ΓPostNorm seqLen dModel ++ ssPostNorm seqLen dModel))
    (hVarEpsPos :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        0 < CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix6 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxVarEps (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix6 seqLen dModel)
            (LayerNorm.layerNormPrefix6 (m := seqLen) (n := dModel) ε) xV) i)
    (hStdNe0 :
      ∀ i : Fin (Shape.size (LayerNorm.VecShape seqLen)),
        CtxVec.get
          (Γ := ΓPostNorm seqLen dModel ++ LayerNorm.ssPrefix7 seqLen dModel)
          (s := LayerNorm.VecShape seqLen)
          (LayerNorm.idxStd (m := seqLen) (n := dModel))
          (Graph.evalVec
            (Γ := ΓPostNorm seqLen dModel)
            (ss := LayerNorm.ssPrefix7 seqLen dModel)
            (LayerNorm.layerNormPrefix7 (m := seqLen) (n := dModel) ε) xV) i ≠ 0) :
    Graph.backpropVec
        (Γ := ΓPostNorm seqLen dModel)
        (ss := ssPostNorm seqLen dModel)
        (postNormGraph (seqLen := seqLen) (dModel := dModel) ε) xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓPostNorm seqLen dModel)
          (ss := ssPostNorm seqLen dModel)
          (postNormGraph (seqLen := seqLen) (dModel := dModel) ε)) xV).adjoint seedV :=
  postNorm_backpropVec_eq_adjoint_fderiv_at
    (seqLen := seqLen) (dModel := dModel) ε xV seedV hVarEpsPos hStdNe0

end

end Transformer
end Autograd
end Proofs
