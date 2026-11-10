/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.LayerNorm
public import TorchLean.NN.Models.Transformer.FeedForward
public import TorchLean.NN.Models.Transformer.ResidualAttention

/-!
# Post-Norm Transformer Sublayers

The residual attention and feed-forward modules prove the smooth residual components

* `x ↦ x + MultiHeadSelfAttention(x)`
* `x ↦ x + W₂ GELU(W₁x + b₁) + b₂`

and this module proves the LayerNorm composition that follows them:

`residual_stream ↦ LayerNorm(residual_stream, gamma, beta)`.

Classical Transformer encoder blocks use this post-norm sublayer shape,
`LayerNorm(x + Sublayer(x))`. Attention and feed-forward sublayers reach the *same* boundary, so
`LayerNorm.backprop_eq_fderiv` serves both of them. The single-graph theorems
`mhaPostNorm_backprop_eq_fderiv` and `seqFFNPostNorm_backprop_eq_fderiv`
append LayerNorm to each residual prefix in one SSA graph. Their attention prefix splits heads as
`MultiHeadAttention.splitHeads` does. All of these are exact-real graph results, and
correspondence with a runtime Transformer or fused kernels needs separate proofs.

The post-norm order is Vaswani et al., Section 3.1, which is also `norm_first=False` in
`torch.nn.TransformerEncoderLayer`. Every theorem here takes the LayerNorm `ε` as an argument and
asks only `0 < ε`, so torch's default `layer_norm_eps = 1e-5` is one instance.

We keep the proof factored at the residual-stream interface on purpose. LayerNorm's certificate is
pointwise (`RealGraph.FDerivCorrectAt`), and the factoring avoids treating it as a global one. It
also gives full-block proofs a clean seam: compose a globally smooth residual graph with the
pointwise `LayerNorm.graph`, threading the unused LayerNorm parameters through the residual prefix
with `DGraph.weakenContext`.

References:

* Vaswani et al. (2017), "Attention Is All You Need", NIPS 2017, arXiv:1706.03762, Section 3.1.
* PyTorch `torch.nn.TransformerEncoderLayer` (PyTorch 2.8 docs).
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.TransformerEncoderLayer.html
* PyTorch `torch.nn.LayerNorm` (PyTorch 2.8 docs).
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.LayerNorm.html
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace Transformer

noncomputable section

open Tape (DGraph)

/-- MHA context extended with the affine parameters for the following LayerNorm. -/
abbrev mhaWithNormCtx (seqLen dModel numHeads headDim : Nat) : List Shape :=
  MultiHeadAttention.ctx seqLen dModel numHeads headDim ++
    [[dModel], [dModel]]

/-- Saved tensors: the residual-MHA intermediates followed by one whole-node LayerNorm output. -/
abbrev ssMHAWithPostNorm (seqLen dModel numHeads headDim : Nat) : List Shape :=
  ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]]

/-- Gamma parameter for the post-norm LayerNorm, weakened through any saved tensors. -/
def idxMHAPostNormGamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (mhaWithNormCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  Idx.weaken
    (Γ := mhaWithNormCtx seqLen dModel numHeads headDim)
    (s := [dModel])
    ⟨⟨5, by simp [mhaWithNormCtx, MultiHeadAttention.ctx]⟩,
      by simp⟩
    ss

/-- Beta parameter for the post-norm LayerNorm, weakened through any saved tensors. -/
def idxMHAPostNormBeta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (mhaWithNormCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  Idx.weaken
    (Γ := mhaWithNormCtx seqLen dModel numHeads headDim)
    (s := [dModel])
    ⟨⟨6, by simp [mhaWithNormCtx, MultiHeadAttention.ctx]⟩,
      by simp⟩
    ss

/-- The residual stream `x + MHA(x)` after the residual-attention prefix graph. -/
def idxMHAResidualForPostNorm {seqLen dModel numHeads headDim : Nat} :
    Idx
      (mhaWithNormCtx seqLen dModel numHeads headDim ++ ssMHAResidual seqLen dModel
        numHeads headDim)
      [seqLen, dModel] :=
  Idx.last (Γ := mhaWithNormCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ss seqLen dModel numHeads headDim)
    (τ := [seqLen, dModel])

/-- LayerNorm input triple after the residual-MHA prefix has run. -/
def mhaPostNormInputs {seqLen dModel numHeads headDim : Nat} :
    LayerNorm.Inputs
      (mhaWithNormCtx seqLen dModel numHeads headDim ++ ssMHAResidual seqLen dModel
        numHeads headDim)
      seqLen dModel :=
  { x := idxMHAResidualForPostNorm (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
    gamma := idxMHAPostNormGamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim)
    beta := idxMHAPostNormBeta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim) }

/--
Residual-MHA graph while carrying the following LayerNorm's affine parameters.

The attention result does not depend on the carried `gamma/beta`. `DGraph.weakenContext`
preserves them in the full context so the appended LayerNorm can read them.
-/
def mhaResidualWithNormDGraph {seqLen dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (mhaWithNormCtx seqLen dModel numHeads headDim)
      (ssMHAResidual seqLen dModel numHeads headDim) :=
  DGraph.weakenContext
    (mhaResidualDGraph (n := seqLen) (dModel := dModel) (numHeads := numHeads)
      (headDim := headDim) c)
    [[dModel], [dModel]]

/--
Single SSA graph for the first post-norm Transformer encoder sublayer:

`LayerNorm(x + MultiHeadSelfAttention(x), gamma, beta)`.

LayerNorm enters as one `LayerNorm.node`, whose certificate comes from the detailed LayerNorm
graph theorem.
-/
def mhaPostNormGraph {seqLen dModel numHeads headDim : Nat} (c ε : ℝ) :
    Tape.Graph (α := ℝ) Unit (mhaWithNormCtx seqLen dModel numHeads headDim)
      (ssMHAWithPostNorm seqLen dModel numHeads headDim) :=
  .snoc
    (mhaResidualWithNormDGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) c).g
    (LayerNorm.node
      (Γ := mhaWithNormCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      (m := seqLen) (n := dModel)
      (mhaPostNormInputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε)

/--
Pointwise correctness of the single-graph residual-MHA post-norm sublayer, for `0 < ε`.
-/
def mhaPostNormFDerivAt
    {seqLen dModel numHeads headDim : Nat} (c ε : ℝ)
    (xV : CtxVec (mhaWithNormCtx seqLen dModel numHeads headDim))
    (hε : 0 < ε) :
    Tape.RealGraph.FDerivCorrectAt
      (Γ := mhaWithNormCtx seqLen dModel numHeads headDim)
      (ss := ssMHAWithPostNorm seqLen dModel numHeads headDim)
      (mhaPostNormGraph (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
        (headDim := headDim) c ε).toData ()
      xV := by
  classical
  let dgPrefix := mhaResidualWithNormDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c
  refine ⟨Tape.RealGraph.FDerivCorrect.at dgPrefix.hg xV, ?_⟩
  exact
    LayerNorm.nodeFDerivAt
      (Γ := mhaWithNormCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      (m := seqLen) (n := dModel)
      (mhaPostNormInputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε
      (Tape.GraphData.evalVec
        (Γ := mhaWithNormCtx seqLen dModel numHeads headDim)
        (ss := ssMHAResidual seqLen dModel numHeads headDim)
        dgPrefix.g.toData () xV)
      hε

/--
End-to-end VJP theorem for the single-graph residual-MHA post-norm sublayer, for `0 < ε`.
Its LayerNorm stage is a `LayerNorm.node`, whose VJP is the adjoint of its derivative by
definition, and `LayerNorm.adjoint_fderiv_eq_backward` identifies the adjoint derivative of the
LayerNorm map with the primitive `LayerNorm.backward`.
-/
theorem mhaPostNorm_backprop_eq_fderiv
    {seqLen dModel numHeads headDim : Nat} (c ε : ℝ)
    (xV : CtxVec (mhaWithNormCtx seqLen dModel numHeads headDim))
    (seedV :
      CtxVec (mhaWithNormCtx seqLen dModel numHeads headDim ++
        ssMHAWithPostNorm seqLen dModel numHeads headDim))
    (hε : 0 < ε) :
    Tape.GraphData.backpropVec
        (Γ := mhaWithNormCtx seqLen dModel numHeads headDim)
        (ss := ssMHAWithPostNorm seqLen dModel numHeads headDim)
        (mhaPostNormGraph (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) c ε).toData ()
        xV seedV
      =
    (fderiv ℝ
        (Tape.GraphData.evalVec
          (Γ := mhaWithNormCtx seqLen dModel numHeads headDim)
          (ss := ssMHAWithPostNorm seqLen dModel numHeads headDim)
          (mhaPostNormGraph (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
            (headDim := headDim) c ε).toData ())
        xV).adjoint seedV :=
  Tape.Graph.backpropVec_eq_fderiv
    (Γ := mhaWithNormCtx seqLen dModel numHeads headDim)
    (ss := ssMHAWithPostNorm seqLen dModel numHeads headDim)
    (g := mhaPostNormGraph (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
      (headDim := headDim) c ε) ()
    xV seedV
    (mhaPostNormFDerivAt (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) c ε xV hε)

/-!
## Sequence feed-forward plus post-norm

The second sublayer of a post-norm Transformer encoder block is

`LayerNorm(X + FFN(X), gamma, beta)`.

Its FFN affine maps and biases act on flattened sequence tensors and are held fixed during
differentiation. A shared position-wise implementation can provide block-diagonal maps, and
correspondence with a runtime or fused kernel is a separate obligation.
-/

/-- Sequence-FFN context extended with the affine parameters for the following LayerNorm. -/
abbrev seqFFNWithNormCtx (seqLen dModel : Nat) : List Shape :=
  seqFFNCtx seqLen dModel ++ [[dModel], [dModel]]

/-- Saved tensors: the sequence-FFN residual intermediates followed by one whole-node LayerNorm
output. -/
abbrev ssSeqFFNWithPostNorm (seqLen dModel dFF : Nat) : List Shape :=
  ssSeqFFNResidual seqLen dModel dFF ++ [[seqLen, dModel]]

/-- Gamma parameter for the FFN post-norm LayerNorm. -/
def idxSeqFFNPostNormGamma {seqLen dModel : Nat} {ss : List Shape} :
    Idx (seqFFNWithNormCtx seqLen dModel ++ ss) [dModel] :=
  Idx.weaken
    (Γ := seqFFNWithNormCtx seqLen dModel)
    (s := [dModel])
    ⟨⟨1, by simp [seqFFNWithNormCtx, seqFFNCtx]⟩, by simp⟩
    ss

/-- Beta parameter for the FFN post-norm LayerNorm. -/
def idxSeqFFNPostNormBeta {seqLen dModel : Nat} {ss : List Shape} :
    Idx (seqFFNWithNormCtx seqLen dModel ++ ss) [dModel] :=
  Idx.weaken
    (Γ := seqFFNWithNormCtx seqLen dModel)
    (s := [dModel])
    ⟨⟨2, by simp [seqFFNWithNormCtx, seqFFNCtx]⟩, by simp⟩
    ss

/-- The residual stream `X + FFN(X)` after the sequence-FFN prefix graph. -/
def idxSeqFFNResidualForPostNorm {seqLen dModel dFF : Nat} :
    Idx (seqFFNWithNormCtx seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
      [seqLen, dModel] :=
  Idx.last (Γ := seqFFNWithNormCtx seqLen dModel)
    (ss := [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
    (τ := [seqLen, dModel])

/-- LayerNorm input triple after the residual-FFN prefix has run. -/
def seqFFNPostNormInputs {seqLen dModel dFF : Nat} :
    LayerNorm.Inputs
      (seqFFNWithNormCtx seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
      seqLen dModel :=
  { x := idxSeqFFNResidualForPostNorm (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
    gamma := idxSeqFFNPostNormGamma (seqLen := seqLen) (dModel := dModel)
      (ss := ssSeqFFNResidual seqLen dModel dFF)
    beta := idxSeqFFNPostNormBeta (seqLen := seqLen) (dModel := dModel)
      (ss := ssSeqFFNResidual seqLen dModel dFF) }

/-- Sequence-FFN graph while carrying the following LayerNorm's affine parameters. -/
def seqFFNResidualWithNormDGraph {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel])) :
    DGraph (seqFFNWithNormCtx seqLen dModel) (ssSeqFFNResidual seqLen dModel dFF) :=
  DGraph.weakenContext
    (seqFFNResidualDGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
      fc1 b1 fc2 b2)
    [[dModel], [dModel]]

/--
Single SSA graph for the second post-norm Transformer encoder sublayer:

`LayerNorm(X + FFN(X), gamma, beta)`.
-/
def seqFFNPostNormGraph {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (ε : ℝ) :
    Tape.Graph (α := ℝ) Unit (seqFFNWithNormCtx seqLen dModel)
      (ssSeqFFNWithPostNorm seqLen dModel dFF) :=
  .snoc
    (seqFFNResidualWithNormDGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
      fc1 b1 fc2 b2).g
    (LayerNorm.node
      (Γ := seqFFNWithNormCtx seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (seqFFNPostNormInputs (seqLen := seqLen) (dModel := dModel) (dFF := dFF))
      ε)

/-- Pointwise correctness of the single-graph residual-FFN post-norm sublayer, for `0 < ε`. -/
def seqFFNPostNormFDerivAt
    {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (ε : ℝ)
    (xV : CtxVec (seqFFNWithNormCtx seqLen dModel))
    (hε : 0 < ε) :
    Tape.RealGraph.FDerivCorrectAt
      (Γ := seqFFNWithNormCtx seqLen dModel)
      (ss := ssSeqFFNWithPostNorm seqLen dModel dFF)
      (seqFFNPostNormGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
        fc1 b1 fc2 b2 ε).toData ()
      xV := by
  classical
  let dgPrefix := seqFFNResidualWithNormDGraph (seqLen := seqLen) (dModel := dModel)
    (dFF := dFF) fc1 b1 fc2 b2
  refine ⟨Tape.RealGraph.FDerivCorrect.at dgPrefix.hg xV, ?_⟩
  exact
    LayerNorm.nodeFDerivAt
      (Γ := seqFFNWithNormCtx seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (seqFFNPostNormInputs (seqLen := seqLen) (dModel := dModel) (dFF := dFF))
      ε
      (Tape.GraphData.evalVec
        (Γ := seqFFNWithNormCtx seqLen dModel)
        (ss := ssSeqFFNResidual seqLen dModel dFF)
        dgPrefix.g.toData () xV)
      hε

/--
End-to-end VJP theorem for the single-graph residual-FFN post-norm sublayer, for `0 < ε`.
Its LayerNorm stage is a `LayerNorm.node`, whose VJP is the adjoint of its derivative by
definition, and `LayerNorm.adjoint_fderiv_eq_backward` identifies the adjoint derivative of the
LayerNorm map with the primitive `LayerNorm.backward`.
-/
theorem seqFFNPostNorm_backprop_eq_fderiv
    {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (ε : ℝ)
    (xV : CtxVec (seqFFNWithNormCtx seqLen dModel))
    (seedV : CtxVec (seqFFNWithNormCtx seqLen dModel ++ ssSeqFFNWithPostNorm seqLen dModel dFF))
    (hε : 0 < ε) :
    Tape.GraphData.backpropVec
        (Γ := seqFFNWithNormCtx seqLen dModel)
        (ss := ssSeqFFNWithPostNorm seqLen dModel dFF)
        (seqFFNPostNormGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
          fc1 b1 fc2 b2 ε).toData ()
        xV seedV
      =
    (fderiv ℝ
        (Tape.GraphData.evalVec
          (Γ := seqFFNWithNormCtx seqLen dModel)
          (ss := ssSeqFFNWithPostNorm seqLen dModel dFF)
          (seqFFNPostNormGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
            fc1 b1 fc2 b2 ε).toData ())
        xV).adjoint seedV :=
  Tape.Graph.backpropVec_eq_fderiv
    (Γ := seqFFNWithNormCtx seqLen dModel)
    (ss := ssSeqFFNWithPostNorm seqLen dModel dFF)
    (g := seqFFNPostNormGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
      fc1 b1 fc2 b2 ε) ()
    xV seedV
    (seqFFNPostNormFDerivAt (seqLen := seqLen) (dModel := dModel)
      (dFF := dFF) fc1 b1 fc2 b2 ε xV hε)

end

end Transformer
end TorchLean
