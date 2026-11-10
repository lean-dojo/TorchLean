/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Transformer.PostNorm

/-!
# Post-Norm Transformer Encoder Block

The end-to-end VJP theorem here covers one concrete SSA graph for the post-norm encoder block

`x ↦ LN₂(FFN(LN₁(MHA(x) + x)) + LN₁(MHA(x) + x))`.

The two sublayers have graph-level VJP theorems of their own in `PostNorm.lean`. `encoderBlockGraph`
assembles them into a single graph over the context
`[x, Wq, Wk, Wv, Wo, gamma₁, beta₁, gamma₂, beta₂]`, with VJP theorem
`encoderBlock_backprop_eq_fderiv`. A model proof that needs the block as one
differentiable map composes the two post-norm sublayers with
`Tape.GraphData.hasFDerivAt_evalVec_comp` and `LayerNorm.graphFDerivAt`.

Attention splits heads as `MultiHeadAttention.splitHeads` does. The FFN maps and biases are fixed
inputs to the construction and sit outside its context. The theorem is about the derivative and
reverse pass of this exact-real graph, and relating it to the public runtime encoder needs a
separate correspondence proof.

We assembled the block from Vaswani et al., Section 3.1, with the GELU FFN of `FeedForward` where
they use ReLU. `encoderBlock_backprop_eq_fderiv` holds for every score scale `c` and every pair of
LayerNorm epsilons with `0 < ε₁` and `0 < ε₂`.

References:
- Vaswani et al. (2017), "Attention Is All You Need", NIPS 2017, arXiv:1706.03762, Section 3.1.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace Transformer


noncomputable section

open Tape (DGraph)

/-!
## Concrete SSA encoder block

The block graph runs over the context

`[x, Wq, Wk, Wv, Wo, gamma₁, beta₁, gamma₂, beta₂]`.

It applies the residual attention prefix, the first LayerNorm, the GELU feed-forward sublayer with
its residual add, and the second LayerNorm. The FFN affine maps are fixed sequence-level linear
maps with biases, the interface of `seqFFNResidualDGraph`.
-/

/-- Full encoder-block context: the input `x`, the MHA weights `Wq, Wk, Wv, Wo` and the two
LayerNorm parameter pairs. -/
abbrev encoderBlockCtx (seqLen dModel numHeads headDim : Nat) : List Shape :=
  mhaWithNormCtx seqLen dModel numHeads headDim ++
    [[dModel], [dModel]]

/-- Saved tensors for one post-norm encoder block. -/
abbrev ssEncoderBlock (seqLen dModel numHeads headDim dFF : Nat) : List Shape :=
  ssMHAResidual seqLen dModel numHeads headDim ++
    [[seqLen, dModel]] ++
    ssSeqFFNResidual seqLen dModel dFF ++
    [[seqLen, dModel]]

/-- First LayerNorm scale parameter in the full encoder-block context. -/
def idxEncoderNorm1Gamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (encoderBlockCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  ⟨⟨5, by simp [encoderBlockCtx]⟩, by simp [encoderBlockCtx]⟩

/-- First LayerNorm shift parameter in the full encoder-block context. -/
def idxEncoderNorm1Beta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (encoderBlockCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  ⟨⟨6, by simp [encoderBlockCtx]⟩, by simp [encoderBlockCtx]⟩

/-- Second LayerNorm scale parameter in the full encoder-block context. -/
def idxEncoderNorm2Gamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (encoderBlockCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  ⟨⟨7, by simp [encoderBlockCtx]⟩, by simp [encoderBlockCtx]⟩

/-- Second LayerNorm shift parameter in the full encoder-block context. -/
def idxEncoderNorm2Beta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (encoderBlockCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  ⟨⟨8, by simp [encoderBlockCtx]⟩, by simp [encoderBlockCtx]⟩

/-- The residual-attention prefix `mhaResidualDGraph`, weakened to carry both LayerNorm parameter
pairs. -/
def encoderMHAResidualDGraph {seqLen dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (encoderBlockCtx seqLen dModel numHeads headDim)
      (ssMHAResidual seqLen dModel numHeads headDim) :=
  DGraph.weakenContext
    (mhaResidualDGraph (n := seqLen) (dModel := dModel) (numHeads := numHeads)
      (headDim := headDim) c)
    [[dModel], [dModel], [dModel], [dModel]]

/-- Residual stream `x + MHA(x)` after the attention prefix. -/
def idxEncoderMHAResidual {seqLen dModel numHeads headDim : Nat} :
    Idx
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      [seqLen, dModel] :=
  Idx.last (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ss seqLen dModel numHeads headDim)
    (τ := [seqLen, dModel])

/-- First LayerNorm input triple in the concrete encoder block. -/
def encoderNorm1Inputs {seqLen dModel numHeads headDim : Nat} :
    LayerNorm.Inputs
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      seqLen dModel :=
  { x := idxEncoderMHAResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
    gamma := idxEncoderNorm1Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim)
    beta := idxEncoderNorm1Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim) }

/-- Encoder graph through the first post-norm attention sublayer. -/
def encoderAfterNorm1Graph {seqLen dModel numHeads headDim : Nat} (c ε₁ : ℝ) :
    Tape.Graph (α := ℝ) Unit (encoderBlockCtx seqLen dModel numHeads headDim)
      (ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]]) :=
  .snoc
    (encoderMHAResidualDGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) c).g
    (LayerNorm.node
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      (m := seqLen) (n := dModel)
      (encoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε₁)

/-- First post-norm sublayer output. -/
def idxEncoderNorm1Out {seqLen dModel numHeads headDim : Nat} :
    Idx
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.last (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim)
    (τ := [seqLen, dModel])

/-- First post-norm output weakened through the FFN intermediates. -/
def idxEncoderNorm1OutAfterFFN {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.weaken
    (idxEncoderNorm1Out (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim))
    [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]]

/-- First FFN affine output in the full encoder graph. -/
def idxEncoderFFNHiddenPre {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        [[seqLen, dFF]])
      [seqLen, dFF] :=
  Idx.last (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]])
    (τ := [seqLen, dFF])

/-- FFN activation output in the full encoder graph. -/
def idxEncoderFFNHiddenAct {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF]])
      [seqLen, dFF] :=
  Idx.last (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel], [seqLen, dFF]])
    (τ := [seqLen, dFF])

/-- FFN projection output in the full encoder graph. -/
def idxEncoderFFNProjected {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.last (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel],
      [seqLen, dFF], [seqLen, dFF]])
    (τ := [seqLen, dModel])

/-- FFN residual output after the second sublayer residual add. -/
def idxEncoderFFNResidual {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        ssSeqFFNResidual seqLen dModel dFF)
      [seqLen, dModel] :=
  Idx.last (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
    (ss := ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel],
      [seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
    (τ := [seqLen, dModel])

/-- Encoder graph through the FFN residual, before the second LayerNorm. -/
def encoderFFNResidualGraph {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ : ℝ) :
    Tape.Graph (α := ℝ) Unit (encoderBlockCtx seqLen dModel numHeads headDim)
      (ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        ssSeqFFNResidual seqLen dModel dFF) :=
  let g1 := encoderAfterNorm1Graph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c ε₁
  let g2 := Tape.Graph.snoc g1
    (Tape.RealNode.affine
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]])
      (sIn := [seqLen, dModel]) (sOut := [seqLen, dFF])
      (idxEncoderNorm1Out (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      fc1 b1)
  let g3 := Tape.Graph.snoc g2
    (Tape.LawfulNode.ofArm (Op.gelu (s := [seqLen, dFF])) rfl
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++ [[seqLen, dFF]])
      (idxEncoderFFNHiddenPre (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF), ())
      (Arm.adj_of_cert₁ fun xs δ => Op.gelu_tanh_backward_eq_adjoint xs.1 δ))
  let g4 := Tape.Graph.snoc g3
    (Tape.RealNode.affine
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF]])
      (sIn := [seqLen, dFF]) (sOut := [seqLen, dModel])
      (idxEncoderFFNHiddenAct (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      fc2 b2)
  Tape.Graph.snoc g4
    (Tape.LawfulNode.ofArm (Op.add (s := [seqLen, dModel])) rfl
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      (idxEncoderNorm1OutAfterFFN (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF),
        idxEncoderFFNProjected (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF), ())
      (Arm.adj_of_cert₂ fun xs δ => Op.add_prod_backward_eq_adjoint xs.1 xs.2.1 δ))

/-- Second LayerNorm input triple in the concrete encoder block. -/
def encoderNorm2Inputs {seqLen dModel numHeads headDim dFF : Nat} :
    LayerNorm.Inputs
      (encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        ssSeqFFNResidual seqLen dModel dFF)
      seqLen dModel :=
  { x := idxEncoderFFNResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
    gamma := idxEncoderNorm2Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim ++
        [[seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF)
    beta := idxEncoderNorm2Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim ++
        [[seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF) }

/-- Concrete SSA graph for one full post-norm Transformer encoder block. -/
def encoderBlockGraph {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ ε₂ : ℝ) :
    Tape.Graph (α := ℝ) Unit (encoderBlockCtx seqLen dModel numHeads headDim)
      (ssEncoderBlock seqLen dModel numHeads headDim dFF) :=
  Tape.Graph.snoc
    (encoderFFNResidualGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      fc1 b1 fc2 b2 c ε₁)
    (LayerNorm.node
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (encoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      ε₂)

/-- Pointwise analytic correctness of the graph through the FFN residual at `xV`, for `0 < ε₁`. -/
def encoderFFNResidualFDerivAt {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ : ℝ)
    (xV : CtxVec (encoderBlockCtx seqLen dModel numHeads headDim))
    (hε₁ : 0 < ε₁) :
    Tape.RealGraph.FDerivCorrectAt
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim ++
        [[seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF)
      (encoderFFNResidualGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        fc1 b1 fc2 b2 c ε₁).toData ()
      xV := by
  classical
  let dgMha := encoderMHAResidualDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c
  let hgMha : Tape.RealGraph.FDerivCorrectAt
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
      (ss := ssMHAResidual seqLen dModel numHeads headDim) dgMha.g.toData () xV :=
    Tape.RealGraph.FDerivCorrect.at dgMha.hg xV
  refine ⟨⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩
  · simpa [encoderAfterNorm1Graph, dgMha] using hgMha
  · exact LayerNorm.nodeFDerivAt
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim)
      (m := seqLen) (n := dModel)
      (encoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε₁ _ hε₁
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.affineFDeriv
        (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
          ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]])
        (sIn := [seqLen, dModel]) (sOut := [seqLen, dFF])
        (idxEncoderNorm1Out (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        fc1 b1)
      _
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.FDerivCorrect.ofArm₁
        (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
          ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++ [[seqLen, dFF]])
        (s := [seqLen, dFF])
        (idx := idxEncoderFFNHiddenPre (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF))
        (cert := Op.gelu_tanh_backward_eq_adjoint))
      _
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.affineFDeriv
        (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
          ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
          [[seqLen, dFF], [seqLen, dFF]])
        (sIn := [seqLen, dFF]) (sOut := [seqLen, dModel])
        (idxEncoderFFNHiddenAct (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
        fc2 b2)
      _
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.FDerivCorrect.ofArm₂
        (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
          ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
          [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
        (s := [seqLen, dModel])
        (a := idxEncoderNorm1OutAfterFFN (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
        (b := idxEncoderFFNProjected (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF))
        (cert := Op.add_prod_backward_eq_adjoint))
      _

/-- Pointwise analytic correctness of the whole encoder-block graph at `xV`, for `0 < ε₁` and
`0 < ε₂`. -/
def encoderBlockFDerivAt {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ ε₂ : ℝ)
    (xV : CtxVec (encoderBlockCtx seqLen dModel numHeads headDim))
    (hε₁ : 0 < ε₁)
    (hε₂ : 0 < ε₂) :
    Tape.RealGraph.FDerivCorrectAt
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
      (ss := ssEncoderBlock seqLen dModel numHeads headDim dFF)
      (encoderBlockGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        fc1 b1 fc2 b2 c ε₁ ε₂).toData ()
      xV :=
  ⟨encoderFFNResidualFDerivAt
    (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
    (dFF := dFF) fc1 b1 fc2 b2 c ε₁ xV hε₁,
    LayerNorm.nodeFDerivAt
      (Γ := encoderBlockCtx seqLen dModel numHeads headDim ++
        ssMHAResidual seqLen dModel numHeads headDim ++ [[seqLen, dModel]] ++
        ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (encoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      ε₂ _ hε₂⟩

/-- End-to-end VJP theorem for the post-norm encoder-block graph: for `0 < ε₁` and `0 < ε₂`,
`backpropVec` at `xV` equals the adjoint of the derivative of `evalVec` at `xV`, applied to
`seedV`. Each LayerNorm stage is a `LayerNorm.node`, whose VJP is the adjoint of its derivative
by definition, and `LayerNorm.adjoint_fderiv_eq_backward` identifies the adjoint derivative of the
LayerNorm map with the primitive `LayerNorm.backward`. -/
theorem encoderBlock_backprop_eq_fderiv
    {seqLen dModel numHeads headDim dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ ε₂ : ℝ)
    (xV : CtxVec (encoderBlockCtx seqLen dModel numHeads headDim))
    (seedV : CtxVec (encoderBlockCtx seqLen dModel numHeads headDim ++
      ssEncoderBlock seqLen dModel numHeads headDim dFF))
    (hε₁ : 0 < ε₁)
    (hε₂ : 0 < ε₂) :
    Tape.GraphData.backpropVec
        (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
        (ss := ssEncoderBlock seqLen dModel numHeads headDim dFF)
        (encoderBlockGraph (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
          fc1 b1 fc2 b2 c ε₁ ε₂).toData ()
        xV seedV
      =
    (fderiv ℝ
        (Tape.GraphData.evalVec
          (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
          (ss := ssEncoderBlock seqLen dModel numHeads headDim dFF)
          (encoderBlockGraph (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
            fc1 b1 fc2 b2 c ε₁ ε₂).toData ())
        xV).adjoint seedV :=
  Tape.Graph.backpropVec_eq_fderiv
    (Γ := encoderBlockCtx seqLen dModel numHeads headDim)
    (ss := ssEncoderBlock seqLen dModel numHeads headDim dFF)
    (g := encoderBlockGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      fc1 b1 fc2 b2 c ε₁ ε₂) ()
    xV seedV
    (encoderBlockFDerivAt
      (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
      (dFF := dFF) fc1 b1 fc2 b2 c ε₁ ε₂ xV hε₁ hε₂)

end

end Transformer
end TorchLean
