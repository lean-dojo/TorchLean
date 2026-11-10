/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Attention.BiasedMultiHead
public import TorchLean.NN.Models.Transformer.PostNorm

/-!
# Post-Norm Decoder Core

The decoder-core SSA graph here has a fixed additive score bias, and its VJP theorem is
`decoderCore_backprop_eq_fderiv`. The graph starts from head tensors that are already
projected. Its merge and FFN affine maps are fixed arguments, so the differentiated context holds
only the head tensors, the residual stream and the LayerNorm parameters. The score bias is finite
real data and does not implement Boolean hard or causal masking. The theorems are exact-real
derivative statements and do not relate the graph to the executable decoder or to LibTorch.

Both LayerNorms are `LayerNorm.node` nodes in the one graph, certified by
`LayerNorm.nodeFDerivAt` for `0 < ε₁` and `0 < ε₂`. Map-level composition goes
through `Tape.GraphData.hasFDerivAt_evalVec_comp`: with `LayerNorm.graph` and
`LayerNorm.graphFDerivAt`, it differentiates LayerNorm after a differentiable map into the
LayerNorm context `LayerNorm.ctx` (`[residual_stream, gamma, beta]`). For
the attention sublayer, the same lemma applied to `biasedCoreDGraph` from
`TorchLean.NN.Attention.BiasedMultiHead` with the caller's
projection derivative gives the derivative of the core after the projection. `HasFDerivAt.comp` with
the derivative of a caller's merge into `LayerNorm.ctx` then gives such a map. Nothing here builds a
full decoder from a token or parameter context.

From Vaswani et al., Section 3.1, we kept the self-attention and FFN sublayers of a post-norm
block, with the tanh GELU in the FFN where they use ReLU. Their decoder also attends to the
encoder output, a third sublayer this core leaves out, and it masks later positions with `−∞`
scores (Section 3.2.3). With the default `bias := 0` the attention here is unmasked.

References:
- Vaswani et al. (2017), "Attention Is All You Need", NIPS 2017, arXiv:1706.03762, Sections 3.1
  and 3.2.3.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace Transformer


noncomputable section

open Tape (DGraph)

/-!
## Concrete fixed-bias decoder-core SSA graph

The context holds the head tensors after the Q/K/V projection split, the residual stream and the
two LayerNorm parameter pairs:

`[Q_heads, Kᵀ_heads, V_heads, residual_stream, gamma₁, beta₁, gamma₂, beta₂]`.

The graph adds the fixed bias to the scaled attention scores, merges the head output through a
supplied affine map and adds the residual stream before the first LayerNorm. The GELU feed-forward
sublayer, its residual add and the second LayerNorm follow.
-/

/-- Concrete decoder-core context: head tensors, residual stream, and two LayerNorm
parameter pairs. -/
abbrev decoderCoreCtx (seqLen dModel numHeads headDim : Nat) : List Shape :=
  MultiHeadAttention.biasedCoreCtx seqLen numHeads headDim ++
    [ [seqLen, dModel]
    , [dModel], [dModel]
    , [dModel], [dModel]
    ]

/-- Saved tensors for the concrete fixed-bias decoder-core block. -/
abbrev ssDecoderCore (seqLen dModel numHeads headDim dFF : Nat) : List Shape :=
  MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
    [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
    ssSeqFFNResidual seqLen dModel dFF ++
    [[seqLen, dModel]]

/-- Residual-stream input in the decoder-core context. -/
def idxDecoderResidualInput {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (decoderCoreCtx seqLen dModel numHeads headDim ++ ss) [seqLen, dModel] :=
  ⟨⟨3, by simp⟩, by simp⟩

/-- First LayerNorm scale parameter in the decoder-core context. -/
def idxDecoderNorm1Gamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (decoderCoreCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  ⟨⟨4, by simp⟩, by simp⟩

/-- First LayerNorm shift parameter in the decoder-core context. -/
def idxDecoderNorm1Beta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (decoderCoreCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  ⟨⟨5, by simp⟩, by simp⟩

/-- Second LayerNorm scale parameter in the decoder-core context. -/
def idxDecoderNorm2Gamma {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (decoderCoreCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  ⟨⟨6, by simp⟩, by simp⟩

/-- Second LayerNorm shift parameter in the decoder-core context. -/
def idxDecoderNorm2Beta {seqLen dModel numHeads headDim : Nat} {ss : List Shape} :
    Idx (decoderCoreCtx seqLen dModel numHeads headDim ++ ss) [dModel] :=
  ⟨⟨7, by simp⟩, by simp⟩

/-- The attention core `MultiHeadAttention.biasedCoreDGraph` with fixed bias, weakened to
carry the residual stream and both LayerNorm parameter pairs. -/
def decoderBiasedCoreDGraph {seqLen dModel numHeads headDim : Nat}
    (c : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.scoresShape seqLen numHeads)) := 0) :
    DGraph (decoderCoreCtx seqLen dModel numHeads headDim)
      (MultiHeadAttention.ssBiasedCore seqLen numHeads headDim) :=
  DGraph.weakenContext
    (MultiHeadAttention.biasedCoreDGraph
      (n := seqLen) (numHeads := numHeads) (headDim := headDim) c bias)
    [ [seqLen, dModel]
    , [dModel], [dModel]
    , [dModel], [dModel]
    ]

/-- Head output after the fixed-bias attention core. -/
def idxDecoderHeadOut {seqLen dModel numHeads headDim : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim)
      (MultiHeadAttention.headsShape seqLen numHeads headDim) :=
  Idx.last (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := [MultiHeadAttention.scoresShape seqLen numHeads,
      MultiHeadAttention.scoresShape seqLen numHeads,
      MultiHeadAttention.scoresShape seqLen numHeads,
      MultiHeadAttention.scoresShape seqLen numHeads])
    (τ := MultiHeadAttention.headsShape seqLen numHeads headDim)

/-- Merged attention output after the supplied output projection. -/
def idxDecoderMergedAttention {seqLen dModel numHeads headDim : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.last (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim)
    (τ := [seqLen, dModel])

/-- Residual input weakened past the merged attention output. -/
def idxDecoderResidualAfterMerge {seqLen dModel numHeads headDim : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.weaken
    (idxDecoderResidualInput (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim))
    [[seqLen, dModel]]

/-- Sum of the residual stream and projected attention output before the first LayerNorm. -/
def idxDecoderAttentionResidual {seqLen dModel numHeads headDim : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.last (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
      [[seqLen, dModel]])
    (τ := [seqLen, dModel])

/-- First LayerNorm input triple for the concrete decoder-core graph. -/
def decoderNorm1Inputs {seqLen dModel numHeads headDim : Nat} :
    LayerNorm.Inputs
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel]])
      seqLen dModel :=
  { x := idxDecoderAttentionResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
    gamma := idxDecoderNorm1Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel]])
    beta := idxDecoderNorm1Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel]]) }

/-- Decoder graph through the first post-norm attention sublayer. -/
def decoderAfterNorm1Graph {seqLen dModel numHeads headDim : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.headsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (mergeBias : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.scoresShape seqLen numHeads)) := 0) :
    Tape.Graph (α := ℝ) Unit (decoderCoreCtx seqLen dModel numHeads headDim)
      (MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]]) :=
  let g0 := (decoderBiasedCoreDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c bias).g
  let g1 := Tape.Graph.snoc g0
    (Tape.RealNode.affine
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim)
      (sIn := MultiHeadAttention.headsShape seqLen numHeads headDim)
      (sOut := [seqLen, dModel])
      (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      merge mergeBias)
  let g2 := Tape.Graph.snoc g1
    (Tape.LawfulNode.ofArm (Op.add (s := [seqLen, dModel])) rfl
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++ [[seqLen, dModel]])
      (idxDecoderResidualAfterMerge (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim),
        idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim), ())
      (Arm.adj_of_cert₂ fun xs δ => Op.add_prod_backward_eq_adjoint xs.1 xs.2.1 δ))
  Tape.Graph.snoc g2
    (LayerNorm.node
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel]])
      (m := seqLen) (n := dModel)
      (decoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε₁)

/-- First decoder post-norm output. -/
def idxDecoderNorm1Out {seqLen dModel numHeads headDim : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.last (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
      [[seqLen, dModel], [seqLen, dModel]])
    (τ := [seqLen, dModel])

/-- First decoder post-norm output weakened through FFN intermediates. -/
def idxDecoderNorm1OutAfterFFN {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.weaken
    (idxDecoderNorm1Out (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim))
    [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]]

/-- First FFN affine output in the concrete decoder graph. -/
def idxDecoderFFNHiddenPre {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
        [[seqLen, dFF]])
      [seqLen, dFF] :=
  Idx.last (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
      [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]])
    (τ := [seqLen, dFF])

/-- FFN activation output in the concrete decoder graph. -/
def idxDecoderFFNHiddenAct {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF]])
      [seqLen, dFF] :=
  Idx.last (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
      [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel], [seqLen, dFF]])
    (τ := [seqLen, dFF])

/-- FFN projection output in the concrete decoder graph. -/
def idxDecoderFFNProjected {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.last (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
      [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel], [seqLen, dFF], [seqLen, dFF]])
    (τ := [seqLen, dModel])

/-- FFN residual output before the second decoder LayerNorm. -/
def idxDecoderFFNResidual {seqLen dModel numHeads headDim dFF : Nat} :
    Idx
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
        ssSeqFFNResidual seqLen dModel dFF)
      [seqLen, dModel] :=
  Idx.last (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
      [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel], [seqLen, dFF],
        [seqLen, dFF], [seqLen, dModel]])
    (τ := [seqLen, dModel])

/-- Decoder graph through the FFN residual. -/
def decoderFFNResidualGraph {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.headsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (mergeBias : Vec (Shape.size [seqLen, dModel]))
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.scoresShape seqLen numHeads)) := 0) :
    Tape.Graph (α := ℝ) Unit (decoderCoreCtx seqLen dModel numHeads headDim)
      (MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel],
          [seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF) :=
  let g1 := decoderAfterNorm1Graph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) merge mergeBias c ε₁ bias
  let g2 := Tape.Graph.snoc g1
    (Tape.RealNode.affine
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]])
      (sIn := [seqLen, dModel]) (sOut := [seqLen, dFF])
      (idxDecoderNorm1Out (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      fc1 b1)
  let g3 := Tape.Graph.snoc g2
    (Tape.LawfulNode.ofArm (Op.gelu (s := [seqLen, dFF])) rfl
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++ [[seqLen, dFF]])
      (idxDecoderFFNHiddenPre (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF), ())
      (Arm.adj_of_cert₁ fun xs δ => Op.gelu_tanh_backward_eq_adjoint xs.1 δ))
  let g4 := Tape.Graph.snoc g3
    (Tape.RealNode.affine
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF]])
      (sIn := [seqLen, dFF]) (sOut := [seqLen, dModel])
      (idxDecoderFFNHiddenAct (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      fc2 b2)
  Tape.Graph.snoc g4
    (Tape.LawfulNode.ofArm (Op.add (s := [seqLen, dModel])) rfl
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
        [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      (idxDecoderNorm1OutAfterFFN (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF),
        idxDecoderFFNProjected (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF), ())
      (Arm.adj_of_cert₂ fun xs δ => Op.add_prod_backward_eq_adjoint xs.1 xs.2.1 δ))

/-- Second LayerNorm input triple for the concrete decoder-core graph. -/
def decoderNorm2Inputs {seqLen dModel numHeads headDim dFF : Nat} :
    LayerNorm.Inputs
      (decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
        ssSeqFFNResidual seqLen dModel dFF)
      seqLen dModel :=
  { x := idxDecoderFFNResidual (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
    gamma := idxDecoderNorm2Gamma (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel],
          [seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF)
    beta := idxDecoderNorm2Beta (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim)
      (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel],
          [seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF) }

/-- Concrete SSA graph for one additive-bias decoder-core block. -/
def decoderCoreGraph {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.headsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (mergeBias : Vec (Shape.size [seqLen, dModel]))
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.scoresShape seqLen numHeads)) := 0) :
    Tape.Graph (α := ℝ) Unit (decoderCoreCtx seqLen dModel numHeads headDim)
      (ssDecoderCore seqLen dModel numHeads headDim dFF) :=
  Tape.Graph.snoc
    (decoderFFNResidualGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      merge mergeBias fc1 b1 fc2 b2 c ε₁ bias)
    (LayerNorm.node
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel],
          [seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (decoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      ε₂)

/-- Pointwise analytic correctness of the decoder graph through the FFN residual at `xV`, for
`0 < ε₁`. -/
def decoderFFNResidualFDerivAt {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.headsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (mergeBias : Vec (Shape.size [seqLen, dModel]))
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.scoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (decoderCoreCtx seqLen dModel numHeads headDim))
    (hε₁ : 0 < ε₁) :
    Tape.RealGraph.FDerivCorrectAt
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
      (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel],
          [seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF)
      (decoderFFNResidualGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        merge mergeBias fc1 b1 fc2 b2 c ε₁ bias).toData ()
      xV := by
  classical
  let dgCore := decoderBiasedCoreDGraph (seqLen := seqLen) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim) c bias
  let hgCore : Tape.RealGraph.FDerivCorrectAt
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
      (ss := MultiHeadAttention.ssBiasedCore seqLen numHeads headDim) dgCore.g.toData () xV :=
    Tape.RealGraph.FDerivCorrect.at dgCore.hg xV
  refine ⟨⟨⟨⟨⟨⟨⟨?_, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩, ?_⟩
  · simpa [decoderAfterNorm1Graph, dgCore] using hgCore
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.affineFDeriv
        (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssBiasedCore seqLen numHeads headDim)
        (sIn := MultiHeadAttention.headsShape seqLen numHeads headDim)
        (sOut := [seqLen, dModel])
        (idxDecoderHeadOut (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        merge mergeBias)
      _
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.FDerivCorrect.ofArm₂
        (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++ [[seqLen, dModel]])
        (s := [seqLen, dModel])
        (a := idxDecoderResidualAfterMerge (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        (b := idxDecoderMergedAttention (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim))
        (cert := Op.add_prod_backward_eq_adjoint))
      _
  · exact LayerNorm.nodeFDerivAt
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel]])
      (m := seqLen) (n := dModel)
      (decoderNorm1Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim))
      ε₁ _ hε₁
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.affineFDeriv
        (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
          [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]])
        (sIn := [seqLen, dModel]) (sOut := [seqLen, dFF])
        (idxDecoderNorm1Out (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim))
        fc1 b1)
      _
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.FDerivCorrect.ofArm₁
        (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
          [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++ [[seqLen, dFF]])
        (s := [seqLen, dFF])
        (idx := idxDecoderFFNHiddenPre (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF))
        (cert := Op.gelu_tanh_backward_eq_adjoint))
      _
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.affineFDeriv
        (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
          [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
          [[seqLen, dFF], [seqLen, dFF]])
        (sIn := [seqLen, dFF]) (sOut := [seqLen, dModel])
        (idxDecoderFFNHiddenAct (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
        fc2 b2)
      _
  · exact Tape.RealNode.FDerivCorrect.at
      (Tape.RealNode.FDerivCorrect.ofArm₂
        (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
          MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
          [[seqLen, dModel], [seqLen, dModel], [seqLen, dModel]] ++
          [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
        (s := [seqLen, dModel])
        (a := idxDecoderNorm1OutAfterFFN (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
        (b := idxDecoderFFNProjected (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim) (dFF := dFF))
        (cert := Op.add_prod_backward_eq_adjoint))
      _

/-- Pointwise analytic correctness of the whole decoder-core graph at `xV`, for `0 < ε₁` and
`0 < ε₂`. -/
def decoderCoreFDerivAt {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.headsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (mergeBias : Vec (Shape.size [seqLen, dModel]))
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.scoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (decoderCoreCtx seqLen dModel numHeads headDim))
    (hε₁ : 0 < ε₁)
    (hε₂ : 0 < ε₂) :
    Tape.RealGraph.FDerivCorrectAt
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
      (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
      (decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
        merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias).toData ()
      xV :=
  ⟨decoderFFNResidualFDerivAt
    (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
    (dFF := dFF) merge mergeBias fc1 b1 fc2 b2 c ε₁ bias xV hε₁,
    LayerNorm.nodeFDerivAt
      (Γ := decoderCoreCtx seqLen dModel numHeads headDim ++
        MultiHeadAttention.ssBiasedCore seqLen numHeads headDim ++
        [[seqLen, dModel], [seqLen, dModel],
          [seqLen, dModel]] ++ ssSeqFFNResidual seqLen dModel dFF)
      (m := seqLen) (n := dModel)
      (decoderNorm2Inputs (seqLen := seqLen) (dModel := dModel)
        (numHeads := numHeads) (headDim := headDim) (dFF := dFF))
      ε₂ _ hε₂⟩

/-- End-to-end VJP theorem for the additive-bias decoder-core graph: for `0 < ε₁` and `0 < ε₂`,
`backpropVec` at `xV` equals the adjoint of the derivative of `evalVec` at `xV`, applied to
`seedV`. Each LayerNorm stage is a `LayerNorm.node`, whose VJP is the adjoint of its derivative
by definition, and `LayerNorm.adjoint_fderiv_eq_backward` identifies the adjoint derivative of the
LayerNorm map with the primitive `LayerNorm.backward`. -/
theorem decoderCore_backprop_eq_fderiv
    {seqLen dModel numHeads headDim dFF : Nat}
    (merge :
      Vec (Shape.size (MultiHeadAttention.headsShape seqLen numHeads headDim)) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (mergeBias : Vec (Shape.size [seqLen, dModel]))
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel]))
    (c ε₁ ε₂ : ℝ)
    (bias : Vec (Shape.size (MultiHeadAttention.scoresShape seqLen numHeads)) := 0)
    (xV : CtxVec (decoderCoreCtx seqLen dModel numHeads headDim))
    (seedV : CtxVec (decoderCoreCtx seqLen dModel numHeads headDim ++
      ssDecoderCore seqLen dModel numHeads headDim dFF))
    (hε₁ : 0 < ε₁)
    (hε₂ : 0 < ε₂) :
    Tape.GraphData.backpropVec
        (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
        (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
        (decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
          (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
          merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias).toData ()
        xV seedV
      =
    (fderiv ℝ
        (Tape.GraphData.evalVec
          (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
          (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
          (decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
            (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
            merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias).toData ())
        xV).adjoint seedV :=
  Tape.Graph.backpropVec_eq_fderiv
    (Γ := decoderCoreCtx seqLen dModel numHeads headDim)
    (ss := ssDecoderCore seqLen dModel numHeads headDim dFF)
    (g := decoderCoreGraph (seqLen := seqLen) (dModel := dModel)
      (numHeads := numHeads) (headDim := headDim) (dFF := dFF)
      merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias) ()
    xV seedV
    (decoderCoreFDerivAt
      (seqLen := seqLen) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
      (dFF := dFF) merge mergeBias fc1 b1 fc2 b2 c ε₁ ε₂ bias xV hε₁ hε₂)

end

end Transformer
end TorchLean
