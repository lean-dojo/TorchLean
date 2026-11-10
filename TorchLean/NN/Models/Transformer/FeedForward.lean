/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.DGraph
public import TorchLean.NN.Tape.RealNode.Arithmetic
public import TorchLean.NN.Tape.RealNode.Arm
public import TorchLean.NN.Op.Pointwise

/-!
# Transformer Feed-Forward Sublayer VJP

This file builds the position-wise Transformer feed-forward sublayer on the flattened
`(seqLen × dModel)` sequence tensor as a proof-carrying graph (`DGraph`):

`X ↦ X + A₂(GELU(A₁ X + b₁)) + b₂`.

The full Transformer encoder block composes this FFN residual with MHA and LayerNorm. This file
gives the proof component for the FFN half of that block.

The shape is eq. (2) of Vaswani et al. with their ReLU replaced by the tanh GELU of Hendrycks and
Gimpel, Section 2, `0.5 x (1 + tanh (√(2 / π) (x + 0.044715 x³)))`, which is PyTorch's
`approximate='tanh'`. It's the default of `Op.gelu`, and `Op.gelu_tanh_backward_eq_adjoint`
certifies that node. The spec record `Transformer.FeedForward` in `TorchLean.NN.Models.Transformer`
keeps the ReLU, so it and this graph compute different FFNs.

References:

* Vaswani et al. (2017), "Attention Is All You Need", NIPS 2017, arXiv:1706.03762, eq. (2).
* Dan Hendrycks and Kevin Gimpel (2016), "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415,
  Section 2.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace Transformer


noncomputable section

open Tape (DGraph)

/-!
## Sequence-shaped FFN residual

The runtime Transformer applies the same FFN to every token in a `(seqLen × dModel)` tensor. The
model-level proof interface packages that operation as two fixed affine maps over the flattened
sequence tensor. A concrete shared-weight implementation instantiates them with the usual
block-diagonal (time-distributed) linear operator and repeated biases. The graph's proof
`Tape.RealGraph.FDerivCorrect` needs only the affine maps and the smooth GELU primitive.
-/

/-- Context for a sequence-level FFN residual block: just the sequence stream. -/
abbrev seqFFNCtx (seqLen dModel : Nat) : List Shape :=
  [[seqLen, dModel]]

/-- Saved tensors for the sequence-level residual FFN. -/
abbrev ssSeqFFNResidual (seqLen dModel dFF : Nat) : List Shape :=
  [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel], [seqLen, dModel]]

/-- Sequence input index, weakened through saved tensors. -/
def seqFFNIdxX {seqLen dModel : Nat} {ss : List Shape} :
    Idx (seqFFNCtx seqLen dModel ++ ss) [seqLen, dModel] :=
  ⟨⟨0, by simp [seqFFNCtx]⟩, by simp [seqFFNCtx]⟩

/-- First sequence affine output. -/
def seqFFNIdxHiddenPre {seqLen dModel dFF : Nat} :
    Idx (seqFFNCtx seqLen dModel ++ [[seqLen, dFF]])
      [seqLen, dFF] :=
  Idx.last (Γ := seqFFNCtx seqLen dModel) (ss := []) (τ := [seqLen, dFF])

/-- Index of the saved sequence-shaped GELU activation output. -/
def seqFFNIdxHiddenAct {seqLen dModel dFF : Nat} :
    Idx (seqFFNCtx seqLen dModel ++ [[seqLen, dFF], [seqLen, dFF]]) [seqLen, dFF] :=
  Idx.last (Γ := seqFFNCtx seqLen dModel) (ss := [[seqLen, dFF]])
    (τ := [seqLen, dFF])

/-- Second sequence affine output. -/
def seqFFNIdxProjected {seqLen dModel dFF : Nat} :
    Idx (seqFFNCtx seqLen dModel ++ [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      [seqLen, dModel] :=
  Idx.last (Γ := seqFFNCtx seqLen dModel)
    (ss := [[seqLen, dFF], [seqLen, dFF]])
    (τ := [seqLen, dModel])

/--
Proof-carrying graph for the sequence-shaped residual FFN:

`X ↦ X + A₂(GELU(A₁ X + b₁)) + b₂`.

The affine maps and biases act on flattened sequence tensors and are held fixed during
differentiation. Correspondence with a concrete runtime or fused kernel requires a separate proof.
-/
def seqFFNResidualDGraph {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size [seqLen, dModel]) →L[ℝ]
        Vec (Shape.size [seqLen, dFF]))
    (b1 : Vec (Shape.size [seqLen, dFF]))
    (fc2 :
      Vec (Shape.size [seqLen, dFF]) →L[ℝ]
        Vec (Shape.size [seqLen, dModel]))
    (b2 : Vec (Shape.size [seqLen, dModel])) :
    DGraph (seqFFNCtx seqLen dModel) (ssSeqFFNResidual seqLen dModel dFF) :=
  let dg0 : DGraph (seqFFNCtx seqLen dModel) [] := DGraph.nil
  let dg1 : DGraph (seqFFNCtx seqLen dModel) [[seqLen, dFF]] :=
    DGraph.snoc (dg := dg0)
      (node := Tape.RealNode.affine
        (Γ := seqFFNCtx seqLen dModel)
        (sIn := [seqLen, dModel])
        (sOut := [seqLen, dFF])
        (seqFFNIdxX (seqLen := seqLen) (dModel := dModel) (ss := [])) fc1 b1)
      (hn := Tape.RealNode.affineFDeriv
        (Γ := seqFFNCtx seqLen dModel)
        (sIn := [seqLen, dModel])
        (sOut := [seqLen, dFF])
        (seqFFNIdxX (seqLen := seqLen) (dModel := dModel) (ss := [])) fc1 b1)
  let dg2 : DGraph (seqFFNCtx seqLen dModel)
      [[seqLen, dFF], [seqLen, dFF]] :=
    DGraph.snoc (dg := dg1)
      (node := Tape.LawfulNode.ofArm (Op.gelu (s := [seqLen, dFF])) rfl
        (Γ := seqFFNCtx seqLen dModel ++ [[seqLen, dFF]])
        (seqFFNIdxHiddenPre (seqLen := seqLen) (dModel := dModel) (dFF := dFF), ())
        (Arm.adj_of_cert₁ fun xs δ => Op.gelu_tanh_backward_eq_adjoint xs.1 δ))
      (hn := Tape.RealNode.FDerivCorrect.ofArm₁ (Γ := seqFFNCtx seqLen dModel ++ [[seqLen, dFF]])
        (s := [seqLen, dFF])
        (idx := seqFFNIdxHiddenPre (seqLen := seqLen) (dModel := dModel) (dFF := dFF))
        (cert := Op.gelu_tanh_backward_eq_adjoint))
  let dg3 : DGraph (seqFFNCtx seqLen dModel)
      [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]] :=
    DGraph.snoc (dg := dg2)
      (node := Tape.RealNode.affine
        (Γ := seqFFNCtx seqLen dModel ++ [[seqLen, dFF], [seqLen, dFF]])
        (sIn := [seqLen, dFF])
        (sOut := [seqLen, dModel])
        (seqFFNIdxHiddenAct (seqLen := seqLen) (dModel := dModel) (dFF := dFF)) fc2 b2)
      (hn := Tape.RealNode.affineFDeriv
        (Γ := seqFFNCtx seqLen dModel ++ [[seqLen, dFF], [seqLen, dFF]])
        (sIn := [seqLen, dFF])
        (sOut := [seqLen, dModel])
        (seqFFNIdxHiddenAct (seqLen := seqLen) (dModel := dModel) (dFF := dFF)) fc2 b2)
  DGraph.snoc (dg := dg3)
    (node := Tape.LawfulNode.ofArm (Op.add (s := [seqLen, dModel])) rfl
      (Γ := seqFFNCtx seqLen dModel ++ [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      (seqFFNIdxX (seqLen := seqLen) (dModel := dModel)
          (ss := [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]]),
        seqFFNIdxProjected (seqLen := seqLen) (dModel := dModel) (dFF := dFF), ())
      (Arm.adj_of_cert₂ fun xs δ => Op.add_prod_backward_eq_adjoint xs.1 xs.2.1 δ))
    (hn := Tape.RealNode.FDerivCorrect.ofArm₂
      (Γ := seqFFNCtx seqLen dModel ++ [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]])
      (s := [seqLen, dModel])
      (a := seqFFNIdxX (seqLen := seqLen) (dModel := dModel)
        (ss := [[seqLen, dFF], [seqLen, dFF], [seqLen, dModel]]))
      (b := seqFFNIdxProjected (seqLen := seqLen) (dModel := dModel) (dFF := dFF))
      (cert := Op.add_prod_backward_eq_adjoint))

end

end Transformer
end TorchLean
