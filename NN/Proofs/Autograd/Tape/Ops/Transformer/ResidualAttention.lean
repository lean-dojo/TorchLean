/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Attention.MultiHeadSelfAttention

/-!
# Residual direct-reshape attention

This file appends a residual addition to the graph in `DirectReshapeAttention`:

`x ↦ x + MHA(x)`.

It can then be composed with post-norm LayerNorm:

`LayerNorm(x + MultiHeadSelfAttention(x))`.

The resulting reverse pass is the adjoint of this graph's Fréchet derivative. It retains the
direct-reshape head split, which differs from `Spec.splitHeadsSpec` and the executable attention
layout. Adding a residual does not establish correspondence with the public MHA or runtime
Transformer; that requires a separate forward-layout bridge.

References:
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- PyTorch `torch.nn.MultiheadAttention`:
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- PyTorch `torch.nn.TransformerEncoderLayer`:
  https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Transformer

open Spec TorchLean
open TapeNodes
open DGraph

noncomputable section

/-- Intermediate list for MHA followed by one residual-add output. -/
abbrev ssMHAResidual (n dModel numHeads headDim : Nat) : List Shape :=
  DirectReshapeAttention.ssMHA n dModel numHeads headDim ++ [DirectReshapeAttention.XShape n dModel]

/-- Original sequence input `x`, weakened into the context after the MHA intermediates. -/
def residualIdxX {n dModel numHeads headDim : Nat} :
    Idx (DirectReshapeAttention.ΓMHA n dModel numHeads headDim ++
        DirectReshapeAttention.ssMHA n dModel numHeads headDim)
      (DirectReshapeAttention.XShape n dModel) :=
  DirectReshapeAttention.idxX (n := n) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim)
    (ss := DirectReshapeAttention.ssMHA n dModel numHeads headDim)

/--
The final output of `mhaDGraph`, i.e. the projected attention result.

The literal index is intentional: `DirectReshapeAttention.ssMHA` has 14 saved values,
and the final entry is the attention output with the same shape as the input sequence.
-/
def residualIdxAttnOut {n dModel numHeads headDim : Nat} :
    Idx (DirectReshapeAttention.ΓMHA n dModel numHeads headDim ++
        DirectReshapeAttention.ssMHA n dModel numHeads headDim)
      (DirectReshapeAttention.XShape n dModel) :=
  ⟨⟨18, by simp [DirectReshapeAttention.ΓMHA, DirectReshapeAttention.ssMHA]⟩,
    by simp [DirectReshapeAttention.ΓMHA, DirectReshapeAttention.ssMHA]⟩

/--
Proof-carrying graph for `x + MHA(x)`, using the direct-reshape attention variant.

Context layout is inherited from MHA:
`[x, Wq, Wk, Wv, Wo]`.
-/
def mhaResidualDGraph {n dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (DirectReshapeAttention.ΓMHA n dModel numHeads headDim)
      (ssMHAResidual n dModel numHeads headDim) :=
  DGraph.snoc
    (dg := DirectReshapeAttention.mhaDGraph (n := n) (dModel := dModel) (numHeads := numHeads)
      (headDim := headDim) c)
    (node := add
      (Γ := DirectReshapeAttention.ΓMHA n dModel numHeads headDim ++
        DirectReshapeAttention.ssMHA n dModel numHeads headDim)
      (s := DirectReshapeAttention.XShape n dModel)
      (a := residualIdxX (n := n) (dModel := dModel) (numHeads := numHeads)
        (headDim := headDim))
      (b := residualIdxAttnOut (n := n) (dModel := dModel) (numHeads := numHeads)
        (headDim := headDim)))
    (hn := addFderiv
      (Γ := DirectReshapeAttention.ΓMHA n dModel numHeads headDim ++
        DirectReshapeAttention.ssMHA n dModel numHeads headDim)
      (s := DirectReshapeAttention.XShape n dModel)
      (a := residualIdxX (n := n) (dModel := dModel) (numHeads := numHeads)
        (headDim := headDim))
      (b := residualIdxAttnOut (n := n) (dModel := dModel) (numHeads := numHeads)
        (headDim := headDim)))

/--
End-to-end VJP theorem for the residual-attention sublayer `x + MHA(x)`.

This graph is composed with LayerNorm in the post-norm proof modules. LayerNorm itself
has its own current-spec VJP theorem in `NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm`; the composed
post-norm attention sublayer and the two-sublayer post-norm bridge are packaged in
`NN.Proofs.Autograd.Tape.Ops.Transformer.PostNorm`.
-/
theorem mhaResidual_backpropVec_eq_adjoint_fderiv
    {n dModel numHeads headDim : Nat} (c : ℝ)
    (xV : CtxVec (DirectReshapeAttention.ΓMHA n dModel numHeads headDim))
    (seedV : CtxVec (DirectReshapeAttention.ΓMHA n dModel numHeads headDim ++
      ssMHAResidual n dModel numHeads headDim)) :
    Graph.backpropVec
        (Γ := DirectReshapeAttention.ΓMHA n dModel numHeads headDim)
        (ss := ssMHAResidual n dModel numHeads headDim)
        (mhaResidualDGraph (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          c).g xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := DirectReshapeAttention.ΓMHA n dModel numHeads headDim)
          (ss := ssMHAResidual n dModel numHeads headDim)
          (mhaResidualDGraph (n := n) (dModel := dModel) (numHeads := numHeads)
            (headDim := headDim) c).g)
        xV).adjoint seedV :=
  DGraph.backpropVec_eq_adjoint_fderiv
    (dg := mhaResidualDGraph (n := n) (dModel := dModel) (numHeads := numHeads)
      (headDim := headDim) c) xV seedV

end

end Transformer
end Autograd
end Proofs
