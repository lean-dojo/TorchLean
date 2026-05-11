/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Attention.MultiHeadSelfAttention

/-!
# Residual Attention Blocks

This file proves the next composition step after full multi-head self-attention:

`x ↦ x + MHA(x)`.

That residual add is the first half of a post-norm Transformer encoder sublayer:

`LayerNorm(x + MultiHeadSelfAttention(x))`.

The existing MHA theorem already proves that the attention graph's reverse pass is the adjoint of
the Fréchet derivative. Here we append the residual add as one more proved tape node, giving a
reusable graph theorem for the residual stream that is passed to post-norm LayerNorm in the runtime
Transformer blocks.

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

open Spec
open TapeNodes
open DGraph

noncomputable section

/-- Intermediate list for MHA followed by one residual-add output. -/
abbrev ssMHAResidual (n dModel numHeads headDim : Nat) : List Shape :=
  MultiHeadAttention.ssMHA n dModel numHeads headDim ++ [MultiHeadAttention.XShape n dModel]

/-- Original sequence input `x`, weakened into the context after the MHA intermediates. -/
def residualIdxX {n dModel numHeads headDim : Nat} :
    Idx (MultiHeadAttention.ΓMHA n dModel numHeads headDim ++ MultiHeadAttention.ssMHA n dModel numHeads headDim)
      (MultiHeadAttention.XShape n dModel) :=
  MultiHeadAttention.idxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
    (ss := MultiHeadAttention.ssMHA n dModel numHeads headDim)

/--
The final output of `mhaDGraph`, i.e. the projected attention result.

The literal index is intentional: `MultiHeadAttention.ssMHA` is a fixed 14-entry saved-tensor list, and the final
entry is the attention output with the same shape as the input sequence.
-/
def residualIdxAttnOut {n dModel numHeads headDim : Nat} :
    Idx (MultiHeadAttention.ΓMHA n dModel numHeads headDim ++ MultiHeadAttention.ssMHA n dModel numHeads headDim)
      (MultiHeadAttention.XShape n dModel) :=
  ⟨⟨18, by simp [MultiHeadAttention.ΓMHA, MultiHeadAttention.ssMHA]⟩,
    by simp [MultiHeadAttention.ΓMHA, MultiHeadAttention.ssMHA]⟩

/--
Proof-carrying graph for `x + MHA(x)`.

Context layout is inherited from MHA:
`[x, Wq, Wk, Wv, Wo]`.
-/
def mhaResidualDGraph {n dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (MultiHeadAttention.ΓMHA n dModel numHeads headDim) (ssMHAResidual n dModel numHeads headDim) := by
  let dgMha := MultiHeadAttention.mhaDGraph (n := n) (dModel := dModel) (numHeads := numHeads)
    (headDim := headDim) c
  exact
    DGraph.snoc (dg := dgMha)
      (node := add
        (Γ := MultiHeadAttention.ΓMHA n dModel numHeads headDim ++ MultiHeadAttention.ssMHA n dModel numHeads headDim)
        (s := MultiHeadAttention.XShape n dModel)
        (a := residualIdxX (n := n) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim))
        (b := residualIdxAttnOut (n := n) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim)))
      (hn := addFderiv
        (Γ := MultiHeadAttention.ΓMHA n dModel numHeads headDim ++ MultiHeadAttention.ssMHA n dModel numHeads headDim)
        (s := MultiHeadAttention.XShape n dModel)
        (a := residualIdxX (n := n) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim))
        (b := residualIdxAttnOut (n := n) (dModel := dModel) (numHeads := numHeads)
          (headDim := headDim)))

/--
End-to-end VJP theorem for the residual-attention sublayer `x + MHA(x)`.

This is the proved residual-stream component used by post-norm Transformer blocks. LayerNorm itself
has its own current-spec VJP theorem in `NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm`; the composed
post-norm attention sublayer and the two-sublayer post-norm bridge are packaged in
`NN.Proofs.Autograd.Tape.Ops.Transformer.PostNorm`.
-/
theorem mhaResidual_backpropVec_eq_adjoint_fderiv
    {n dModel numHeads headDim : Nat} (c : ℝ)
    (xV : CtxVec (MultiHeadAttention.ΓMHA n dModel numHeads headDim))
    (seedV : CtxVec (MultiHeadAttention.ΓMHA n dModel numHeads headDim ++ ssMHAResidual n dModel numHeads
      headDim)) :
    Graph.backpropVec
        (Γ := MultiHeadAttention.ΓMHA n dModel numHeads headDim)
        (ss := ssMHAResidual n dModel numHeads headDim)
        (mhaResidualDGraph (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim)
          c).g xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := MultiHeadAttention.ΓMHA n dModel numHeads headDim)
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
