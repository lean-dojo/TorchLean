/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Attention.MultiHead

/-!
# Residual multi-head attention

`mhaResidualDGraph` adds a residual connection to the `MultiHeadAttention` graph:

`x ↦ x + MHA(x)`.

Post-norm LayerNorm then composes with it:

`LayerNorm(x + MultiHeadSelfAttention(x))`.

The graph's reverse pass is the adjoint of its Fréchet derivative. We add the residual with one
`Op.add` node, certified by `Op.add_prod_backward_eq_adjoint`, after the attention graph of
`TorchLean.NN.Attention.MultiHead`. That graph splits heads by `splitHeadsEquiv`, the coordinate
map of `MultiHeadAttention.splitHeads`, and takes the score scale `c` as an argument, where
eq. (1) of Vaswani et al. uses `1/√d_k`. Equality of the graph with `MultiHeadAttention.forward`
or with the runtime Transformer isn't proved.

References:
- Vaswani et al. (2017), "Attention Is All You Need", NIPS 2017, arXiv:1706.03762, eq. (1) and
  Section 3.2.2.
- PyTorch `torch.nn.MultiheadAttention` (PyTorch 2.8 docs):
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.MultiheadAttention.html
- PyTorch `torch.nn.TransformerEncoderLayer` (PyTorch 2.8 docs):
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.TransformerEncoderLayer.html
-/

@[expose] public section

namespace TorchLean
namespace Transformer


noncomputable section

open Tape (DGraph)

/-- Saved tensors: the MHA intermediates followed by the residual-add output. -/
abbrev ssMHAResidual (n dModel numHeads headDim : Nat) : List Shape :=
  MultiHeadAttention.ss n dModel numHeads headDim ++ [[n, dModel]]

/-- Original sequence input `x`, weakened into the context after the MHA intermediates. -/
def residualIdxX {n dModel numHeads headDim : Nat} :
    Idx (MultiHeadAttention.ctx n dModel numHeads headDim ++
        MultiHeadAttention.ss n dModel numHeads headDim)
      [n, dModel] :=
  MultiHeadAttention.idxX (n := n) (dModel := dModel)
    (numHeads := numHeads) (headDim := headDim)
    (ss := MultiHeadAttention.ss n dModel numHeads headDim)

/--
The final output of `dGraph`, the projected attention result.

The literal index `18` points past the context `[x, Wq, Wk, Wv, Wo]` to the last saved value of
`MultiHeadAttention.ss`, the attention output, which has the shape of the input sequence.
-/
def residualIdxAttnOut {n dModel numHeads headDim : Nat} :
    Idx (MultiHeadAttention.ctx n dModel numHeads headDim ++
        MultiHeadAttention.ss n dModel numHeads headDim)
      [n, dModel] :=
  ⟨⟨18, by simp [MultiHeadAttention.ctx, MultiHeadAttention.ss]⟩,
    by simp [MultiHeadAttention.ctx, MultiHeadAttention.ss]⟩

/--
Proof-carrying graph for `x + MHA(x)`, with heads split as `MultiHeadAttention.splitHeads` does.

Context layout is inherited from MHA:
`[x, Wq, Wk, Wv, Wo]`.
-/
def mhaResidualDGraph {n dModel numHeads headDim : Nat} (c : ℝ) :
    DGraph (MultiHeadAttention.ctx n dModel numHeads headDim)
      (ssMHAResidual n dModel numHeads headDim) :=
  DGraph.snoc
    (dg := MultiHeadAttention.dGraph (n := n) (dModel := dModel) (numHeads := numHeads)
      (headDim := headDim) c)
    (node := Tape.LawfulNode.ofArm (Op.add (s := [n, dModel])) rfl
      (Γ := MultiHeadAttention.ctx n dModel numHeads headDim ++
        MultiHeadAttention.ss n dModel numHeads headDim)
      (residualIdxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim),
        residualIdxAttnOut (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim),
        ())
      (Arm.adj_of_cert₂ fun xs δ => Op.add_prod_backward_eq_adjoint xs.1 xs.2.1 δ))
    (hn := Tape.RealNode.FDerivCorrect.ofArm₂
      (Γ := MultiHeadAttention.ctx n dModel numHeads headDim ++
        MultiHeadAttention.ss n dModel numHeads headDim)
      (s := [n, dModel])
      (a := residualIdxX (n := n) (dModel := dModel) (numHeads := numHeads) (headDim := headDim))
      (b := residualIdxAttnOut (n := n) (dModel := dModel) (numHeads := numHeads)
        (headDim := headDim))
      (cert := Op.add_prod_backward_eq_adjoint))

end

end Transformer
end TorchLean
