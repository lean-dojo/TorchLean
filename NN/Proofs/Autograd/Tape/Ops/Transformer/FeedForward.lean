/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition
public import NN.Proofs.Autograd.Tape.Util.Idx

/-!
# Transformer Feed-Forward Sublayer VJP

This file proves the standard position-wise Transformer feed-forward sublayer, at the vector level:

`x ↦ x + W₂ GELU(W₁ x + b₁) + b₂`.

The theorem is intentionally about one token/vector. Batched sequence application is a map over
positions, and the full Transformer encoder block additionally composes this FFN residual with MHA
and LayerNorm. This file gives the clean proof component for the FFN half of that block.

References:

* Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
* Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Transformer

open Spec
open TapeNodes
open DGraph

noncomputable section

/-- Model-vector shape. -/
abbrev FFNModelShape (dModel : Nat) : Shape := .dim dModel .scalar

/-- Hidden feed-forward shape. -/
abbrev FFNHiddenShape (dFF : Nat) : Shape := .dim dFF .scalar

/-- Context for one position-wise FFN: just the input vector. -/
abbrev ΓFFN (dModel : Nat) : List Shape := [FFNModelShape dModel]

/-- Saved tensors: first affine, activation, second affine, residual output. -/
abbrev ssFFNResidual (dModel dFF : Nat) : List Shape :=
  [FFNHiddenShape dFF, FFNHiddenShape dFF, FFNModelShape dModel, FFNModelShape dModel]

/-- Input vector index. -/
def ffnIdxX {dModel : Nat} {ss : List Shape} :
    Idx (ΓFFN dModel ++ ss) (FFNModelShape dModel) :=
  ⟨⟨0, by simp [ΓFFN]⟩, by simp [ΓFFN, FFNModelShape]⟩

/-- Most recently appended tensor helper. -/
def idxLast {Γ : List Shape} {ss : List Shape} {τ : Shape} :
    Idx (Γ ++ ss ++ [τ]) τ :=
  _root_.Proofs.Autograd.Idx.last (Γ := Γ) (ss := ss) (τ := τ)

/-- First affine output index. -/
def ffnIdxHiddenPre {dModel dFF : Nat} :
    Idx (ΓFFN dModel ++ [FFNHiddenShape dFF]) (FFNHiddenShape dFF) :=
  idxLast (Γ := ΓFFN dModel) (ss := []) (τ := FFNHiddenShape dFF)

/-- GELU activation output index. -/
def ffnIdxHiddenAct {dModel dFF : Nat} :
    Idx (ΓFFN dModel ++ [FFNHiddenShape dFF, FFNHiddenShape dFF]) (FFNHiddenShape dFF) :=
  idxLast (Γ := ΓFFN dModel) (ss := [FFNHiddenShape dFF]) (τ := FFNHiddenShape dFF)

/-- Second affine output index. -/
def ffnIdxProjected {dModel dFF : Nat} :
    Idx (ΓFFN dModel ++ [FFNHiddenShape dFF, FFNHiddenShape dFF, FFNModelShape dModel])
      (FFNModelShape dModel) :=
  idxLast (Γ := ΓFFN dModel) (ss := [FFNHiddenShape dFF, FFNHiddenShape dFF])
    (τ := FFNModelShape dModel)

/--
Proof-carrying graph for a residual Transformer FFN sublayer.

The two affine maps are fixed `LinearSpec`s here, so the theorem covers the VJP with respect to the
input vector. Parameter-gradient theorems live at the trainable-parameter/runtime layer.
-/
def ffnResidualDGraph {dModel dFF : Nat}
    (fc1 : Spec.LinearSpec ℝ dModel dFF)
    (fc2 : Spec.LinearSpec ℝ dFF dModel) :
    DGraph (ΓFFN dModel) (ssFFNResidual dModel dFF) := by
  let dg0 : DGraph (ΓFFN dModel) [] := DGraph.nil
  let dg1 : DGraph (ΓFFN dModel) [FFNHiddenShape dFF] :=
    DGraph.snoc (dg := dg0)
      (node := linear
        (Γ := ΓFFN dModel) (inDim := dModel) (outDim := dFF)
        (ffnIdxX (dModel := dModel) (ss := [])) fc1)
      (hn := linearFderiv
        (Γ := ΓFFN dModel) (inDim := dModel) (outDim := dFF)
        (ffnIdxX (dModel := dModel) (ss := [])) fc1)
  let dg2 : DGraph (ΓFFN dModel) [FFNHiddenShape dFF, FFNHiddenShape dFF] :=
    DGraph.snoc (dg := dg1)
      (node := gelu
        (Γ := ΓFFN dModel ++ [FFNHiddenShape dFF]) (s := FFNHiddenShape dFF)
        (ffnIdxHiddenPre (dModel := dModel) (dFF := dFF)))
      (hn := geluFderiv
        (Γ := ΓFFN dModel ++ [FFNHiddenShape dFF]) (s := FFNHiddenShape dFF)
        (ffnIdxHiddenPre (dModel := dModel) (dFF := dFF)))
  let dg3 :
      DGraph (ΓFFN dModel) [FFNHiddenShape dFF, FFNHiddenShape dFF, FFNModelShape dModel] :=
    DGraph.snoc (dg := dg2)
      (node := linear
        (Γ := ΓFFN dModel ++ [FFNHiddenShape dFF, FFNHiddenShape dFF])
        (inDim := dFF) (outDim := dModel)
        (ffnIdxHiddenAct (dModel := dModel) (dFF := dFF)) fc2)
      (hn := linearFderiv
        (Γ := ΓFFN dModel ++ [FFNHiddenShape dFF, FFNHiddenShape dFF])
        (inDim := dFF) (outDim := dModel)
        (ffnIdxHiddenAct (dModel := dModel) (dFF := dFF)) fc2)
  exact
    DGraph.snoc (dg := dg3)
      (node := add
        (Γ := ΓFFN dModel ++ [FFNHiddenShape dFF, FFNHiddenShape dFF, FFNModelShape dModel])
        (s := FFNModelShape dModel)
        (ffnIdxX (dModel := dModel)
          (ss := [FFNHiddenShape dFF, FFNHiddenShape dFF, FFNModelShape dModel]))
        (ffnIdxProjected (dModel := dModel) (dFF := dFF)))
      (hn := addFderiv
        (Γ := ΓFFN dModel ++ [FFNHiddenShape dFF, FFNHiddenShape dFF, FFNModelShape dModel])
        (s := FFNModelShape dModel)
        (ffnIdxX (dModel := dModel)
          (ss := [FFNHiddenShape dFF, FFNHiddenShape dFF, FFNModelShape dModel]))
        (ffnIdxProjected (dModel := dModel) (dFF := dFF)))

/--
End-to-end VJP theorem for the residual Transformer feed-forward sublayer.
-/
theorem ffnResidual_backpropVec_eq_adjoint_fderiv
    {dModel dFF : Nat}
    (fc1 : Spec.LinearSpec ℝ dModel dFF)
    (fc2 : Spec.LinearSpec ℝ dFF dModel)
    (xV : CtxVec (ΓFFN dModel))
    (seedV : CtxVec (ΓFFN dModel ++ ssFFNResidual dModel dFF)) :
    Graph.backpropVec
        (Γ := ΓFFN dModel)
        (ss := ssFFNResidual dModel dFF)
        (ffnResidualDGraph (dModel := dModel) (dFF := dFF) fc1 fc2).g xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓFFN dModel)
          (ss := ssFFNResidual dModel dFF)
          (ffnResidualDGraph (dModel := dModel) (dFF := dFF) fc1 fc2).g)
        xV).adjoint seedV :=
  DGraph.backpropVec_eq_adjoint_fderiv
    (dg := ffnResidualDGraph (dModel := dModel) (dFF := dFF) fc1 fc2) xV seedV

/-!
## Sequence-shaped FFN residual

The runtime Transformer applies the same FFN to every token in a `(seqLen × dModel)` tensor. For the
model-level proof interface below, we package that operation as two fixed affine maps over the
flattened sequence tensor. A concrete shared-weight implementation instantiates these maps with the
usual block-diagonal/time-distributed linear operator; the VJP theorem itself only needs the affine
maps and the smooth GELU primitive.
-/

/-- Sequence-shaped model stream. -/
abbrev SeqFFNModelShape (seqLen dModel : Nat) : Shape :=
  .dim seqLen (.dim dModel .scalar)

/-- Sequence-shaped FFN hidden stream. -/
abbrev SeqFFNHiddenShape (seqLen dFF : Nat) : Shape :=
  .dim seqLen (.dim dFF .scalar)

/-- Context for a sequence-level FFN residual block: just the sequence stream. -/
abbrev ΓSeqFFN (seqLen dModel : Nat) : List Shape :=
  [SeqFFNModelShape seqLen dModel]

/-- Saved tensors for the sequence-level residual FFN. -/
abbrev ssSeqFFNResidual (seqLen dModel dFF : Nat) : List Shape :=
  [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
    SeqFFNModelShape seqLen dModel, SeqFFNModelShape seqLen dModel]

/-- Sequence input index, weakened through saved tensors. -/
def seqFfnIdxX {seqLen dModel : Nat} {ss : List Shape} :
    Idx (ΓSeqFFN seqLen dModel ++ ss) (SeqFFNModelShape seqLen dModel) :=
  ⟨⟨0, by simp [ΓSeqFFN]⟩, by simp [ΓSeqFFN, SeqFFNModelShape]⟩

/-- First sequence affine output. -/
def seqFfnIdxHiddenPre {seqLen dModel dFF : Nat} :
    Idx (ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF])
      (SeqFFNHiddenShape seqLen dFF) :=
  idxLast (Γ := ΓSeqFFN seqLen dModel) (ss := []) (τ := SeqFFNHiddenShape seqLen dFF)

/-- GELU activation output. -/
def seqFfnIdxHiddenAct {seqLen dModel dFF : Nat} :
    Idx (ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF,
      SeqFFNHiddenShape seqLen dFF]) (SeqFFNHiddenShape seqLen dFF) :=
  idxLast (Γ := ΓSeqFFN seqLen dModel) (ss := [SeqFFNHiddenShape seqLen dFF])
    (τ := SeqFFNHiddenShape seqLen dFF)

/-- Second sequence affine output. -/
def seqFfnIdxProjected {seqLen dModel dFF : Nat} :
    Idx (ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF,
      SeqFFNHiddenShape seqLen dFF, SeqFFNModelShape seqLen dModel])
      (SeqFFNModelShape seqLen dModel) :=
  idxLast (Γ := ΓSeqFFN seqLen dModel)
    (ss := [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF])
    (τ := SeqFFNModelShape seqLen dModel)

/--
Proof-carrying graph for the sequence-shaped residual FFN:

`X ↦ X + A₂(GELU(A₁ X + b₁)) + b₂`.

The affine maps are supplied explicitly over flattened sequence tensors. This keeps the theorem
usable for shared-weight position-wise FFNs, fused FFN kernels, and future compiler-generated
linearizations, as long as they expose the same affine map.
-/
def seqFfnResidualDGraph {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel))) :
    DGraph (ΓSeqFFN seqLen dModel) (ssSeqFFNResidual seqLen dModel dFF) := by
  let dg0 : DGraph (ΓSeqFFN seqLen dModel) [] := DGraph.nil
  let dg1 : DGraph (ΓSeqFFN seqLen dModel) [SeqFFNHiddenShape seqLen dFF] :=
    DGraph.snoc (dg := dg0)
      (node := TapeNodes.affine
        (Γ := ΓSeqFFN seqLen dModel)
        (sIn := SeqFFNModelShape seqLen dModel)
        (sOut := SeqFFNHiddenShape seqLen dFF)
        (seqFfnIdxX (seqLen := seqLen) (dModel := dModel) (ss := [])) fc1 b1)
      (hn := TapeNodes.affineFderiv
        (Γ := ΓSeqFFN seqLen dModel)
        (sIn := SeqFFNModelShape seqLen dModel)
        (sOut := SeqFFNHiddenShape seqLen dFF)
        (seqFfnIdxX (seqLen := seqLen) (dModel := dModel) (ss := [])) fc1 b1)
  let dg2 : DGraph (ΓSeqFFN seqLen dModel)
      [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF] :=
    DGraph.snoc (dg := dg1)
      (node := gelu
        (Γ := ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF])
        (s := SeqFFNHiddenShape seqLen dFF)
        (seqFfnIdxHiddenPre (seqLen := seqLen) (dModel := dModel) (dFF := dFF)))
      (hn := geluFderiv
        (Γ := ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF])
        (s := SeqFFNHiddenShape seqLen dFF)
        (seqFfnIdxHiddenPre (seqLen := seqLen) (dModel := dModel) (dFF := dFF)))
  let dg3 : DGraph (ΓSeqFFN seqLen dModel)
      [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
        SeqFFNModelShape seqLen dModel] :=
    DGraph.snoc (dg := dg2)
      (node := TapeNodes.affine
        (Γ := ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF,
          SeqFFNHiddenShape seqLen dFF])
        (sIn := SeqFFNHiddenShape seqLen dFF)
        (sOut := SeqFFNModelShape seqLen dModel)
        (seqFfnIdxHiddenAct (seqLen := seqLen) (dModel := dModel) (dFF := dFF)) fc2 b2)
      (hn := TapeNodes.affineFderiv
        (Γ := ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF,
          SeqFFNHiddenShape seqLen dFF])
        (sIn := SeqFFNHiddenShape seqLen dFF)
        (sOut := SeqFFNModelShape seqLen dModel)
        (seqFfnIdxHiddenAct (seqLen := seqLen) (dModel := dModel) (dFF := dFF)) fc2 b2)
  exact
    DGraph.snoc (dg := dg3)
      (node := add
        (Γ := ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF,
          SeqFFNHiddenShape seqLen dFF, SeqFFNModelShape seqLen dModel])
        (s := SeqFFNModelShape seqLen dModel)
        (seqFfnIdxX (seqLen := seqLen) (dModel := dModel)
          (ss := [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
            SeqFFNModelShape seqLen dModel]))
        (seqFfnIdxProjected (seqLen := seqLen) (dModel := dModel) (dFF := dFF)))
      (hn := addFderiv
        (Γ := ΓSeqFFN seqLen dModel ++ [SeqFFNHiddenShape seqLen dFF,
          SeqFFNHiddenShape seqLen dFF, SeqFFNModelShape seqLen dModel])
        (s := SeqFFNModelShape seqLen dModel)
        (seqFfnIdxX (seqLen := seqLen) (dModel := dModel)
          (ss := [SeqFFNHiddenShape seqLen dFF, SeqFFNHiddenShape seqLen dFF,
            SeqFFNModelShape seqLen dModel]))
        (seqFfnIdxProjected (seqLen := seqLen) (dModel := dModel) (dFF := dFF)))

/-- End-to-end VJP theorem for the sequence-shaped residual FFN. -/
theorem seqFfnResidual_backpropVec_eq_adjoint_fderiv
    {seqLen dModel dFF : Nat}
    (fc1 :
      Vec (Shape.size (SeqFFNModelShape seqLen dModel)) →L[ℝ]
        Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (b1 : Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)))
    (fc2 :
      Vec (Shape.size (SeqFFNHiddenShape seqLen dFF)) →L[ℝ]
        Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (b2 : Vec (Shape.size (SeqFFNModelShape seqLen dModel)))
    (xV : CtxVec (ΓSeqFFN seqLen dModel))
    (seedV : CtxVec (ΓSeqFFN seqLen dModel ++ ssSeqFFNResidual seqLen dModel dFF)) :
    Graph.backpropVec
        (Γ := ΓSeqFFN seqLen dModel)
        (ss := ssSeqFFNResidual seqLen dModel dFF)
        (seqFfnResidualDGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
          fc1 b1 fc2 b2).g xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓSeqFFN seqLen dModel)
          (ss := ssSeqFFNResidual seqLen dModel dFF)
          (seqFfnResidualDGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
            fc1 b1 fc2 b2).g)
        xV).adjoint seedV :=
  DGraph.backpropVec_eq_adjoint_fderiv
    (dg := seqFfnResidualDGraph (seqLen := seqLen) (dModel := dModel) (dFF := dFF)
      fc1 b1 fc2 b2) xV seedV

end

end Transformer
end Autograd
end Proofs
