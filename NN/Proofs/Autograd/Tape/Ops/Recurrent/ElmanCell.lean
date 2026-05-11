/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition
public import NN.Proofs.Autograd.Tape.Util.Idx

/-!
# Elman RNN Cell VJP

This file proves the core differentiable cell used by a vanilla tanh RNN:

`h' = tanh(W [x; h] + b)`.

The theorem is deliberately cell-level. Runtime sequence layers unroll this cell over time and
scatter the hidden states into an output sequence; the full BPTT theorem is the induction over that
unroll plus the existing `gather`/`scatter` adjoint facts. We prove the cell first because it is the
right reusable grain size: vector concatenation, affine maps, and smooth elementwise `tanh`.

References:

* Elman, "Finding Structure in Time", Cognitive Science 1990.
* PyTorch `torch.nn.RNN`:
  https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Recurrent

open Spec
open TapeNodes
open DGraph

universe u

noncomputable section

/-- Input vector shape for one RNN step. -/
abbrev XShape (inputSize : Nat) : Shape := .dim inputSize .scalar

/-- Hidden vector shape for one RNN step. -/
abbrev HShape (hiddenSize : Nat) : Shape := .dim hiddenSize .scalar

/-- Context for a one-step Elman cell: current input and previous hidden state. -/
abbrev ΓElman (inputSize hiddenSize : Nat) : List Shape :=
  [XShape inputSize, HShape hiddenSize]

/-- Saved tensors: concatenated `[x; h]`, affine preactivation, and next hidden state. -/
abbrev ssElmanCell (inputSize hiddenSize : Nat) : List Shape :=
  [.dim (inputSize + hiddenSize) .scalar, HShape hiddenSize, HShape hiddenSize]

/-- Current input index. -/
def idxInput {inputSize hiddenSize : Nat} {ss : List Shape} :
    Idx (ΓElman inputSize hiddenSize ++ ss) (XShape inputSize) :=
  ⟨⟨0, by simp [ΓElman]⟩, by simp [ΓElman, XShape]⟩

/-- Previous hidden-state index. -/
def idxHidden {inputSize hiddenSize : Nat} {ss : List Shape} :
    Idx (ΓElman inputSize hiddenSize ++ ss) (HShape hiddenSize) :=
  ⟨⟨1, by simp [ΓElman]⟩, by simp [ΓElman, HShape]⟩

/-- Most recently appended tensor helper. -/
def idxLast {Γ : List Shape} {ss : List Shape} {τ : Shape} :
    Idx (Γ ++ ss ++ [τ]) τ :=
  _root_.Proofs.Autograd.Idx.last (Γ := Γ) (ss := ss) (τ := τ)

/-- Index of the concatenated `[x; h]` vector. -/
def idxConcat {inputSize hiddenSize : Nat} :
    Idx (ΓElman inputSize hiddenSize ++ [.dim (inputSize + hiddenSize) .scalar])
      (.dim (inputSize + hiddenSize) .scalar) :=
  idxLast (Γ := ΓElman inputSize hiddenSize) (ss := [])
    (τ := .dim (inputSize + hiddenSize) .scalar)

/-- Index of the affine preactivation. -/
def idxPre {inputSize hiddenSize : Nat} :
    Idx (ΓElman inputSize hiddenSize ++
      [.dim (inputSize + hiddenSize) .scalar, HShape hiddenSize]) (HShape hiddenSize) :=
  idxLast (Γ := ΓElman inputSize hiddenSize)
    (ss := [.dim (inputSize + hiddenSize) .scalar]) (τ := HShape hiddenSize)

/--
Proof-carrying graph for one Elman RNN cell.

The affine map is represented by a fixed `LinearSpec`, so this theorem covers the VJP with respect
to the cell inputs `(x, h)`. Parameter-gradient theorems are a separate layer over the trainable
runtime parameter list.
-/
def elmanCellDGraph {inputSize hiddenSize : Nat}
    (cell : Spec.LinearSpec ℝ (inputSize + hiddenSize) hiddenSize) :
    DGraph (ΓElman inputSize hiddenSize) (ssElmanCell inputSize hiddenSize) := by
  let dg0 : DGraph (ΓElman inputSize hiddenSize) [] := DGraph.nil
  let dg1 :
      DGraph (ΓElman inputSize hiddenSize) [.dim (inputSize + hiddenSize) .scalar] :=
    DGraph.snoc (dg := dg0)
      (node := concatVectors
        (Γ := ΓElman inputSize hiddenSize) (n := inputSize) (m := hiddenSize)
        (idxInput (inputSize := inputSize) (hiddenSize := hiddenSize) (ss := []))
        (idxHidden (inputSize := inputSize) (hiddenSize := hiddenSize) (ss := [])))
      (hn := concatVectorsFderiv
        (Γ := ΓElman inputSize hiddenSize) (n := inputSize) (m := hiddenSize)
        (idxInput (inputSize := inputSize) (hiddenSize := hiddenSize) (ss := []))
        (idxHidden (inputSize := inputSize) (hiddenSize := hiddenSize) (ss := [])))
  let dg2 :
      DGraph (ΓElman inputSize hiddenSize)
        [.dim (inputSize + hiddenSize) .scalar, HShape hiddenSize] :=
    DGraph.snoc (dg := dg1)
      (node := linear
        (Γ := ΓElman inputSize hiddenSize ++ [.dim (inputSize + hiddenSize) .scalar])
        (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
        (idxConcat (inputSize := inputSize) (hiddenSize := hiddenSize)) cell)
      (hn := linearFderiv
        (Γ := ΓElman inputSize hiddenSize ++ [.dim (inputSize + hiddenSize) .scalar])
        (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
        (idxConcat (inputSize := inputSize) (hiddenSize := hiddenSize)) cell)
  exact
    DGraph.snoc (dg := dg2)
      (node := tanh
        (Γ := ΓElman inputSize hiddenSize ++
          [.dim (inputSize + hiddenSize) .scalar, HShape hiddenSize])
        (s := HShape hiddenSize)
        (idxPre (inputSize := inputSize) (hiddenSize := hiddenSize)))
      (hn := tanhFderiv
        (Γ := ΓElman inputSize hiddenSize ++
          [.dim (inputSize + hiddenSize) .scalar, HShape hiddenSize])
        (s := HShape hiddenSize)
        (idxPre (inputSize := inputSize) (hiddenSize := hiddenSize)))

/--
End-to-end VJP theorem for one vanilla RNN cell.

This is the recurrent analogue of the attention block theorems: the graph-level reverse pass equals
the adjoint of the Fréchet derivative of the cell evaluation function.
-/
theorem elmanCell_backpropVec_eq_adjoint_fderiv
    {inputSize hiddenSize : Nat}
    (cell : Spec.LinearSpec ℝ (inputSize + hiddenSize) hiddenSize)
    (xV : CtxVec (ΓElman inputSize hiddenSize))
    (seedV : CtxVec (ΓElman inputSize hiddenSize ++ ssElmanCell inputSize hiddenSize)) :
    Graph.backpropVec
        (Γ := ΓElman inputSize hiddenSize)
        (ss := ssElmanCell inputSize hiddenSize)
        (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g xV seedV
      =
    (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓElman inputSize hiddenSize)
          (ss := ssElmanCell inputSize hiddenSize)
          (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g)
        xV).adjoint seedV :=
  DGraph.backpropVec_eq_adjoint_fderiv
    (dg := elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell) xV seedV

/--
Forward evaluation of one Elman cell is differentiable at every input context.

This is the recurrent analogue of the Transformer sublayer calculus bridges: it exposes the cell as
a differentiable map that can be composed repeatedly when proving BPTT for an unrolled RNN.
-/
theorem elmanCell_eval_hasFDerivAt
    {inputSize hiddenSize : Nat}
    (cell : Spec.LinearSpec ℝ (inputSize + hiddenSize) hiddenSize)
    (xV : CtxVec (ΓElman inputSize hiddenSize)) :
    HasFDerivAt
      (Graph.evalVec
        (Γ := ΓElman inputSize hiddenSize)
        (ss := ssElmanCell inputSize hiddenSize)
        (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g)
      (fderiv ℝ
        (Graph.evalVec
          (Γ := ΓElman inputSize hiddenSize)
          (ss := ssElmanCell inputSize hiddenSize)
          (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g)
        xV)
      xV := by
  rcases Graph.hasFDerivAt_evalVec_and_jvp
      (Γ := ΓElman inputSize hiddenSize)
      (ss := ssElmanCell inputSize hiddenSize)
      (g := (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g)
      (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).hg xV with
    ⟨D, hD, _hJ⟩
  rw [hD.fderiv]
  exact hD

/--
Two-step recurrent composition bridge.

Suppose `firstCtx` builds the first cell context from some outer state `E`, and `secondCtx` builds
the next cell context from the evaluated first-cell graph (for example by selecting the next input
and the hidden state produced by the first step). If both context builders are differentiable, then
the two-cell unroll is differentiable.

The theorem is intentionally abstract over `secondCtx`: different sequence layouts store inputs,
hidden states, and caches differently, but every vanilla RNN BPTT proof follows this same chain-rule
shape.
-/
theorem elmanTwoStep_hasFDerivAt
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {inputSize hiddenSize : Nat}
    (cell : Spec.LinearSpec ℝ (inputSize + hiddenSize) hiddenSize)
    (firstCtx : E → CtxVec (ΓElman inputSize hiddenSize))
    (DfirstCtx : E →L[ℝ] CtxVec (ΓElman inputSize hiddenSize))
    (secondCtx :
      CtxVec (ΓElman inputSize hiddenSize ++ ssElmanCell inputSize hiddenSize) →
        CtxVec (ΓElman inputSize hiddenSize))
    (DsecondCtx :
      CtxVec (ΓElman inputSize hiddenSize ++ ssElmanCell inputSize hiddenSize) →L[ℝ]
        CtxVec (ΓElman inputSize hiddenSize))
    (x : E)
    (hFirstCtx : HasFDerivAt firstCtx DfirstCtx x)
    (hSecondCtx :
      HasFDerivAt secondCtx DsecondCtx
        (Graph.evalVec
          (Γ := ΓElman inputSize hiddenSize)
          (ss := ssElmanCell inputSize hiddenSize)
          (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g
          (firstCtx x))) :
    HasFDerivAt
      (fun z : E =>
        Graph.evalVec
          (Γ := ΓElman inputSize hiddenSize)
          (ss := ssElmanCell inputSize hiddenSize)
          (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g
          (secondCtx
            (Graph.evalVec
              (Γ := ΓElman inputSize hiddenSize)
              (ss := ssElmanCell inputSize hiddenSize)
              (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g
              (firstCtx z))))
      ((fderiv ℝ
          (Graph.evalVec
            (Γ := ΓElman inputSize hiddenSize)
            (ss := ssElmanCell inputSize hiddenSize)
            (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g)
          (secondCtx
            (Graph.evalVec
              (Γ := ΓElman inputSize hiddenSize)
              (ss := ssElmanCell inputSize hiddenSize)
              (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g
              (firstCtx x)))).comp
        (DsecondCtx.comp
          ((fderiv ℝ
            (Graph.evalVec
              (Γ := ΓElman inputSize hiddenSize)
              (ss := ssElmanCell inputSize hiddenSize)
              (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g)
            (firstCtx x)).comp DfirstCtx)))
      x := by
  let cellEval :=
    Graph.evalVec
      (Γ := ΓElman inputSize hiddenSize)
      (ss := ssElmanCell inputSize hiddenSize)
      (elmanCellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g
  have hFirstEval :
      HasFDerivAt (fun z : E => cellEval (firstCtx z))
        ((fderiv ℝ cellEval (firstCtx x)).comp DfirstCtx) x :=
    (elmanCell_eval_hasFDerivAt (inputSize := inputSize) (hiddenSize := hiddenSize)
      cell (firstCtx x)).comp x hFirstCtx
  have hSecondCtxAfterFirst :
      HasFDerivAt (fun z : E => secondCtx (cellEval (firstCtx z)))
        (DsecondCtx.comp ((fderiv ℝ cellEval (firstCtx x)).comp DfirstCtx)) x :=
    hSecondCtx.comp x hFirstEval
  simpa [cellEval] using
    (elmanCell_eval_hasFDerivAt (inputSize := inputSize) (hiddenSize := hiddenSize)
      cell (secondCtx (cellEval (firstCtx x)))).comp x hSecondCtxAfterFirst

end

end Recurrent
end Autograd
end Proofs
