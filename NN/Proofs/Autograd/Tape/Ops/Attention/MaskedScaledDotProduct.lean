/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Ops.Attention.ScaledDotProduct

/-!
# Additive-Bias Scaled-Dot-Product Attention

This file proves the differentiable fixed-score-bias form:

`softmax(c · QKᵀ + bias) V`.

The `bias` tensor is fixed data. This is useful for relative-position biases or other intentional
finite score shifts. It is not the boolean causal-mask semantics: boolean attention masks in the
spec/runtime path use hard masking, where blocked entries contribute zero softmax numerator.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec

open scoped BigOperators

noncomputable section

namespace Attention

open TapeNodes
open DGraph

/-- Saved tensors for fixed-bias scaled-dot-product attention. -/
abbrev ssMaskedScaledDotProduct (m d : Nat) : List Shape :=
  [ .dim d (.dim m .scalar)           -- Kᵀ
  , .dim m (.dim m .scalar)           -- Q*Kᵀ
  , .dim m (.dim m .scalar)           -- scaled logits
  , .dim m (.dim m .scalar)           -- logits plus fixed mask/bias
  , .dim m (.dim m .scalar)           -- softmax probs
  , .dim m (.dim d .scalar)           -- output
  ]

/--
Scaled dot-product attention with a fixed additive score bias.

The proof follows the unmasked graph with one extra affine identity node between scaling and
softmax.  Because the bias is fixed, its derivative is the identity on the scaled logits.
-/
def maskedScaledDotProductDGraph {m d : Nat}
    (c : ℝ) (bias : Vec (Shape.size (.dim m (.dim m .scalar))) := 0) :
    DGraph (ΓQKV m d) (ssMaskedScaledDotProduct m d) := by
  classical

  let dg0 : DGraph (ΓQKV m d) [] := DGraph.nil

  let nodeKt : Node (ΓQKV m d) (.dim d (.dim m .scalar)) :=
    TapeNodes.matrixTranspose (Γ := ΓQKV m d) (m := m) (n := d) (A := idxK (m := m) (d := d))
  let dg1 :=
    DGraph.snoc (dg := dg0) (node := nodeKt)
      (hn := TapeNodes.matrixTransposeFderiv (Γ := ΓQKV m d) (m := m) (n := d)
        (A := idxK (m := m) (d := d)))

  let idxKt : Idx (ΓQKV m d ++ [.dim d (.dim m .scalar)]) (.dim d (.dim m .scalar)) :=
    Idx.last (Γ := ΓQKV m d) (ss := []) (τ := .dim d (.dim m .scalar))
  let nodeLogits :
      Node (ΓQKV m d ++ [.dim d (.dim m .scalar)]) (.dim m (.dim m .scalar)) :=
    TapeNodes.matmul (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar)])
      (m := m) (n := d) (p := m)
      (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
      (B := idxKt)
  let dg2 :=
    DGraph.snoc (dg := dg1) (node := nodeLogits)
      (hn := TapeNodes.matmulFderiv (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar)])
        (m := m) (n := d) (p := m)
        (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
        (B := idxKt))

  let idxLogits :
      Idx (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Idx.last (Γ := ΓQKV m d) (ss := [.dim d (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeScaled :
      Node (ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    TapeNodes.scale
      (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (s := .dim m (.dim m .scalar)) (idx := idxLogits) c
  let dg3 :=
    DGraph.snoc (dg := dg2) (node := nodeScaled)
      (hn := TapeNodes.scaleFderiv
        (Γ := ΓQKV m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (s := .dim m (.dim m .scalar)) (idx := idxLogits) (c := c))

  let idxScaled :
      Idx
        (ΓQKV m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Idx.last (Γ := ΓQKV m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeMasked :
      Node
        (ΓQKV m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    TapeNodes.affine
      (Γ := ΓQKV m d ++
        [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (sIn := .dim m (.dim m .scalar)) (sOut := .dim m (.dim m .scalar))
      idxScaled (1 : Vec (Shape.size (.dim m (.dim m .scalar))) →L[ℝ]
        Vec (Shape.size (.dim m (.dim m .scalar)))) bias
  let dg4 :=
    DGraph.snoc (dg := dg3) (node := nodeMasked)
      (hn := TapeNodes.affineFderiv
        (Γ := ΓQKV m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (sIn := .dim m (.dim m .scalar)) (sOut := .dim m (.dim m .scalar))
        idxScaled (1 : Vec (Shape.size (.dim m (.dim m .scalar))) →L[ℝ]
          Vec (Shape.size (.dim m (.dim m .scalar)))) bias)

  let idxMasked :
      Idx
        (ΓQKV m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Idx.last (Γ := ΓQKV m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeProbs :
      Node
        (ΓQKV m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    TapeNodes.softmaxLast
      (Γ := ΓQKV m d ++
        [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
        , .dim m (.dim m .scalar)])
      (m := m) (n := m) (idx := idxMasked)
  let dg5 :=
    DGraph.snoc (dg := dg4) (node := nodeProbs)
      (hn := TapeNodes.softmaxLastFderiv
        (Γ := ΓQKV m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar)])
        (m := m) (n := m) (idx := idxMasked))

  let idxProbs :
      Idx
        (ΓQKV m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Idx.last (Γ := ΓQKV m d)
      (ss :=
        [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
        , .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeOut :
      Node
        (ΓQKV m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim d .scalar)) :=
    TapeNodes.matmul
      (Γ := ΓQKV m d ++
        [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
        , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (m := m) (n := m) (p := d)
      (A := idxProbs)
      (B := idxV (m := m) (d := d)
        (ss :=
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar), .dim m (.dim m .scalar)]))
  let dg6 :=
    DGraph.snoc (dg := dg5) (node := nodeOut)
      (hn := TapeNodes.matmulFderiv
        (Γ := ΓQKV m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (m := m) (n := m) (p := d)
        (A := idxProbs)
        (B := idxV (m := m) (d := d)
          (ss :=
            [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
            , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])))

  simpa using dg6

/--
Reverse-mode theorem for finite additive-mask scaled-dot-product attention.
-/
theorem backprop_eq_adjoint_fderiv_maskedScaledDotProduct
    {m d : Nat} (c : ℝ) (bias : Vec (Shape.size (.dim m (.dim m .scalar))) := 0) :
    ∀ (xV : CtxVec (ΓQKV m d))
      (seedV : CtxVec (ΓQKV m d ++ ssMaskedScaledDotProduct m d)),
      Graph.backpropVec (Γ := ΓQKV m d) (ss := ssMaskedScaledDotProduct m d)
          (maskedScaledDotProductDGraph (m := m) (d := d) c bias).g xV seedV
        =
      (fderiv ℝ
          (Graph.evalVec (Γ := ΓQKV m d) (ss := ssMaskedScaledDotProduct m d)
            (maskedScaledDotProductDGraph (m := m) (d := d) c bias).g)
          xV).adjoint seedV :=
  DGraph.backpropVec_eq_adjoint_fderiv
    (dg := maskedScaledDotProductDGraph (m := m) (d := d) c bias)

end Attention

end

end Autograd
end Proofs
