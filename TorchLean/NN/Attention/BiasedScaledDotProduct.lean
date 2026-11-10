/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Attention.ScaledDotProduct

/-!
# Additive-Bias Scaled-Dot-Product Attention

Attention with a fixed additive bias on the scores,

`softmax(c · QKᵀ + bias) V`,

built as a `DGraph` in which every node carries a `RealNode.FDerivCorrect` certificate. The `bias`
tensor is fixed data, which suits relative-position biases and other finite score shifts. Boolean
attention masks in the spec and runtime path work differently: they use hard masking, where a
blocked entry has a zero softmax numerator.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open scoped BigOperators

noncomputable section

namespace Attention

open Tape (DGraph)


/-- Saved tensors for fixed-bias scaled-dot-product attention. -/
abbrev ssBiasedScaledDotProduct (m d : Nat) : List Shape :=
  [ .dim d (.dim m .scalar)           -- Kᵀ
  , .dim m (.dim m .scalar)           -- Q*Kᵀ
  , .dim m (.dim m .scalar)           -- scaled logits
  , .dim m (.dim m .scalar)           -- logits plus fixed bias
  , .dim m (.dim m .scalar)           -- softmax probs
  , .dim m (.dim d .scalar)           -- output
  ]

/--
Scaled dot-product attention with a fixed additive score bias.

The construction repeats the unmasked graph of `scaledDotProductDGraph` and inserts one
`Tape.RealNode.affine` node between scaling and softmax, with the identity as its linear part and
`bias` as its offset. Since the bias is fixed, that node's derivative is the identity on the scaled
logits.
-/
def biasedScaledDotProductDGraph {m d : Nat}
    (c : ℝ) (bias : Vec (Shape.size (Shape.dim m (.dim m .scalar))) := 0) :
    DGraph (qkvCtx m d) (ssBiasedScaledDotProduct m d) := by
  let dg0 : DGraph (qkvCtx m d) [] := DGraph.nil

  let nodeKt : Tape.LawfulNode (α := ℝ) Unit (qkvCtx m d) (.dim d (.dim m .scalar)) :=
    Tape.RealNode.matrixTranspose (Γ := qkvCtx m d) (m := m) (n := d) (A := idxK (m := m) (d := d))
  let dg1 :=
    DGraph.snoc (dg := dg0) (node := nodeKt)
      (hn := Tape.RealNode.matrixTransposeFDeriv (Γ := qkvCtx m d) (m := m) (n := d)
        (A := idxK (m := m) (d := d)))

  let idxKt : Idx (qkvCtx m d ++ [.dim d (.dim m .scalar)]) (.dim d (.dim m .scalar)) :=
    Idx.last (Γ := qkvCtx m d) (ss := []) (τ := .dim d (.dim m .scalar))
  let nodeLogits :
      Tape.LawfulNode (α := ℝ) Unit (qkvCtx m d ++ [.dim d (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Tape.RealNode.matmul (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar)])
      (m := m) (n := d) (p := m)
      (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
      (B := idxKt)
  let dg2 :=
    DGraph.snoc (dg := dg1) (node := nodeLogits)
      (hn := Tape.RealNode.matmulFDeriv (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar)])
        (m := m) (n := d) (p := m)
        (A := idxQ (m := m) (d := d) (ss := [.dim d (.dim m .scalar)]))
        (B := idxKt))

  let idxLogits :
      Idx (qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Idx.last (Γ := qkvCtx m d) (ss := [.dim d (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeScaled :
      Tape.LawfulNode (α := ℝ) Unit
        (qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Tape.LawfulNode.ofArm (Op.scale (s := .dim m (.dim m .scalar))) rfl
      (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (idxLogits, Tensor.scalar c, ())
      (Arm.adj_of_cert₁ fun xs δ => Op.scale_backward_eq_adjoint xs.1 δ xs.2.1)
  let dg3 :=
    DGraph.snoc (dg := dg2) (node := nodeScaled)
      (hn := Tape.RealNode.FDerivCorrect.ofArm₁
        (Γ := qkvCtx m d ++ [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
        (s := .dim m (.dim m .scalar)) (idx := idxLogits)
        (cert := fun x δ => Op.scale_backward_eq_adjoint x δ (Tensor.scalar c)))

  let idxScaled :
      Idx
        (qkvCtx m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Idx.last (Γ := qkvCtx m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeMasked :
      Tape.LawfulNode (α := ℝ) Unit
        (qkvCtx m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Tape.RealNode.affine
      (Γ := qkvCtx m d ++
        [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (sIn := .dim m (.dim m .scalar)) (sOut := .dim m (.dim m .scalar))
      idxScaled (1 : Vec (Shape.size (Shape.dim m (.dim m .scalar))) →L[ℝ]
        Vec (Shape.size (Shape.dim m (.dim m .scalar)))) bias
  let dg4 :=
    DGraph.snoc (dg := dg3) (node := nodeMasked)
      (hn := Tape.RealNode.affineFDeriv
        (Γ := qkvCtx m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (sIn := .dim m (.dim m .scalar)) (sOut := .dim m (.dim m .scalar))
        idxScaled (1 : Vec (Shape.size (Shape.dim m (.dim m .scalar))) →L[ℝ]
          Vec (Shape.size (Shape.dim m (.dim m .scalar)))) bias)

  let idxMasked :
      Idx
        (qkvCtx m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Idx.last (Γ := qkvCtx m d)
      (ss := [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeProbs :
      Tape.LawfulNode (α := ℝ) Unit
        (qkvCtx m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Tape.LawfulNode.ofArm (Op.softmax (s := .dim m (.dim m .scalar)) ⟨1, Nat.one_lt_two⟩) rfl
      (Γ := qkvCtx m d ++
        [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
          .dim m (.dim m .scalar)])
      (idxMasked, ()) (Arm.adj_of_cert₁ fun xs δ => Op.softmax_backward_eq_adjoint _ xs.1 δ)
  let dg5 :=
    DGraph.snoc (dg := dg4) (node := nodeProbs)
      (hn := Tape.RealNode.FDerivCorrect.ofArm₁
        (Γ := qkvCtx m d ++
          [.dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar),
            .dim m (.dim m .scalar)])
        (idx := idxMasked)
        (cert := Op.softmax_backward_eq_adjoint (s := .dim m (.dim m .scalar)) ⟨1, Nat.one_lt_two⟩))

  let idxProbs :
      Idx
        (qkvCtx m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim m .scalar)) :=
    Idx.last (Γ := qkvCtx m d)
      (ss :=
        [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
        , .dim m (.dim m .scalar)])
      (τ := .dim m (.dim m .scalar))
  let nodeOut :
      Tape.LawfulNode (α := ℝ) Unit
        (qkvCtx m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (.dim m (.dim d .scalar)) :=
    Tape.RealNode.matmul
      (Γ := qkvCtx m d ++
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
      (hn := Tape.RealNode.matmulFDeriv
        (Γ := qkvCtx m d ++
          [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
          , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])
        (m := m) (n := m) (p := d)
        (A := idxProbs)
        (B := idxV (m := m) (d := d)
          (ss :=
            [ .dim d (.dim m .scalar), .dim m (.dim m .scalar), .dim m (.dim m .scalar)
            , .dim m (.dim m .scalar), .dim m (.dim m .scalar)])))

  exact dg6

end Attention

end

end TorchLean
