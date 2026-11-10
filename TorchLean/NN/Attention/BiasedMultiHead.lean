/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Attention.MultiHead

/-!
# Additive-Bias Multi-Head Attention Core

`biasedCoreDGraph` is the head-wise attention core with a fixed score bias, as a proof-carrying
graph. The surrounding projection, split and merge stages live in `MultiHead.lean`,
and this graph is a reusable additive-bias replacement for the score, probability and value part
of that graph:

`softmax(c · QKᵀ + bias) V`.

Its bundled certificate, read at the input with `.at`, gives the reverse-mode theorem through
`Tape.Graph.backpropVec_eq_fderiv`. Read at the projected input, it gives the composition
with a differentiable projection/split stage through `Tape.GraphData.hasFDerivAt_evalVec_comp`.

The bias has shape `(heads × seq × seq)` and is meant for finite additive score biases. Boolean
attention masks in the spec and runtime path work differently: they use hard masking, where a
blocked entry has a zero softmax numerator.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open scoped BigOperators

noncomputable section

namespace MultiHeadAttention

open Tape (DGraph)


/-- Context for a fixed-bias attention core: `[Q_heads, Kᵀ_heads, V_heads]`. -/
abbrev biasedCoreCtx (n numHeads headDim : Nat) : List Shape :=
  [ headsShape n numHeads headDim
  , ktShape n numHeads headDim
  , headsShape n numHeads headDim
  ]

/-- Saved tensors for the fixed-bias multi-head attention core. -/
abbrev ssBiasedCore (n numHeads headDim : Nat) : List Shape :=
  [ scoresShape n numHeads       -- QKᵀ
  , scoresShape n numHeads       -- scaled scores
  , scoresShape n numHeads       -- scaled scores plus fixed bias
  , scoresShape n numHeads       -- probabilities
  , headsShape n numHeads headDim -- per-head output
  ]

/-- Query-head index in the fixed-bias attention context. -/
def idxBiasedCoreQ {n numHeads headDim : Nat} {ss : List Shape} :
    Idx (biasedCoreCtx n numHeads headDim ++ ss) (headsShape n numHeads headDim) :=
  ⟨⟨0, by simp [biasedCoreCtx]⟩, by simp [biasedCoreCtx]⟩

/-- Transposed-key index in the fixed-bias attention context. -/
def idxBiasedCoreKt {n numHeads headDim : Nat} {ss : List Shape} :
    Idx (biasedCoreCtx n numHeads headDim ++ ss) (ktShape n numHeads headDim) :=
  ⟨⟨1, by simp [biasedCoreCtx]⟩, by simp [biasedCoreCtx, ktShape]⟩

/-- Value-head index in the fixed-bias attention context. -/
def idxBiasedCoreV {n numHeads headDim : Nat} {ss : List Shape} :
    Idx (biasedCoreCtx n numHeads headDim ++ ss) (headsShape n numHeads headDim) :=
  ⟨⟨2, by simp [biasedCoreCtx]⟩, by simp [biasedCoreCtx]⟩

/--
Proof-carrying fixed-bias attention core.

The fixed `bias` is added after scaling the score tensor and before the row-wise softmax.
-/
def biasedCoreDGraph {n numHeads headDim : Nat}
    (c : ℝ) (bias : Vec (Shape.size (scoresShape n numHeads)) := 0) :
    DGraph (biasedCoreCtx n numHeads headDim) (ssBiasedCore n numHeads headDim) := by
  classical
  let dg0 : DGraph (biasedCoreCtx n numHeads headDim) [] := DGraph.nil

  let nodeScores :
      Tape.LawfulNode (α := ℝ) Unit (biasedCoreCtx n numHeads headDim) (scoresShape n numHeads) :=
    Tape.RealNode.Batched.matmul
      (Γ := biasedCoreCtx n numHeads headDim)
      (h := numHeads) (m := n) (n := headDim) (p := n)
      (A := idxBiasedCoreQ (n := n) (numHeads := numHeads) (headDim := headDim) (ss := []))
      (B := idxBiasedCoreKt (n := n) (numHeads := numHeads) (headDim := headDim) (ss := []))
  let dg1 :=
    DGraph.snoc (dg := dg0) (node := nodeScores)
      (hn := Tape.RealNode.Batched.matmulFDeriv
        (Γ := biasedCoreCtx n numHeads headDim)
        (h := numHeads) (m := n) (n := headDim) (p := n)
        (A := idxBiasedCoreQ (n := n) (numHeads := numHeads) (headDim := headDim) (ss := []))
        (B := idxBiasedCoreKt (n := n) (numHeads := numHeads) (headDim := headDim) (ss := [])))

  let idxScores :
      Idx (biasedCoreCtx n numHeads headDim ++ [scoresShape n numHeads])
        (scoresShape n numHeads) :=
    Idx.last (Γ := biasedCoreCtx n numHeads headDim) (ss := []) (τ := scoresShape n numHeads)
  let nodeScaled :
      Tape.LawfulNode (α := ℝ) Unit (biasedCoreCtx n numHeads headDim ++ [scoresShape n numHeads])
        (scoresShape n numHeads) :=
    Tape.LawfulNode.ofArm (Op.scale (s := scoresShape n numHeads)) rfl
      (Γ := biasedCoreCtx n numHeads headDim ++ [scoresShape n numHeads])
      (idxScores, Tensor.scalar c, ())
      (Arm.adj_of_cert₁ fun xs δ => Op.scale_backward_eq_adjoint xs.1 δ xs.2.1)
  let dg2 :=
    DGraph.snoc (dg := dg1) (node := nodeScaled)
      (hn := Tape.RealNode.FDerivCorrect.ofArm₁
        (Γ := biasedCoreCtx n numHeads headDim ++ [scoresShape n numHeads])
        (s := scoresShape n numHeads) (idx := idxScores)
        (cert := fun x δ => Op.scale_backward_eq_adjoint x δ (Tensor.scalar c)))

  let idxScaled :
      Idx (biasedCoreCtx n numHeads headDim ++ [scoresShape n numHeads, scoresShape n numHeads])
        (scoresShape n numHeads) :=
    Idx.last (Γ := biasedCoreCtx n numHeads headDim) (ss := [scoresShape n numHeads])
      (τ := scoresShape n numHeads)
  let nodeMasked :
      Tape.LawfulNode (α := ℝ) Unit
        (biasedCoreCtx n numHeads headDim ++ [scoresShape n numHeads, scoresShape n numHeads])
        (scoresShape n numHeads) :=
    Tape.RealNode.affine
      (Γ := biasedCoreCtx n numHeads headDim ++ [scoresShape n numHeads, scoresShape n numHeads])
      (sIn := scoresShape n numHeads) (sOut := scoresShape n numHeads)
      idxScaled
      (1 : Vec (Shape.size (scoresShape n numHeads)) →L[ℝ]
        Vec (Shape.size (scoresShape n numHeads)))
      bias
  let dg3 :=
    DGraph.snoc (dg := dg2) (node := nodeMasked)
      (hn := Tape.RealNode.affineFDeriv
        (Γ := biasedCoreCtx n numHeads headDim ++ [scoresShape n numHeads, scoresShape n numHeads])
        (sIn := scoresShape n numHeads) (sOut := scoresShape n numHeads)
        idxScaled
        (1 : Vec (Shape.size (scoresShape n numHeads)) →L[ℝ]
          Vec (Shape.size (scoresShape n numHeads)))
        bias)

  let idxMasked :
      Idx (biasedCoreCtx n numHeads headDim ++
        [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads])
        (scoresShape n numHeads) :=
    Idx.last (Γ := biasedCoreCtx n numHeads headDim)
      (ss := [scoresShape n numHeads, scoresShape n numHeads])
      (τ := scoresShape n numHeads)
  let nodeProbs :
      Tape.LawfulNode (α := ℝ) Unit (biasedCoreCtx n numHeads headDim ++
        [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads])
        (scoresShape n numHeads) :=
    Tape.LawfulNode.ofArm
      (Op.softmax (s := .dim numHeads (.dim n (.dim n .scalar))) ⟨2, Nat.lt_succ_self 2⟩) rfl
      (Γ := biasedCoreCtx n numHeads headDim ++
        [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads])
      (idxMasked, ()) (Arm.adj_of_cert₁ fun xs δ => Op.softmax_backward_eq_adjoint _ xs.1 δ)
  let dg4 :=
    DGraph.snoc (dg := dg3) (node := nodeProbs)
      (hn := Tape.RealNode.FDerivCorrect.ofArm₁
        (Γ := biasedCoreCtx n numHeads headDim ++
          [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads])
        (idx := idxMasked)
        (cert := Op.softmax_backward_eq_adjoint (s := .dim numHeads (.dim n (.dim n .scalar)))
          ⟨2, Nat.lt_succ_self 2⟩))

  let idxProbs :
      Idx (biasedCoreCtx n numHeads headDim ++
        [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads,
          scoresShape n numHeads])
        (scoresShape n numHeads) :=
    Idx.last (Γ := biasedCoreCtx n numHeads headDim)
      (ss := [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads])
      (τ := scoresShape n numHeads)
  let nodeOut :
      Tape.LawfulNode (α := ℝ) Unit (biasedCoreCtx n numHeads headDim ++
        [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads,
          scoresShape n numHeads])
        (headsShape n numHeads headDim) :=
    Tape.RealNode.Batched.matmul
      (Γ := biasedCoreCtx n numHeads headDim ++
        [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads,
          scoresShape n numHeads])
      (h := numHeads) (m := n) (n := n) (p := headDim)
      (A := idxProbs)
      (B := idxBiasedCoreV (n := n) (numHeads := numHeads) (headDim := headDim)
        (ss := [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads,
          scoresShape n numHeads]))
  let dg5 :=
    DGraph.snoc (dg := dg4) (node := nodeOut)
      (hn := Tape.RealNode.Batched.matmulFDeriv
        (Γ := biasedCoreCtx n numHeads headDim ++
          [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads,
            scoresShape n numHeads])
        (h := numHeads) (m := n) (n := n) (p := headDim)
        (A := idxProbs)
        (B := idxBiasedCoreV (n := n) (numHeads := numHeads) (headDim := headDim)
          (ss := [scoresShape n numHeads, scoresShape n numHeads, scoresShape n numHeads,
            scoresShape n numHeads])))

  exact dg5

end MultiHeadAttention

end

end TorchLean
